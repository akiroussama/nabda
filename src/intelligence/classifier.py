"""
Work Classification Engine for Strategic Execution Gap analysis.
Classifies Jira tickets into strategic categories using NLP and embeddings.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from loguru import logger
import pickle
from pathlib import Path

# Lazy imports for GPU-optional dependencies
SentenceTransformer = None
util = None
xgb = None

def _load_sentence_transformers():
    """Lazy load sentence_transformers to handle missing CUDA gracefully."""
    global SentenceTransformer, util
    if SentenceTransformer is None:
        try:
            import os
            # Force CPU mode to avoid CUDA issues
            os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
            from sentence_transformers import SentenceTransformer as ST, util as st_util
            SentenceTransformer = ST
            util = st_util
        except Exception as e:
            logger.warning(f"Could not load sentence_transformers: {e}")
            SentenceTransformer = None
            util = None
    return SentenceTransformer, util

def _load_xgboost():
    """Lazy load xgboost."""
    global xgb
    if xgb is None:
        try:
            import xgboost as xgb_module
            xgb = xgb_module
        except ImportError:
            logger.warning("xgboost not available")
            xgb = None
    return xgb

# Strategic Categories
CATEGORY_NEW_VALUE = "New Value"
CATEGORY_MAINTENANCE = "Maintenance"
CATEGORY_TECH_DEBT = "Tech Debt"
CATEGORY_FIREFIGHTING = "Firefighting"
CATEGORY_DEPENDENCY = "Dependency/Blocked"
CATEGORY_REWORK = "Rework"

ALL_CATEGORIES = [
    CATEGORY_NEW_VALUE,
    CATEGORY_MAINTENANCE,
    CATEGORY_TECH_DEBT,
    CATEGORY_FIREFIGHTING,
    CATEGORY_DEPENDENCY,
    CATEGORY_REWORK
]

# Archetypes for Zero-Shot Classification (when no trained model exists)
ARCHETYPES = {
    CATEGORY_NEW_VALUE: [
        "Implement new feature",
        "Add user capability",
        "Create new dashboard",
        "Launch product integration",
        "User story for new functionality"
    ],
    CATEGORY_MAINTENANCE: [
        "Update dependencies",
        "Routine maintenance",
        "Bump version",
        "Documentation update",
        "Minor tweak",
        "Config change"
    ],
    CATEGORY_TECH_DEBT: [
        "Refactor legacy code",
        "Clean up database",
        "Improve code quality",
        "Remove unused endpoint",
        "Architecture migration",
        "Switch to new library"
    ],
    CATEGORY_FIREFIGHTING: [
        "Fix critical bug",
        "Production incident",
        "System outage",
        "Urgent patch",
        "Hotfix deployment",
        "Crash investigation",
        "Performance degradation"
    ],
    CATEGORY_DEPENDENCY: [
        "Waiting for API",
        "Blocked by team",
        "Dependency update required",
        "Third-party issue"
    ],
    CATEGORY_REWORK: [
        "Revert changes",
        "Redo implementation",
        "Fix regression from last sprint",
        "Address code review feedback"
    ]
}

class WorkClassifier:
    """
    Classifies Jira tickets into work categories using embeddings and XGBoost/Heuristics.
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self._embedder = None
        self._classifier = None
        self._archetype_embeddings: Dict[str, np.ndarray] = {}
        self._embedder_available = None  # Lazy check

    def _get_embedder(self):
        """Get sentence transformer embedder, returns None if not available."""
        if self._embedder_available is False:
            return None

        if self._embedder is None:
            ST, _ = _load_sentence_transformers()
            if ST is None:
                self._embedder_available = False
                logger.warning("Sentence transformers not available - using fallback classification")
                return None

            logger.info("Loading sentence-transformers model...")
            try:
                self._embedder = ST('all-MiniLM-L6-v2')
                self._embedder_available = True
            except Exception as e:
                logger.warning(f"Could not load embedding model: {e}")
                self._embedder_available = False
                return None

        return self._embedder

    def _generate_archetype_embeddings(self):
        """Pre-compute embeddings for archetypes for zero-shot classification."""
        if self._archetype_embeddings:
            return

        embedder = self._get_embedder()
        if embedder is None:
            return

        for category, phrases in ARCHETYPES.items():
            self._archetype_embeddings[category] = embedder.encode(phrases)

    def generate_embeddings(self, texts: List[str]) -> Optional[np.ndarray]:
        """Generate vectors for a list of texts. Returns None if embedder unavailable."""
        embedder = self._get_embedder()
        if embedder is None:
            return None
        return embedder.encode(texts, show_progress_bar=True)

    def _keyword_classify(self, text: str) -> Tuple[str, float]:
        """Fallback keyword-based classification when embeddings unavailable."""
        text_lower = text.lower()

        # Firefighting keywords (highest priority)
        firefighting_keywords = ['bug', 'fix', 'incident', 'crash', 'urgent', 'hotfix', 'broken', 'error', 'outage']
        if any(kw in text_lower for kw in firefighting_keywords):
            return CATEGORY_FIREFIGHTING, 0.7

        # Tech debt keywords
        tech_debt_keywords = ['refactor', 'cleanup', 'debt', 'legacy', 'migrate', 'upgrade', 'deprecate']
        if any(kw in text_lower for kw in tech_debt_keywords):
            return CATEGORY_TECH_DEBT, 0.7

        # Maintenance keywords
        maintenance_keywords = ['update', 'maintenance', 'version', 'bump', 'dependency', 'routine']
        if any(kw in text_lower for kw in maintenance_keywords):
            return CATEGORY_MAINTENANCE, 0.6

        # Rework keywords
        rework_keywords = ['revert', 'redo', 'again', 'retry', 'regression']
        if any(kw in text_lower for kw in rework_keywords):
            return CATEGORY_REWORK, 0.6

        # Dependency/blocked keywords
        blocked_keywords = ['blocked', 'waiting', 'depends', 'dependency']
        if any(kw in text_lower for kw in blocked_keywords):
            return CATEGORY_DEPENDENCY, 0.6

        # Default to new value
        return CATEGORY_NEW_VALUE, 0.5

    def classify_tickets(self, df: pd.DataFrame, text_column: str = "summary") -> pd.DataFrame:
        """
        Classify tickets in the DataFrame.
        Adds/Updates 'predicted_category' and 'confidence' columns.
        """
        if df.empty:
            return df

        logger.info(f"Classifying {len(df)} tickets...")

        texts = df[text_column].fillna("").tolist()

        # Try embedding-based classification first
        embeddings = self.generate_embeddings(texts)

        categories = []
        confidences = []
        shadow_flags = []

        if embeddings is not None and self._archetype_embeddings is not None:
            # Use Zero-Shot / Similarity based on Archetypes
            self._generate_archetype_embeddings()
            _, st_util = _load_sentence_transformers()

            if st_util and self._archetype_embeddings:
                for i, embedding in enumerate(embeddings):
                    best_cat = CATEGORY_NEW_VALUE
                    max_score = -1.0

                    for category, arch_embeds in self._archetype_embeddings.items():
                        scores = st_util.cos_sim(embedding, arch_embeds)
                        score = float(scores.max())

                        if score > max_score:
                            max_score = score
                            best_cat = category

                    categories.append(best_cat)
                    confidences.append(max_score)

                    # Shadow Work Detection
                    is_shadow = False
                    original_type = df.iloc[i].get('issue_type', '').lower() if 'issue_type' in df.columns else ''
                    if best_cat in [CATEGORY_MAINTENANCE, CATEGORY_FIREFIGHTING, CATEGORY_TECH_DEBT]:
                        if 'story' in original_type or 'feature' in original_type or 'epic' in original_type:
                            if max_score > 0.4:
                                is_shadow = True
                    shadow_flags.append(is_shadow)
            else:
                # Fallback to keyword classification
                embeddings = None

        if embeddings is None or not categories:
            # Use keyword-based fallback
            logger.info("Using keyword-based classification fallback")
            categories = []
            confidences = []
            shadow_flags = []

            for i, text in enumerate(texts):
                cat, conf = self._keyword_classify(text)
                categories.append(cat)
                confidences.append(conf)

                # Shadow Work Detection
                is_shadow = False
                original_type = df.iloc[i].get('issue_type', '').lower() if 'issue_type' in df.columns else ''
                if cat in [CATEGORY_MAINTENANCE, CATEGORY_FIREFIGHTING, CATEGORY_TECH_DEBT]:
                    if 'story' in original_type or 'feature' in original_type or 'epic' in original_type:
                        if conf > 0.5:
                            is_shadow = True
                shadow_flags.append(is_shadow)

        df['predicted_category'] = categories
        df['classification_confidence'] = confidences
        df['is_shadow_work'] = shadow_flags

        return df

    def train(self, df: pd.DataFrame, label_column: str):
        """
        Train a classifier on labeled data.
        """
        # Placeholder for future implementation
        logger.info("Training functionality not fully implemented yet.")
        pass
