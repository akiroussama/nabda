"""
Work Classification Engine for Strategic Execution Gap analysis.
Classifies Jira tickets into strategic categories using NLP and embeddings.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from loguru import logger
from sentence_transformers import SentenceTransformer, util
import xgboost as xgb
import pickle
from pathlib import Path

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
        self._embedder: Optional[SentenceTransformer] = None
        self._classifier: Optional[xgb.Booster] = None
        self._archetype_embeddings: Dict[str, np.ndarray] = {}
        
    def _get_embedder(self) -> SentenceTransformer:
        if self._embedder is None:
            logger.info("Loading sentence-transformers model...")
            # using a small, fast model
            self._embedder = SentenceTransformer('all-MiniLM-L6-v2')
        return self._embedder

    def _generate_archetype_embeddings(self):
        """Pre-compute embeddings for archetypes for zero-shot classification."""
        if self._archetype_embeddings:
            return
            
        embedder = self._get_embedder()
        for category, phrases in ARCHETYPES.items():
            self._archetype_embeddings[category] = embedder.encode(phrases)

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate vectors for a list of texts."""
        embedder = self._get_embedder()
        return embedder.encode(texts, show_progress_bar=True)

    def classify_tickets(self, df: pd.DataFrame, text_column: str = "summary") -> pd.DataFrame:
        """
        Classify tickets in the DataFrame. 
        Adds/Updates 'predicted_category' and 'confidence' columns.
        """
        if df.empty:
            return df
        
        logger.info(f"Classifying {len(df)} tickets...")
        
        # 1. Combine summary and description for better context if available
        # But for now, let's stick to summary or passed column to keep it simple/fast
        texts = df[text_column].fillna("").tolist()
        
        # 2. Heuristic/Keyword First (Fastest)
        # We can implement simple keyword matching overrides here
        
        # 3. Embedding-based Classification
        embeddings = self.generate_embeddings(texts)
        
        # If we have a trained XGBoost model, use it
        if self._classifier:
            # TODO: Implement XGBoost prediction using embeddings as features
            pass
        else:
            # Use Zero-Shot / Similarity based on Archetypes
            self._generate_archetype_embeddings()
            categories = []
            confidences = []
            shadow_flags = []
            
            for i, embedding in enumerate(embeddings):
                best_cat = CATEGORY_NEW_VALUE # Default
                max_score = -1.0
                
                # Check against each category's archetypes
                for category, arch_embeds in self._archetype_embeddings.items():
                    # Calculate max cosine similarity with any phrase in this category
                    # embedding is (384,), arch_embeds is (N, 384)
                    scores = util.cos_sim(embedding, arch_embeds)
                    score = float(scores.max())
                    
                    if score > max_score:
                        max_score = score
                        best_cat = category
                
                categories.append(best_cat)
                confidences.append(max_score)
                
                # Shadow Work Detection
                # Check if it was labeled "New Feature" or "Story" but classified as "Maintenance" or "Firefighting"
                # This requires knowing the original issue type, let's assume valid input df has 'issue_type'
                is_shadow = False
                original_type = df.iloc[i].get('issue_type', '').lower()
                if best_cat in [CATEGORY_MAINTENANCE, CATEGORY_FIREFIGHTING, CATEGORY_TECH_DEBT]:
                    if 'story' in original_type or 'feature' in original_type or 'epic' in original_type:
                        # High similarity to maintenance/firefighting despite being a Feature
                        if max_score > 0.4: # Threshold
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
