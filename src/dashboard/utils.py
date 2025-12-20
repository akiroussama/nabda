"""Shared utilities for the dashboard."""

import streamlit as st
import duckdb
from pathlib import Path

def get_connection():
    """Get database connection with caching."""
    db_path = Path("data/jira.duckdb")
    if not db_path.exists():
        raise FileNotFoundError("Database not found. Run 'jira-copilot init' first.")
    return duckdb.connect(str(db_path))


def get_predictor():
    """Get unified predictor with caching."""
    from src.models.predictor import Predictor
    return Predictor.from_model_dir("models")


def get_intelligence():
    """Get intelligence orchestrator."""
    from src.intelligence.orchestrator import JiraIntelligence
    return JiraIntelligence()


def apply_custom_css():
    """Apply custom CSS to the page."""
    st.markdown("""
    <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: 700;
            color: #1f77b4;
            margin-bottom: 1rem;
        }
        .metric-card {
            background-color: #f0f2f6;
            border-radius: 10px;
            padding: 1rem;
            margin: 0.5rem 0;
        }
        .risk-high { color: #ff4b4b; font-weight: bold; }
        .risk-medium { color: #ffa726; font-weight: bold; }
        .risk-low { color: #66bb6a; font-weight: bold; }
        .stMetric > div { background-color: #f8f9fa; border-radius: 8px; padding: 10px; }
    </style>
    """, unsafe_allow_html=True)
