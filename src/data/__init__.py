"""Data module for DuckDB schema, loading, and queries."""

from src.data.loader import DataLoader
from src.data.queries import JiraQueries
from src.data.schema import initialize_database

__all__ = [
    "initialize_database",
    "DataLoader",
    "JiraQueries",
]
