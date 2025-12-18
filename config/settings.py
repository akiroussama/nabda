"""
Application settings using Pydantic Settings.

Loads configuration from environment variables and .env file.
"""

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class JiraSettings(BaseSettings):
    """Jira connection settings."""

    model_config = SettingsConfigDict(env_prefix="JIRA_")

    url: str = Field(..., description="Jira instance URL")
    email: str = Field(..., description="Jira user email")
    api_token: str = Field(..., description="Jira API token")
    project_key: str = Field(..., description="Default project key")
    board_id: int = Field(1, description="Default board ID")

    # Custom field mappings
    story_points_field: str = Field("customfield_10016", description="Story points field ID")
    epic_link_field: str = Field("customfield_10014", description="Epic link field ID")
    sprint_field: str = Field("customfield_10020", description="Sprint field ID")

    @field_validator("url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        """Ensure URL doesn't have trailing slash."""
        return v.rstrip("/")


class LLMSettings(BaseSettings):
    """LLM (Gemini) settings."""

    model_config = SettingsConfigDict(env_prefix="")

    google_api_key: str = Field(..., description="Google API key for Gemini")
    llm_model: str = Field("gemini-2.0-flash", description="Gemini model to use")
    temperature: float = Field(0.7, ge=0, le=2, description="Generation temperature")
    max_tokens: int = Field(4096, description="Maximum tokens in response")


class DatabaseSettings(BaseSettings):
    """Database settings."""

    model_config = SettingsConfigDict(env_prefix="")

    database_path: str = Field("data/jira.duckdb", description="Path to DuckDB database")

    @property
    def full_path(self) -> Path:
        """Get absolute path to database."""
        return Path(self.database_path).resolve()


class AppSettings(BaseSettings):
    """Application-wide settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        "INFO", description="Logging level"
    )

    # Sync settings
    sync_interval_minutes: int = Field(30, description="Auto-sync interval in minutes")

    # Privacy
    anonymize_developers: bool = Field(True, description="Pseudonymize developer names")

    # Feature flags
    enable_worklog_sync: bool = Field(True, description="Enable worklog synchronization")
    enable_llm_features: bool = Field(True, description="Enable LLM-powered features")

    # Sub-settings
    jira: JiraSettings = Field(default_factory=JiraSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)


# Global settings instance
_settings: AppSettings | None = None


def get_settings() -> AppSettings:
    """Get or create settings instance."""
    global _settings
    if _settings is None:
        _settings = AppSettings()
    return _settings


def reload_settings() -> AppSettings:
    """Force reload settings from environment."""
    global _settings
    _settings = AppSettings()
    return _settings
