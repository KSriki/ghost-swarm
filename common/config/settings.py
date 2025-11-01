"""Configuration management using Pydantic Settings."""

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # AI Provider Settings
    anthropic_api_key: str = Field(
        description="Anthropic API key for Claude models"
    )
    openai_api_key: str = Field(
        default="",
        description="OpenAI API key or compatible API key",
    )
    openai_api_base: str = Field(
        default="https://api.openai.com/v1",
        description="OpenAI API base URL or compatible endpoint",
    )

    # Application Settings
    environment: Literal["development", "staging", "production"] = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    log_format: Literal["json", "text"] = "json"

    # Redis Configuration
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: str = ""

    # Agent2Agent Communication
    a2a_host: str = "0.0.0.0"
    a2a_port: int = 8765
    a2a_protocol: Literal["ws", "wss"] = "ws"

    # MCP Server Configuration
    mcp_host: str = "0.0.0.0"
    mcp_port: int = 3000

    # RAG Configuration
    chroma_persist_dir: Path = Path("./data/chroma")
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Worker Configuration
    max_workers: int = Field(default=4, ge=1, le=32)
    worker_timeout: int = Field(default=300, ge=30, le=3600)

    @field_validator("chroma_persist_dir", mode="before")
    @classmethod
    def create_chroma_dir(cls, v: str | Path) -> Path:
        """Ensure Chroma directory exists."""
        path = Path(v)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def redis_url(self) -> str:
        """Get Redis connection URL."""
        if self.redis_password:
            return f"redis://:{self.redis_password}@{self.redis_host}:{self.redis_port}/{self.redis_db}"
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"

    @property
    def a2a_url(self) -> str:
        """Get Agent2Agent WebSocket URL."""
        return f"{self.a2a_protocol}://{self.a2a_host}:{self.a2a_port}"

    @property
    def is_production(self) -> bool:
        """Check if running in production."""
        return self.environment == "production"

    @property
    def is_development(self) -> bool:
        """Check if running in development."""
        return self.environment == "development"


# Global settings instance
_settings: Settings | None = None


def get_settings() -> Settings:
    """Get or create the global settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()  # type: ignore[call-arg]
    return _settings


def reload_settings() -> Settings:
    """Reload settings from environment."""
    global _settings
    _settings = Settings()  # type: ignore[call-arg]
    return _settings