"""Configuration models and environment settings for closr."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMProvider(str, Enum):
    """Supported LLM and embedding providers."""

    OPENAI = "openai"
    GOOGLE = "google"


class MCPServerConfig(BaseModel):
    """Configuration for a single Model Context Protocol (MCP) server."""

    transport: Literal["stdio", "streamable_http", "sse"] = "stdio"
    command: str | None = None
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    url: str | None = None
    headers: dict[str, str] = Field(default_factory=dict)
    enabled: bool = True


class ClosrSettings(BaseSettings):
    """Application settings loaded from environment variables and `.env`."""

    model_config = SettingsConfigDict(
        env_prefix="CLOSR_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    openai_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "openai_api_key",
            "OPENAI_API_KEY",
            "CLOSR_OPENAI_API_KEY",
        ),
    )
    google_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "google_api_key",
            "GOOGLE_API_KEY",
            "CLOSR_GOOGLE_API_KEY",
        ),
    )
    default_provider: LLMProvider = LLMProvider.OPENAI
    fallback_provider: LLMProvider | None = LLMProvider.GOOGLE
    openai_chat_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"
    google_chat_model: str = "gemini-2.5-flash"
    google_embedding_model: str = "models/text-embedding-004"
    embedding_provider: LLMProvider | None = None
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    mcp_config_path: Path = Path("mcp_servers.json")
    vectorstore_dir: Path = Path(".closr_data/vectorstore")

    @field_validator(
        "openai_api_key",
        "google_api_key",
        "fallback_provider",
        "embedding_provider",
        mode="before",
    )
    @classmethod
    def _empty_str_to_none(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    def get_api_key(self, provider: LLMProvider) -> str | None:
        """Return the configured API key for the given provider, if any."""
        if provider is LLMProvider.OPENAI:
            return self.openai_api_key
        if provider is LLMProvider.GOOGLE:
            return self.google_api_key
        return None

    def get_available_providers(self) -> list[LLMProvider]:
        """Return all providers that currently have a non-empty API key configured."""
        return [
            provider
            for provider in (LLMProvider.OPENAI, LLMProvider.GOOGLE)
            if self.get_api_key(provider)
        ]


def get_settings() -> ClosrSettings:
    """Load and return a fresh `ClosrSettings` instance."""
    return ClosrSettings()
