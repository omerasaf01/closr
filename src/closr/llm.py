"""Multi-provider LangChain chat and embedding factory with task routing and fallback support."""

from __future__ import annotations

from typing import Literal

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import SecretStr

from closr.config import ClosrSettings, LLMProvider, get_settings

TaskType = Literal["document_analysis", "audience_discovery", "strategy", "agent"]


class LLMFactory:
    """Factory for creating provider-specific, task-routed, and resilient LangChain models."""

    def __init__(self, settings: ClosrSettings | None = None) -> None:
        self.settings = settings or get_settings()

    def _resolve_provider(
        self,
        preferred: LLMProvider | None,
        allow_fallback: bool = True,
    ) -> tuple[LLMProvider, bool]:
        """Resolve which provider to use and whether a fallback occurred.

        Returns `(resolved_provider, did_fallback)`.
        Raises `ValueError` if the required API key(s) are not configured.
        """
        target = preferred or self.settings.default_provider
        if self.settings.get_api_key(target):
            return target, False

        if not allow_fallback:
            raise ValueError(
                f"API key for provider '{target.value}' is not configured, "
                "and fallback is disabled."
            )

        candidates: list[LLMProvider] = []
        if self.settings.fallback_provider is not None:
            candidates.append(self.settings.fallback_provider)
        candidates.extend(
            [self.settings.default_provider, LLMProvider.OPENAI, LLMProvider.GOOGLE]
        )

        for candidate in candidates:
            if candidate is not target and self.settings.get_api_key(candidate):
                return candidate, True

        raise ValueError(
            "No LLM API keys are configured. Set OPENAI_API_KEY and/or GOOGLE_API_KEY "
            "in your environment or .env file."
        )

    def _build_chat_model(
        self,
        provider: LLMProvider,
        model: str | None = None,
        temperature: float | None = None,
    ) -> BaseChatModel:
        """Instantiate a `BaseChatModel` for a resolved provider that has an API key."""
        api_key = self.settings.get_api_key(provider)
        if not api_key:
            raise ValueError(f"Missing API key for provider '{provider.value}'.")

        effective_temp = self.settings.temperature if temperature is None else temperature
        secret_key = SecretStr(api_key)

        if provider is LLMProvider.OPENAI:
            return ChatOpenAI(
                model=model or self.settings.openai_chat_model,
                api_key=secret_key,
                temperature=effective_temp,
            )

        return ChatGoogleGenerativeAI(
            model=model or self.settings.google_chat_model,
            api_key=secret_key,
            temperature=effective_temp,
        )

    def get_chat_model(
        self,
        provider: LLMProvider | None = None,
        model: str | None = None,
        temperature: float | None = None,
        allow_fallback: bool = True,
    ) -> BaseChatModel:
        """Return a configured chat model for the requested or default provider."""
        resolved, did_fallback = self._resolve_provider(
            preferred=provider,
            allow_fallback=allow_fallback,
        )
        # Avoid passing a provider-specific model override to a different fallback provider.
        effective_model = None if (did_fallback and provider is not None) else model
        return self._build_chat_model(
            provider=resolved,
            model=effective_model,
            temperature=temperature,
        )

    def get_model_for_task(
        self,
        task: TaskType,
        allow_fallback: bool = True,
    ) -> BaseChatModel:
        """Return the most appropriate chat model for a given workflow task."""
        task_preferences: dict[TaskType, LLMProvider] = {
            "document_analysis": LLMProvider.GOOGLE,
            "audience_discovery": self.settings.default_provider,
            "strategy": LLMProvider.OPENAI,
            "agent": LLMProvider.OPENAI,
        }
        return self.get_chat_model(
            provider=task_preferences[task],
            allow_fallback=allow_fallback,
        )

    def get_resilient_chat_model(
        self,
        primary_provider: LLMProvider | None = None,
        temperature: float | None = None,
    ) -> Runnable:
        """Return a chat model with automatic runtime fallback when both providers are configured."""
        resolved_primary, _ = self._resolve_provider(
            preferred=primary_provider,
            allow_fallback=True,
        )
        primary_model = self._build_chat_model(
            provider=resolved_primary,
            temperature=temperature,
        )

        available = self.settings.get_available_providers()
        if len(available) < 2:
            return primary_model

        fallback_candidates = [
            p
            for p in (
                self.settings.fallback_provider,
                LLMProvider.GOOGLE,
                LLMProvider.OPENAI,
            )
            if p is not None and p is not resolved_primary and p in available
        ]
        if not fallback_candidates:
            return primary_model

        fallback_model = self._build_chat_model(
            provider=fallback_candidates[0],
            temperature=temperature,
        )
        return primary_model.with_fallbacks([fallback_model])

    def get_embeddings(self, provider: LLMProvider | None = None) -> Embeddings:
        """Return a configured embeddings model with graceful provider fallback."""
        preferred = provider or self.settings.embedding_provider or self.settings.default_provider
        resolved, _ = self._resolve_provider(preferred=preferred, allow_fallback=True)
        api_key = self.settings.get_api_key(resolved)
        if not api_key:
            raise ValueError(f"Missing API key for embeddings provider '{resolved.value}'.")

        secret_key = SecretStr(api_key)
        if resolved is LLMProvider.OPENAI:
            return OpenAIEmbeddings(
                model=self.settings.openai_embedding_model,
                api_key=secret_key,
            )

        return GoogleGenerativeAIEmbeddings(
            model=self.settings.google_embedding_model,
            google_api_key=secret_key,
        )
