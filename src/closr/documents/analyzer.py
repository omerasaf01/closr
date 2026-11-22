"""Organization document analysis and structured needs extraction."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import Runnable

from closr.documents.loader import load_documents, split_documents
from closr.documents.vectorstore import OrganizationKnowledgeBase
from closr.schemas import OrganizationProfile

ORGANIZATION_ANALYSIS_SYSTEM_PROMPT = """You are an expert B2B/B2C marketing strategist and business analyst.
Your task is to deeply analyze an organization's internal documents and extract a thorough, evidence-based `OrganizationProfile`.

Focus on extracting:
1. Core identity: official organization name, primary industry/vertical, and a crisp executive summary.
2. Products or services and the core value proposition.
3. Strategic business and growth goals stated or implied in the materials.
4. Explicit and implicit marketing needs, capability gaps, funnel bottlenecks, and growth priorities.
5. Customer pain points addressed by the organization's offerings.
6. Competitive differentiators, proof points, and brand tone / voice guidelines.
7. Geographic focus, budget limits, compliance rules, or operational constraints.
8. Existing customer segments, buyer personas, or ideal customer profile (ICP) hints.

Be concrete and grounded strictly in the provided documents. Do not invent unsupported facts or metrics.
"""

_TRUNCATION_MARKER = "\n...[truncated due to context limit]"


def _extract_filename(doc: Document) -> str:
    filename = doc.metadata.get("filename")
    if isinstance(filename, str) and filename.strip():
        return filename.strip()
    source = doc.metadata.get("source")
    if isinstance(source, str) and source.strip():
        return Path(source).name
    return "unknown"


class OrganizationAnalyzer:
    """Extracts a structured `OrganizationProfile` from internal organization documents."""

    def __init__(
        self,
        llm: BaseChatModel | Runnable[Any, Any],
        knowledge_base: OrganizationKnowledgeBase | None = None,
        max_context_chars: int = 60_000,
    ) -> None:
        if max_context_chars <= 0:
            raise ValueError("max_context_chars must be a positive integer.")
        self.llm = llm
        self.knowledge_base = knowledge_base
        self.max_context_chars = max_context_chars

    def _build_context(self, documents: list[Document]) -> tuple[str, list[str]]:
        if not documents:
            raise ValueError("Cannot analyze an empty list of documents.")

        unique_filenames: list[str] = []
        seen_filenames: set[str] = set()
        sections: list[str] = []
        current_length = 0

        for doc in documents:
            body = doc.page_content.strip()
            if not body:
                continue

            filename = _extract_filename(doc)
            page = doc.metadata.get("page")
            page_suffix = f" (page {page})" if page is not None else ""
            header = f"=== Document: {filename}{page_suffix} ===\n"
            section = f"{header}{body}"
            separator_len = 2 if sections else 0
            projected = current_length + separator_len + len(section)

            if projected <= self.max_context_chars:
                sections.append(section)
                current_length = projected
                if filename not in seen_filenames:
                    seen_filenames.add(filename)
                    unique_filenames.append(filename)
                continue

            remaining = self.max_context_chars - current_length - separator_len
            if remaining > len(_TRUNCATION_MARKER):
                truncated = (
                    section[: remaining - len(_TRUNCATION_MARKER)].rstrip()
                    + _TRUNCATION_MARKER
                )
                sections.append(truncated)
                if filename not in seen_filenames:
                    seen_filenames.add(filename)
                    unique_filenames.append(filename)
            elif not sections and remaining > 0:
                sections.append(section[: self.max_context_chars])
                if filename not in seen_filenames:
                    seen_filenames.add(filename)
                    unique_filenames.append(filename)
            break

        if not sections:
            raise ValueError("Provided documents contain no non-empty text content.")

        return "\n\n".join(sections), unique_filenames

    def _build_messages(self, context: str) -> list[BaseMessage]:
        return [
            SystemMessage(content=ORGANIZATION_ANALYSIS_SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    "Analyze the following internal organization documents and extract "
                    "a complete OrganizationProfile.\n\n"
                    f"{context}"
                )
            ),
        ]

    def _get_structured_runnable(self) -> Runnable[Any, Any]:
        with_structured = getattr(self.llm, "with_structured_output", None)
        if callable(with_structured):
            return with_structured(OrganizationProfile)

        # Support RunnableWithFallbacks wrapping BaseChatModels.
        primary = getattr(self.llm, "runnable", None)
        fallbacks = getattr(self.llm, "fallbacks", None)
        if primary is not None and callable(getattr(primary, "with_structured_output", None)):
            structured_primary = primary.with_structured_output(OrganizationProfile)
            structured_fallbacks = [
                fb.with_structured_output(OrganizationProfile)
                for fb in (fallbacks or ())
                if callable(getattr(fb, "with_structured_output", None))
            ]
            if structured_fallbacks:
                return structured_primary.with_fallbacks(structured_fallbacks)
            return structured_primary

        return self.llm

    @staticmethod
    def _finalize_profile(
        raw_result: Any,
        source_filenames: list[str],
    ) -> OrganizationProfile:
        if isinstance(raw_result, OrganizationProfile):
            profile = raw_result
        elif isinstance(raw_result, dict):
            profile = OrganizationProfile.model_validate(raw_result)
        elif isinstance(raw_result, BaseMessage):
            content = raw_result.content
            if not isinstance(content, str):
                raise TypeError(
                    f"Expected string message content for OrganizationProfile, got {type(content)!r}"
                )
            profile = OrganizationProfile.model_validate_json(content)
        elif isinstance(raw_result, str):
            profile = OrganizationProfile.model_validate_json(raw_result)
        else:
            raise TypeError(
                f"Expected OrganizationProfile from structured LLM output, got {type(raw_result)!r}"
            )

        merged_sources = list(dict.fromkeys([*profile.source_documents, *source_filenames]))
        return profile.model_copy(update={"source_documents": merged_sources})

    def analyze_documents(self, documents: list[Document]) -> OrganizationProfile:
        """Analyze pre-loaded `Document` objects and return an `OrganizationProfile`."""
        context, source_filenames = self._build_context(documents)
        messages = self._build_messages(context)
        structured_llm = self._get_structured_runnable()
        result = structured_llm.invoke(messages)
        return self._finalize_profile(result, source_filenames)

    async def aanalyze_documents(
        self,
        documents: list[Document],
    ) -> OrganizationProfile:
        """Asynchronously analyze pre-loaded `Document` objects into an `OrganizationProfile`."""
        context, source_filenames = self._build_context(documents)
        messages = self._build_messages(context)
        structured_llm = self._get_structured_runnable()
        result = await structured_llm.ainvoke(messages)
        return self._finalize_profile(result, source_filenames)

    def _ingest_to_knowledge_base(self, documents: list[Document]) -> None:
        if self.knowledge_base is None:
            return
        chunks = split_documents(documents)
        if chunks:
            self.knowledge_base.add_documents(chunks)
        if self.knowledge_base.persist_path is not None:
            self.knowledge_base.save(self.knowledge_base.persist_path)

    def analyze_paths(
        self,
        paths: Path | str | Sequence[Path | str],
    ) -> OrganizationProfile:
        """Load documents from `paths`, optionally index into `knowledge_base`, and extract profile."""
        documents = load_documents(paths)
        if not documents:
            raise ValueError(f"No analyzable document content found at: {paths}")
        self._ingest_to_knowledge_base(documents)
        return self.analyze_documents(documents)

    async def aanalyze_paths(
        self,
        paths: Path | str | Sequence[Path | str],
    ) -> OrganizationProfile:
        """Async version of `analyze_paths`."""
        documents = load_documents(paths)
        if not documents:
            raise ValueError(f"No analyzable document content found at: {paths}")
        self._ingest_to_knowledge_base(documents)
        return await self.aanalyze_documents(documents)
