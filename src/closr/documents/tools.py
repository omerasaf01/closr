"""LangChain tools exposing organization document analysis and RAG search."""

from __future__ import annotations

from pathlib import Path

from langchain_core.tools import BaseTool, tool

from closr.documents.analyzer import OrganizationAnalyzer
from closr.documents.vectorstore import OrganizationKnowledgeBase


def create_document_tools(
    analyzer: OrganizationAnalyzer,
    knowledge_base: OrganizationKnowledgeBase | None = None,
) -> list[BaseTool]:
    """Create LangChain tools for analyzing organization docs and querying the knowledge base."""
    kb = knowledge_base if knowledge_base is not None else analyzer.knowledge_base

    @tool
    def analyze_organization_documents(path: str) -> str:
        """Analyze an organization's internal file or directory of documents and return a structured OrganizationProfile JSON."""
        try:
            profile = analyzer.analyze_paths(path)
        except (FileNotFoundError, IsADirectoryError, ValueError, OSError) as exc:
            return (
                f"Error analyzing organization documents at '{path}': {exc}. "
                "Verify that the path exists and contains supported formats "
                "(.md, .txt, .pdf, .docx, .json, .csv, .yaml, .yml)."
            )
        return profile.model_dump_json(indent=2)

    @tool
    def search_organization_knowledge(query: str, k: int = 4) -> str:
        """Search the indexed organization knowledge base for relevant facts, metrics, products, or constraints."""
        if kb is None:
            return (
                "Organization knowledge base is not configured. "
                "Initialize a knowledge base before searching."
            )
        if len(kb) == 0:
            return (
                "Organization knowledge base is empty. "
                "Run analyze_organization_documents or ingest documents first."
            )
        if not query.strip():
            return "Please provide a non-empty search query."

        results = kb.search(query, k=max(1, k))
        if not results:
            return f"No relevant excerpts found in organization knowledge base for '{query}'."

        formatted_chunks: list[str] = []
        for doc in results:
            filename = str(
                doc.metadata.get("filename")
                or Path(str(doc.metadata.get("source", "unknown"))).name
                or "unknown"
            )
            page = doc.metadata.get("page")
            header = (
                f"[source: {filename} (page {page})]"
                if page is not None
                else f"[source: {filename}]"
            )
            formatted_chunks.append(f"{header}\n{doc.page_content.strip()}")
        return "\n\n---\n\n".join(formatted_chunks)

    return [analyze_organization_documents, search_organization_knowledge]
