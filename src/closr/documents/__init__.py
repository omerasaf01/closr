"""Organization document loading, vector RAG, and structured profile analysis."""

from __future__ import annotations

from closr.documents.analyzer import OrganizationAnalyzer
from closr.documents.loader import (
    SUPPORTED_EXTENSIONS,
    load_document,
    load_documents,
    split_documents,
)
from closr.documents.tools import create_document_tools
from closr.documents.vectorstore import OrganizationKnowledgeBase

__all__ = [
    "SUPPORTED_EXTENSIONS",
    "OrganizationAnalyzer",
    "OrganizationKnowledgeBase",
    "create_document_tools",
    "load_document",
    "load_documents",
    "split_documents",
]
