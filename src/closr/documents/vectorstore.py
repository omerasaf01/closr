"""In-memory vector store and persistence wrapper for organization RAG."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore, VectorStoreRetriever

from closr.documents.loader import load_documents, split_documents


class OrganizationKnowledgeBase:
    """Vector knowledge base over an organization's internal documents."""

    def __init__(
        self,
        embeddings: Embeddings,
        persist_path: Path | str | None = None,
    ) -> None:
        self.embeddings = embeddings
        self.persist_path: Path | None = (
            Path(persist_path).expanduser() if persist_path is not None else None
        )
        if self.persist_path is not None and self.persist_path.is_file():
            self.store = InMemoryVectorStore.load(
                str(self.persist_path),
                embedding=embeddings,
            )
        else:
            self.store = InMemoryVectorStore(embedding=embeddings)

    def __len__(self) -> int:
        return len(self.store.store)

    @property
    def source_files(self) -> list[str]:
        """Return sorted unique source filenames currently indexed in the store."""
        filenames: set[str] = set()
        for entry in self.store.store.values():
            metadata = entry.get("metadata") or {}
            filename = metadata.get("filename")
            if isinstance(filename, str) and filename.strip():
                filenames.add(filename)
                continue
            source = metadata.get("source")
            if isinstance(source, str) and source.strip():
                filenames.add(Path(source).name)
        return sorted(filenames)

    def add_documents(self, documents: list[Document]) -> list[str]:
        """Add document chunks to the vector store and return assigned IDs."""
        if not documents:
            return []
        return self.store.add_documents(documents)

    def ingest_paths(
        self,
        paths: Path | str | Sequence[Path | str],
        chunk_size: int = 1200,
        chunk_overlap: int = 200,
    ) -> int:
        """Load, split, index, and optionally persist documents from `paths`."""
        documents = load_documents(paths)
        chunks = split_documents(
            documents,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        if chunks:
            self.add_documents(chunks)
        if self.persist_path is not None:
            self.save(self.persist_path)
        return len(chunks)

    def search(self, query: str, k: int = 5) -> list[Document]:
        """Return the top-`k` most relevant document chunks for `query`."""
        if not self.store.store:
            return []
        effective_k = max(1, min(k, len(self.store.store)))
        return self.store.similarity_search(query, k=effective_k)

    def as_retriever(self, k: int = 5) -> VectorStoreRetriever:
        """Return a LangChain `VectorStoreRetriever` configured with `k` results."""
        return self.store.as_retriever(search_kwargs={"k": max(1, k)})

    def save(self, path: Path | str | None = None) -> Path:
        """Persist the vector store to disk as JSON and return the target path."""
        target = (
            Path(path).expanduser()
            if path is not None
            else self.persist_path
        )
        if target is None:
            raise ValueError(
                "No target path specified. Pass `path` or initialize with `persist_path`."
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        self.store.dump(str(target))
        return target

    @classmethod
    def load(
        cls,
        path: Path | str,
        embeddings: Embeddings,
    ) -> OrganizationKnowledgeBase:
        """Load a persisted `OrganizationKnowledgeBase` from disk."""
        source_path = Path(path).expanduser()
        if not source_path.exists():
            raise FileNotFoundError(f"Persisted knowledge base not found: {source_path}")
        instance = cls(embeddings=embeddings, persist_path=None)
        instance.persist_path = source_path
        instance.store = InMemoryVectorStore.load(
            str(source_path),
            embedding=embeddings,
        )
        return instance
