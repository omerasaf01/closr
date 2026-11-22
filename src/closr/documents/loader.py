"""Document loading and chunking utilities for organization intelligence."""

from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from pathlib import Path

import yaml
from docx import Document as DocxDocument
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

SUPPORTED_EXTENSIONS: set[str] = {
    ".md",
    ".txt",
    ".pdf",
    ".docx",
    ".json",
    ".csv",
    ".yaml",
    ".yml",
}


def _base_metadata(resolved_path: Path, ext: str) -> dict[str, str]:
    return {
        "source": str(resolved_path),
        "filename": resolved_path.name,
        "file_type": ext,
    }


def _load_text(resolved_path: Path, metadata: dict[str, str]) -> list[Document]:
    content = resolved_path.read_text(encoding="utf-8")
    if not content.strip():
        return []
    return [Document(page_content=content, metadata=metadata)]


def _load_pdf(resolved_path: Path, metadata: dict[str, str]) -> list[Document]:
    reader = PdfReader(str(resolved_path))
    documents: list[Document] = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            documents.append(
                Document(
                    page_content=text,
                    metadata={**metadata, "page": page_number},
                )
            )
    return documents


def _load_docx(resolved_path: Path, metadata: dict[str, str]) -> list[Document]:
    doc = DocxDocument(str(resolved_path))
    parts: list[str] = []

    for paragraph in doc.paragraphs:
        text = paragraph.text.strip()
        if text:
            parts.append(text)

    for table in doc.tables:
        table_lines: list[str] = []
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells]
            if any(cells):
                table_lines.append(" | ".join(cells))
        if table_lines:
            parts.append("\n".join(table_lines))

    combined = "\n\n".join(parts).strip()
    if not combined:
        return []
    return [Document(page_content=combined, metadata=metadata)]


def _load_json(resolved_path: Path, metadata: dict[str, str]) -> list[Document]:
    raw = resolved_path.read_text(encoding="utf-8")
    if not raw.strip():
        return []
    parsed = json.loads(raw)
    formatted = json.dumps(parsed, indent=2, ensure_ascii=False)
    return [Document(page_content=formatted, metadata=metadata)]


def _load_csv(resolved_path: Path, metadata: dict[str, str]) -> list[Document]:
    with resolved_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))

    if not rows:
        return []

    headers = [header.strip() for header in rows[0]]
    lines: list[str] = []
    for row_index, row in enumerate(rows[1:], start=1):
        if not any(cell.strip() for cell in row):
            continue
        pairs: list[str] = []
        for col_index, cell in enumerate(row):
            value = cell.strip()
            if not value:
                continue
            header = (
                headers[col_index]
                if col_index < len(headers) and headers[col_index]
                else f"col_{col_index + 1}"
            )
            pairs.append(f"{header}: {value}")
        if pairs:
            lines.append(f"Row {row_index}: " + " | ".join(pairs))

    if not lines:
        non_empty_headers = [h for h in headers if h]
        if not non_empty_headers:
            return []
        lines.append("Headers: " + ", ".join(non_empty_headers))

    return [Document(page_content="\n".join(lines), metadata=metadata)]


def _load_yaml(resolved_path: Path, metadata: dict[str, str]) -> list[Document]:
    raw = resolved_path.read_text(encoding="utf-8")
    if not raw.strip():
        return []
    parsed = yaml.safe_load(raw)
    if parsed is None:
        return [Document(page_content=raw.strip(), metadata=metadata)]
    formatted = yaml.safe_dump(parsed, sort_keys=False, allow_unicode=True).strip()
    return [Document(page_content=formatted, metadata=metadata)]


def load_document(path: Path | str) -> list[Document]:
    """Load a single supported organization document into LangChain `Document` objects.

    Raises:
        FileNotFoundError: If `path` does not exist.
        ValueError: If `path` is a directory or has an unsupported file extension.
    """
    resolved_path = Path(path).expanduser().resolve()
    if not resolved_path.exists():
        raise FileNotFoundError(f"Document not found: {resolved_path}")
    if resolved_path.is_dir():
        raise ValueError(
            f"Expected a file path but received directory: {resolved_path}. "
            "Use load_documents() for directories."
        )

    ext = resolved_path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise ValueError(
            f"Unsupported file extension '{ext}' for '{resolved_path.name}'. "
            f"Supported extensions: {supported}"
        )

    metadata = _base_metadata(resolved_path, ext)

    if ext in {".md", ".txt"}:
        return _load_text(resolved_path, metadata)
    if ext == ".pdf":
        return _load_pdf(resolved_path, metadata)
    if ext == ".docx":
        return _load_docx(resolved_path, metadata)
    if ext == ".json":
        return _load_json(resolved_path, metadata)
    if ext == ".csv":
        return _load_csv(resolved_path, metadata)
    return _load_yaml(resolved_path, metadata)


def _iter_directory_files(directory: Path) -> list[Path]:
    """Recursively collect supported, non-hidden files in deterministic sorted order."""
    discovered: list[Path] = []
    for candidate in directory.rglob("*"):
        if not candidate.is_file():
            continue
        relative_parts = candidate.relative_to(directory).parts
        if any(part.startswith(".") for part in relative_parts):
            continue
        if candidate.suffix.lower() in SUPPORTED_EXTENSIONS:
            discovered.append(candidate.resolve())
    return sorted(discovered)


def load_documents(paths: Path | str | Sequence[Path | str]) -> list[Document]:
    """Load documents from a file path, a directory, or a sequence of files/directories."""
    raw_paths: Sequence[Path | str] = (
        [paths] if isinstance(paths, (str, Path)) else paths
    )

    files_to_load: list[Path] = []
    seen_files: set[Path] = set()

    for raw_path in raw_paths:
        resolved = Path(raw_path).expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Path not found: {resolved}")

        if resolved.is_dir():
            for file_path in _iter_directory_files(resolved):
                if file_path not in seen_files:
                    seen_files.add(file_path)
                    files_to_load.append(file_path)
        else:
            if resolved not in seen_files:
                seen_files.add(resolved)
                files_to_load.append(resolved)

    documents: list[Document] = []
    for file_path in files_to_load:
        documents.extend(load_document(file_path))
    return documents


def split_documents(
    documents: list[Document],
    chunk_size: int = 1200,
    chunk_overlap: int = 200,
) -> list[Document]:
    """Split documents into overlapping text chunks suitable for vector indexing."""
    if not documents:
        return []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(documents)
