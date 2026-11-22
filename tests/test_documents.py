"""Offline unit tests for organization document loading, RAG vector store, analyzer, and tools."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from docx import Document as DocxDocument
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable, RunnableLambda

from closr.documents import (
    SUPPORTED_EXTENSIONS,
    OrganizationAnalyzer,
    OrganizationKnowledgeBase,
    create_document_tools,
    load_document,
    load_documents,
    split_documents,
)
from closr.schemas import OrganizationProfile

SAMPLE_DOCS_DIR = Path(__file__).resolve().parent.parent / "examples" / "sample_org_docs"


class FakeStructuredRunnable(Runnable[Any, OrganizationProfile]):
    """Deterministic fake runnable returning a pre-configured OrganizationProfile."""

    def __init__(self, profile: OrganizationProfile, recorder: list[list[BaseMessage]]) -> None:
        self._profile = profile
        self._recorder = recorder

    def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> OrganizationProfile:
        self._recorder.append(list(input))
        return self._profile

    async def ainvoke(
        self,
        input: Any,
        config: Any = None,
        **kwargs: Any,
    ) -> OrganizationProfile:
        self._recorder.append(list(input))
        return self._profile


class FakeStructuredLLM:
    """Fake chat model exposing `with_structured_output` without network calls."""

    def __init__(self, profile: OrganizationProfile) -> None:
        self.profile = profile
        self.received_messages: list[list[BaseMessage]] = []
        self.requested_schemas: list[Any] = []

    def with_structured_output(self, schema: Any) -> FakeStructuredRunnable:
        self.requested_schemas.append(schema)
        return FakeStructuredRunnable(self.profile, self.received_messages)


def _sample_profile() -> OrganizationProfile:
    return OrganizationProfile(
        organization_name="NovaFlow Analytics",
        industry="B2B SaaS / Revenue Intelligence",
        summary="AI-powered RevOps and churn prediction platform for mid-market SaaS.",
        products_or_services=["NovaFlow Pulse", "NovaFlow Retain", "NovaFlow Sync"],
        value_proposition="Predictable net revenue retention in 48 hours.",
        business_goals=["Grow qualified pipeline by 65%", "Expand EMEA ARR share to 35%"],
        marketing_needs=["LinkedIn ABM for RevOps", "EMEA GDPR trust positioning"],
        pain_points_addressed=["Unexplained logo churn", "Fragmented CRM and billing data"],
        brand_tone="Authoritative, pragmatic, and analytical",
        differentiators=["Explainable 90-day churn scoring", "48-hour zero-ETL onboarding"],
        geographic_focus=["North America", "EMEA"],
        budget_or_constraints=["$180,000 per quarter", "Strict GDPR compliance"],
        existing_audience_hints=["VP of RevOps at Series B-D SaaS ($10M-$120M ARR)"],
        source_documents=[],
    )


# ---------------------------------------------------------------------------
# Loader & Splitter Tests
# ---------------------------------------------------------------------------


def test_supported_extensions_constant() -> None:
    assert SUPPORTED_EXTENSIONS == {
        ".md",
        ".txt",
        ".pdf",
        ".docx",
        ".json",
        ".csv",
        ".yaml",
        ".yml",
    }


def test_load_document_markdown_and_txt(tmp_path: Path) -> None:
    md_file = tmp_path / "overview.md"
    md_file.write_text("# Company\nWe build analytics.", encoding="utf-8")

    txt_file = tmp_path / "notes.txt"
    txt_file.write_text("Plain text brief.", encoding="utf-8")

    md_docs = load_document(md_file)
    assert len(md_docs) == 1
    assert md_docs[0].page_content == "# Company\nWe build analytics."
    assert md_docs[0].metadata == {
        "source": str(md_file.resolve()),
        "filename": "overview.md",
        "file_type": ".md",
    }

    txt_docs = load_document(txt_file)
    assert len(txt_docs) == 1
    assert txt_docs[0].metadata["file_type"] == ".txt"
    assert txt_docs[0].page_content == "Plain text brief."


def test_load_document_json_csv_yaml(tmp_path: Path) -> None:
    json_file = tmp_path / "catalog.json"
    json_file.write_text('{"name":"NovaFlow","tiers":["Growth","Scale"]}', encoding="utf-8")

    csv_file = tmp_path / "metrics.csv"
    csv_file.write_text("metric,target,owner\nSQLs,140,Demand Gen\nNRR,120%,RevOps\n", encoding="utf-8")

    yaml_file = tmp_path / "positioning.yaml"
    yaml_file.write_text("brand:\n  tone: pragmatic\n  regions:\n    - EMEA\n    - NA\n", encoding="utf-8")

    json_docs = load_document(json_file)
    assert len(json_docs) == 1
    assert json.loads(json_docs[0].page_content)["name"] == "NovaFlow"
    assert "\n" in json_docs[0].page_content  # pretty formatted

    csv_docs = load_document(csv_file)
    assert len(csv_docs) == 1
    assert "Row 1: metric: SQLs | target: 140 | owner: Demand Gen" in csv_docs[0].page_content
    assert "Row 2: metric: NRR | target: 120% | owner: RevOps" in csv_docs[0].page_content

    yaml_docs = load_document(yaml_file)
    assert len(yaml_docs) == 1
    assert "tone: pragmatic" in yaml_docs[0].page_content
    assert "EMEA" in yaml_docs[0].page_content


def test_load_document_docx(tmp_path: Path) -> None:
    docx_path = tmp_path / "brief.docx"
    doc = DocxDocument()
    doc.add_heading("NovaFlow Marketing Brief", level=1)
    doc.add_paragraph("Target mid-market SaaS teams across North America and EMEA.")
    table = doc.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "Tier"
    table.cell(0, 1).text = "Annual Price"
    table.cell(1, 0).text = "Scale"
    table.cell(1, 1).text = "$36,000"
    doc.save(str(docx_path))

    loaded = load_document(docx_path)
    assert len(loaded) == 1
    assert loaded[0].metadata["filename"] == "brief.docx"
    assert loaded[0].metadata["file_type"] == ".docx"
    assert "Target mid-market SaaS teams" in loaded[0].page_content
    assert "Tier | Annual Price" in loaded[0].page_content
    assert "Scale | $36,000" in loaded[0].page_content


def test_load_document_pdf_pages(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pdf_path = tmp_path / "deck.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n%fake\n")

    fake_pages = [
        SimpleNamespace(extract_text=lambda: "Page one executive summary."),
        SimpleNamespace(extract_text=lambda: "   "),
        SimpleNamespace(extract_text=lambda: "Page three pricing overview."),
    ]
    monkeypatch.setattr(
        "closr.documents.loader.PdfReader",
        lambda _path: SimpleNamespace(pages=fake_pages),
    )

    loaded = load_document(pdf_path)
    assert len(loaded) == 2
    assert loaded[0].metadata["page"] == 1
    assert loaded[0].page_content == "Page one executive summary."
    assert loaded[1].metadata["page"] == 3
    assert loaded[1].page_content == "Page three pricing overview."


def test_load_document_errors(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_document(tmp_path / "does_not_exist.md")

    bad_file = tmp_path / "image.png"
    bad_file.write_bytes(b"png")
    with pytest.raises(ValueError, match="Unsupported file extension"):
        load_document(bad_file)


def test_load_documents_directory_recursive_skips_hidden_and_unsupported(tmp_path: Path) -> None:
    (tmp_path / "b_note.txt").write_text("Note B", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "a_brief.md").write_text("Brief A", encoding="utf-8")
    (tmp_path / "sub" / "ignored.bin").write_bytes(b"\x00\x01")
    (tmp_path / ".hidden.md").write_text("Should skip hidden file", encoding="utf-8")
    (tmp_path / ".hidden_dir").mkdir()
    (tmp_path / ".hidden_dir" / "secret.txt").write_text("Should skip hidden dir", encoding="utf-8")

    docs = load_documents(tmp_path)
    filenames = [doc.metadata["filename"] for doc in docs]
    assert filenames == ["b_note.txt", "a_brief.md"]  # sorted full paths: root before sub/


def test_load_sample_org_docs_directory() -> None:
    docs = load_documents(SAMPLE_DOCS_DIR)
    filenames = [doc.metadata["filename"] for doc in docs]
    assert filenames == [
        "company_overview.md",
        "marketing_brief.md",
        "product_catalog.json",
    ]


def test_split_documents_chunks_with_overlap() -> None:
    long_text = " ".join(f"token{i}" for i in range(400))
    doc = Document(page_content=long_text, metadata={"filename": "long.txt"})

    chunks = split_documents([doc], chunk_size=200, chunk_overlap=40)
    assert len(chunks) > 2
    assert all(len(chunk.page_content) <= 200 for chunk in chunks)
    assert all(chunk.metadata["filename"] == "long.txt" for chunk in chunks)
    assert split_documents([]) == []


# ---------------------------------------------------------------------------
# Knowledge Base (InMemoryVectorStore) Tests
# ---------------------------------------------------------------------------


def test_knowledge_base_ingest_search_save_and_load(tmp_path: Path) -> None:
    embeddings = DeterministicFakeEmbedding(size=32)
    persist_file = tmp_path / "nested" / "kb.json"

    kb = OrganizationKnowledgeBase(embeddings=embeddings, persist_path=persist_file)
    assert kb.search("churn") == []
    assert kb.source_files == []

    chunk_count = kb.ingest_paths(SAMPLE_DOCS_DIR, chunk_size=600, chunk_overlap=100)
    assert chunk_count >= 3
    assert len(kb) == chunk_count
    assert persist_file.exists()
    assert kb.source_files == [
        "company_overview.md",
        "marketing_brief.md",
        "product_catalog.json",
    ]

    results = kb.search("predictive churn 90 days", k=2)
    assert len(results) == 2

    retriever = kb.as_retriever(k=2)
    retrieved = retriever.invoke("EMEA Frankfurt GDPR")
    assert len(retrieved) == 2

    # Explicit save to another path and reload via classmethod
    alt_path = tmp_path / "backup" / "store.json"
    saved_path = kb.save(alt_path)
    assert saved_path == alt_path
    assert alt_path.exists()

    reloaded = OrganizationKnowledgeBase.load(alt_path, embeddings=embeddings)
    assert len(reloaded) == chunk_count
    assert reloaded.source_files == kb.source_files
    assert len(reloaded.search("NovaFlow Pulse", k=2)) == 2

    # Auto-load when initialized with existing persist_path
    auto_loaded = OrganizationKnowledgeBase(embeddings=embeddings, persist_path=persist_file)
    assert len(auto_loaded) == chunk_count
    assert auto_loaded.source_files == kb.source_files


def test_knowledge_base_load_missing_and_save_without_path(tmp_path: Path) -> None:
    embeddings = DeterministicFakeEmbedding(size=16)
    kb = OrganizationKnowledgeBase(embeddings=embeddings)

    with pytest.raises(ValueError, match="No target path"):
        kb.save()

    with pytest.raises(FileNotFoundError):
        OrganizationKnowledgeBase.load(tmp_path / "missing.json", embeddings=embeddings)


# ---------------------------------------------------------------------------
# OrganizationAnalyzer Tests
# ---------------------------------------------------------------------------


def test_analyzer_sync_and_knowledge_base_ingestion(tmp_path: Path) -> None:
    embeddings = DeterministicFakeEmbedding(size=32)
    persist_path = tmp_path / "kb.json"
    kb = OrganizationKnowledgeBase(embeddings=embeddings, persist_path=persist_path)

    fake_llm = FakeStructuredLLM(_sample_profile())
    analyzer = OrganizationAnalyzer(
        llm=fake_llm,  # type: ignore[arg-type]
        knowledge_base=kb,
    )

    profile = analyzer.analyze_paths(SAMPLE_DOCS_DIR)
    assert fake_llm.requested_schemas == [OrganizationProfile]
    assert profile.organization_name == "NovaFlow Analytics"
    assert profile.source_documents == [
        "company_overview.md",
        "marketing_brief.md",
        "product_catalog.json",
    ]
    # Knowledge base should also have been populated and persisted
    assert len(kb) > 0
    assert persist_path.exists()

    human_message = fake_llm.received_messages[0][1]
    assert isinstance(human_message.content, str)
    assert "=== Document: company_overview.md ===" in human_message.content
    assert "=== Document: marketing_brief.md ===" in human_message.content
    assert "=== Document: product_catalog.json ===" in human_message.content


def test_analyzer_respects_max_context_chars() -> None:
    fake_llm = FakeStructuredLLM(_sample_profile())
    analyzer = OrganizationAnalyzer(
        llm=fake_llm,  # type: ignore[arg-type]
        max_context_chars=500,
    )

    profile = analyzer.analyze_paths(SAMPLE_DOCS_DIR)
    assert profile.source_documents == ["company_overview.md"]

    human_message = fake_llm.received_messages[0][1]
    assert isinstance(human_message.content, str)
    prefix = (
        "Analyze the following internal organization documents and extract "
        "a complete OrganizationProfile.\n\n"
    )
    context_body = human_message.content.removeprefix(prefix)
    assert len(context_body) <= 500
    assert "=== Document: company_overview.md ===" in context_body
    assert "truncated due to context limit" in context_body


async def test_analyzer_async_and_empty_error() -> None:
    fake_llm = FakeStructuredLLM(_sample_profile())
    analyzer = OrganizationAnalyzer(llm=fake_llm)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="empty list"):
        analyzer.analyze_documents([])

    profile = await analyzer.aanalyze_paths(SAMPLE_DOCS_DIR)
    assert profile.organization_name == "NovaFlow Analytics"
    assert set(profile.source_documents) == {
        "company_overview.md",
        "marketing_brief.md",
        "product_catalog.json",
    }


def test_analyzer_supports_plain_runnable_returning_profile() -> None:
    expected = _sample_profile()
    runnable = RunnableLambda(lambda _msgs: expected)
    analyzer = OrganizationAnalyzer(llm=runnable)

    docs = [Document(page_content="NovaFlow summary", metadata={"filename": "summary.md"})]
    profile = analyzer.analyze_documents(docs)
    assert profile.organization_name == "NovaFlow Analytics"
    assert profile.source_documents == ["summary.md"]


# ---------------------------------------------------------------------------
# Document Tools Tests
# ---------------------------------------------------------------------------


def test_document_tools_analyze_and_search(tmp_path: Path) -> None:
    embeddings = DeterministicFakeEmbedding(size=32)
    kb = OrganizationKnowledgeBase(embeddings=embeddings)
    fake_llm = FakeStructuredLLM(_sample_profile())
    analyzer = OrganizationAnalyzer(llm=fake_llm, knowledge_base=kb)  # type: ignore[arg-type]

    tools = create_document_tools(analyzer=analyzer, knowledge_base=kb)
    tool_map = {t.name: t for t in tools}
    assert set(tool_map) == {
        "analyze_organization_documents",
        "search_organization_knowledge",
    }

    # Searching empty KB returns helpful guidance
    empty_msg = tool_map["search_organization_knowledge"].invoke({"query": "churn", "k": 2})
    assert "empty" in empty_msg.lower()

    # Invalid path returns helpful error string rather than crashing
    error_msg = tool_map["analyze_organization_documents"].invoke(
        {"path": str(tmp_path / "missing_dir")}
    )
    assert "error analyzing organization documents" in error_msg.lower()

    # Valid path analyzes docs, returns JSON, and populates KB
    profile_json = tool_map["analyze_organization_documents"].invoke(
        {"path": str(SAMPLE_DOCS_DIR)}
    )
    parsed = OrganizationProfile.model_validate_json(profile_json)
    assert parsed.organization_name == "NovaFlow Analytics"
    assert len(kb) > 0

    # Searching populated KB returns formatted [source: filename] excerpts
    search_output = tool_map["search_organization_knowledge"].invoke(
        {"query": "GDPR Frankfurt residency", "k": 3}
    )
    assert "[source: " in search_output
    assert any(
        name in search_output
        for name in ("company_overview.md", "marketing_brief.md", "product_catalog.json")
    )


def test_search_tool_without_knowledge_base() -> None:
    fake_llm = FakeStructuredLLM(_sample_profile())
    analyzer = OrganizationAnalyzer(llm=fake_llm)  # type: ignore[arg-type]
    tools = create_document_tools(analyzer=analyzer)
    search_tool = next(t for t in tools if t.name == "search_organization_knowledge")

    output = search_tool.invoke({"query": "pricing"})
    assert "not configured" in output.lower()
