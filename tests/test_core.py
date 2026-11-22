"""Unit tests for closr core configuration, schemas, and multi-provider LLM factory."""

from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.runnables import RunnableWithFallbacks
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import ValidationError

from closr.config import ClosrSettings, LLMProvider, MCPServerConfig, get_settings
from closr.llm import LLMFactory
from closr.schemas import (
    AudienceSegment,
    CampaignRecommendation,
    MarketSignal,
    MarketingPlan,
    OrganizationProfile,
    TargetAudienceReport,
)

ENV_KEYS_TO_CLEAR = (
    "OPENAI_API_KEY",
    "GOOGLE_API_KEY",
    "CLOSR_OPENAI_API_KEY",
    "CLOSR_GOOGLE_API_KEY",
    "CLOSR_DEFAULT_PROVIDER",
    "CLOSR_FALLBACK_PROVIDER",
    "CLOSR_OPENAI_CHAT_MODEL",
    "CLOSR_OPENAI_EMBEDDING_MODEL",
    "CLOSR_GOOGLE_CHAT_MODEL",
    "CLOSR_GOOGLE_EMBEDDING_MODEL",
    "CLOSR_EMBEDDING_PROVIDER",
    "CLOSR_TEMPERATURE",
    "CLOSR_MCP_CONFIG_PATH",
    "CLOSR_VECTORSTORE_DIR",
)


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clear API key and CLOSR_ environment variables so tests are deterministic."""
    for key in ENV_KEYS_TO_CLEAR:
        monkeypatch.delenv(key, raising=False)


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


def test_settings_defaults(clean_env: None) -> None:
    settings = ClosrSettings(_env_file=None)  # type: ignore[call-arg]
    assert settings.openai_api_key is None
    assert settings.google_api_key is None
    assert settings.default_provider is LLMProvider.OPENAI
    assert settings.fallback_provider is LLMProvider.GOOGLE
    assert settings.openai_chat_model == "gpt-4o-mini"
    assert settings.openai_embedding_model == "text-embedding-3-small"
    assert settings.google_chat_model == "gemini-2.5-flash"
    assert settings.google_embedding_model == "models/text-embedding-004"
    assert settings.embedding_provider is None
    assert settings.temperature == pytest.approx(0.2)
    assert settings.mcp_config_path == Path("mcp_servers.json")
    assert settings.vectorstore_dir == Path(".closr_data/vectorstore")
    assert settings.get_available_providers() == []


def test_settings_loads_standard_and_prefixed_env_vars(
    clean_env: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-openai")
    monkeypatch.setenv("GOOGLE_API_KEY", "gk-test-google")
    monkeypatch.setenv("CLOSR_DEFAULT_PROVIDER", "google")
    monkeypatch.setenv("CLOSR_FALLBACK_PROVIDER", "openai")
    monkeypatch.setenv("CLOSR_OPENAI_CHAT_MODEL", "gpt-4o")
    monkeypatch.setenv("CLOSR_GOOGLE_CHAT_MODEL", "gemini-2.5-pro")
    monkeypatch.setenv("CLOSR_EMBEDDING_PROVIDER", "openai")
    monkeypatch.setenv("CLOSR_TEMPERATURE", "0.5")
    monkeypatch.setenv("CLOSR_MCP_CONFIG_PATH", "/tmp/custom_mcp.json")
    monkeypatch.setenv("CLOSR_VECTORSTORE_DIR", "/tmp/custom_vs")

    settings = ClosrSettings(_env_file=None)  # type: ignore[call-arg]
    assert settings.openai_api_key == "sk-test-openai"
    assert settings.google_api_key == "gk-test-google"
    assert settings.default_provider is LLMProvider.GOOGLE
    assert settings.fallback_provider is LLMProvider.OPENAI
    assert settings.openai_chat_model == "gpt-4o"
    assert settings.google_chat_model == "gemini-2.5-pro"
    assert settings.embedding_provider is LLMProvider.OPENAI
    assert settings.temperature == pytest.approx(0.5)
    assert settings.mcp_config_path == Path("/tmp/custom_mcp.json")
    assert settings.vectorstore_dir == Path("/tmp/custom_vs")
    assert settings.get_available_providers() == [
        LLMProvider.OPENAI,
        LLMProvider.GOOGLE,
    ]


def test_settings_empty_strings_normalized_to_none(
    clean_env: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "   ")
    monkeypatch.setenv("GOOGLE_API_KEY", "")
    monkeypatch.setenv("CLOSR_FALLBACK_PROVIDER", "")
    monkeypatch.setenv("CLOSR_EMBEDDING_PROVIDER", "")

    settings = ClosrSettings(_env_file=None)  # type: ignore[call-arg]
    assert settings.openai_api_key is None
    assert settings.google_api_key is None
    assert settings.fallback_provider is None
    assert settings.embedding_provider is None
    assert settings.get_available_providers() == []


def test_settings_loads_from_dotenv_file(
    clean_env: None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "OPENAI_API_KEY=sk-from-file\nCLOSR_TEMPERATURE=0.7\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    settings = get_settings()
    assert settings.openai_api_key == "sk-from-file"
    assert settings.temperature == pytest.approx(0.7)
    assert settings.get_available_providers() == [LLMProvider.OPENAI]


def test_mcp_server_config_defaults_and_http() -> None:
    stdio_cfg = MCPServerConfig(command="npx", args=["-y", "server-example"])
    assert stdio_cfg.transport == "stdio"
    assert stdio_cfg.enabled is True
    assert stdio_cfg.env == {}
    assert stdio_cfg.headers == {}

    http_cfg = MCPServerConfig(
        transport="streamable_http",
        url="https://mcp.example.com/mcp",
        headers={"Authorization": "Bearer token"},
        enabled=False,
    )
    assert http_cfg.transport == "streamable_http"
    assert http_cfg.url == "https://mcp.example.com/mcp"
    assert http_cfg.enabled is False


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------


def _sample_profile() -> OrganizationProfile:
    return OrganizationProfile(
        organization_name="Acme Cloud",
        industry="B2B SaaS",
        summary="Automated cloud cost optimization for engineering teams.",
        products_or_services=["CostRadar", "K8s Autoscaler"],
        value_proposition="Cut cloud spend by 30% in two weeks.",
        business_goals=["Grow ARR by 2x"],
        marketing_needs=["Pipeline generation"],
        pain_points_addressed=["Unpredictable AWS bills"],
        brand_tone="Pragmatic and technical",
        differentiators=["Zero-agent deployment"],
        geographic_focus=["North America", "EU"],
        budget_or_constraints=["$25k/month paid media cap"],
        existing_audience_hints=["VP of Engineering", "FinOps Lead"],
        source_documents=["pitch_deck.pdf"],
    )


def _sample_segment() -> AudienceSegment:
    return AudienceSegment(
        name="Series B Engineering Leaders",
        description="VPs of Engineering and DevOps leads scaling Kubernetes.",
        demographics_or_firmographics="50-500 employee SaaS companies on AWS",
        roles_or_personas=["VP Engineering", "Platform Lead"],
        pain_points=["Spiking cloud compute bills"],
        buying_triggers=["Post-funding infrastructure scale-up"],
        preferred_channels=["LinkedIn", "Technical Blogs"],
        targeting_keywords=["kubernetes cost optimization", "aws finops"],
        communities_and_hashtags=["#FinOps", "r/devops"],
        fit_score=92,
        rationale="Directly aligned with zero-agent cost savings value prop.",
        messaging_angle="Reclaim 30% of cloud budget without slowing deploys.",
    )


def test_schemas_validate_and_roundtrip_json() -> None:
    profile = _sample_profile()
    signal = MarketSignal(
        query="kubernetes cost optimization tools",
        source="reddit",
        title="How are teams cutting EKS costs?",
        snippet="Looking for automated ways to reduce idle nodes.",
        url="https://reddit.com/r/devops/example",
        detected_keywords=["EKS", "idle nodes"],
        platform_hint="Reddit",
    )
    segment = _sample_segment()
    report = TargetAudienceReport(
        organization_name=profile.organization_name,
        scan_queries=[signal.query],
        signals_analyzed=1,
        segments=[segment],
        recommended_primary_segment=segment.name,
        channel_priority=["LinkedIn", "Developer Communities"],
    )
    campaign = CampaignRecommendation(
        campaign_name="Zero-Agent FinOps Audit",
        target_segment=segment.name,
        objective="Generate qualified demo requests",
        channels=["LinkedIn", "Reddit"],
        key_messages=["Cut AWS bills by 30% in 14 days"],
        content_ideas=["EKS waste benchmark report"],
        kpis=["Demo bookings", "CPL"],
        recommended_mcp_actions=["create_linkedin_draft"],
    )
    plan = MarketingPlan(
        organization_profile=profile,
        audience_report=report,
        executive_summary="Focus Q1 on FinOps-minded engineering leaders.",
        campaigns=[campaign],
        next_steps=["Publish EKS benchmark guide"],
    )

    restored = MarketingPlan.model_validate_json(plan.model_dump_json())
    assert restored == plan


@pytest.mark.parametrize("bad_score", [-1, 101])
def test_audience_segment_fit_score_bounds(bad_score: int) -> None:
    base = _sample_segment().model_dump()
    base["fit_score"] = bad_score
    with pytest.raises(ValidationError):
        AudienceSegment.model_validate(base)


def test_schemas_work_with_structured_output_binding(clean_env: None) -> None:
    settings = ClosrSettings(
        _env_file=None,  # type: ignore[call-arg]
        openai_api_key="sk-test",
        google_api_key="gk-test",
    )
    factory = LLMFactory(settings)
    openai_model = factory.get_chat_model(LLMProvider.OPENAI)
    google_model = factory.get_chat_model(LLMProvider.GOOGLE)

    for schema in (
        OrganizationProfile,
        MarketSignal,
        AudienceSegment,
        TargetAudienceReport,
        CampaignRecommendation,
        MarketingPlan,
    ):
        assert openai_model.with_structured_output(schema) is not None
        assert google_model.with_structured_output(schema) is not None


# ---------------------------------------------------------------------------
# LLMFactory tests
# ---------------------------------------------------------------------------


def test_factory_get_chat_model_explicit_providers(clean_env: None) -> None:
    settings = ClosrSettings(
        _env_file=None,  # type: ignore[call-arg]
        openai_api_key="sk-test",
        google_api_key="gk-test",
        temperature=0.3,
    )
    factory = LLMFactory(settings)

    oai = factory.get_chat_model(LLMProvider.OPENAI, model="gpt-4o", temperature=0.1)
    assert isinstance(oai, ChatOpenAI)
    assert oai.model_name == "gpt-4o"
    assert oai.temperature == pytest.approx(0.1)

    gem = factory.get_chat_model(LLMProvider.GOOGLE)
    assert isinstance(gem, ChatGoogleGenerativeAI)
    assert gem.model == "gemini-2.5-flash"
    assert gem.temperature == pytest.approx(0.3)


def test_factory_chat_model_fallback_and_no_fallback_errors(clean_env: None) -> None:
    google_only = ClosrSettings(
        _env_file=None,  # type: ignore[call-arg]
        google_api_key="gk-only",
    )
    factory_google_only = LLMFactory(google_only)

    # Requesting OpenAI with a specific model override falls back to Google's default model.
    fallback_model = factory_google_only.get_chat_model(
        provider=LLMProvider.OPENAI,
        model="gpt-4o",
        allow_fallback=True,
    )
    assert isinstance(fallback_model, ChatGoogleGenerativeAI)
    assert fallback_model.model == "gemini-2.5-flash"

    with pytest.raises(ValueError, match="openai"):
        factory_google_only.get_chat_model(
            provider=LLMProvider.OPENAI,
            allow_fallback=False,
        )

    openai_only = ClosrSettings(
        _env_file=None,  # type: ignore[call-arg]
        openai_api_key="sk-only",
    )
    factory_openai_only = LLMFactory(openai_only)
    fallback_oai = factory_openai_only.get_chat_model(
        provider=LLMProvider.GOOGLE,
        allow_fallback=True,
    )
    assert isinstance(fallback_oai, ChatOpenAI)


def test_factory_raises_when_no_keys_configured(clean_env: None) -> None:
    settings = ClosrSettings(_env_file=None)  # type: ignore[call-arg]
    factory = LLMFactory(settings)

    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        factory.get_chat_model()
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        factory.get_resilient_chat_model()
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        factory.get_embeddings()


def test_factory_task_routing_both_keys(clean_env: None) -> None:
    settings = ClosrSettings(
        _env_file=None,  # type: ignore[call-arg]
        openai_api_key="sk-test",
        google_api_key="gk-test",
        default_provider=LLMProvider.GOOGLE,
    )
    factory = LLMFactory(settings)

    assert isinstance(
        factory.get_model_for_task("document_analysis"),
        ChatGoogleGenerativeAI,
    )
    assert isinstance(factory.get_model_for_task("strategy"), ChatOpenAI)
    assert isinstance(factory.get_model_for_task("agent"), ChatOpenAI)
    assert isinstance(
        factory.get_model_for_task("audience_discovery"),
        ChatGoogleGenerativeAI,
    )


def test_factory_task_routing_single_key_fallback(clean_env: None) -> None:
    openai_only = LLMFactory(
        ClosrSettings(_env_file=None, openai_api_key="sk-only")  # type: ignore[call-arg]
    )
    assert isinstance(
        openai_only.get_model_for_task("document_analysis", allow_fallback=True),
        ChatOpenAI,
    )
    with pytest.raises(ValueError):
        openai_only.get_model_for_task("document_analysis", allow_fallback=False)

    google_only = LLMFactory(
        ClosrSettings(_env_file=None, google_api_key="gk-only")  # type: ignore[call-arg]
    )
    assert isinstance(
        google_only.get_model_for_task("strategy", allow_fallback=True),
        ChatGoogleGenerativeAI,
    )
    with pytest.raises(ValueError):
        google_only.get_model_for_task("agent", allow_fallback=False)


def test_factory_resilient_chat_model(clean_env: None) -> None:
    both = LLMFactory(
        ClosrSettings(
            _env_file=None,  # type: ignore[call-arg]
            openai_api_key="sk-test",
            google_api_key="gk-test",
        )
    )
    resilient = both.get_resilient_chat_model(
        primary_provider=LLMProvider.OPENAI,
        temperature=0.4,
    )
    assert isinstance(resilient, RunnableWithFallbacks)
    assert isinstance(resilient.runnable, ChatOpenAI)
    assert len(resilient.fallbacks) == 1
    assert isinstance(resilient.fallbacks[0], ChatGoogleGenerativeAI)

    single = LLMFactory(
        ClosrSettings(_env_file=None, google_api_key="gk-only")  # type: ignore[call-arg]
    )
    single_model = single.get_resilient_chat_model(primary_provider=LLMProvider.OPENAI)
    assert isinstance(single_model, ChatGoogleGenerativeAI)


def test_factory_get_embeddings_and_fallback(clean_env: None) -> None:
    both = LLMFactory(
        ClosrSettings(
            _env_file=None,  # type: ignore[call-arg]
            openai_api_key="sk-test",
            google_api_key="gk-test",
            embedding_provider=LLMProvider.GOOGLE,
        )
    )
    assert isinstance(both.get_embeddings(), GoogleGenerativeAIEmbeddings)
    assert isinstance(both.get_embeddings(LLMProvider.OPENAI), OpenAIEmbeddings)

    google_only = LLMFactory(
        ClosrSettings(
            _env_file=None,  # type: ignore[call-arg]
            google_api_key="gk-only",
            default_provider=LLMProvider.OPENAI,
        )
    )
    emb = google_only.get_embeddings()
    assert isinstance(emb, GoogleGenerativeAIEmbeddings)
