"""End-to-end offline integration tests covering documents, audience discovery, MCP, agent chat, config, and CLI."""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from docx import Document as DocxDocument
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda, RunnableWithFallbacks
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from pydantic import Field

from closr.agent import MarketingAgent, StrategyDraft
from closr.audience import OfflineHeuristicScanner, ScanQueryPlan
from closr.config import ClosrSettings, LLMProvider
from closr.documents import OrganizationAnalyzer, OrganizationKnowledgeBase
from closr.llm import LLMFactory
from closr.schemas import (
    AudienceSegment,
    CampaignRecommendation,
    MarketingPlan,
    OrganizationProfile,
    TargetAudienceReport,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_DOCS_DIR = REPO_ROOT / "examples" / "sample_org_docs"


# ---------------------------------------------------------------------------
# Deterministic offline fakes
# ---------------------------------------------------------------------------


def _build_novaflow_profile(source_documents: list[str] | None = None) -> OrganizationProfile:
    return OrganizationProfile(
        organization_name="NovaFlow Analytics",
        industry="B2B SaaS / Revenue Intelligence",
        summary=(
            "AI-powered RevOps and predictive churn intelligence platform unifying CRM, "
            "billing, and product telemetry in 48 hours for mid-market SaaS teams."
        ),
        products_or_services=["NovaFlow Pulse", "NovaFlow Retain", "NovaFlow Sync"],
        value_proposition=(
            "Predictable net revenue retention in 48 hours via explainable 90-day churn "
            "alerts and automated expansion playbooks."
        ),
        business_goals=[
            "Grow qualified sales pipeline by 65% YoY ($14M to $23M)",
            "Increase EMEA share of new bookings from 22% to 35%",
            "Reduce blended CAC payback period below 11 months",
        ],
        marketing_needs=[
            "High-intent LinkedIn ABM and search demand capture",
            "EMEA GDPR trust positioning with Frankfurt data residency proof",
            "Technical NRR benchmark authority content and RevOps community growth",
        ],
        pain_points_addressed=[
            "12-18% preventable ARR loss from fragmented CRM/billing/telemetry data",
            "Late churn detection occurring only during renewal calls",
        ],
        brand_tone="Authoritative, pragmatic, analytical, and approachable",
        differentiators=[
            "Explainable 90-day churn scoring",
            "48-hour zero-ETL onboarding",
            "SOC 2 Type II and Frankfurt EU GDPR data residency",
        ],
        geographic_focus=["North America", "EMEA"],
        budget_or_constraints=[
            "$180,000 quarterly program budget",
            "Strict GDPR opt-in and CCPA compliance",
            "Exclude SMB companies under 30 employees",
        ],
        existing_audience_hints=[
            "VP / Director of Revenue Operations",
            "Chief Revenue Officer",
            "VP of Customer Success",
            "Head of Finance / FP&A",
        ],
        source_documents=source_documents or [],
    )


def _build_unsorted_audience_report() -> TargetAudienceReport:
    """Return segments intentionally unsorted by `fit_score` to verify pipeline ranking."""
    return TargetAudienceReport(
        organization_name="NovaFlow Analytics",
        scan_queries=[
            "RevOps churn prediction software B2B SaaS",
            "EMEA GDPR NRR forecasting platform",
        ],
        signals_analyzed=0,
        segments=[
            AudienceSegment(
                name="Subscription E-Commerce Finance Leaders",
                description="FP&A and Retention Ops leaders at high-frequency subscription brands.",
                demographics_or_firmographics="100-500 employees, North America & UK",
                roles_or_personas=["Head of Finance", "VP of Retention"],
                pain_points=["Manual Stripe/Chargebee cohort reconciliation"],
                buying_triggers=["Quarterly board audit", "Rising involuntary churn"],
                preferred_channels=["Google Search", "Email", "LinkedIn"],
                targeting_keywords=["subscription nrr forecasting", "stripe churn analytics"],
                communities_and_hashtags=["#SaaSFinance", "#SubscriptionGrowth"],
                fit_score=78,
                rationale="Strong billing reconciliation fit; secondary to core SaaS RevOps.",
                messaging_angle="Reconcile billing and CRM retention cohorts in 48 hours.",
            ),
            AudienceSegment(
                name="Mid-Market B2B SaaS RevOps Leaders",
                description="RevOps and Revenue Leaders at Series B-D B2B SaaS companies ($10M-$120M ARR).",
                demographics_or_firmographics="50-500 employees, B2B SaaS in North America & EMEA",
                roles_or_personas=[
                    "VP of Revenue Operations",
                    "Chief Revenue Officer",
                    "VP of Customer Success",
                ],
                pain_points=["Fragmented CRM and billing telemetry", "Preventable renewal churn"],
                buying_triggers=["Missed NRR targets for 2+ quarters", "Loss of data analyst"],
                preferred_channels=["LinkedIn", "Google Search", "Email"],
                targeting_keywords=["revops churn intelligence", "nrr forecasting software"],
                communities_and_hashtags=["RevOps Co-op", "Pavilion", "#RevOps"],
                fit_score=94,
                rationale="Highest urgency and direct alignment with zero-ETL 90-day churn scoring.",
                messaging_angle="Explainable 90-day churn risk and board-ready NRR in 48 hours.",
            ),
            AudienceSegment(
                name="EMEA SaaS Customer Success Executives",
                description="EMEA CS and Revenue leaders requiring GDPR-compliant retention tooling.",
                demographics_or_firmographics="UK, DACH, Benelux, and Nordics mid-market B2B SaaS",
                roles_or_personas=["VP of Customer Success", "Director of Revenue Operations"],
                pain_points=["US-only data hosting blockers", "Reactive account health scores"],
                buying_triggers=["EMEA enterprise expansion", "GDPR procurement review"],
                preferred_channels=["LinkedIn", "Google Search", "YouTube"],
                targeting_keywords=["gdpr churn prediction", "eu data residency revops"],
                communities_and_hashtags=["#CustomerSuccess", "#EMEA_SaaS"],
                fit_score=87,
                rationale="Dedicated Frankfurt EU residency unlocks stalled EMEA enterprise deals.",
                messaging_angle="GDPR-native churn intelligence with dedicated Frankfurt residency.",
            ),
        ],
        recommended_primary_segment="",
        channel_priority=["LinkedIn", "Google Search", "Email", "YouTube"],
        notes="Lead with LinkedIn ABM and search capture; exclude <30 employee SMBs.",
    )


def _build_strategy_draft() -> StrategyDraft:
    return StrategyDraft(
        executive_summary=(
            "Deploy a capital-efficient LinkedIn ABM, Google Search, and lifecycle email program "
            "for NovaFlow Analytics targeting mid-market B2B SaaS RevOps and EMEA CS leaders."
        ),
        campaigns=[
            CampaignRecommendation(
                campaign_name="90-Day Explainable Churn ABM",
                target_segment="Mid-Market B2B SaaS RevOps Leaders",
                objective="Generate 95+ SQLs per quarter at <= $1,250 cost per SQL",
                channels=["LinkedIn", "Google Search"],
                key_messages=[
                    "Predictable net revenue retention in 48 hours with zero-ETL setup",
                    "Explainable 90-day churn risk alerts pushed straight to Slack and Salesforce",
                ],
                content_ideas=[
                    "FY2026 Mid-Market NRR Benchmark Report",
                    "Zero-ETL RevOps Architecture Playbook",
                ],
                kpis=["SQLs >= 95/quarter", "Demo-to-SQL >= 42%", "Cost per SQL <= $1,250"],
                recommended_mcp_actions=[
                    "Run build_utm_tracking_links for LinkedIn and Google Search variants",
                    "Calibrate monthly spend via estimate_campaign_budget",
                ],
            ),
            CampaignRecommendation(
                campaign_name="EMEA GDPR Trust & Expansion Engine",
                target_segment="EMEA SaaS Customer Success Executives",
                objective="Grow EMEA share of new bookings from 22% to 35%",
                channels=["LinkedIn", "YouTube", "Email"],
                key_messages=[
                    "Dedicated Frankfurt EU GDPR data residency with SOC 2 Type II certification",
                ],
                content_ideas=[
                    "EMEA Procurement & GDPR Readiness Brief",
                    "2-minute Frankfurt Residency & Churn Playbook Walkthrough",
                ],
                kpis=["EMEA SQLs >= 45/quarter", "EMEA booking share >= 35%"],
                recommended_mcp_actions=[
                    "Benchmark regional CPCs with get_channel_benchmarks",
                ],
            ),
        ],
        next_steps=[
            "Week 1: Generate standardized UTM links via build_utm_tracking_links.",
            "Week 2: Launch role-specific RevOps and EMEA GDPR landing pages.",
            "Weeks 3-4: Activate LinkedIn ABM and high-intent comparison search campaigns.",
        ],
    )


class RecordingStructuredLLM:
    """Deterministic fake structured-output LLM capturing all prompts and schemas."""

    def __init__(self, responses: dict[type[Any], Any]) -> None:
        self.responses = responses
        self.calls: list[tuple[type[Any], list[BaseMessage]]] = []

    def with_structured_output(self, schema: type[Any]) -> RunnableLambda[Any, Any]:
        def _invoke(messages: Any) -> Any:
            msg_list = list(messages)
            self.calls.append((schema, msg_list))
            response = self.responses[schema]
            return response(msg_list) if callable(response) else response

        return RunnableLambda(_invoke)


class MultiTurnChainingChatModel(BaseChatModel):
    """Deterministic multi-turn fake chat model chaining local + real MCP tool calls."""

    docs_path: str
    executed_tools: list[str] = Field(default_factory=list)
    bound_tool_names: list[str] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "multi-turn-chaining-fake"

    def bind_tools(
        self,
        tools: Sequence[Any],
        *,
        tool_choice: Any = None,
        **kwargs: Any,
    ) -> MultiTurnChainingChatModel:
        names = [getattr(t, "name", str(t)) for t in tools]
        return self.model_copy(update={"bound_tool_names": names})

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        last = messages[-1]
        human_turns = sum(1 for m in messages if isinstance(m, HumanMessage))

        # Turn 1: analyze internal docs directory
        if human_turns == 1 and isinstance(last, HumanMessage):
            ai_msg = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "analyze_organization_documents",
                        "args": {"path": self.docs_path},
                        "id": "call_t1_analyze",
                        "type": "tool_call",
                    }
                ],
            )
            return ChatResult(generations=[ChatGeneration(message=ai_msg)])

        if isinstance(last, ToolMessage) and last.name == "analyze_organization_documents":
            self.executed_tools.append("analyze_organization_documents")
            profile = OrganizationProfile.model_validate_json(str(last.content))
            ai_msg = AIMessage(
                content=(
                    f"Analyzed {profile.organization_name} ({profile.industry}) across "
                    f"{len(profile.source_documents)} documents: {', '.join(profile.source_documents)}."
                )
            )
            return ChatResult(generations=[ChatGeneration(message=ai_msg)])

        # Turn 2 step 1: discover audiences relying on automatically synced current_profile
        if human_turns == 2 and isinstance(last, HumanMessage):
            ai_msg = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "discover_target_audiences",
                        "args": {"focus_area": "EMEA SaaS"},
                        "id": "call_t2_discover",
                        "type": "tool_call",
                    }
                ],
            )
            return ChatResult(generations=[ChatGeneration(message=ai_msg)])

        # Turn 2 step 2: chain directly into real built-in MCP tool `build_utm_tracking_links`
        if isinstance(last, ToolMessage) and last.name == "discover_target_audiences":
            self.executed_tools.append("discover_target_audiences")
            ai_msg = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "build_utm_tracking_links",
                        "args": {
                            "base_url": "https://novaflow.io/demo",
                            "campaign_name": "EMEA RevOps Launch",
                            "channels": ["linkedin", "google_search"],
                        },
                        "id": "call_t2_utm",
                        "type": "tool_call",
                    }
                ],
            )
            return ChatResult(generations=[ChatGeneration(message=ai_msg)])

        # Turn 2 final response: synthesize audience + real MCP UTM outputs
        if isinstance(last, ToolMessage) and last.name == "build_utm_tracking_links":
            self.executed_tools.append("build_utm_tracking_links")
            prior_reports = [
                m for m in messages if isinstance(m, ToolMessage) and m.name == "discover_target_audiences"
            ]
            report = TargetAudienceReport.model_validate_json(str(prior_reports[-1].content))
            utm_raw = str(last.content)
            ai_msg = AIMessage(
                content=(
                    f"Primary audience segment: {report.recommended_primary_segment} "
                    f"(signals={report.signals_analyzed}). "
                    f"Generated UTM tracking links: {utm_raw}"
                )
            )
            return ChatResult(generations=[ChatGeneration(message=ai_msg)])

        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content="Unexpected conversation state."))]
        )

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        return self._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


# ---------------------------------------------------------------------------
# 1. Multi-format Document Intelligence + Persistence + RAG
# ---------------------------------------------------------------------------


async def test_multiformat_document_intelligence_persistence_and_rag(tmp_path: Path) -> None:
    docx_path = tmp_path / "extra_playbook.docx"
    docx_doc = DocxDocument()
    docx_doc.add_heading("NovaFlow EMEA Field Playbook", level=1)
    docx_doc.add_paragraph(
        "Target DACH and UK RevOps leaders with GDPR Frankfurt residency proof and zero-ETL demos."
    )
    table = docx_doc.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "Region"
    table.cell(0, 1).text = "Target Share"
    table.cell(1, 0).text = "EMEA"
    table.cell(1, 1).text = "35% of new bookings"
    docx_doc.save(str(docx_path))

    csv_path = tmp_path / "campaign_metrics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["channel", "target_cpa_usd", "demo_to_sql_pct"])
        writer.writerow(["linkedin", "1180", "44"])
        writer.writerow(["google_search", "920", "48"])

    embeddings = DeterministicFakeEmbedding(size=32)
    persist_path = tmp_path / "vectorstore" / "org_kb.json"

    kb = OrganizationKnowledgeBase(embeddings=embeddings, persist_path=persist_path)
    chunk_count = kb.ingest_paths([SAMPLE_DOCS_DIR, docx_path, csv_path])
    assert chunk_count >= 5
    assert persist_path.is_file()

    expected_files = sorted(
        [
            "company_overview.md",
            "marketing_brief.md",
            "product_catalog.json",
            "extra_playbook.docx",
            "campaign_metrics.csv",
        ]
    )
    assert kb.source_files == expected_files

    reloaded_kb = OrganizationKnowledgeBase.load(persist_path, embeddings=embeddings)
    assert reloaded_kb.source_files == expected_files
    assert len(reloaded_kb) == len(kb)

    retrieved = reloaded_kb.search("EMEA Frankfurt GDPR residency RevOps", k=3)
    assert len(retrieved) == 3
    assert all(doc.page_content.strip() for doc in retrieved)

    fake_doc_llm = RecordingStructuredLLM({OrganizationProfile: _build_novaflow_profile()})
    analyzer = OrganizationAnalyzer(
        llm=fake_doc_llm,  # type: ignore[arg-type]
        knowledge_base=reloaded_kb,
    )
    profile = await analyzer.aanalyze_paths([SAMPLE_DOCS_DIR, docx_path, csv_path])

    assert profile.organization_name == "NovaFlow Analytics"
    assert sorted(profile.source_documents) == expected_files
    assert len(fake_doc_llm.calls) == 1
    prompt_text = str(fake_doc_llm.calls[0][1][1].content)
    assert "NovaFlow EMEA Field Playbook" in prompt_text
    assert "campaign_metrics.csv" in prompt_text
    assert "NovaFlow Pulse" in prompt_text


# ---------------------------------------------------------------------------
# 2. Full End-to-End Pipeline (Docs -> Audience -> Real Built-in MCP -> Plan)
# ---------------------------------------------------------------------------


async def test_end_to_end_campaign_pipeline_with_real_builtin_mcp(tmp_path: Path) -> None:
    fake_llm = RecordingStructuredLLM(
        {
            OrganizationProfile: _build_novaflow_profile(),
            ScanQueryPlan: ScanQueryPlan(
                queries=[
                    "RevOps churn prediction software B2B SaaS",
                    "EMEA GDPR NRR forecasting platform",
                ]
            ),
            TargetAudienceReport: _build_unsorted_audience_report(),
            StrategyDraft: _build_strategy_draft(),
        }
    )

    settings = ClosrSettings(
        _env_file=None,  # type: ignore[call-arg]
        openai_api_key=None,
        google_api_key=None,
        mcp_config_path=tmp_path / "missing_mcp.json",
        vectorstore_dir=tmp_path / "vectorstore",
    )
    agent = MarketingAgent(
        settings=settings,
        doc_llm=fake_llm,  # type: ignore[arg-type]
        audience_llm=fake_llm,  # type: ignore[arg-type]
        strategy_llm=fake_llm,  # type: ignore[arg-type]
        embeddings=DeterministicFakeEmbedding(size=32),
        scanner=OfflineHeuristicScanner(),
        include_builtin_mcp=True,
    )

    plan = await agent.run_full_campaign_pipeline(
        SAMPLE_DOCS_DIR,
        focus_area="EMEA SaaS RevOps",
        monthly_budget_usd=25_000.0,
    )

    assert isinstance(plan, MarketingPlan)
    assert plan.organization_profile.organization_name == "NovaFlow Analytics"
    assert sorted(plan.organization_profile.source_documents) == [
        "company_overview.md",
        "marketing_brief.md",
        "product_catalog.json",
    ]

    # Segments must be sorted descending by fit_score and primary segment inferred
    fit_scores = [seg.fit_score for seg in plan.audience_report.segments]
    assert fit_scores == sorted(fit_scores, reverse=True)
    assert fit_scores == [94, 87, 78]
    assert plan.audience_report.recommended_primary_segment == "Mid-Market B2B SaaS RevOps Leaders"
    assert plan.audience_report.signals_analyzed > 0

    # Verify real built-in MCP enrichment data reached strategy synthesis
    strategy_calls = [msgs for schema, msgs in fake_llm.calls if schema is StrategyDraft]
    assert len(strategy_calls) == 1
    strategy_prompt = str(strategy_calls[0][1].content)
    assert '"get_channel_benchmarks"' in strategy_prompt
    assert '"suggest_audience_channels"' in strategy_prompt
    assert '"estimate_campaign_budget"' in strategy_prompt
    assert "25000.0" in strategy_prompt

    assert len(plan.campaigns) == 2
    assert len(plan.next_steps) == 3
    assert agent.current_profile == plan.organization_profile
    assert agent.current_audience_report == plan.audience_report


# ---------------------------------------------------------------------------
# 3. Autonomous Multi-Turn Chat with Document + Audience + Real MCP Chaining
# ---------------------------------------------------------------------------


async def test_autonomous_multiturn_chat_chains_docs_audience_and_real_mcp(tmp_path: Path) -> None:
    fake_llm = RecordingStructuredLLM(
        {
            OrganizationProfile: _build_novaflow_profile(),
            ScanQueryPlan: ScanQueryPlan(
                queries=[
                    "RevOps churn intelligence EMEA B2B SaaS",
                    "GDPR NRR forecasting tools",
                ]
            ),
            TargetAudienceReport: _build_unsorted_audience_report(),
        }
    )
    chat_model = MultiTurnChainingChatModel(docs_path=str(SAMPLE_DOCS_DIR))

    settings = ClosrSettings(
        _env_file=None,  # type: ignore[call-arg]
        openai_api_key=None,
        google_api_key=None,
        mcp_config_path=tmp_path / "missing_mcp.json",
        vectorstore_dir=tmp_path / "vectorstore",
    )
    agent = MarketingAgent(
        settings=settings,
        doc_llm=fake_llm,  # type: ignore[arg-type]
        audience_llm=fake_llm,  # type: ignore[arg-type]
        strategy_llm=fake_llm,  # type: ignore[arg-type]
        agent_llm=chat_model,
        embeddings=DeterministicFakeEmbedding(size=32),
        scanner=OfflineHeuristicScanner(),
        include_builtin_mcp=True,
    )

    assert agent.current_profile is None
    assert agent.current_audience_report is None

    # Turn 1: analyze docs directory via agent tool call
    turn1_reply = await agent.chat(
        f"Analyze our company docs in {SAMPLE_DOCS_DIR}.",
        thread_id="integration-thread",
    )
    assert "NovaFlow Analytics" in turn1_reply
    assert agent.current_profile is not None
    assert sorted(agent.current_profile.source_documents) == [
        "company_overview.md",
        "marketing_brief.md",
        "product_catalog.json",
    ]

    # Turn 2: discover audiences using synced profile, then chain into real MCP tool
    turn2_reply = await agent.chat(
        "Discover our target audiences in EMEA SaaS and generate UTM links for LinkedIn and Google Search.",
        thread_id="integration-thread",
    )
    assert "Mid-Market B2B SaaS RevOps Leaders" in turn2_reply
    assert "utm_source=linkedin" in turn2_reply
    assert "utm_source=google" in turn2_reply
    assert "utm_campaign=emea-revops-launch" in turn2_reply

    assert agent.current_audience_report is not None
    assert agent.current_audience_report.recommended_primary_segment == "Mid-Market B2B SaaS RevOps Leaders"
    assert chat_model.executed_tools == [
        "analyze_organization_documents",
        "discover_target_audiences",
        "build_utm_tracking_links",
    ]


# ---------------------------------------------------------------------------
# 4. Dual-Provider Config, Task Routing & Clean Import Integration
# ---------------------------------------------------------------------------


def test_dual_provider_routing_resilient_fallback_and_clean_import(tmp_path: Path) -> None:
    settings = ClosrSettings(
        _env_file=None,  # type: ignore[call-arg]
        openai_api_key="sk-test-openai",
        google_api_key="gk-test-google",
        default_provider=LLMProvider.OPENAI,
        fallback_provider=LLMProvider.GOOGLE,
    )
    factory = LLMFactory(settings)

    resilient = factory.get_resilient_chat_model()
    assert isinstance(resilient, RunnableWithFallbacks)
    assert isinstance(resilient.runnable, ChatOpenAI)
    assert len(resilient.fallbacks) == 1
    assert isinstance(resilient.fallbacks[0], ChatGoogleGenerativeAI)

    doc_model = factory.get_model_for_task("document_analysis")
    agent_model = factory.get_model_for_task("agent")
    strategy_model = factory.get_model_for_task("strategy")
    assert isinstance(doc_model, ChatGoogleGenerativeAI)
    assert isinstance(agent_model, ChatOpenAI)
    assert isinstance(strategy_model, ChatOpenAI)

    # Clean import in a fresh Python process with zero API keys set and no .env in cwd
    clean_env = {
        k: v
        for k, v in os.environ.items()
        if k not in {"OPENAI_API_KEY", "GOOGLE_API_KEY"} and not k.startswith("CLOSR_")
    }
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import closr; "
                "assert hasattr(closr, 'MarketingAgent'); "
                "assert hasattr(closr, 'LLMFactory'); "
                "assert hasattr(closr, 'MCPManager'); "
                "assert closr.get_settings().get_available_providers() == []"
            ),
        ],
        cwd=str(tmp_path),
        env=clean_env,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert proc.returncode == 0, f"stderr: {proc.stderr}"


# ---------------------------------------------------------------------------
# 5. CLI Subprocess Smoke Tests
# ---------------------------------------------------------------------------


def test_cli_subprocess_help_and_mcp_tools(tmp_path: Path) -> None:
    closr_bin = REPO_ROOT / ".venv" / "bin" / "closr"
    if not closr_bin.is_file():
        closr_bin = Path(sys.executable).parent / "closr"
    assert closr_bin.is_file(), f"CLI executable not found at {closr_bin}"

    clean_env = {
        k: v
        for k, v in os.environ.items()
        if k not in {"OPENAI_API_KEY", "GOOGLE_API_KEY"} and not k.startswith("CLOSR_")
    }
    clean_env["COLUMNS"] = "160"

    help_proc = subprocess.run(
        [str(closr_bin), "--help"],
        cwd=str(tmp_path),
        env=clean_env,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert help_proc.returncode == 0, f"stderr: {help_proc.stderr}"
    for command in ("analyze", "discover", "campaign", "chat", "mcp-tools"):
        assert command in help_proc.stdout

    missing_cfg = tmp_path / "nonexistent_mcp.json"
    mcp_proc = subprocess.run(
        [str(closr_bin), "mcp-tools", "--config", str(missing_cfg)],
        cwd=str(tmp_path),
        env=clean_env,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert mcp_proc.returncode == 0, f"stderr: {mcp_proc.stderr}"
    for expected in (
        "closr-marketing",
        "get_channel_benchmarks",
        "estimate_campaign_budget",
        "build_utm_tracking_links",
        "suggest_audience_channels",
    ):
        assert expected in mcp_proc.stdout
