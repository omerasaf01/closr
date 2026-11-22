"""Offline unit and integration tests for the LangGraph MarketingAgent and CLI."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool
from pydantic import Field
from typer.testing import CliRunner

from closr.agent import (
    MarketingAgent,
    StrategyDraft,
    _parse_mcp_tool_output,
    build_marketing_pipeline,
)
from closr.audience import OfflineHeuristicScanner, ScanQueryPlan, TargetAudienceDiscovery
from closr.cli import _create_agent, app
from closr.config import ClosrSettings
from closr.documents import OrganizationAnalyzer
from closr.schemas import (
    AudienceSegment,
    CampaignRecommendation,
    MarketingPlan,
    OrganizationProfile,
    TargetAudienceReport,
)

SAMPLE_DOCS_DIR = Path(__file__).resolve().parent.parent / "examples" / "sample_org_docs"


# ---------------------------------------------------------------------------
# Deterministic fakes for offline tests
# ---------------------------------------------------------------------------


def _sample_profile() -> OrganizationProfile:
    return OrganizationProfile(
        organization_name="NovaFlow Analytics",
        industry="B2B SaaS / Revenue Intelligence",
        summary=(
            "AI-powered RevOps and predictive churn intelligence platform for mid-market "
            "SaaS and subscription businesses."
        ),
        products_or_services=["NovaFlow Pulse", "NovaFlow Retain", "NovaFlow Sync"],
        value_proposition=(
            "Predictable net revenue retention in 48 hours by unifying CRM, billing, "
            "and product telemetry."
        ),
        business_goals=[
            "Grow qualified pipeline by 65% YoY",
            "Expand EMEA share of new bookings from 22% to 35%",
        ],
        marketing_needs=[
            "High-intent LinkedIn ABM and search demand capture",
            "EMEA GDPR trust positioning and technical authority content",
        ],
        pain_points_addressed=[
            "Preventable ARR churn from fragmented funnel visibility",
            "Delayed renewal risk detection across CRM and billing",
        ],
        brand_tone="Authoritative, pragmatic, analytical, and approachable",
        differentiators=[
            "Explainable 90-day churn scoring",
            "48-hour zero-ETL onboarding",
            "EU Frankfurt data residency and GDPR compliance",
        ],
        geographic_focus=["North America", "EMEA"],
        budget_or_constraints=[
            "$180,000 per quarter program budget",
            "Strict GDPR opt-in compliance",
            "Exclude SMB companies under 30 employees",
        ],
        existing_audience_hints=[
            "VP / Director of Revenue Operations",
            "VP of Customer Success",
            "Chief Revenue Officer",
        ],
        source_documents=[],
    )


def _sample_audience_report() -> TargetAudienceReport:
    return TargetAudienceReport(
        organization_name="NovaFlow Analytics",
        scan_queries=[
            "RevOps churn prediction software B2B SaaS",
            "VP Customer Success NRR forecasting tools",
        ],
        signals_analyzed=6,
        segments=[
            AudienceSegment(
                name="Mid-Market B2B SaaS RevOps Leaders",
                description="RevOps and Revenue Leaders at Series B-D B2B SaaS companies ($10M-$120M ARR).",
                demographics_or_firmographics="50-500 employees, B2B SaaS in North America & EMEA",
                roles_or_personas=[
                    "VP of Revenue Operations",
                    "Chief Revenue Officer",
                    "VP of Customer Success",
                ],
                pain_points=["Fragmented CRM and billing data", "Surprise renewal churn"],
                buying_triggers=["Missed NRR targets", "CRM/billing migration"],
                preferred_channels=["LinkedIn", "Google Search", "Email"],
                targeting_keywords=["nrr forecasting", "churn prediction software", "revops analytics"],
                communities_and_hashtags=["#RevOps", "RevOps Co-op", "Pavilion"],
                fit_score=93,
                rationale="Direct alignment with 48-hour zero-ETL onboarding and 90-day churn scoring.",
                messaging_angle="Spot renewal risk 90 days early with zero-ETL CRM & billing unification.",
            ),
            AudienceSegment(
                name="EMEA SaaS Finance & CS Executives",
                description="EMEA-based CS and Finance leaders needing GDPR-compliant NRR visibility.",
                demographics_or_firmographics="UK, DACH, Benelux, and Nordics mid-market SaaS",
                roles_or_personas=["Head of Finance", "VP of Customer Success"],
                pain_points=["EU data residency concerns", "Manual board NRR reporting"],
                buying_triggers=["Upcoming Series C/D diligence", "EU procurement audit"],
                preferred_channels=["LinkedIn", "Google Search", "YouTube"],
                targeting_keywords=["gdpr churn analytics", "emea revops platform"],
                communities_and_hashtags=["#CustomerSuccess", "#SaaS"],
                fit_score=86,
                rationale="Dedicated Frankfurt residency directly unlocks EMEA enterprise procurement.",
                messaging_angle="GDPR-native NRR forecasting with Frankfurt EU data residency.",
            ),
        ],
        recommended_primary_segment="Mid-Market B2B SaaS RevOps Leaders",
        channel_priority=["LinkedIn", "Google Search", "Email", "YouTube"],
        notes="Prioritize ABM on LinkedIn paired with high-intent search capture.",
    )


def _sample_strategy_draft() -> StrategyDraft:
    return StrategyDraft(
        executive_summary=(
            "Focus NovaFlow Analytics' Q1-Q2 growth budget on high-intent LinkedIn ABM and "
            "Google Search demand capture targeting mid-market B2B SaaS RevOps and CS leaders "
            "in North America and EMEA, supported by GDPR trust assets and lifecycle email nurture."
        ),
        campaigns=[
            CampaignRecommendation(
                campaign_name="90-Day Churn Early Warning ABM",
                target_segment="Mid-Market B2B SaaS RevOps Leaders",
                objective="Generate 95 marketing-sourced SQLs per quarter at <= $1,250 cost per SQL",
                channels=["LinkedIn", "Google Search"],
                key_messages=[
                    "Predictable net revenue retention in 48 hours without SQL pipelines",
                    "Explainable 90-day churn risk drivers pushed straight to Slack and Salesforce",
                ],
                content_ideas=[
                    "FY2026 NRR Benchmark & Churn Driver Teardown",
                    "Interactive 48-hour Zero-ETL Architecture Walkthrough",
                ],
                kpis=["SQLs >= 95/quarter", "Demo-to-SQL >= 42%", "Blended CPA <= $1,250"],
                recommended_mcp_actions=[
                    "Invoke build_utm_tracking_links for LinkedIn and Google Search landing pages",
                    "Invoke estimate_campaign_budget to rebalance monthly spend across LinkedIn and Search",
                ],
            ),
            CampaignRecommendation(
                campaign_name="EMEA GDPR-Native NRR Expansion",
                target_segment="EMEA SaaS Finance & CS Executives",
                objective="Lift EMEA pipeline share to 35% across UK, DACH, and Nordic SaaS",
                channels=["LinkedIn", "YouTube", "Email"],
                key_messages=[
                    "Dedicated Frankfurt EU data residency with SOC 2 Type II & ISO 27001 alignment",
                ],
                content_ideas=[
                    "EMEA RevOps Security & GDPR Readiness Brief",
                    "2-minute Frankfurt Architecture & NRR Forecasting Demo",
                ],
                kpis=["EMEA SQLs >= 45/quarter", "CTR >= 0.55% on LinkedIn"],
                recommended_mcp_actions=[
                    "Invoke get_channel_benchmarks for SaaS regional calibration",
                ],
            ),
        ],
        next_steps=[
            "Week 1: Generate standardized UTM links via build_utm_tracking_links for all ABM & Search variants.",
            "Week 2: Launch role-specific RevOps and EMEA landing pages with GDPR consent tracking.",
            "Weeks 3-4: Activate LinkedIn Thought Leader Ads and high-intent comparison search campaigns.",
        ],
    )


class FakeStructuredLLM:
    """Deterministic fake LLM implementing `with_structured_output` and capturing prompts."""

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


class FakeToolCallingChatModel(BaseChatModel):
    """Deterministic tool-calling fake `BaseChatModel` for testing `MarketingAgent.chat`."""

    tool_to_call: str = "search_organization_knowledge"
    tool_args: dict[str, Any] = Field(
        default_factory=lambda: {"query": "GDPR compliance and pipeline goals", "k": 2}
    )
    call_history: list[list[BaseMessage]] = Field(default_factory=list)
    bound_tools: list[Any] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "fake-tool-calling-chat"

    def bind_tools(
        self,
        tools: Sequence[Any],
        *,
        tool_choice: Any = None,
        **kwargs: Any,
    ) -> FakeToolCallingChatModel:
        return self.model_copy(update={"bound_tools": list(tools)})

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.call_history.append(list(messages))
        last = messages[-1]

        # If we just got a ToolMessage back, synthesize the final answer citing its output.
        if isinstance(last, ToolMessage):
            snippet = str(last.content).strip().splitlines()[0][:160]
            ai_msg = AIMessage(
                content=(
                    f"Tool '{last.name}' returned evidence: {snippet}. "
                    f"(history_messages={len(messages)})"
                )
            )
            return ChatResult(generations=[ChatGeneration(message=ai_msg)])

        # On the first turn when the user asks to inspect/search/check, issue a real tool call.
        human_count = sum(1 for m in messages if isinstance(m, HumanMessage))
        last_text = str(getattr(last, "content", "")).lower()
        if human_count == 1 and any(w in last_text for w in ("search", "benchmark", "check", "docs")):
            ai_msg = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": self.tool_to_call,
                        "args": dict(self.tool_args),
                        "id": "fake_call_1",
                        "type": "tool_call",
                    }
                ],
            )
            return ChatResult(generations=[ChatGeneration(message=ai_msg)])

        # Subsequent turns confirm conversation memory was preserved.
        prior_humans = [str(m.content) for m in messages if isinstance(m, HumanMessage)]
        ai_msg = AIMessage(
            content=(
                f"Follow-up summary across {len(prior_humans)} user turns "
                f"(first='{prior_humans[0]}', total_messages={len(messages)})."
            )
        )
        return ChatResult(generations=[ChatGeneration(message=ai_msg)])

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        return self._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


def _make_test_agent(
    tmp_path: Path,
    *,
    include_builtin_mcp: bool = True,
    agent_llm: BaseChatModel | None = None,
) -> tuple[MarketingAgent, FakeStructuredLLM]:
    profile = _sample_profile()
    report = _sample_audience_report()
    draft = _sample_strategy_draft()

    fake_llm = FakeStructuredLLM(
        {
            OrganizationProfile: profile,
            ScanQueryPlan: ScanQueryPlan(
                queries=[
                    "RevOps churn prediction software B2B SaaS",
                    "VP Customer Success NRR forecasting tools",
                ]
            ),
            TargetAudienceReport: report,
            StrategyDraft: draft,
        }
    )

    settings = ClosrSettings(
        openai_api_key=None,
        google_api_key=None,
        mcp_config_path=tmp_path / "nonexistent_mcp.json",
        vectorstore_dir=tmp_path / "vectorstore",
    )
    agent = MarketingAgent(
        settings=settings,
        doc_llm=fake_llm,  # type: ignore[arg-type]
        audience_llm=fake_llm,  # type: ignore[arg-type]
        strategy_llm=fake_llm,  # type: ignore[arg-type]
        agent_llm=agent_llm,
        embeddings=DeterministicFakeEmbedding(size=16),
        scanner=OfflineHeuristicScanner(),
        include_builtin_mcp=include_builtin_mcp,
    )
    return agent, fake_llm


# ---------------------------------------------------------------------------
# _parse_mcp_tool_output tests
# ---------------------------------------------------------------------------


def test_parse_mcp_tool_output_normalizes_all_formats() -> None:
    # LangChain content block list wrapping JSON
    assert _parse_mcp_tool_output([{"type": "text", "text": '{"industry": "saas", "ctr": 0.55}'}]) == {
        "industry": "saas",
        "ctr": 0.55,
    }

    # Multiple JSON content blocks
    assert _parse_mcp_tool_output(
        [
            {"type": "text", "text": '{"a": 1}'},
            {"type": "text", "text": '{"b": 2}'},
        ]
    ) == [{"a": 1}, {"b": 2}]

    # Single dict content block
    assert _parse_mcp_tool_output({"type": "text", "text": '{"ok": true}'}) == {"ok": True}

    # Raw JSON string
    assert _parse_mcp_tool_output('{"budget": 10000}') == {"budget": 10000}

    # Plain text content block and raw plain string
    assert _parse_mcp_tool_output([{"type": "text", "text": "plain text message"}]) == "plain text message"
    assert _parse_mcp_tool_output("hello world") == "hello world"

    # Regular dicts and lists of dicts (non-content-blocks) pass through unchanged
    raw_dict = {"channel": "linkedin", "cpc": 5.8}
    assert _parse_mcp_tool_output(raw_dict) == raw_dict

    raw_list = [{"channel": "linkedin", "url": "https://example.com"}]
    assert _parse_mcp_tool_output(raw_list) == raw_list

    assert _parse_mcp_tool_output(None) is None


# ---------------------------------------------------------------------------
# End-to-end LangGraph pipeline tests (with real built-in FastMCP server)
# ---------------------------------------------------------------------------


async def test_full_campaign_pipeline_with_real_builtin_mcp(tmp_path: Path) -> None:
    agent, fake_llm = _make_test_agent(tmp_path, include_builtin_mcp=True)

    plan = await agent.run_full_campaign_pipeline(
        SAMPLE_DOCS_DIR,
        focus_area="EMEA mid-market SaaS",
        monthly_budget_usd=15_000.0,
    )

    assert isinstance(plan, MarketingPlan)
    assert plan.organization_profile.organization_name == "NovaFlow Analytics"
    assert set(plan.organization_profile.source_documents) >= {
        "company_overview.md",
        "marketing_brief.md",
        "product_catalog.json",
    }
    assert plan.audience_report.signals_analyzed > 0
    assert plan.audience_report.segments[0].fit_score >= plan.audience_report.segments[-1].fit_score
    assert len(plan.campaigns) == 2
    assert len(plan.next_steps) == 3

    # Verify cached state and indexed vector knowledge base
    assert agent.current_profile is plan.organization_profile
    assert agent.current_audience_report is plan.audience_report
    assert agent.knowledge_base is not None
    assert len(agent.knowledge_base) > 0
    rag_hits = agent.knowledge_base.search("Frankfurt GDPR compliance", k=2)
    assert len(rag_hits) > 0

    # Verify MCP tools were loaded from real FastMCP server and enriched strategy synthesis
    tool_names = {t.name for t in agent.tools}
    assert {
        "analyze_organization_documents",
        "search_organization_knowledge",
        "scan_market_signals",
        "discover_target_audiences",
        "evaluate_audience_segment",
        "get_channel_benchmarks",
        "estimate_campaign_budget",
        "build_utm_tracking_links",
        "suggest_audience_channels",
    }.issubset(tool_names)

    strategy_calls = [msgs for schema, msgs in fake_llm.calls if schema is StrategyDraft]
    assert len(strategy_calls) == 1
    strategy_prompt_text = str(strategy_calls[0][-1].content)
    assert "get_channel_benchmarks" in strategy_prompt_text
    assert "suggest_audience_channels" in strategy_prompt_text
    assert "estimate_campaign_budget" in strategy_prompt_text
    assert '"industry": "saas"' in strategy_prompt_text
    assert '"monthly_budget_usd": 15000.0' in strategy_prompt_text


async def test_pipeline_supports_prefixed_mcp_tools_and_isolates_tool_errors(tmp_path: Path) -> None:
    profile = _sample_profile()
    report = _sample_audience_report()
    draft = _sample_strategy_draft()

    fake_llm = FakeStructuredLLM(
        {
            OrganizationProfile: profile,
            ScanQueryPlan: ScanQueryPlan(queries=["revops churn prediction"]),
            TargetAudienceReport: report,
            StrategyDraft: draft,
        }
    )

    @tool(name_or_callable="closr-marketing_get_channel_benchmarks")
    async def prefixed_benchmarks(industry: str = "saas", channel: str = "all") -> list[dict[str, str]]:
        """Prefixed mock benchmark tool returning LangChain content blocks."""
        return [
            {
                "type": "text",
                "text": json.dumps({"industry": industry, "channel": channel, "benchmarks": {"linkedin": {"avg_cpc_usd": 6.67}}}),
            }
        ]

    @tool(name_or_callable="closr-marketing_estimate_campaign_budget")
    async def failing_budget(
        monthly_budget_usd: float,
        primary_goal: str = "lead_generation",
        channels: list[str] | None = None,
        industry: str = "saas",
    ) -> str:
        """Simulated failing MCP tool."""
        raise RuntimeError("Simulated upstream budget MCP timeout")

    pipeline = build_marketing_pipeline(
        org_analyzer=OrganizationAnalyzer(llm=fake_llm),  # type: ignore[arg-type]
        audience_discovery=TargetAudienceDiscovery(
            llm=fake_llm,  # type: ignore[arg-type]
            scanner=OfflineHeuristicScanner(),
        ),
        strategy_llm=fake_llm,  # type: ignore[arg-type]
        mcp_tools=[prefixed_benchmarks, failing_budget],
    )

    final_state = await pipeline.ainvoke(
        {
            "organization_profile": profile,
            "monthly_budget_usd": 12_000.0,
            "mcp_insights": {},
            "errors": [],
        }
    )

    assert final_state["marketing_plan"] is not None
    assert final_state["mcp_insights"]["get_channel_benchmarks"]["industry"] == "saas"
    assert len(final_state["errors"]) == 1
    assert "Simulated upstream budget MCP timeout" in final_state["errors"][0]


# ---------------------------------------------------------------------------
# MarketingAgent.chat() tests (tool calling + thread memory + real MCP tool)
# ---------------------------------------------------------------------------


async def test_agent_chat_tool_calling_and_thread_memory(tmp_path: Path) -> None:
    fake_chat = FakeToolCallingChatModel(
        tool_to_call="search_organization_knowledge",
        tool_args={"query": "Frankfurt GDPR compliance", "k": 2},
    )
    agent, _ = _make_test_agent(
        tmp_path,
        include_builtin_mcp=False,
        agent_llm=fake_chat,
    )
    await agent.analyze_organization(SAMPLE_DOCS_DIR)

    # Turn 1: invokes real search_organization_knowledge tool against indexed sample docs
    reply_1 = await agent.chat(
        "Search our organization docs for GDPR and Frankfurt data residency.",
        thread_id="thread-alpha",
    )
    assert "Tool 'search_organization_knowledge' returned evidence:" in reply_1
    assert "[source:" in reply_1

    # Turn 2 on same thread: InMemorySaver retains Turn 1 messages
    reply_2 = await agent.chat(
        "Summarize what we just discussed.",
        thread_id="thread-alpha",
    )
    assert "Follow-up summary across 2 user turns" in reply_2
    assert "first='Search our organization docs for GDPR and Frankfurt data residency.'" in reply_2

    # Separate thread has isolated memory
    reply_other = await agent.chat("Hello on a fresh thread.", thread_id="thread-beta")
    assert "Follow-up summary across 1 user turns" in reply_other
    assert "first='Hello on a fresh thread.'" in reply_other


async def test_agent_chat_can_invoke_real_mcp_tool(tmp_path: Path) -> None:
    fake_chat = FakeToolCallingChatModel(
        tool_to_call="get_channel_benchmarks",
        tool_args={"industry": "saas", "channel": "linkedin"},
    )
    agent, _ = _make_test_agent(
        tmp_path,
        include_builtin_mcp=True,
        agent_llm=fake_chat,
    )

    reply = await agent.chat(
        "Check SaaS LinkedIn benchmark metrics for us.",
        thread_id="thread-mcp",
    )
    assert "Tool 'get_channel_benchmarks' returned evidence:" in reply
    assert "saas" in reply.lower()


async def test_agent_chat_syncs_profile_and_audience_state_across_tool_calls(tmp_path: Path) -> None:
    fake_chat = FakeToolCallingChatModel(
        tool_to_call="analyze_organization_documents",
        tool_args={"path": str(SAMPLE_DOCS_DIR)},
    )
    agent, _ = _make_test_agent(
        tmp_path,
        include_builtin_mcp=False,
        agent_llm=fake_chat,
    )
    assert agent.current_profile is None
    assert agent.current_audience_report is None

    # Agent autonomously calls analyze_organization_documents during chat
    reply = await agent.chat(
        "Search and analyze the docs directory.",
        thread_id="thread-sync",
    )
    assert "Tool 'analyze_organization_documents' returned evidence:" in reply
    assert agent.current_profile is not None
    assert agent.current_profile.organization_name == "NovaFlow Analytics"

    # Follow-up discover_target_audiences tool call works without explicit JSON because current_profile synced
    discover_tool = next(t for t in agent.tools if t.name == "discover_target_audiences")
    report_json = await discover_tool.ainvoke({"focus_area": "EMEA SaaS"})
    parsed_report = TargetAudienceReport.model_validate_json(report_json)
    assert parsed_report.organization_name == "NovaFlow Analytics"
    assert agent.current_audience_report is not None
    assert agent.current_audience_report.recommended_primary_segment == parsed_report.recommended_primary_segment


# ---------------------------------------------------------------------------
# Typer CLI tests
# ---------------------------------------------------------------------------


def test_cli_help_lists_all_commands() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for cmd in ("analyze", "discover", "campaign", "chat", "mcp-tools"):
        assert cmd in result.output


def test_cli_mcp_tools_discovers_builtin_server(tmp_path: Path) -> None:
    runner = CliRunner()
    missing_cfg = tmp_path / "missing_mcp.json"
    result = runner.invoke(
        app,
        ["mcp-tools", "--config", str(missing_cfg)],
        env={"COLUMNS": "120"},
    )
    assert result.exit_code == 0
    assert "closr-marketing" in result.output
    assert "get_channel_benchmarks" in result.output
    assert "estimate_campaign_budget" in result.output
    assert "build_utm_tracking_links" in result.output
    assert "suggest_audience_channels" in result.output


def test_cli_analyze_discover_and_campaign_with_injected_agent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = CliRunner()
    recorded_no_mcp_flags: list[bool] = []

    def _fake_create_agent(
        *,
        provider: Any = None,
        offline: bool = False,
        no_mcp: bool = False,
    ) -> MarketingAgent:
        recorded_no_mcp_flags.append(no_mcp)
        agent, _ = _make_test_agent(tmp_path, include_builtin_mcp=not no_mcp)
        return agent

    monkeypatch.setattr("closr.cli._create_agent", _fake_create_agent)

    profile_out = tmp_path / "out" / "profile.json"
    analyze_res = runner.invoke(
        app,
        ["analyze", str(SAMPLE_DOCS_DIR), "--output", str(profile_out)],
    )
    assert analyze_res.exit_code == 0
    assert "NovaFlow Analytics" in analyze_res.output
    assert profile_out.is_file()
    saved_profile = OrganizationProfile.model_validate_json(profile_out.read_text(encoding="utf-8"))
    assert saved_profile.organization_name == "NovaFlow Analytics"

    audience_out = tmp_path / "out" / "audience.json"
    discover_res = runner.invoke(
        app,
        ["discover", str(SAMPLE_DOCS_DIR), "--offline", "--output", str(audience_out)],
    )
    assert discover_res.exit_code == 0
    assert "Mid-Market B2B SaaS RevOps Leaders" in discover_res.output
    saved_report = TargetAudienceReport.model_validate_json(audience_out.read_text(encoding="utf-8"))
    assert saved_report.recommended_primary_segment == "Mid-Market B2B SaaS RevOps Leaders"

    plan_out = tmp_path / "out" / "plan.json"
    campaign_res = runner.invoke(
        app,
        [
            "campaign",
            str(SAMPLE_DOCS_DIR),
            "--offline",
            "--budget",
            "15000",
            "--no-mcp",
            "--output",
            str(plan_out),
        ],
    )
    assert campaign_res.exit_code == 0
    assert "90-Day Churn Early Warning ABM" in campaign_res.output
    assert recorded_no_mcp_flags[-1] is True
    saved_plan = MarketingPlan.model_validate_json(plan_out.read_text(encoding="utf-8"))
    assert len(saved_plan.campaigns) == 2

    # Error handling check on missing path
    bad_res = runner.invoke(app, ["analyze", str(tmp_path / "does_not_exist")])
    assert bad_res.exit_code == 1
    assert "Error:" in bad_res.output


def test_cli_no_mcp_disables_all_configured_and_builtin_servers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mcp_cfg = tmp_path / "mcp_servers.json"
    mcp_cfg.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "external-crm": {
                        "transport": "streamable_http",
                        "url": "https://mcp.example.com/crm",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("CLOSR_MCP_CONFIG_PATH", str(mcp_cfg))

    agent_with_mcp = _create_agent(no_mcp=False)
    assert set(agent_with_mcp.mcp_manager.servers.keys()) == {"external-crm", "closr-marketing"}

    agent_no_mcp = _create_agent(no_mcp=True)
    assert agent_no_mcp.mcp_manager.servers == {}
