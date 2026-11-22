"""LangGraph workflow orchestrator and autonomous MarketingAgent."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, TypeVar

from langchain.agents import create_agent
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from langchain_core.runnables.fallbacks import RunnableWithFallbacks
from langchain_core.tools import BaseTool, StructuredTool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field, ValidationError

from closr.agent.prompts import (
    MARKETING_AGENT_SYSTEM_PROMPT,
    STRATEGY_SYNTHESIS_SYSTEM_PROMPT,
)
from closr.agent.state import MarketingWorkflowState
from closr.audience import (
    AudienceScanner,
    TargetAudienceDiscovery,
    create_audience_tools,
)
from closr.config import ClosrSettings, LLMProvider, get_settings
from closr.documents import (
    OrganizationAnalyzer,
    OrganizationKnowledgeBase,
    create_document_tools,
)
from closr.llm import LLMFactory, TaskType
from closr.mcp import MCPManager
from closr.schemas import (
    CampaignRecommendation,
    MarketingPlan,
    OrganizationProfile,
    TargetAudienceReport,
)

logger = logging.getLogger(__name__)

SchemaT = TypeVar("SchemaT", bound=BaseModel)

_SUPPORTED_MCP_CHANNELS: dict[str, str] = {
    "linkedin": "linkedin",
    "linkedin_ads": "linkedin",
    "google": "google_search",
    "google_ads": "google_search",
    "google_search": "google_search",
    "search": "google_search",
    "sem": "google_search",
    "paid_search": "google_search",
    "meta": "meta",
    "meta_ads": "meta",
    "facebook": "meta",
    "instagram": "meta",
    "tiktok": "tiktok",
    "tiktok_ads": "tiktok",
    "youtube": "youtube",
    "youtube_ads": "youtube",
    "email": "email",
    "newsletter": "email",
}


class StrategyDraft(BaseModel):
    """Structured strategy payload synthesized before assembling the full `MarketingPlan`."""

    executive_summary: str = Field(
        description="High-level strategic summary of the marketing plan and expected impact.",
    )
    campaigns: list[CampaignRecommendation] = Field(
        default_factory=list,
        description="Prioritized list of recommended marketing campaigns.",
    )
    next_steps: list[str] = Field(
        default_factory=list,
        description="Concrete, ordered next steps for execution.",
    )


def _parse_mcp_tool_output(raw: Any) -> Any:
    """Normalize MCP tool outputs (LangChain content blocks, JSON strings, or raw values)."""
    if raw is None:
        return None

    if isinstance(raw, str):
        stripped = raw.strip()
        if not stripped:
            return raw
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            return raw

    if isinstance(raw, dict):
        if (
            raw.get("type") == "text"
            and isinstance(raw.get("text"), str)
            and set(raw.keys()).issubset({"type", "text", "id", "annotations"})
        ):
            return _parse_mcp_tool_output(raw["text"])
        return raw

    if isinstance(raw, list):
        has_text_block = any(
            isinstance(item, dict)
            and item.get("type") == "text"
            and isinstance(item.get("text"), str)
            for item in raw
        )
        if not has_text_block:
            return raw

        texts: list[str] = []
        for item in raw:
            if (
                isinstance(item, dict)
                and item.get("type") == "text"
                and isinstance(item.get("text"), str)
            ):
                texts.append(item["text"])
            elif isinstance(item, str):
                texts.append(item)

        if not texts:
            return raw
        if len(texts) == 1:
            return _parse_mcp_tool_output(texts[0])

        joined = "\n".join(texts).strip()
        try:
            return json.loads(joined)
        except json.JSONDecodeError:
            parsed_parts: list[Any] = []
            all_json = True
            for text in texts:
                parsed = _parse_mcp_tool_output(text)
                if isinstance(parsed, str) and parsed == text:
                    all_json = False
                parsed_parts.append(parsed)
            return parsed_parts if all_json else joined

    return raw


def _bind_structured_output(
    llm: BaseChatModel | Runnable[Any, Any],
    schema: type[SchemaT],
) -> Runnable[Any, Any]:
    """Bind `schema` via `with_structured_output`, supporting `RunnableWithFallbacks`."""
    with_structured = getattr(llm, "with_structured_output", None)
    if callable(with_structured):
        return with_structured(schema)

    if isinstance(llm, RunnableWithFallbacks):
        primary = _bind_structured_output(llm.runnable, schema)
        fallbacks = [_bind_structured_output(fb, schema) for fb in llm.fallbacks]
        return primary.with_fallbacks(fallbacks)

    raise TypeError(
        f"LLM object of type {type(llm).__name__} does not support 'with_structured_output'."
    )


def _validate_structured_result(result: Any, schema: type[SchemaT]) -> SchemaT:
    """Validate structured output returned as a model instance, dict, message, or JSON string."""
    if isinstance(result, schema):
        return result
    if isinstance(result, BaseMessage):
        content = result.content
        if isinstance(content, str):
            return schema.model_validate_json(content)
        raise TypeError(
            f"Expected string message content for {schema.__name__}, got {type(content)!r}"
        )
    if isinstance(result, str):
        return schema.model_validate_json(result)
    return schema.model_validate(result)


def _find_mcp_tool(tools: Sequence[BaseTool], target_name: str) -> BaseTool | None:
    """Find an MCP tool by exact name or prefixed suffix (when `tool_name_prefix=True`)."""
    for tool in tools:
        if tool.name == target_name:
            return tool
    for tool in tools:
        for sep in ("_", ":", ".", "-"):
            if tool.name.endswith(f"{sep}{target_name}"):
                return tool
    return None


def _infer_mcp_industry(industry: str) -> str:
    """Map free-form industry descriptions to built-in MCP benchmark verticals."""
    normalized = industry.strip().lower()
    keyword_map = (
        ("saas", "saas"),
        ("software", "saas"),
        ("cloud", "saas"),
        ("ecommerce", "ecommerce"),
        ("e-commerce", "ecommerce"),
        ("retail", "ecommerce"),
        ("fintech", "fintech"),
        ("finance", "fintech"),
        ("banking", "fintech"),
        ("healthcare", "healthcare"),
        ("health", "healthcare"),
        ("medtech", "healthcare"),
        ("education", "education"),
        ("edtech", "education"),
        ("b2b", "b2b"),
    )
    for keyword, canonical in keyword_map:
        if keyword in normalized:
            return canonical
    return industry.strip() or "general"


def _infer_business_model(
    profile: OrganizationProfile,
    report: TargetAudienceReport,
) -> str:
    """Infer MCP-compatible business model (`b2b`, `b2c`, `b2b2c`, `marketplace`)."""
    text = " ".join(
        [
            profile.industry,
            profile.summary,
            *profile.existing_audience_hints,
            *(seg.demographics_or_firmographics for seg in report.segments),
        ]
    ).lower()
    if "b2b2c" in text:
        return "b2b2c"
    if "marketplace" in text:
        return "marketplace"
    if "b2c" in text or "direct-to-consumer" in text or "consumer" in text:
        if "b2b" not in text:
            return "b2c"
    return "b2b"


def _infer_target_roles(
    profile: OrganizationProfile,
    report: TargetAudienceReport,
) -> list[str]:
    """Extract a deduplicated list of target buyer roles from audience segments and profile."""
    roles: list[str] = []
    for segment in report.segments:
        roles.extend(segment.roles_or_personas)
    if not roles:
        roles.extend(profile.existing_audience_hints)
    cleaned = list(dict.fromkeys(r.strip() for r in roles if r.strip()))
    return cleaned or ["Marketing Leader"]


def _infer_primary_goal(profile: OrganizationProfile) -> str:
    """Infer MCP-compatible primary goal (`awareness`, `lead_generation`, `conversions`, `retention`)."""
    blob = " ".join([*profile.business_goals, *profile.marketing_needs]).lower()
    if any(tok in blob for tok in ("retention", "churn", "renewal", "nrr", "expansion")):
        if not any(tok in blob for tok in ("pipeline", "lead", "sql", "bookings", "demand")):
            return "retention"
    if any(tok in blob for tok in ("awareness", "brand", "reach")):
        if not any(tok in blob for tok in ("pipeline", "lead", "sql", "conversion", "sales")):
            return "awareness"
    if any(tok in blob for tok in ("conversion", "checkout", "purchase", "ecommerce")):
        if not any(tok in blob for tok in ("pipeline", "lead", "sql", "b2b")):
            return "conversions"
    return "lead_generation"


def _map_supported_mcp_channels(channels: Sequence[str]) -> list[str] | None:
    """Map free-form channel names to supported MCP channel keys, or None if none match."""
    mapped: list[str] = []
    for raw in channels:
        key = raw.strip().lower().replace(" ", "_").replace("-", "_")
        canonical = _SUPPORTED_MCP_CHANNELS.get(key)
        if canonical and canonical not in mapped:
            mapped.append(canonical)
    return mapped or None


def _extract_assistant_text(result: Any) -> str:
    """Extract the final assistant text response from an agent graph invocation result."""
    if isinstance(result, str):
        return result
    if not isinstance(result, dict):
        return str(result)

    messages = result.get("messages") or []
    if not messages:
        return ""

    last_message = messages[-1]
    content = getattr(last_message, "content", last_message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(part for part in parts if part).strip()
    return str(content)


class _LazyTaskLLM:
    """Lazy wrapper deferring LLMFactory lookup until an organization/audience tool is invoked."""

    def __init__(
        self,
        factory: LLMFactory,
        task: TaskType,
        provider: LLMProvider | None = None,
    ) -> None:
        self._factory = factory
        self._task = task
        self._provider = provider

    def _resolve(self) -> BaseChatModel:
        if self._provider is not None:
            return self._factory.get_chat_model(provider=self._provider)
        return self._factory.get_model_for_task(self._task)

    def with_structured_output(self, schema: type[Any]) -> Any:
        return self._resolve().with_structured_output(schema)

    def invoke(self, *args: Any, **kwargs: Any) -> Any:
        return self._resolve().invoke(*args, **kwargs)

    async def ainvoke(self, *args: Any, **kwargs: Any) -> Any:
        return await self._resolve().ainvoke(*args, **kwargs)


def build_marketing_pipeline(
    org_analyzer: OrganizationAnalyzer,
    audience_discovery: TargetAudienceDiscovery,
    strategy_llm: BaseChatModel | Runnable[Any, Any],
    mcp_tools: Sequence[BaseTool] = (),
) -> CompiledStateGraph[Any, Any, Any, Any]:
    """Build the 4-stage LangGraph marketing pipeline.

    Flow: `START -> analyze_organization -> discover_audience -> enrich_with_mcp -> synthesize_plan -> END`.
    """

    async def analyze_organization(state: MarketingWorkflowState) -> dict[str, Any]:
        existing = state.get("organization_profile")
        if existing is not None:
            return {"organization_profile": existing}

        docs_path = state.get("docs_path")
        if not docs_path:
            raise ValueError(
                "Either 'organization_profile' or 'docs_path' must be provided to run the pipeline."
            )
        profile = await org_analyzer.aanalyze_paths(docs_path)
        return {"organization_profile": profile}

    async def discover_audience(state: MarketingWorkflowState) -> dict[str, Any]:
        profile = state.get("organization_profile")
        if profile is None:
            raise ValueError("Cannot discover audiences without an 'organization_profile'.")
        report = await audience_discovery.discover_audiences(
            profile,
            focus_area=state.get("focus_area"),
        )
        return {"audience_report": report}

    async def enrich_with_mcp(state: MarketingWorkflowState) -> dict[str, Any]:
        profile = state.get("organization_profile")
        report = state.get("audience_report")
        if profile is None or report is None:
            raise ValueError("Both 'organization_profile' and 'audience_report' are required for MCP enrichment.")

        insights: dict[str, Any] = dict(state.get("mcp_insights") or {})
        errors: list[str] = list(state.get("errors") or [])

        industry = _infer_mcp_industry(profile.industry)
        business_model = _infer_business_model(profile, report)
        target_roles = _infer_target_roles(profile, report)
        region = ", ".join(profile.geographic_focus) if profile.geographic_focus else "global"
        primary_goal = _infer_primary_goal(profile)
        mapped_channels = _map_supported_mcp_channels(report.channel_priority)

        raw_budget = state.get("monthly_budget_usd")
        effective_budget = (
            float(raw_budget)
            if (raw_budget is not None and raw_budget > 0)
            else (10_000.0 if raw_budget is None else 0.0)
        )

        tool_invocations: list[tuple[str, dict[str, Any]]] = [
            ("get_channel_benchmarks", {"industry": industry, "channel": "all"}),
            (
                "suggest_audience_channels",
                {
                    "business_model": business_model,
                    "target_roles": target_roles,
                    "region": region,
                },
            ),
        ]
        if effective_budget > 0:
            budget_args: dict[str, Any] = {
                "monthly_budget_usd": effective_budget,
                "primary_goal": primary_goal,
                "industry": industry,
            }
            if mapped_channels is not None:
                budget_args["channels"] = mapped_channels
            tool_invocations.append(("estimate_campaign_budget", budget_args))

        for tool_key, args in tool_invocations:
            tool = _find_mcp_tool(mcp_tools, tool_key)
            if tool is None:
                continue
            try:
                raw_output = await tool.ainvoke(args)
                parsed = _parse_mcp_tool_output(raw_output)
                if isinstance(parsed, str) and parsed.lower().startswith("error"):
                    errors.append(f"MCP tool '{tool.name}' returned error: {parsed}")
                else:
                    insights[tool_key] = parsed
            except Exception as exc:
                logger.warning("MCP enrichment tool %r failed: %s", tool.name, exc)
                errors.append(f"MCP tool '{tool.name}' failed: {exc}")

        return {"mcp_insights": insights, "errors": errors}

    async def synthesize_plan(state: MarketingWorkflowState) -> dict[str, Any]:
        profile = state.get("organization_profile")
        report = state.get("audience_report")
        if profile is None or report is None:
            raise ValueError("Cannot synthesize MarketingPlan without profile and audience report.")

        mcp_insights = state.get("mcp_insights") or {}
        errors = state.get("errors") or []
        focus_area = state.get("focus_area")
        monthly_budget = state.get("monthly_budget_usd")

        context_sections = [
            f"Organization Profile:\n{profile.model_dump_json(indent=2)}",
            f"Target Audience Report:\n{report.model_dump_json(indent=2)}",
            f"MCP Insights:\n{json.dumps(mcp_insights, indent=2, default=str)}",
        ]
        if focus_area:
            context_sections.append(f"Focus Area: {focus_area}")
        if monthly_budget is not None:
            context_sections.append(f"Target Monthly Budget (USD): {monthly_budget:,.2f}")
        if errors:
            context_sections.append("Workflow Warnings/Errors:\n" + "\n".join(f"- {e}" for e in errors))

        messages = [
            SystemMessage(content=STRATEGY_SYNTHESIS_SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    "Synthesize an actionable StrategyDraft grounded in the following inputs:\n\n"
                    + "\n\n".join(context_sections)
                )
            ),
        ]

        structured_llm = _bind_structured_output(strategy_llm, StrategyDraft)
        raw_draft = await structured_llm.ainvoke(messages)
        draft = _validate_structured_result(raw_draft, StrategyDraft)

        plan = MarketingPlan(
            organization_profile=profile,
            audience_report=report,
            executive_summary=draft.executive_summary,
            campaigns=draft.campaigns,
            next_steps=draft.next_steps,
        )
        return {"marketing_plan": plan}

    graph: StateGraph[MarketingWorkflowState] = StateGraph(MarketingWorkflowState)
    graph.add_node("analyze_organization", analyze_organization)
    graph.add_node("discover_audience", discover_audience)
    graph.add_node("enrich_with_mcp", enrich_with_mcp)
    graph.add_node("synthesize_plan", synthesize_plan)

    graph.add_edge(START, "analyze_organization")
    graph.add_edge("analyze_organization", "discover_audience")
    graph.add_edge("discover_audience", "enrich_with_mcp")
    graph.add_edge("enrich_with_mcp", "synthesize_plan")
    graph.add_edge("synthesize_plan", END)

    return graph.compile()


class MarketingAgent:
    """High-level orchestrator combining document RAG, audience discovery, MCP tools, and LangGraph."""

    def __init__(
        self,
        settings: ClosrSettings | None = None,
        *,
        llm_factory: LLMFactory | None = None,
        doc_llm: BaseChatModel | Runnable[Any, Any] | None = None,
        audience_llm: BaseChatModel | Runnable[Any, Any] | None = None,
        strategy_llm: BaseChatModel | Runnable[Any, Any] | None = None,
        agent_llm: BaseChatModel | None = None,
        embeddings: Embeddings | None = None,
        scanner: AudienceScanner | None = None,
        mcp_manager: MCPManager | None = None,
        include_builtin_mcp: bool = True,
        provider: LLMProvider | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.llm_factory = llm_factory or LLMFactory(self.settings)
        self.provider = provider

        self._doc_llm = doc_llm
        self._audience_llm = audience_llm
        self._strategy_llm = strategy_llm
        self._agent_llm = agent_llm
        self._embeddings = embeddings
        self._scanner = scanner
        self._include_builtin_mcp = include_builtin_mcp

        self.mcp_manager: MCPManager = (
            mcp_manager
            if mcp_manager is not None
            else MCPManager.from_config_file(
                self.settings.mcp_config_path,
                include_builtin_marketing_server=include_builtin_mcp,
            )
        )

        self.knowledge_base: OrganizationKnowledgeBase | None = self._build_knowledge_base()
        self.current_profile: OrganizationProfile | None = None
        self.current_audience_report: TargetAudienceReport | None = None

        self._mcp_tools: list[BaseTool] = []
        self._tools: list[BaseTool] = []
        self._mcp_loaded: bool = False
        self._checkpointer = InMemorySaver()
        self._chat_agent: CompiledStateGraph[Any, Any, Any, Any] | None = None

    def _build_knowledge_base(self) -> OrganizationKnowledgeBase | None:
        embeddings = self._embeddings
        if embeddings is None:
            try:
                embeddings = self.llm_factory.get_embeddings(provider=self.provider)
            except ValueError:
                return None

        persist_path = self.settings.vectorstore_dir / "org_kb.json"
        return OrganizationKnowledgeBase(embeddings=embeddings, persist_path=persist_path)

    def _ensure_knowledge_base(self) -> OrganizationKnowledgeBase | None:
        if self.knowledge_base is None:
            self.knowledge_base = self._build_knowledge_base()
        return self.knowledge_base

    def _resolve_task_llm(
        self,
        injected: BaseChatModel | Runnable[Any, Any] | None,
        task: TaskType,
    ) -> BaseChatModel | Runnable[Any, Any]:
        if injected is not None:
            return injected
        if self.provider is not None:
            return self.llm_factory.get_chat_model(provider=self.provider)
        return self.llm_factory.get_model_for_task(task)

    def _get_org_analyzer(self, *, allow_lazy: bool = False) -> OrganizationAnalyzer:
        kb = self._ensure_knowledge_base()
        if allow_lazy and self._doc_llm is None and not self.settings.get_available_providers():
            llm: BaseChatModel | Runnable[Any, Any] = _LazyTaskLLM(  # type: ignore[assignment]
                self.llm_factory,
                "document_analysis",
                self.provider,
            )
        else:
            llm = self._resolve_task_llm(self._doc_llm, "document_analysis")
        return OrganizationAnalyzer(llm=llm, knowledge_base=kb)

    def _get_audience_discovery(self, *, allow_lazy: bool = False) -> TargetAudienceDiscovery:
        if allow_lazy and self._audience_llm is None and not self.settings.get_available_providers():
            llm: BaseChatModel | Runnable[Any, Any] = _LazyTaskLLM(  # type: ignore[assignment]
                self.llm_factory,
                "audience_discovery",
                self.provider,
            )
        else:
            llm = self._resolve_task_llm(self._audience_llm, "audience_discovery")
        return TargetAudienceDiscovery(llm=llm, scanner=self._scanner)

    def _get_strategy_llm(self) -> BaseChatModel | Runnable[Any, Any]:
        return self._resolve_task_llm(self._strategy_llm, "strategy")

    def _get_agent_llm(self) -> BaseChatModel:
        if self._agent_llm is not None:
            return self._agent_llm
        if self.provider is not None:
            return self.llm_factory.get_chat_model(provider=self.provider)
        return self.llm_factory.get_model_for_task("agent")

    def _maybe_cache_profile(self, output: Any) -> None:
        if not isinstance(output, str):
            return
        try:
            self.current_profile = OrganizationProfile.model_validate_json(output)
        except ValidationError:
            pass

    def _maybe_cache_audience_report(self, output: Any) -> None:
        if not isinstance(output, str):
            return
        try:
            self.current_audience_report = TargetAudienceReport.model_validate_json(output)
        except ValidationError:
            pass

    @staticmethod
    def _wrap_tool_with_hook(
        base_tool: BaseTool,
        on_output: Callable[[Any], None],
    ) -> BaseTool:
        def _sync_call(**kwargs: Any) -> Any:
            output = base_tool.invoke(kwargs)
            on_output(output)
            return output

        async def _async_call(**kwargs: Any) -> Any:
            output = await base_tool.ainvoke(kwargs)
            on_output(output)
            return output

        return StructuredTool.from_function(
            func=_sync_call,
            coroutine=_async_call,
            name=base_tool.name,
            description=base_tool.description,
            args_schema=base_tool.args_schema,
        )

    async def initialize(self, *, reload_mcp: bool = False) -> None:
        """Load MCP tools (if needed) and register local + MCP tools for the agent."""
        if not self._mcp_loaded or reload_mcp:
            self._mcp_tools = await self.mcp_manager.load_tools()
            self._mcp_loaded = True
            self._chat_agent = None

        raw_doc_tools = create_document_tools(
            self._get_org_analyzer(allow_lazy=True),
            knowledge_base=self._ensure_knowledge_base(),
        )
        doc_tools = [
            self._wrap_tool_with_hook(t, self._maybe_cache_profile)
            if t.name == "analyze_organization_documents"
            else t
            for t in raw_doc_tools
        ]

        raw_audience_tools = create_audience_tools(
            self._get_audience_discovery(allow_lazy=True),
            profile_getter=lambda: self.current_profile,
        )
        audience_tools = [
            self._wrap_tool_with_hook(t, self._maybe_cache_audience_report)
            if t.name == "discover_target_audiences"
            else t
            for t in raw_audience_tools
        ]
        self._tools = [*doc_tools, *audience_tools, *self._mcp_tools]

    @property
    def tools(self) -> list[BaseTool]:
        """Return all currently registered LangChain and MCP tools."""
        return list(self._tools)

    async def analyze_organization(self, docs_path: Path | str) -> OrganizationProfile:
        """Load and analyze organization documents, updating `self.current_profile`."""
        analyzer = self._get_org_analyzer(allow_lazy=False)
        profile = await analyzer.aanalyze_paths(docs_path)
        self.current_profile = profile
        return profile

    async def discover_target_audience(
        self,
        profile: OrganizationProfile | None = None,
        focus_area: str | None = None,
    ) -> TargetAudienceReport:
        """Discover and prioritize target audience segments for `profile` or `current_profile`."""
        target_profile = profile if profile is not None else self.current_profile
        if target_profile is None:
            raise ValueError(
                "No OrganizationProfile provided or cached. Run analyze_organization first."
            )
        discovery = self._get_audience_discovery(allow_lazy=False)
        report = await discovery.discover_audiences(target_profile, focus_area=focus_area)
        self.current_audience_report = report
        return report

    async def run_full_campaign_pipeline(
        self,
        docs_path: Path | str | None = None,
        *,
        profile: OrganizationProfile | None = None,
        focus_area: str | None = None,
        monthly_budget_usd: float | None = None,
    ) -> MarketingPlan:
        """Execute the end-to-end LangGraph workflow and return a validated `MarketingPlan`."""
        await self.initialize()

        effective_profile = profile
        if effective_profile is None and docs_path is None:
            effective_profile = self.current_profile

        pipeline = build_marketing_pipeline(
            org_analyzer=self._get_org_analyzer(allow_lazy=False),
            audience_discovery=self._get_audience_discovery(allow_lazy=False),
            strategy_llm=self._get_strategy_llm(),
            mcp_tools=self._mcp_tools,
        )
        initial_state: MarketingWorkflowState = {
            "docs_path": str(docs_path) if docs_path is not None else None,
            "focus_area": focus_area,
            "monthly_budget_usd": monthly_budget_usd,
            "organization_profile": effective_profile,
            "mcp_insights": {},
            "errors": [],
        }

        final_state = await pipeline.ainvoke(initial_state)
        resolved_profile = final_state.get("organization_profile")
        resolved_report = final_state.get("audience_report")
        plan = final_state.get("marketing_plan")

        if resolved_profile is not None:
            self.current_profile = resolved_profile
        if resolved_report is not None:
            self.current_audience_report = resolved_report
        if plan is None:
            raise RuntimeError("Marketing pipeline finished without producing a MarketingPlan.")
        return plan

    async def chat(self, message: str, thread_id: str = "default") -> str:
        """Run a conversational turn with tool-calling and per-thread memory."""
        await self.initialize()
        if self._chat_agent is None:
            agent_llm = self._get_agent_llm()
            self._chat_agent = create_agent(
                model=agent_llm,
                tools=self.tools,
                system_prompt=MARKETING_AGENT_SYSTEM_PROMPT,
                checkpointer=self._checkpointer,
            )

        result = await self._chat_agent.ainvoke(
            {"messages": [HumanMessage(content=message)]},
            config={"configurable": {"thread_id": thread_id}},
        )
        return _extract_assistant_text(result)
