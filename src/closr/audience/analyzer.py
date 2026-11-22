"""LLM-driven target audience discovery, query planning, and ICP fit evaluation."""

from __future__ import annotations

from typing import Any, TypeVar

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from langchain_core.runnables.fallbacks import RunnableWithFallbacks
from pydantic import BaseModel, Field

from closr.audience.scanner import (
    AudienceScanner,
    CompositeAudienceScanner,
    DuckDuckGoWebScanner,
    OfflineHeuristicScanner,
    deduplicate_signals,
)
from closr.schemas import (
    AudienceSegment,
    MarketSignal,
    OrganizationProfile,
    TargetAudienceReport,
)

SchemaT = TypeVar("SchemaT", bound=BaseModel)


class ScanQueryPlan(BaseModel):
    """Structured plan of search queries for external market and audience discovery."""

    queries: list[str] = Field(
        default_factory=list,
        description=(
            "Targeted search queries designed to uncover buyer communities, "
            "pain-point discussions, competitor alternatives, and audience intent signals."
        ),
    )


def _bind_structured_output(
    llm: BaseChatModel | Runnable[Any, Any],
    schema: type[SchemaT],
) -> Runnable[Any, Any]:
    """Bind `schema` via `with_structured_output`, including `RunnableWithFallbacks`."""
    if hasattr(llm, "with_structured_output"):
        return llm.with_structured_output(schema)  # type: ignore[union-attr]

    if isinstance(llm, RunnableWithFallbacks):
        primary = _bind_structured_output(llm.runnable, schema)
        fallbacks = [_bind_structured_output(fb, schema) for fb in llm.fallbacks]
        return primary.with_fallbacks(fallbacks)

    raise TypeError(
        f"LLM object of type {type(llm).__name__} does not support 'with_structured_output'."
    )


def _validate_structured_result(result: Any, schema: type[SchemaT]) -> SchemaT:
    if isinstance(result, schema):
        return result
    return schema.model_validate(result)


def _format_profile_context(profile: OrganizationProfile) -> str:
    return profile.model_dump_json(indent=2)


def _format_signals_context(signals: list[MarketSignal], max_signals: int = 40) -> str:
    if not signals:
        return "No external market signals were returned; rely on organization profile evidence and note research gaps."

    lines: list[str] = []
    for idx, signal in enumerate(signals[:max_signals], start=1):
        keywords = ", ".join(signal.detected_keywords) if signal.detected_keywords else "none"
        lines.append(
            f"[{idx}] source={signal.source} | platform={signal.platform_hint or 'unknown'} | "
            f"query={signal.query!r}\n"
            f"    title: {signal.title}\n"
            f"    url: {signal.url or 'N/A'}\n"
            f"    keywords: {keywords}\n"
            f"    snippet: {signal.snippet}"
        )
    if len(signals) > max_signals:
        lines.append(f"... ({len(signals) - max_signals} additional signals omitted for brevity)")
    return "\n".join(lines)


def _deduplicate_queries(queries: list[str], limit: int) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for raw in queries:
        cleaned = " ".join(raw.split()).strip()
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(cleaned)
        if len(unique) >= limit:
            break
    return unique


def _fallback_queries_from_profile(
    profile: OrganizationProfile,
    focus_area: str | None = None,
    num_queries: int = 5,
) -> list[str]:
    """Build deterministic research queries from `OrganizationProfile` fields."""
    candidates: list[str] = []
    industry = profile.industry.strip()
    products = [p.strip() for p in profile.products_or_services if p.strip()]
    pains = [p.strip() for p in profile.pain_points_addressed if p.strip()]
    hints = [h.strip() for h in profile.existing_audience_hints if h.strip()]
    needs = [n.strip() for n in profile.marketing_needs if n.strip()]
    geos = [g.strip() for g in profile.geographic_focus if g.strip()]

    primary_product = products[0] if products else profile.organization_name
    geo_suffix = f" {geos[0]}" if geos else ""

    if focus_area and focus_area.strip():
        candidates.append(f"{focus_area.strip()} {industry} buyer pain points")

    if pains:
        candidates.append(f"{pains[0]} {industry} solutions reddit OR community")

    if hints:
        candidates.append(f"{hints[0]} {primary_product} challenges linkedin")

    if products:
        candidates.append(f"{products[0]} alternatives reviews G2 Capterra")

    if pains and hints:
        candidates.append(f"how {hints[0]} solve {pains[0]}{geo_suffix}")

    if needs:
        candidates.append(f"{industry} {needs[0]} target audience strategy{geo_suffix}")

    candidates.append(f"{profile.organization_name} {industry} customer reviews discussions")
    candidates.append(f"{industry} software buyer intent trends{geo_suffix}")

    return _deduplicate_queries(candidates, limit=num_queries)


class TargetAudienceDiscovery:
    """Discovers, scores, and synthesizes target audience segments (ICPs) for an organization."""

    def __init__(
        self,
        llm: BaseChatModel | Runnable[Any, Any],
        scanner: AudienceScanner | None = None,
    ) -> None:
        self.llm = llm
        self.scanner: AudienceScanner = (
            scanner
            if scanner is not None
            else CompositeAudienceScanner(
                [DuckDuckGoWebScanner()],
                fallback_scanner=OfflineHeuristicScanner(),
            )
        )

    async def generate_scan_queries(
        self,
        profile: OrganizationProfile,
        focus_area: str | None = None,
        num_queries: int = 5,
    ) -> list[str]:
        """Generate targeted market-scanning queries grounded in the organization profile."""
        if num_queries <= 0:
            return []

        system_prompt = (
            "You are a senior B2B/B2C growth researcher and market intelligence strategist. "
            "Given an organization's profile, generate high-signal web search queries that "
            "surface where target buyers gather, what pain points they discuss, which "
            "competitor alternatives they evaluate, and what buyer-intent terms they use.\n"
            "Focus on practical search queries across communities (Reddit, LinkedIn, Hacker News), "
            "review sites (G2, Capterra), and industry discussions, grounded in the organization's "
            "industry, products/services, pain points, geographies, marketing needs, and existing "
            "audience hints."
        )
        focus_text = (
            f"\nOptional Focus Area: {focus_area.strip()}"
            if focus_area and focus_area.strip()
            else ""
        )
        human_prompt = (
            f"Generate up to {num_queries} distinct, high-signal market research search queries "
            f"for the following organization.{focus_text}\n\n"
            f"Organization Profile:\n{_format_profile_context(profile)}"
        )

        structured_llm = _bind_structured_output(self.llm, ScanQueryPlan)
        raw_plan = await structured_llm.ainvoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt),
            ]
        )
        plan = _validate_structured_result(raw_plan, ScanQueryPlan)
        queries = _deduplicate_queries(plan.queries, limit=num_queries)
        if queries:
            return queries

        return _fallback_queries_from_profile(
            profile,
            focus_area=focus_area,
            num_queries=num_queries,
        )

    async def discover_audiences(
        self,
        profile: OrganizationProfile,
        focus_area: str | None = None,
        extra_signals: list[MarketSignal] | None = None,
        max_results_per_query: int = 5,
        num_queries: int = 5,
    ) -> TargetAudienceReport:
        """Execute query planning, market scanning, signal merging, and ICP synthesis."""
        queries = await self.generate_scan_queries(
            profile,
            focus_area=focus_area,
            num_queries=num_queries,
        )
        scanned_signals = (
            await self.scanner.scan(queries, max_results_per_query=max_results_per_query)
            if queries
            else []
        )
        merged_signals = deduplicate_signals(
            [*scanned_signals, *(extra_signals or [])]
        )

        system_prompt = (
            "You are a principal digital marketing strategist and ICP architect. "
            "Synthesize the organization profile and collected external market signals into an "
            "actionable TargetAudienceReport.\n"
            "Requirements:\n"
            "- Build evidence-based audience segments with specific personas/roles, "
            "demographics or firmographics, pain points, and buying triggers.\n"
            "- Assign realistic fit_score values (0-100) reflecting product-market fit, "
            "urgency of pain, reachability, and budget/constraints alignment.\n"
            "- Recommend concrete preferred_channels, targeting_keywords, and "
            "communities_and_hashtags observed or strongly indicated by the signals.\n"
            "- Craft a sharp, differentiated messaging_angle for each segment aligned with "
            "the organization's brand_tone and differentiators.\n"
            "- Identify the single best recommended_primary_segment and ordered channel_priority."
        )
        focus_line = (
            f"Focus Area: {focus_area.strip()}\n\n"
            if focus_area and focus_area.strip()
            else ""
        )
        human_prompt = (
            f"{focus_line}"
            f"Organization Profile:\n{_format_profile_context(profile)}\n\n"
            f"Executed Scan Queries ({len(queries)}):\n"
            + "\n".join(f"- {q}" for q in queries)
            + f"\n\nCollected Market Signals ({len(merged_signals)}):\n"
            + _format_signals_context(merged_signals)
        )

        structured_llm = _bind_structured_output(self.llm, TargetAudienceReport)
        raw_report = await structured_llm.ainvoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt),
            ]
        )
        report = _validate_structured_result(raw_report, TargetAudienceReport)

        sorted_segments = sorted(
            report.segments,
            key=lambda segment: segment.fit_score,
            reverse=True,
        )
        recommended_primary = report.recommended_primary_segment.strip()
        if not recommended_primary and sorted_segments:
            recommended_primary = sorted_segments[0].name

        return report.model_copy(
            update={
                "organization_name": profile.organization_name,
                "scan_queries": queries,
                "signals_analyzed": len(merged_signals),
                "segments": sorted_segments,
                "recommended_primary_segment": recommended_primary,
            }
        )

    async def evaluate_segment_fit(
        self,
        profile: OrganizationProfile,
        segment_description: str,
    ) -> AudienceSegment:
        """Score and critique a proposed target audience segment against an organization profile."""
        cleaned_description = segment_description.strip()
        if not cleaned_description:
            raise ValueError("segment_description must not be empty.")

        system_prompt = (
            "You are a senior digital marketing strategist evaluating a proposed target "
            "audience segment against an organization's capabilities, value proposition, "
            "constraints, and goals.\n"
            "Return a complete AudienceSegment evaluation with an objective fit_score (0-100), "
            "clear rationale citing strengths and risks/gaps, actionable preferred_channels, "
            "targeting_keywords, communities_and_hashtags, and a tailored messaging_angle."
        )
        human_prompt = (
            f"Evaluate the following proposed target audience segment:\n"
            f"{cleaned_description}\n\n"
            f"Organization Profile:\n{_format_profile_context(profile)}"
        )

        structured_llm = _bind_structured_output(self.llm, AudienceSegment)
        raw_segment = await structured_llm.ainvoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt),
            ]
        )
        return _validate_structured_result(raw_segment, AudienceSegment)
