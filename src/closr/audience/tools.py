"""LangChain tools exposing market signal scanning and target audience discovery."""

from __future__ import annotations

import json
from collections.abc import Callable

from langchain_core.tools import BaseTool, tool
from pydantic import ValidationError

from closr.audience.analyzer import TargetAudienceDiscovery
from closr.schemas import OrganizationProfile


def _resolve_organization_profile(
    organization_profile_json: str,
    profile_getter: Callable[[], OrganizationProfile | None] | None,
) -> tuple[OrganizationProfile | None, str | None]:
    """Resolve an `OrganizationProfile` from explicit JSON or fallback `profile_getter`."""
    raw_json = organization_profile_json.strip()
    if raw_json:
        try:
            return OrganizationProfile.model_validate_json(raw_json), None
        except ValidationError as exc:
            return None, f"Error: Invalid organization_profile_json: {exc}"

    if profile_getter is not None:
        profile = profile_getter()
        if profile is not None:
            return profile, None

    return (
        None,
        "Error: No organization profile available. Provide 'organization_profile_json' "
        "or analyze organization documents first.",
    )


def create_audience_tools(
    discovery: TargetAudienceDiscovery,
    profile_getter: Callable[[], OrganizationProfile | None] | None = None,
) -> list[BaseTool]:
    """Create async LangChain tools for market signal scanning and ICP discovery."""

    @tool
    async def scan_market_signals(
        queries: list[str],
        max_results_per_query: int = 5,
    ) -> str:
        """Scan web and heuristic market sources for audience discussions and intent signals."""
        signals = await discovery.scanner.scan(
            queries,
            max_results_per_query=max_results_per_query,
        )
        return json.dumps([signal.model_dump() for signal in signals], indent=2)

    @tool
    async def discover_target_audiences(
        organization_profile_json: str = "",
        focus_area: str = "",
    ) -> str:
        """Discover and prioritize target audience segments (ICPs) for an organization."""
        profile, error = _resolve_organization_profile(
            organization_profile_json,
            profile_getter,
        )
        if profile is None:
            return error or "Error: No organization profile available."

        report = await discovery.discover_audiences(
            profile,
            focus_area=focus_area.strip() or None,
        )
        return report.model_dump_json(indent=2)

    @tool
    async def evaluate_audience_segment(
        segment_description: str,
        organization_profile_json: str = "",
    ) -> str:
        """Score and evaluate a proposed target audience segment against an organization profile."""
        if not segment_description.strip():
            return "Error: 'segment_description' must not be empty."

        profile, error = _resolve_organization_profile(
            organization_profile_json,
            profile_getter,
        )
        if profile is None:
            return error or "Error: No organization profile available."

        segment = await discovery.evaluate_segment_fit(
            profile,
            segment_description=segment_description,
        )
        return segment.model_dump_json(indent=2)

    return [
        scan_market_signals,
        discover_target_audiences,
        evaluate_audience_segment,
    ]
