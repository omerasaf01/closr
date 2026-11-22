"""LangGraph workflow state definitions for the closr marketing pipeline."""

from __future__ import annotations

from typing import Any, TypedDict

from closr.schemas import (
    MarketingPlan,
    OrganizationProfile,
    TargetAudienceReport,
)


class MarketingWorkflowState(TypedDict, total=False):
    """Shared state passed across the end-to-end LangGraph marketing pipeline."""

    docs_path: str | None
    focus_area: str | None
    monthly_budget_usd: float | None
    organization_profile: OrganizationProfile | None
    audience_report: TargetAudienceReport | None
    mcp_insights: dict[str, Any]
    marketing_plan: MarketingPlan | None
    errors: list[str]
