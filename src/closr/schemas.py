"""Shared Pydantic v2 domain schemas for organization analysis, audience discovery, and marketing strategy."""

from __future__ import annotations

from pydantic import BaseModel, Field


class OrganizationProfile(BaseModel):
    """Structured profile extracted from an organization's internal documents and materials."""

    organization_name: str = Field(
        description="Official or primary name of the organization.",
    )
    industry: str = Field(
        description="Primary industry, vertical, or market category.",
    )
    summary: str = Field(
        description="Concise executive overview of what the organization does and who it serves.",
    )
    products_or_services: list[str] = Field(
        default_factory=list,
        description="Core products, services, or platform capabilities offered.",
    )
    value_proposition: str = Field(
        default="",
        description="Core value proposition explaining why customers choose the organization.",
    )
    business_goals: list[str] = Field(
        default_factory=list,
        description="Strategic business or growth objectives identified in the materials.",
    )
    marketing_needs: list[str] = Field(
        default_factory=list,
        description="Explicit or implied marketing needs, gaps, or priorities.",
    )
    pain_points_addressed: list[str] = Field(
        default_factory=list,
        description="Customer problems and pain points solved by the organization's offerings.",
    )
    brand_tone: str = Field(
        default="",
        description="Observed or desired brand voice and communication tone.",
    )
    differentiators: list[str] = Field(
        default_factory=list,
        description="Key competitive advantages and differentiators.",
    )
    geographic_focus: list[str] = Field(
        default_factory=list,
        description="Target geographies, regions, or markets.",
    )
    budget_or_constraints: list[str] = Field(
        default_factory=list,
        description="Known budget limits, compliance constraints, or operational guardrails.",
    )
    existing_audience_hints: list[str] = Field(
        default_factory=list,
        description="Existing customer segments, ICP hints, or buyer personas mentioned in source docs.",
    )
    source_documents: list[str] = Field(
        default_factory=list,
        description="Filenames or identifiers of source documents analyzed to build this profile.",
    )


class MarketSignal(BaseModel):
    """External market signal collected from web search, communities, or MCP data sources."""

    query: str = Field(
        description="Search or discovery query used to surface this market signal.",
    )
    source: str = Field(
        description="Origin of the signal (for example: web_search, reddit, linkedin, mcp_tool).",
    )
    title: str = Field(
        description="Headline or title of the discovered result or discussion.",
    )
    snippet: str = Field(
        description="Relevant excerpt or summary of the market signal.",
    )
    url: str | None = Field(
        default=None,
        description="Optional canonical URL associated with the signal.",
    )
    detected_keywords: list[str] = Field(
        default_factory=list,
        description="Keywords, topics, or buyer-intent terms extracted from the signal.",
    )
    platform_hint: str | None = Field(
        default=None,
        description="Suggested social, search, or community platform indicated by this signal.",
    )


class AudienceSegment(BaseModel):
    """Detailed target audience segment or ideal customer profile derived from profile and market signals."""

    name: str = Field(
        description="Short descriptive name for the audience segment.",
    )
    description: str = Field(
        description="Clear summary of who belongs to this segment and why they matter.",
    )
    demographics_or_firmographics: str = Field(
        description="Demographic traits (B2C) or firmographic traits (B2B) such as company size and vertical.",
    )
    roles_or_personas: list[str] = Field(
        default_factory=list,
        description="Decision makers, buyers, influencers, or end-user personas in this segment.",
    )
    pain_points: list[str] = Field(
        default_factory=list,
        description="Top challenges and friction points experienced by this segment.",
    )
    buying_triggers: list[str] = Field(
        default_factory=list,
        description="Events or situations that prompt this segment to evaluate solutions.",
    )
    preferred_channels: list[str] = Field(
        default_factory=list,
        description="Marketing and communication channels where this segment is most reachable.",
    )
    targeting_keywords: list[str] = Field(
        default_factory=list,
        description="Search terms, intent keywords, and ad-targeting topics for this segment.",
    )
    communities_and_hashtags: list[str] = Field(
        default_factory=list,
        description="Relevant online communities, forums, newsletters, or hashtags.",
    )
    fit_score: int = Field(
        ge=0,
        le=100,
        description="Estimated audience-product fit score from 0 to 100.",
    )
    rationale: str = Field(
        description="Evidence-based explanation of why this segment is a strong fit.",
    )
    messaging_angle: str = Field(
        description="Recommended positioning and messaging hook tailored to this segment.",
    )


class TargetAudienceReport(BaseModel):
    """Consolidated audience discovery report across analyzed market signals."""

    organization_name: str = Field(
        description="Name of the organization the audience report applies to.",
    )
    scan_queries: list[str] = Field(
        default_factory=list,
        description="Queries executed during external market and audience scanning.",
    )
    signals_analyzed: int = Field(
        default=0,
        ge=0,
        description="Total number of external market signals synthesized in this report.",
    )
    segments: list[AudienceSegment] = Field(
        default_factory=list,
        description="Prioritized list of identified target audience segments.",
    )
    recommended_primary_segment: str = Field(
        description="Name of the single highest-priority audience segment to target first.",
    )
    channel_priority: list[str] = Field(
        default_factory=list,
        description="Ordered list of recommended marketing channels from highest to lowest priority.",
    )
    notes: str = Field(
        default="",
        description="Additional strategic observations, caveats, or research gaps.",
    )


class CampaignRecommendation(BaseModel):
    """Actionable marketing campaign proposal tailored to a specific audience segment."""

    campaign_name: str = Field(
        description="Concise, memorable name for the campaign.",
    )
    target_segment: str = Field(
        description="Name of the audience segment this campaign targets.",
    )
    objective: str = Field(
        description="Primary measurable marketing objective (e.g., lead generation, awareness, activation).",
    )
    channels: list[str] = Field(
        default_factory=list,
        description="Distribution channels selected for this campaign.",
    )
    key_messages: list[str] = Field(
        default_factory=list,
        description="Core messaging pillars and value propositions used in campaign copy.",
    )
    content_ideas: list[str] = Field(
        default_factory=list,
        description="Concrete content assets, creative formats, or campaign deliverables.",
    )
    kpis: list[str] = Field(
        default_factory=list,
        description="Key performance indicators used to measure campaign success.",
    )
    recommended_mcp_actions: list[str] = Field(
        default_factory=list,
        description="Suggested follow-up actions or tool calls to execute via connected MCP servers.",
    )


class MarketingPlan(BaseModel):
    """End-to-end marketing strategy combining organization analysis, audience discovery, and campaigns."""

    organization_profile: OrganizationProfile = Field(
        description="Extracted profile of the organization.",
    )
    audience_report: TargetAudienceReport = Field(
        description="Synthesized target audience discovery report.",
    )
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
