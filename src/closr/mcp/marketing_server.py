"""Built-in FastMCP digital marketing tools server for closr."""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("closr-marketing-mcp")

_SUPPORTED_CHANNELS: dict[str, dict[str, Any]] = {
    "linkedin": {
        "avg_ctr_pct": 0.55,
        "avg_cpc_usd": 5.80,
        "avg_conversion_rate_pct": 2.90,
        "recommended_formats": [
            "Single-image sponsored content",
            "Document carousel ads",
            "Thought Leader Ads",
            "Lead Gen Forms",
        ],
        "best_for": "B2B decision-maker targeting, ABM, and high-ACV pipeline generation",
    },
    "google_search": {
        "avg_ctr_pct": 3.40,
        "avg_cpc_usd": 3.20,
        "avg_conversion_rate_pct": 4.20,
        "recommended_formats": [
            "Responsive Search Ads",
            "High-intent competitor & category keywords",
            "Sitelink & callout extensions",
        ],
        "best_for": "Capturing active in-market demand and high-intent comparison searches",
    },
    "meta": {
        "avg_ctr_pct": 1.35,
        "avg_cpc_usd": 1.45,
        "avg_conversion_rate_pct": 2.60,
        "recommended_formats": [
            "Short-form Reels video",
            "Carousel product/feature explainers",
            "Retargeting dynamic ads",
        ],
        "best_for": "Scalable demand generation, retargeting, and visual product storytelling",
    },
    "tiktok": {
        "avg_ctr_pct": 1.65,
        "avg_cpc_usd": 0.95,
        "avg_conversion_rate_pct": 1.40,
        "recommended_formats": [
            "Creator-style Spark Ads",
            "Native vertical 15-30s demos",
            "Hook-first problem/solution clips",
        ],
        "best_for": "Top-of-funnel awareness, creator-led discovery, and consumer/prosumer reach",
    },
    "youtube": {
        "avg_ctr_pct": 0.75,
        "avg_cpc_usd": 1.85,
        "avg_conversion_rate_pct": 1.75,
        "recommended_formats": [
            "Skippable in-stream demos",
            "In-feed educational walkthroughs",
            "6-second bumper retargeting",
        ],
        "best_for": "Category education, product walkthroughs, and mid-funnel consideration",
    },
    "email": {
        "avg_ctr_pct": 2.80,
        "avg_cpc_usd": 0.45,
        "avg_conversion_rate_pct": 5.10,
        "recommended_formats": [
            "Segmented lifecycle nurture sequences",
            "Case-study & ROI digest emails",
            "Interactive product launch announcements",
        ],
        "best_for": "Lead nurturing, trial-to-paid activation, expansion, and retention",
    },
}

_CHANNEL_ALIASES: dict[str, str] = {
    "google": "google_search",
    "google_ads": "google_search",
    "search": "google_search",
    "sem": "google_search",
    "facebook": "meta",
    "instagram": "meta",
    "meta_ads": "meta",
    "linkedin_ads": "linkedin",
    "tiktok_ads": "tiktok",
    "youtube_ads": "youtube",
    "newsletter": "email",
}

# Multipliers: (ctr_mult, cpc_mult, cvr_mult)
_INDUSTRY_MULTIPLIERS: dict[str, tuple[float, float, float]] = {
    "saas": (1.00, 1.15, 1.05),
    "ecommerce": (1.20, 0.75, 1.15),
    "fintech": (0.90, 1.35, 0.95),
    "healthcare": (0.95, 1.18, 0.90),
    "education": (1.10, 0.85, 1.10),
    "b2b": (0.95, 1.10, 1.00),
    "general": (1.00, 1.00, 1.00),
}

_INDUSTRY_ALIASES: dict[str, str] = {
    "b2b_saas": "saas",
    "software": "saas",
    "tech": "saas",
    "ecom": "ecommerce",
    "retail": "ecommerce",
    "dtc": "ecommerce",
    "d2c": "ecommerce",
    "finance": "fintech",
    "health": "healthcare",
    "medtech": "healthcare",
    "edtech": "education",
}

_GOAL_DEFAULTS: dict[str, list[str]] = {
    "awareness": ["youtube", "meta", "tiktok", "linkedin"],
    "lead_generation": ["linkedin", "google_search", "meta"],
    "conversions": ["google_search", "meta", "email"],
    "retention": ["email", "meta", "youtube"],
}

_GOAL_ALIASES: dict[str, str] = {
    "brand_awareness": "awareness",
    "reach": "awareness",
    "leads": "lead_generation",
    "lead_gen": "lead_generation",
    "demand_gen": "lead_generation",
    "conversion": "conversions",
    "sales": "conversions",
    "revenue": "conversions",
    "lifecycle": "retention",
    "churn_reduction": "retention",
}

_GOAL_CHANNEL_WEIGHTS: dict[str, dict[str, float]] = {
    "awareness": {
        "youtube": 1.3,
        "tiktok": 1.3,
        "meta": 1.2,
        "linkedin": 0.9,
        "google_search": 0.6,
        "email": 0.4,
    },
    "lead_generation": {
        "linkedin": 1.4,
        "google_search": 1.3,
        "meta": 1.0,
        "email": 0.9,
        "youtube": 0.8,
        "tiktok": 0.6,
    },
    "conversions": {
        "google_search": 1.5,
        "email": 1.2,
        "meta": 1.1,
        "linkedin": 1.0,
        "youtube": 0.7,
        "tiktok": 0.6,
    },
    "retention": {
        "email": 1.6,
        "meta": 0.9,
        "youtube": 0.8,
        "linkedin": 0.7,
        "google_search": 0.5,
        "tiktok": 0.5,
    },
}

_UTM_CHANNEL_MAP: dict[str, tuple[str, str]] = {
    "linkedin": ("linkedin", "paid_social"),
    "google_search": ("google", "cpc"),
    "meta": ("meta", "paid_social"),
    "tiktok": ("tiktok", "paid_social"),
    "youtube": ("youtube", "video"),
    "email": ("newsletter", "email"),
}


def _normalize_key(raw: str) -> str:
    return re.sub(r"[\s\-]+", "_", raw.strip().lower())


def _resolve_industry(industry: str) -> tuple[str, bool]:
    key = _normalize_key(industry)
    key = _INDUSTRY_ALIASES.get(key, key)
    if key in _INDUSTRY_MULTIPLIERS:
        return key, False
    return "general", True


def _resolve_channel(channel: str) -> str:
    key = _normalize_key(channel)
    key = _CHANNEL_ALIASES.get(key, key)
    if key not in _SUPPORTED_CHANNELS:
        supported = ", ".join(sorted(_SUPPORTED_CHANNELS))
        raise ValueError(f"Unsupported channel {channel!r}. Supported channels: {supported}.")
    return key


def _resolve_goal(primary_goal: str) -> str:
    key = _normalize_key(primary_goal)
    key = _GOAL_ALIASES.get(key, key)
    if key not in _GOAL_DEFAULTS:
        supported = ", ".join(sorted(_GOAL_DEFAULTS))
        raise ValueError(
            f"Unsupported primary_goal {primary_goal!r}. Supported goals: {supported}."
        )
    return key


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.strip().lower()).strip("-")
    return slug or "untitled"


@mcp.tool()
def get_channel_benchmarks(industry: str = "saas", channel: str = "all") -> dict[str, Any]:
    """Return calibrated digital marketing benchmarks by industry and channel.

    Args:
        industry: Industry vertical (`saas`, `ecommerce`, `fintech`, `healthcare`,
            `education`, `b2b`, `general`). Unknown values fall back to `general`.
        channel: Target channel (`all`, `linkedin`, `google_search`, `meta`,
            `tiktok`, `youtube`, or `email`).
    """
    resolved_industry, used_fallback = _resolve_industry(industry)
    ctr_mult, cpc_mult, cvr_mult = _INDUSTRY_MULTIPLIERS[resolved_industry]

    channel_key = _normalize_key(channel)
    if channel_key == "all":
        target_channels = list(_SUPPORTED_CHANNELS.keys())
        resolved_channel = "all"
    else:
        resolved_channel = _resolve_channel(channel)
        target_channels = [resolved_channel]

    benchmarks: dict[str, dict[str, Any]] = {}
    for ch in target_channels:
        base = _SUPPORTED_CHANNELS[ch]
        benchmarks[ch] = {
            "avg_ctr_pct": round(base["avg_ctr_pct"] * ctr_mult, 2),
            "avg_cpc_usd": round(base["avg_cpc_usd"] * cpc_mult, 2),
            "avg_conversion_rate_pct": round(base["avg_conversion_rate_pct"] * cvr_mult, 2),
            "recommended_formats": list(base["recommended_formats"]),
            "best_for": base["best_for"],
        }

    return {
        "industry": resolved_industry,
        "requested_industry": industry,
        "industry_fallback_applied": used_fallback,
        "channel": resolved_channel,
        "benchmarks": benchmarks,
        "supported_industries": sorted(_INDUSTRY_MULTIPLIERS.keys()),
        "supported_channels": sorted(_SUPPORTED_CHANNELS.keys()),
    }


@mcp.tool()
def estimate_campaign_budget(
    monthly_budget_usd: float,
    primary_goal: str = "lead_generation",
    channels: list[str] | None = None,
    industry: str = "saas",
) -> dict[str, Any]:
    """Allocate a monthly marketing budget across channels and forecast clicks and conversions.

    Args:
        monthly_budget_usd: Positive monthly media/campaign budget in USD.
        primary_goal: Campaign objective (`awareness`, `lead_generation`,
            `conversions`, or `retention`).
        channels: Optional list of channels to allocate across. Defaults to
            recommended channels for the selected `primary_goal`.
        industry: Industry vertical used to calibrate CPC and conversion benchmarks.
    """
    if monthly_budget_usd <= 0:
        raise ValueError("monthly_budget_usd must be greater than 0.")

    resolved_goal = _resolve_goal(primary_goal)
    resolved_industry, used_fallback = _resolve_industry(industry)

    raw_channels = channels if channels else _GOAL_DEFAULTS[resolved_goal]
    selected_channels: list[str] = []
    for ch in raw_channels:
        resolved_ch = _resolve_channel(ch)
        if resolved_ch not in selected_channels:
            selected_channels.append(resolved_ch)

    benchmark_data = get_channel_benchmarks(industry=resolved_industry, channel="all")["benchmarks"]
    goal_weights = _GOAL_CHANNEL_WEIGHTS[resolved_goal]

    raw_weights = [goal_weights.get(ch, 1.0) for ch in selected_channels]
    weight_sum = sum(raw_weights)
    normalized_weights = [w / weight_sum for w in raw_weights]

    allocations: list[dict[str, Any]] = []
    remaining_budget = round(float(monthly_budget_usd), 2)
    total_clicks = 0
    total_conversions = 0.0

    for idx, (ch, weight) in enumerate(zip(selected_channels, normalized_weights, strict=True)):
        if idx == len(selected_channels) - 1:
            allocated = round(remaining_budget, 2)
        else:
            allocated = round(monthly_budget_usd * weight, 2)
            remaining_budget = round(remaining_budget - allocated, 2)

        ch_bench = benchmark_data[ch]
        est_cpc = float(ch_bench["avg_cpc_usd"])
        est_cvr_pct = float(ch_bench["avg_conversion_rate_pct"])

        est_clicks = max(0, round(allocated / est_cpc)) if est_cpc > 0 else 0
        est_conversions = round(est_clicks * (est_cvr_pct / 100.0), 1)
        est_cpa = round(allocated / est_conversions, 2) if est_conversions > 0 else None

        total_clicks += est_clicks
        total_conversions += est_conversions

        allocations.append(
            {
                "channel": ch,
                "allocation_pct": round(weight * 100.0, 1),
                "allocated_budget_usd": allocated,
                "assumed_cpc_usd": est_cpc,
                "assumed_conversion_rate_pct": est_cvr_pct,
                "estimated_clicks": est_clicks,
                "estimated_conversions": est_conversions,
                "estimated_cpa_usd": est_cpa,
            }
        )

    total_conversions_rounded = round(total_conversions, 1)
    blended_cpc = round(monthly_budget_usd / total_clicks, 2) if total_clicks > 0 else None
    blended_cpa = (
        round(monthly_budget_usd / total_conversions_rounded, 2)
        if total_conversions_rounded > 0
        else None
    )

    assumptions = [
        f"Benchmarks calibrated for industry={resolved_industry!r} and goal={resolved_goal!r}.",
        "Channel budget weights reflect historical funnel efficiency for the selected goal.",
        "Email CPC represents amortized ESP/creative production cost per click equivalent.",
        "Actual CPA varies with landing-page conversion rate, creative fatigue, and bidding.",
    ]
    if used_fallback:
        assumptions.append(
            f"Requested industry {industry!r} mapped to 'general' baseline multipliers."
        )

    return {
        "monthly_budget_usd": round(float(monthly_budget_usd), 2),
        "primary_goal": resolved_goal,
        "industry": resolved_industry,
        "allocations": allocations,
        "totals": {
            "total_allocated_usd": round(sum(a["allocated_budget_usd"] for a in allocations), 2),
            "total_estimated_clicks": total_clicks,
            "total_estimated_conversions": total_conversions_rounded,
            "blended_cpc_usd": blended_cpc,
            "blended_cpa_usd": blended_cpa,
        },
        "assumptions": assumptions,
    }


@mcp.tool()
def build_utm_tracking_links(
    base_url: str,
    campaign_name: str,
    channels: list[str],
    content_variants: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Generate slugified, URL-encoded UTM tracking links while preserving existing query params.

    Args:
        base_url: Destination landing page URL (must include http:// or https://).
        campaign_name: Human-readable campaign name to slugify into `utm_campaign`.
        channels: List of distribution channels to generate links for.
        content_variants: Optional A/B creative variants mapped to `utm_content`.
    """
    cleaned_url = base_url.strip()
    if not cleaned_url:
        raise ValueError("base_url must not be empty.")
    parsed = urlsplit(cleaned_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"base_url must be a valid http(s) URL, got {base_url!r}.")

    if not campaign_name.strip():
        raise ValueError("campaign_name must not be empty.")
    if not channels:
        raise ValueError("channels must contain at least one channel.")

    campaign_slug = _slugify(campaign_name)
    existing_params = parse_qsl(parsed.query, keep_blank_values=True)
    utm_keys = {"utm_source", "utm_medium", "utm_campaign", "utm_content"}
    preserved_params = [(k, v) for k, v in existing_params if k not in utm_keys]

    variants: list[str | None]
    if content_variants:
        variants = [v for v in content_variants if v.strip()]
        if not variants:
            variants = [None]
    else:
        variants = [None]

    links: list[dict[str, Any]] = []
    for raw_channel in channels:
        if not raw_channel.strip():
            continue
        norm_key = _normalize_key(raw_channel)
        resolved_key = _CHANNEL_ALIASES.get(norm_key, norm_key)
        utm_source, utm_medium = _UTM_CHANNEL_MAP.get(
            resolved_key, (_slugify(raw_channel), "digital")
        )

        for variant in variants:
            utm_params: list[tuple[str, str]] = [
                ("utm_source", utm_source),
                ("utm_medium", utm_medium),
                ("utm_campaign", campaign_slug),
            ]
            utm_content: str | None = None
            if variant is not None:
                utm_content = _slugify(variant)
                utm_params.append(("utm_content", utm_content))

            query_string = urlencode([*preserved_params, *utm_params], doseq=True)
            full_url = urlunsplit(
                (parsed.scheme, parsed.netloc, parsed.path, query_string, parsed.fragment)
            )
            links.append(
                {
                    "channel": resolved_key,
                    "utm_source": utm_source,
                    "utm_medium": utm_medium,
                    "utm_campaign": campaign_slug,
                    "utm_content": utm_content,
                    "url": full_url,
                }
            )

    if not links:
        raise ValueError("No valid non-empty channels were provided.")
    return links


@mcp.tool()
def suggest_audience_channels(
    business_model: str,
    target_roles: list[str],
    region: str = "global",
) -> dict[str, Any]:
    """Recommend prioritized acquisition channels, tactics, and community hubs for an audience.

    Args:
        business_model: Organization model (`b2b`, `b2c`, `b2b2c`, or `marketplace`).
        target_roles: List of target buyer roles, personas, or customer segments.
        region: Geographic focus (e.g. `global`, `north_america`, `europe`, `apac`).
    """
    model_key = _normalize_key(business_model)
    supported_models = {"b2b", "b2c", "b2b2c", "marketplace"}
    if model_key not in supported_models:
        raise ValueError(
            f"Unsupported business_model {business_model!r}. "
            f"Supported: {', '.join(sorted(supported_models))}."
        )

    roles_clean = [r.strip() for r in target_roles if r.strip()]
    roles_blob = " ".join(roles_clean).lower()

    is_technical = any(
        kw in roles_blob
        for kw in ("engineer", "developer", "devops", "cto", "architect", "data", "security")
    )
    is_executive = any(
        kw in roles_blob
        for kw in ("ceo", "cfo", "cmo", "coo", "vp", "founder", "head of", "director", "exec")
    )
    is_marketing_ops = any(
        kw in roles_blob for kw in ("marketing", "growth", "revops", "sales", "product")
    )
    is_consumer = any(
        kw in roles_blob
        for kw in ("consumer", "shopper", "parent", "student", "creator", "freelancer", "patient")
    )

    if model_key == "b2b":
        channel_order = ["linkedin", "google_search", "youtube", "email", "meta"]
    elif model_key == "b2c":
        channel_order = ["meta", "tiktok", "google_search", "youtube", "email"]
    elif model_key == "b2b2c":
        channel_order = ["linkedin", "meta", "google_search", "youtube", "email"]
    else:  # marketplace
        channel_order = ["google_search", "meta", "tiktok", "linkedin", "email"]

    if is_technical and "google_search" in channel_order:
        channel_order.remove("google_search")
        channel_order.insert(0, "google_search")
    if is_consumer and "tiktok" in channel_order and model_key != "b2b":
        channel_order.remove("tiktok")
        channel_order.insert(1, "tiktok")

    tactics_by_channel: dict[str, list[str]] = {
        "linkedin": [
            "Job-title + seniority + company-size layered targeting",
            "Thought Leader Ads from founder/practitioner profiles",
            "Retargeting website & pricing-page visitors with case studies",
        ],
        "google_search": [
            "High-intent problem, category, and competitor comparison keywords",
            "Dedicated landing pages matched to buyer role & use case",
            "Negative keyword pruning for non-commercial queries",
        ],
        "meta": [
            "Broad + lookalike audiences seeded by converted customers",
            "Short problem/solution Reels paired with static proof cards",
            "7-day and 30-day multi-touch retargeting sequences",
        ],
        "tiktok": [
            "Creator/UGC Spark Ads demonstrating real workflow or product outcomes",
            "3-second visual hook testing with native captions",
            "Trend-aligned educational explainers",
        ],
        "youtube": [
            "Custom-intent audiences built from high-value search terms",
            "2-minute product walkthroughs and customer proof stories",
            "Channel/topic placements on respected industry creators",
        ],
        "email": [
            "Role-segmented onboarding and benchmark-led nurture flows",
            "Trigger-based lifecycle messages on key activation milestones",
            "Curated monthly insights newsletter to stay top-of-mind",
        ],
    }

    prioritized_channels: list[dict[str, Any]] = []
    for idx, ch in enumerate(channel_order, start=1):
        bench = _SUPPORTED_CHANNELS[ch]
        prioritized_channels.append(
            {
                "channel": ch,
                "priority_rank": idx,
                "rationale": bench["best_for"],
                "recommended_formats": list(bench["recommended_formats"]),
                "targeting_tactics": tactics_by_channel[ch],
            }
        )

    community_hubs: list[str] = []
    if is_technical:
        community_hubs.extend(
            [
                "Hacker News (Show HN / technical discussions)",
                "Reddit developer & DevOps subreddits (e.g., r/programming, r/devops)",
                "GitHub Discussions & open-source ecosystem sponsorships",
                "Developer newsletters (TLDR, Pragmatic Engineer, Changelog)",
            ]
        )
    if is_executive or model_key in {"b2b", "b2b2c"}:
        community_hubs.extend(
            [
                "LinkedIn executive peer groups & niche Substack newsletters",
                "RevOps / SaaS Slack & Discord communities (e.g., Pavilion, Exit Five)",
                "Category podcasts & analyst/review marketplaces (G2, Capterra)",
            ]
        )
    if is_marketing_ops:
        community_hubs.extend(
            [
                "Demand Curve / GrowthHackers / Product Hunt launches",
                "LinkedIn creator conversations and tactical teardowns",
            ]
        )
    if is_consumer or model_key in {"b2c", "marketplace"}:
        community_hubs.extend(
            [
                "Niche Reddit buyer communities and Discord hobby servers",
                "TikTok & Instagram micro-creator ecosystems",
                "YouTube review & comparison channels",
            ]
        )
    # Deduplicate preserving order
    community_hubs = list(dict.fromkeys(community_hubs))

    region_norm = _normalize_key(region)
    if any(tok in region_norm for tok in ("europe", "eu", "uk", "emea")):
        regional_notes = (
            f"Region {region!r}: Enforce GDPR consent-mode tracking, localize ad copy "
            "and landing pages for top EMEA markets, and prefer contextual/first-party signals."
        )
    elif any(tok in region_norm for tok in ("apac", "asia", "japan", "korea", "sea")):
        regional_notes = (
            f"Region {region!r}: Account for mobile-first journeys and local super-app/messaging "
            "ecosystems alongside global search and YouTube."
        )
    elif any(tok in region_norm for tok in ("north_america", "us", "usa", "canada")):
        regional_notes = (
            f"Region {region!r}: Higher auction CPCs in core B2B/SaaS keywords; differentiate "
            "with sharp category positioning, ROI calculators, and proof-driven retargeting."
        )
    else:
        regional_notes = (
            f"Region {region!r}: Start with English-first flagship campaigns in top-converting "
            "metros, then localize creatives and pricing currency for regional winners."
        )

    quick_wins = [
        f"Launch a role-specific landing page tailored to: {', '.join(roles_clean) or 'primary ICP'}.",
        "Instrument consistent UTM parameters across every paid and community link.",
        "Pair high-intent search capture with retargeting social proof within week 1.",
    ]

    return {
        "business_model": model_key,
        "target_roles": roles_clean,
        "region": region,
        "prioritized_channels": prioritized_channels,
        "community_hubs": community_hubs,
        "regional_notes": regional_notes,
        "quick_wins": quick_wins,
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")
