"""Target audience scanning, ICP discovery, and LangChain audience tools."""

from closr.audience.analyzer import ScanQueryPlan, TargetAudienceDiscovery
from closr.audience.scanner import (
    AudienceScanner,
    CompositeAudienceScanner,
    DuckDuckGoWebScanner,
    OfflineHeuristicScanner,
    extract_keywords,
    infer_platform_hint,
    parse_duckduckgo_html,
)
from closr.audience.tools import create_audience_tools

__all__ = [
    "AudienceScanner",
    "CompositeAudienceScanner",
    "DuckDuckGoWebScanner",
    "OfflineHeuristicScanner",
    "ScanQueryPlan",
    "TargetAudienceDiscovery",
    "create_audience_tools",
    "extract_keywords",
    "infer_platform_hint",
    "parse_duckduckgo_html",
]
