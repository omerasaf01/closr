"""closr: Automated Digital Marketing Agent powered by LangChain, LangGraph, and MCP."""

from closr.agent import MarketingAgent
from closr.config import ClosrSettings, LLMProvider, get_settings
from closr.llm import LLMFactory
from closr.mcp import MCPManager
from closr.schemas import (
    AudienceSegment,
    CampaignRecommendation,
    MarketSignal,
    MarketingPlan,
    OrganizationProfile,
    TargetAudienceReport,
    VideoGenerationRequest,
    VideoGenerationResult,
    VideoPromptSpec,
)
from closr.video import (
    PromoVideoPromptBuilder,
    PromoVideoService,
    SeedanceAPIError,
    SeedanceClient,
)

__version__ = "0.1.0"


def main() -> None:
    """CLI entry point (lazily delegates to `closr.cli.main`)."""
    from closr.cli import main as cli_main

    cli_main()


__all__ = [
    "__version__",
    "AudienceSegment",
    "CampaignRecommendation",
    "ClosrSettings",
    "LLMFactory",
    "LLMProvider",
    "MCPManager",
    "MarketSignal",
    "MarketingAgent",
    "MarketingPlan",
    "OrganizationProfile",
    "PromoVideoPromptBuilder",
    "PromoVideoService",
    "SeedanceAPIError",
    "SeedanceClient",
    "TargetAudienceReport",
    "VideoGenerationRequest",
    "VideoGenerationResult",
    "VideoPromptSpec",
    "get_settings",
    "main",
]
