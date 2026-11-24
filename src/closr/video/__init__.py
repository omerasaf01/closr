"""Promotional video prompt drafting (ChatGPT) and video generation (Seedance) module."""

from closr.video.prompt_builder import (
    PLATFORM_RATIO_MAP,
    SEEDANCE_PROMPT_ENGINEER_SYSTEM_PROMPT,
    PromoVideoPromptBuilder,
    VideoRatio,
    clean_video_prompt,
    infer_ratio_for_platform,
)
from closr.video.seedance_client import (
    TERMINAL_STATUSES,
    SeedanceAPIError,
    SeedanceClient,
)
from closr.video.service import PromoVideoService, VideoResolution
from closr.video.tools import create_video_tools

__all__ = [
    "PLATFORM_RATIO_MAP",
    "SEEDANCE_PROMPT_ENGINEER_SYSTEM_PROMPT",
    "TERMINAL_STATUSES",
    "PromoVideoPromptBuilder",
    "PromoVideoService",
    "SeedanceAPIError",
    "SeedanceClient",
    "VideoRatio",
    "VideoResolution",
    "clean_video_prompt",
    "create_video_tools",
    "infer_ratio_for_platform",
]
