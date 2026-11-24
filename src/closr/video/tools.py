"""LangChain tools exposing ChatGPT video prompt drafting and Seedance video generation."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

from langchain_core.tools import BaseTool, tool
from pydantic import ValidationError

from closr.schemas import OrganizationProfile, TargetAudienceReport
from closr.video.prompt_builder import VideoRatio
from closr.video.seedance_client import SeedanceAPIError
from closr.video.service import PromoVideoService, VideoResolution

_VALID_RATIOS: frozenset[str] = frozenset(
    {"16:9", "9:16", "1:1", "4:3", "3:4", "21:9", "adaptive"}
)
_VALID_RESOLUTIONS: frozenset[str] = frozenset({"480p", "720p", "1080p", "4k"})


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


def _parse_ratio(raw_ratio: str) -> tuple[VideoRatio | None, str | None]:
    cleaned = raw_ratio.strip()
    if not cleaned:
        return None, None
    if cleaned not in _VALID_RATIOS:
        allowed = ", ".join(sorted(_VALID_RATIOS))
        return None, f"Error: Invalid ratio '{cleaned}'. Allowed values: {allowed}."
    return cast(VideoRatio, cleaned), None


def _parse_resolution(raw_resolution: str) -> tuple[VideoResolution | None, str | None]:
    cleaned = raw_resolution.strip().lower()
    if not cleaned:
        return None, None
    if cleaned not in _VALID_RESOLUTIONS:
        allowed = ", ".join(("480p", "720p", "1080p", "4k"))
        return None, f"Error: Invalid resolution '{cleaned}'. Allowed values: {allowed}."
    return cast(VideoResolution, cleaned), None


def create_video_tools(
    video_service: PromoVideoService,
    profile_getter: Callable[[], OrganizationProfile | None] | None = None,
    audience_getter: Callable[[], TargetAudienceReport | None] | None = None,
) -> list[BaseTool]:
    """Create async LangChain tools for promotional video prompt drafting and Seedance generation."""

    @tool
    async def draft_promo_video_prompt(
        campaign_goal: str = "",
        platform: str = "linkedin",
        duration: int = 5,
        ratio: str = "",
        language: str = "en",
        organization_profile_json: str = "",
    ) -> str:
        """Draft a simple, brand-grounded Seedance promotional video prompt via OpenAI ChatGPT."""
        profile, profile_error = _resolve_organization_profile(
            organization_profile_json,
            profile_getter,
        )
        if profile is None:
            return profile_error or "Error: No organization profile available."

        parsed_ratio, ratio_error = _parse_ratio(ratio)
        if ratio_error is not None:
            return ratio_error

        audience_report = audience_getter() if audience_getter is not None else None
        try:
            spec = await video_service.draft_prompt(
                profile,
                audience_report=audience_report,
                campaign_goal=campaign_goal.strip() or None,
                platform=platform.strip() or None,
                duration=duration,
                ratio=parsed_ratio,
                language=language.strip() or "en",
            )
        except (ValueError, ValidationError) as exc:
            return f"Error: Could not draft promotional video prompt: {exc}"

        return spec.model_dump_json(indent=2)

    @tool
    async def generate_promo_video(
        campaign_goal: str = "",
        platform: str = "linkedin",
        duration: int = 5,
        ratio: str = "",
        resolution: str = "720p",
        language: str = "en",
        wait_for_completion: bool = True,
        dry_run: bool = False,
        save_path: str = "",
        organization_profile_json: str = "",
    ) -> str:
        """Draft a promotional video prompt via ChatGPT and generate the video using ByteDance Seedance."""
        profile, profile_error = _resolve_organization_profile(
            organization_profile_json,
            profile_getter,
        )
        if profile is None:
            return profile_error or "Error: No organization profile available."

        parsed_ratio, ratio_error = _parse_ratio(ratio)
        if ratio_error is not None:
            return ratio_error

        parsed_resolution, resolution_error = _parse_resolution(resolution)
        if resolution_error is not None:
            return resolution_error

        audience_report = audience_getter() if audience_getter is not None else None
        download_to = save_path.strip() or None

        try:
            result = await video_service.create_promo_video(
                profile,
                audience_report=audience_report,
                campaign_goal=campaign_goal.strip() or None,
                platform=platform.strip() or None,
                duration=duration,
                ratio=parsed_ratio,
                resolution=parsed_resolution,
                language=language.strip() or "en",
                wait=wait_for_completion,
                download_to=download_to,
                dry_run=dry_run,
            )
        except (ValueError, ValidationError, SeedanceAPIError, TimeoutError, OSError) as exc:
            return f"Error: Promotional video generation failed: {exc}"

        return result.model_dump_json(indent=2)

    @tool
    async def check_video_generation_status(task_id: str) -> str:
        """Check the status and output URL of an existing Seedance promotional video task."""
        try:
            result = await video_service.get_video_status(task_id)
        except (ValueError, SeedanceAPIError) as exc:
            return f"Error: Could not check video generation status: {exc}"

        return result.model_dump_json(indent=2)

    return [
        draft_promo_video_prompt,
        generate_promo_video,
        check_video_generation_status,
    ]
