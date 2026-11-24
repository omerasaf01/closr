"""Service layer orchestrating ChatGPT prompt drafting and Seedance video generation."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, cast

from closr.config import ClosrSettings, get_settings
from closr.llm import LLMFactory
from closr.schemas import (
    OrganizationProfile,
    TargetAudienceReport,
    VideoGenerationRequest,
    VideoGenerationResult,
    VideoPromptSpec,
)
from closr.video.prompt_builder import (
    PromoVideoPromptBuilder,
    VideoRatio,
    infer_ratio_for_platform,
)
from closr.video.seedance_client import SeedanceClient

VideoResolution = Literal["480p", "720p", "1080p", "4k"]


class PromoVideoService:
    """High-level service combining ChatGPT prompt engineering and Seedance video generation."""

    def __init__(
        self,
        prompt_builder: PromoVideoPromptBuilder | None = None,
        seedance_client: SeedanceClient | None = None,
        *,
        settings: ClosrSettings | None = None,
        llm_factory: LLMFactory | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        factory = llm_factory or LLMFactory(self.settings)
        self.prompt_builder = (
            prompt_builder
            if prompt_builder is not None
            else PromoVideoPromptBuilder(llm_factory=factory)
        )
        self.seedance_client = (
            seedance_client
            if seedance_client is not None
            else SeedanceClient(settings=self.settings)
        )

    async def draft_prompt(
        self,
        profile: OrganizationProfile,
        *,
        audience_report: TargetAudienceReport | None = None,
        campaign_goal: str | None = None,
        platform: str | None = None,
        duration: int | None = None,
        ratio: VideoRatio | None = None,
        language: str = "en",
    ) -> VideoPromptSpec:
        """Draft a simple, brand-grounded Seedance `VideoPromptSpec` via ChatGPT."""
        resolved_duration = (
            duration if duration is not None else self.settings.seedance_default_duration
        )
        resolved_ratio: VideoRatio = (
            ratio
            if ratio is not None
            else infer_ratio_for_platform(
                platform,
                default=cast(VideoRatio, self.settings.seedance_default_ratio),
            )
        )
        return await self.prompt_builder.abuild_prompt(
            profile,
            audience_report=audience_report,
            campaign_goal=campaign_goal,
            platform=platform,
            duration=resolved_duration,
            ratio=resolved_ratio,
            language=language,
        )

    async def create_promo_video(
        self,
        profile: OrganizationProfile,
        *,
        audience_report: TargetAudienceReport | None = None,
        campaign_goal: str | None = None,
        platform: str | None = None,
        duration: int | None = None,
        ratio: VideoRatio | None = None,
        resolution: VideoResolution | None = None,
        model: str | None = None,
        watermark: bool = False,
        generate_audio: bool | None = None,
        seed: int | None = None,
        reference_image_url: str | None = None,
        language: str = "en",
        wait: bool = True,
        download_to: Path | str | None = None,
        dry_run: bool = False,
    ) -> VideoGenerationResult:
        """Draft a promotional video prompt via ChatGPT and generate the video via Seedance."""
        prompt_spec = await self.draft_prompt(
            profile,
            audience_report=audience_report,
            campaign_goal=campaign_goal,
            platform=platform,
            duration=duration,
            ratio=ratio,
            language=language,
        )

        effective_resolution: VideoResolution = resolution or cast(
            VideoResolution,
            self.settings.seedance_default_resolution,
        )
        effective_model = model or self.seedance_client.default_model

        if dry_run:
            return VideoGenerationResult(
                task_id="dry-run",
                status="dry_run",
                model=effective_model,
                prompt=prompt_spec.prompt,
                prompt_spec=prompt_spec,
                duration=prompt_spec.recommended_duration,
                ratio=prompt_spec.recommended_ratio,
                resolution=effective_resolution,
            )

        request = VideoGenerationRequest(
            prompt=prompt_spec.prompt,
            model=effective_model,
            ratio=prompt_spec.recommended_ratio,
            resolution=effective_resolution,
            duration=prompt_spec.recommended_duration,
            watermark=watermark,
            generate_audio=generate_audio,
            seed=seed,
            reference_image_url=reference_image_url,
        )
        result = await self.seedance_client.generate_video(
            request,
            wait=wait,
            download_to=download_to,
        )
        return result.model_copy(update={"prompt_spec": prompt_spec})

    async def get_video_status(self, task_id: str) -> VideoGenerationResult:
        """Check the current status and metadata of an existing Seedance video task."""
        return await self.seedance_client.get_task(task_id)
