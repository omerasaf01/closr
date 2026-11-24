"""ChatGPT-powered Seedance promotional video prompt builder."""

from __future__ import annotations

import re
from typing import Any, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from langchain_core.runnables.fallbacks import RunnableWithFallbacks

from closr.llm import LLMFactory
from closr.schemas import (
    AudienceSegment,
    OrganizationProfile,
    TargetAudienceReport,
    VideoPromptSpec,
)

VideoRatio = Literal["16:9", "9:16", "1:1", "4:3", "3:4", "21:9", "adaptive"]

PLATFORM_RATIO_MAP: dict[str, VideoRatio] = {
    "linkedin": "16:9",
    "youtube": "16:9",
    "web": "16:9",
    "website": "16:9",
    "landing_page": "16:9",
    "tiktok": "9:16",
    "reels": "9:16",
    "instagram_reels": "9:16",
    "youtube_shorts": "9:16",
    "shorts": "9:16",
    "instagram": "1:1",
    "meta": "1:1",
    "facebook": "1:1",
    "feed": "1:1",
}

SEEDANCE_PROMPT_ENGINEER_SYSTEM_PROMPT = """\
You are a senior promotional video prompt engineer specializing in ByteDance Seedance text-to-video generation.

Your job is to turn verified organization and audience information into one simple, clear, and visually concrete promotional video prompt ("sade ve anlaşılır") that anyone understands immediately and Seedance renders cleanly.

Always build the `prompt` field in 4 clean, natural visual beats:
1. Subject & Brand Context: Show the real person, team, customer, or product in action, grounded in the organization's actual offerings.
2. Clear Action: Describe one smooth, observable action or visual progression that fits comfortably within the target clip duration.
3. Setting, Lighting & Mood: Place the scene in an authentic environment with natural lighting and a mood aligned to the organization's brand tone.
4. Simple Camera Movement: End with one steady, easy-to-render camera move (such as a gentle dolly-in, slow tracking shot, or calm pan).

Strict simplicity and quality rules:
- Write 2 to 4 plain, concrete sentences (roughly 40 to 85 words) in natural prose.
- Ground every visual detail strictly in the provided organization profile, audience segment, and campaign goal. Never invent fake metrics, awards, statistics, or unsupported claims.
- FORBIDDEN: tag soup or quality buzzwords (e.g., "8k", "masterpiece", "hyperrealistic", "trending on artstation", "ultra-detailed"), keyword lists, dense on-screen text instructions (video models distort text), or abrupt multi-scene jump cuts in short 5-10s clips.
- Respect the requested output language (`language`, such as "en" or "tr") across `title`, `prompt`, `target_message`, and `visual_highlights`.
- Provide 3 to 5 concise `visual_highlights` describing the key visual beats and one clear `target_message` capturing the core value proposition.
"""


def infer_ratio_for_platform(
    platform: str | None,
    default: VideoRatio = "16:9",
) -> VideoRatio:
    """Infer an aspect ratio from a target distribution platform name."""
    if platform is None:
        return default
    normalized = "_".join(
        platform.strip().lower().replace("-", " ").replace("_", " ").split()
    )
    if not normalized:
        return default
    return PLATFORM_RATIO_MAP.get(normalized, default)


def clean_video_prompt(text: str) -> str:
    """Strip wrappers/fences/quotes and normalize whitespace in a video prompt."""
    cleaned = text.strip()
    quote_pairs = (('"', '"'), ("'", "'"), ("“", "”"), ("‘", "’"))

    while cleaned:
        previous = cleaned
        fence_match = re.fullmatch(
            r"```[a-zA-Z0-9_-]*\s*(.*?)\s*```",
            cleaned,
            flags=re.DOTALL,
        )
        if fence_match:
            cleaned = fence_match.group(1).strip()
        for left, right in quote_pairs:
            if len(cleaned) >= 2 and cleaned.startswith(left) and cleaned.endswith(right):
                cleaned = cleaned[len(left) : len(cleaned) - len(right)].strip()
        if cleaned == previous:
            break

    return " ".join(cleaned.split())


def _format_list(items: list[str], fallback: str = "Not specified") -> str:
    cleaned = [item.strip() for item in items if item.strip()]
    return "; ".join(cleaned) if cleaned else fallback


def _select_primary_segment(
    audience_report: TargetAudienceReport,
) -> AudienceSegment | None:
    if not audience_report.segments:
        return None
    target_name = audience_report.recommended_primary_segment.strip().lower()
    if target_name:
        for segment in audience_report.segments:
            if segment.name.strip().lower() == target_name:
                return segment
    return max(audience_report.segments, key=lambda segment: segment.fit_score)


def _bind_structured_output(
    llm: BaseChatModel | Runnable[Any, Any],
) -> Runnable[Any, Any]:
    """Bind `VideoPromptSpec` via `with_structured_output`."""
    with_structured = getattr(llm, "with_structured_output", None)
    if callable(with_structured):
        return with_structured(VideoPromptSpec)

    if isinstance(llm, RunnableWithFallbacks):
        primary = _bind_structured_output(llm.runnable)
        fallbacks = [_bind_structured_output(fb) for fb in llm.fallbacks]
        return primary.with_fallbacks(fallbacks)

    raise TypeError(
        f"LLM object of type {type(llm).__name__} does not support 'with_structured_output'."
    )


class PromoVideoPromptBuilder:
    """Builds simple, brand-grounded Seedance promotional video prompts via OpenAI ChatGPT."""

    def __init__(
        self,
        llm: BaseChatModel | Runnable[Any, Any] | None = None,
        *,
        llm_factory: LLMFactory | None = None,
        model: str | None = None,
        temperature: float | None = 0.4,
    ) -> None:
        self._llm = llm
        self._llm_factory = llm_factory
        self._model = model
        self._temperature = temperature

    def _resolve_llm(self) -> BaseChatModel | Runnable[Any, Any]:
        """Return the injected LLM or strictly instantiate an OpenAI ChatGPT model."""
        if self._llm is not None:
            return self._llm
        factory = self._llm_factory or LLMFactory()
        return factory.get_chatgpt_model(
            model=self._model,
            temperature=self._temperature,
        )

    def _build_messages(
        self,
        profile: OrganizationProfile,
        *,
        audience_report: TargetAudienceReport | None = None,
        campaign_goal: str | None = None,
        platform: str | None = None,
        duration: int = 5,
        ratio: VideoRatio = "16:9",
        language: str = "en",
    ) -> list[BaseMessage]:
        """Construct the system and human prompt messages for structured Seedance prompt synthesis."""
        normalized_language = language.strip() or "en"
        normalized_goal = campaign_goal.strip() if campaign_goal else ""
        normalized_platform = platform.strip() if platform else ""

        lines = [
            "Create a simple, visually concrete promotional video prompt specification for Seedance.",
            "",
            "Target Video Settings:",
            f"- Platform: {normalized_platform or 'General promotional placement'}",
            f"- Aspect Ratio: {ratio}",
            f"- Duration: {duration} seconds (keep action smooth and realistic for a single {duration}s clip)",
            f"- Output Language: {normalized_language}",
        ]
        if normalized_goal:
            lines.append(f"- Campaign Goal: {normalized_goal}")

        lines.extend(
            [
                "",
                "Organization Profile (strict grounding source):",
                f"- Organization Name: {profile.organization_name}",
                f"- Industry: {profile.industry}",
                f"- Summary: {profile.summary}",
                f"- Products / Services: {_format_list(profile.products_or_services)}",
                f"- Value Proposition: {profile.value_proposition.strip() or 'Not specified'}",
                f"- Pain Points Addressed: {_format_list(profile.pain_points_addressed)}",
                f"- Differentiators: {_format_list(profile.differentiators)}",
                f"- Brand Tone: {profile.brand_tone.strip() or 'Clear, authentic, and modern'}",
            ]
        )

        if profile.existing_audience_hints:
            lines.append(
                f"- Existing Audience Hints: {_format_list(profile.existing_audience_hints)}"
            )

        if audience_report is not None:
            lines.extend(
                [
                    "",
                    "Target Audience Context:",
                    f"- Recommended Primary Segment: {audience_report.recommended_primary_segment}",
                ]
            )
            if audience_report.channel_priority:
                lines.append(
                    f"- Priority Channels: {_format_list(audience_report.channel_priority)}"
                )
            primary_segment = _select_primary_segment(audience_report)
            if primary_segment is not None:
                lines.extend(
                    [
                        f"- Segment Description: {primary_segment.description}",
                        f"- Demographics / Firmographics: {primary_segment.demographics_or_firmographics}",
                        f"- Key Roles / Personas: {_format_list(primary_segment.roles_or_personas)}",
                        f"- Segment Pain Points: {_format_list(primary_segment.pain_points)}",
                        f"- Messaging Angle: {primary_segment.messaging_angle}",
                    ]
                )

        lines.extend(
            [
                "",
                "Output Requirements:",
                f"- Write `title`, `prompt`, `target_message`, and `visual_highlights` in language '{normalized_language}'.",
                "- Keep `prompt` to 2-4 plain sentences (about 40-85 words): Subject -> Clear Action -> Environment & Lighting -> Simple Camera Movement.",
                "- Avoid buzzword tags, jump cuts, and dense on-screen text instructions.",
                f"- Set `recommended_ratio` to '{ratio}', `recommended_duration` to {duration}, and `language` to '{normalized_language}'.",
            ]
        )

        return [
            SystemMessage(content=SEEDANCE_PROMPT_ENGINEER_SYSTEM_PROMPT),
            HumanMessage(content="\n".join(lines)),
        ]

    def _prepare_invocation(
        self,
        profile: OrganizationProfile,
        *,
        audience_report: TargetAudienceReport | None,
        campaign_goal: str | None,
        platform: str | None,
        duration: int,
        ratio: VideoRatio | None,
        language: str,
    ) -> tuple[Runnable[Any, Any], list[BaseMessage], VideoRatio, str]:
        if not 4 <= duration <= 30:
            raise ValueError(
                f"Video duration must be between 4 and 30 seconds; got {duration}."
            )

        resolved_ratio: VideoRatio = (
            ratio if ratio is not None else infer_ratio_for_platform(platform)
        )
        resolved_language = language.strip() or "en"
        structured_llm = _bind_structured_output(self._resolve_llm())
        messages = self._build_messages(
            profile,
            audience_report=audience_report,
            campaign_goal=campaign_goal,
            platform=platform,
            duration=duration,
            ratio=resolved_ratio,
            language=resolved_language,
        )
        return structured_llm, messages, resolved_ratio, resolved_language

    def _finalize_spec(
        self,
        raw_result: Any,
        *,
        duration: int,
        ratio: VideoRatio,
        language: str,
    ) -> VideoPromptSpec:
        spec = (
            raw_result
            if isinstance(raw_result, VideoPromptSpec)
            else VideoPromptSpec.model_validate(raw_result)
        )
        cleaned_prompt = clean_video_prompt(spec.prompt)
        if not cleaned_prompt:
            raise ValueError("Generated video prompt is empty after cleaning.")

        return spec.model_copy(
            update={
                "title": spec.title.strip(),
                "prompt": cleaned_prompt,
                "target_message": spec.target_message.strip(),
                "recommended_duration": duration,
                "recommended_ratio": ratio,
                "language": language,
            }
        )

    def build_prompt(
        self,
        profile: OrganizationProfile,
        *,
        audience_report: TargetAudienceReport | None = None,
        campaign_goal: str | None = None,
        platform: str | None = None,
        duration: int = 5,
        ratio: VideoRatio | None = None,
        language: str = "en",
    ) -> VideoPromptSpec:
        """Synchronously synthesize a clean, brand-grounded Seedance `VideoPromptSpec`."""
        structured_llm, messages, resolved_ratio, resolved_language = (
            self._prepare_invocation(
                profile,
                audience_report=audience_report,
                campaign_goal=campaign_goal,
                platform=platform,
                duration=duration,
                ratio=ratio,
                language=language,
            )
        )
        raw_result = structured_llm.invoke(messages)
        return self._finalize_spec(
            raw_result,
            duration=duration,
            ratio=resolved_ratio,
            language=resolved_language,
        )

    async def abuild_prompt(
        self,
        profile: OrganizationProfile,
        *,
        audience_report: TargetAudienceReport | None = None,
        campaign_goal: str | None = None,
        platform: str | None = None,
        duration: int = 5,
        ratio: VideoRatio | None = None,
        language: str = "en",
    ) -> VideoPromptSpec:
        """Asynchronously synthesize a clean, brand-grounded Seedance `VideoPromptSpec`."""
        structured_llm, messages, resolved_ratio, resolved_language = (
            self._prepare_invocation(
                profile,
                audience_report=audience_report,
                campaign_goal=campaign_goal,
                platform=platform,
                duration=duration,
                ratio=ratio,
                language=language,
            )
        )
        raw_result = await structured_llm.ainvoke(messages)
        return self._finalize_spec(
            raw_result,
            duration=duration,
            ratio=resolved_ratio,
            language=resolved_language,
        )
