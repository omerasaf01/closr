"""Offline unit tests for ChatGPT-powered Seedance promotional video prompt engineering."""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from closr.config import ClosrSettings
from closr.llm import LLMFactory
from closr.schemas import (
    AudienceSegment,
    OrganizationProfile,
    TargetAudienceReport,
    VideoPromptSpec,
)
from closr.video.prompt_builder import (
    PLATFORM_RATIO_MAP,
    SEEDANCE_PROMPT_ENGINEER_SYSTEM_PROMPT,
    PromoVideoPromptBuilder,
    VideoRatio,
    clean_video_prompt,
    infer_ratio_for_platform,
)


@pytest.fixture
def clean_openai_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clear provider API key env vars so settings tests remain deterministic."""
    for key in (
        "OPENAI_API_KEY",
        "GOOGLE_API_KEY",
        "CLOSR_OPENAI_API_KEY",
        "CLOSR_GOOGLE_API_KEY",
        "CLOSR_DEFAULT_PROVIDER",
        "CLOSR_FALLBACK_PROVIDER",
    ):
        monkeypatch.delenv(key, raising=False)


def _sample_profile() -> OrganizationProfile:
    return OrganizationProfile(
        organization_name="Looma Robotics",
        industry="Warehouse Automation",
        summary="Autonomous inventory drones and vision analytics for mid-sized logistics hubs.",
        products_or_services=["Looma Scout Drone", "Looma ShelfVision"],
        value_proposition="Complete overnight warehouse cycle counts in two hours without shutting down aisles.",
        pain_points_addressed=[
            "Manual barcode counting errors",
            "Costly overnight warehouse shutdowns",
        ],
        brand_tone="Calm, practical, and trustworthy",
        differentiators=[
            "Zero-infrastructure optical navigation",
            "Deploys in a single afternoon",
        ],
        existing_audience_hints=["Operations Directors at 3PL fulfillment centers"],
    )


def _sample_audience_report() -> TargetAudienceReport:
    return TargetAudienceReport(
        organization_name="Looma Robotics",
        scan_queries=["warehouse cycle count automation"],
        signals_analyzed=6,
        segments=[
            AudienceSegment(
                name="3PL Operations Directors",
                description="Operations leaders running multi-client fulfillment warehouses.",
                demographics_or_firmographics="Mid-market 3PL providers with 50k-250k sq ft hubs",
                roles_or_personas=["Director of Operations", "Inventory Control Manager"],
                pain_points=["Missed stock discrepancies", "Overtime labor for manual audits"],
                buying_triggers=["Peak-season SKU expansion"],
                preferred_channels=["LinkedIn", "YouTube"],
                targeting_keywords=["warehouse cycle count", "inventory drone"],
                communities_and_hashtags=["#SupplyChain", "#3PL"],
                fit_score=93,
                rationale="Direct budget ownership for inventory accuracy improvements.",
                messaging_angle="Count every pallet overnight without stopping daytime fulfillment.",
            ),
            AudienceSegment(
                name="Retail Distribution VPs",
                description="Regional distribution leaders for omnichannel retailers.",
                demographics_or_firmographics="Enterprise retail DCs",
                roles_or_personas=["VP Distribution"],
                pain_points=["Out-of-stock write-offs"],
                buying_triggers=["ERP migration"],
                preferred_channels=["LinkedIn"],
                targeting_keywords=["retail dc inventory"],
                communities_and_hashtags=["#Logistics"],
                fit_score=80,
                rationale="High order volume sensitivity.",
                messaging_angle="Eliminate surprise stockouts before morning dispatch.",
            ),
        ],
        recommended_primary_segment="3PL Operations Directors",
        channel_priority=["LinkedIn", "YouTube"],
    )


class FakeStructuredRunnable:
    """Fake runnable supporting both sync `invoke` and async `ainvoke`."""

    def __init__(
        self,
        result: VideoPromptSpec,
        captured_calls: list[list[BaseMessage]],
    ) -> None:
        self._result = result
        self._captured_calls = captured_calls

    def invoke(
        self,
        input: list[BaseMessage],
        config: Any = None,
        **kwargs: Any,
    ) -> VideoPromptSpec:
        self._captured_calls.append(list(input))
        return self._result

    async def ainvoke(
        self,
        input: list[BaseMessage],
        config: Any = None,
        **kwargs: Any,
    ) -> VideoPromptSpec:
        self._captured_calls.append(list(input))
        return self._result


class FakeStructuredLLM:
    """Fake LLM exposing `with_structured_output` for offline prompt builder tests."""

    def __init__(self, result: VideoPromptSpec) -> None:
        self._result = result
        self.requested_schemas: list[Any] = []
        self.captured_calls: list[list[BaseMessage]] = []

    def with_structured_output(self, schema: Any) -> FakeStructuredRunnable:
        self.requested_schemas.append(schema)
        return FakeStructuredRunnable(self._result, self.captured_calls)


# ---------------------------------------------------------------------------
# Helper tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("platform", "expected"),
    [
        ("linkedin", "16:9"),
        ("LinkedIn", "16:9"),
        ("YouTube", "16:9"),
        ("web", "16:9"),
        ("website", "16:9"),
        ("landing_page", "16:9"),
        ("landing-page", "16:9"),
        ("TikTok", "9:16"),
        ("reels", "9:16"),
        ("Instagram Reels", "9:16"),
        ("instagram-reels", "9:16"),
        ("YouTube Shorts", "9:16"),
        ("shorts", "9:16"),
        ("Instagram", "1:1"),
        ("Meta", "1:1"),
        ("Facebook", "1:1"),
        ("feed", "1:1"),
        ("unknown-channel", "16:9"),
        ("", "16:9"),
        ("   ", "16:9"),
        (None, "16:9"),
    ],
)
def test_infer_ratio_for_platform(platform: str | None, expected: VideoRatio) -> None:
    assert infer_ratio_for_platform(platform) == expected
    assert PLATFORM_RATIO_MAP["linkedin"] == "16:9"
    assert PLATFORM_RATIO_MAP["tiktok"] == "9:16"
    assert PLATFORM_RATIO_MAP["instagram"] == "1:1"


def test_infer_ratio_for_platform_respects_custom_default() -> None:
    assert infer_ratio_for_platform("podcast", default="9:16") == "9:16"
    assert infer_ratio_for_platform(None, default="1:1") == "1:1"


@pytest.mark.parametrize(
    ("raw_text", "expected"),
    [
        (
            '  "A calm warehouse aisle at dawn.\n\nA small drone glides past neatly stacked pallets."  ',
            "A calm warehouse aisle at dawn. A small drone glides past neatly stacked pallets.",
        ),
        (
            "```text\n'A product engineer reviews a clean dashboard.\t\tWarm morning light fills the room.'\n```",
            "A product engineer reviews a clean dashboard. Warm morning light fills the room.",
        ),
        (
            "```\n“A courier loads a compact parcel van. The camera tracks smoothly beside the open door.”\n```",
            "A courier loads a compact parcel van. The camera tracks smoothly beside the open door.",
        ),
        (
            "  A team's operations manager checks inventory on a tablet.   Gentle dolly-in toward the screen.  ",
            "A team's operations manager checks inventory on a tablet. Gentle dolly-in toward the screen.",
        ),
        ("   ", ""),
    ],
)
def test_clean_video_prompt(raw_text: str, expected: str) -> None:
    assert clean_video_prompt(raw_text) == expected


def test_system_prompt_enforces_simplicity_and_four_beats() -> None:
    prompt_text = SEEDANCE_PROMPT_ENGINEER_SYSTEM_PROMPT.lower()
    assert "subject" in prompt_text
    assert "action" in prompt_text
    assert "lighting" in prompt_text
    assert "camera" in prompt_text
    assert "2 to 4" in prompt_text
    assert "40 to 85 words" in prompt_text
    assert "8k" in prompt_text  # Explicitly listed as forbidden tag soup
    assert "forbidden" in prompt_text


# ---------------------------------------------------------------------------
# Strict ChatGPT enforcement tests
# ---------------------------------------------------------------------------


def test_strict_chatgpt_enforcement_raises_when_only_google_key_configured(
    clean_openai_env: None,
) -> None:
    settings = ClosrSettings(
        _env_file=None,  # type: ignore[call-arg]
        openai_api_key=None,
        google_api_key="google-only-key",
    )
    builder = PromoVideoPromptBuilder(llm_factory=LLMFactory(settings=settings))

    with pytest.raises(ValueError, match=r"OPENAI_API_KEY|ChatGPT"):
        builder.build_prompt(_sample_profile())


async def test_strict_chatgpt_enforcement_async_raises_without_openai_key(
    clean_openai_env: None,
) -> None:
    settings = ClosrSettings(
        _env_file=None,  # type: ignore[call-arg]
        openai_api_key=None,
        google_api_key="google-only-key",
    )
    builder = PromoVideoPromptBuilder(llm_factory=LLMFactory(settings=settings))

    with pytest.raises(ValueError, match=r"OPENAI_API_KEY|ChatGPT"):
        await builder.abuild_prompt(_sample_profile())


def test_resolve_llm_returns_chat_openai_when_openai_key_set(
    clean_openai_env: None,
) -> None:
    settings = ClosrSettings(
        _env_file=None,  # type: ignore[call-arg]
        openai_api_key="sk-test-openai-key",
        google_api_key=None,
    )
    builder = PromoVideoPromptBuilder(
        llm_factory=LLMFactory(settings=settings),
        model="gpt-4o",
        temperature=0.35,
    )

    resolved = builder._resolve_llm()
    assert isinstance(resolved, ChatOpenAI)
    assert resolved.model_name == "gpt-4o"
    assert resolved.temperature == 0.35


# ---------------------------------------------------------------------------
# Prompt builder behavior tests
# ---------------------------------------------------------------------------


def test_build_prompt_grounds_org_audience_goal_and_cleans_output() -> None:
    raw_spec = VideoPromptSpec(
        title="  Overnight Clarity with Looma Robotics  ",
        prompt=(
            '```markdown\n'
            '"In a quiet, well-lit fulfillment center, a compact Looma Scout Drone glides smoothly '
            'along high inventory racks while an operations lead checks progress on a tablet.\n'
            'Soft morning daylight highlights clean aisles and calm, orderly shelves.   '
            'The camera performs a gentle tracking shot beside the drone."\n'
            '```'
        ),
        target_message="  Complete overnight cycle counts without stopping warehouse operations.  ",
        recommended_ratio="16:9",  # Will be overridden by TikTok's inferred 9:16
        recommended_duration=5,  # Will be overridden by requested duration=7
        visual_highlights=[
            "Looma Scout Drone scanning high racks",
            "Operations lead reviewing live status on tablet",
            "Smooth lateral tracking camera movement",
        ],
        language="en",
    )
    fake_llm = FakeStructuredLLM(raw_spec)
    builder = PromoVideoPromptBuilder(llm=fake_llm)

    spec = builder.build_prompt(
        _sample_profile(),
        audience_report=_sample_audience_report(),
        campaign_goal="Drive demo bookings from 3PL operations leaders",
        platform="TikTok",
        duration=7,
        language="en",
    )

    assert fake_llm.requested_schemas == [VideoPromptSpec]
    assert len(fake_llm.captured_calls) == 1
    messages = fake_llm.captured_calls[0]
    assert isinstance(messages[0], SystemMessage)
    assert isinstance(messages[1], HumanMessage)

    human_text = str(messages[1].content)
    assert "Looma Robotics" in human_text
    assert "Looma Scout Drone" in human_text
    assert "Complete overnight warehouse cycle counts" in human_text
    assert "3PL Operations Directors" in human_text
    assert "Count every pallet overnight without stopping daytime fulfillment." in human_text
    assert "Drive demo bookings from 3PL operations leaders" in human_text
    assert "TikTok" in human_text
    assert "9:16" in human_text
    assert "7 seconds" in human_text

    assert spec.title == "Overnight Clarity with Looma Robotics"
    assert spec.target_message == "Complete overnight cycle counts without stopping warehouse operations."
    assert not spec.prompt.startswith("```")
    assert not spec.prompt.startswith('"')
    assert "\n" not in spec.prompt
    assert "   " not in spec.prompt
    assert spec.recommended_ratio == "9:16"
    assert spec.recommended_duration == 7
    assert spec.language == "en"


async def test_abuild_prompt_respects_explicit_ratio_and_language() -> None:
    raw_spec = VideoPromptSpec(
        title="Looma ile Kesintisiz Sayim",
        prompt=(
            "Aydinlik bir lojistik deposunda Looma Scout Drone raflar boyunca sakin bir sekilde ilerler. "
            "Operasyon yoneticisi elindeki tabletten envanter sonucunu dogrular. "
            "Yumusak dogal isik altinda kamera araca dogru yavasca yakinlasir."
        ),
        target_message="Depoyu durdurmadan iki saatte tam envanter sayimi.",
        recommended_ratio="16:9",
        recommended_duration=5,
        visual_highlights=[
            "Depo koridorunda ilerleyen otonom drone",
            "Tabletten anlik dogrulama yapan yonetici",
            "Yavas ve akici kamera yakinlasmasi",
        ],
        language="en",
    )
    fake_llm = FakeStructuredLLM(raw_spec)
    builder = PromoVideoPromptBuilder(llm=fake_llm)

    spec = await builder.abuild_prompt(
        _sample_profile(),
        platform="LinkedIn",
        ratio="1:1",
        duration=10,
        language=" tr ",
    )

    assert len(fake_llm.captured_calls) == 1
    human_text = str(fake_llm.captured_calls[0][1].content)
    assert "Aspect Ratio: 1:1" in human_text
    assert "Output Language: tr" in human_text
    assert spec.recommended_ratio == "1:1"
    assert spec.recommended_duration == 10
    assert spec.language == "tr"


@pytest.mark.parametrize("bad_duration", [0, 3, 31, 60])
def test_invalid_duration_raises_value_error(bad_duration: int) -> None:
    fake_llm = FakeStructuredLLM(
        VideoPromptSpec(
            title="Test",
            prompt="Valid prompt text.",
            target_message="Test message.",
        )
    )
    builder = PromoVideoPromptBuilder(llm=fake_llm)

    with pytest.raises(ValueError, match="between 4 and 30 seconds"):
        builder.build_prompt(_sample_profile(), duration=bad_duration)
    assert fake_llm.captured_calls == []


async def test_invalid_duration_async_raises_value_error() -> None:
    fake_llm = FakeStructuredLLM(
        VideoPromptSpec(
            title="Test",
            prompt="Valid prompt text.",
            target_message="Test message.",
        )
    )
    builder = PromoVideoPromptBuilder(llm=fake_llm)

    with pytest.raises(ValueError, match="between 4 and 30 seconds"):
        await builder.abuild_prompt(_sample_profile(), duration=2)
    assert fake_llm.captured_calls == []


def test_blank_llm_prompt_raises_value_error() -> None:
    fake_llm = FakeStructuredLLM(
        VideoPromptSpec(
            title="Empty",
            prompt='```\n  "   "  \n```',
            target_message="Should fail.",
        )
    )
    builder = PromoVideoPromptBuilder(llm=fake_llm)

    with pytest.raises(ValueError, match="empty"):
        builder.build_prompt(_sample_profile())


async def test_blank_llm_prompt_async_raises_value_error() -> None:
    fake_llm = FakeStructuredLLM(
        VideoPromptSpec(
            title="Empty",
            prompt="   \n\t   ",
            target_message="Should fail.",
        )
    )
    builder = PromoVideoPromptBuilder(llm=fake_llm)

    with pytest.raises(ValueError, match="empty"):
        await builder.abuild_prompt(_sample_profile())


def test_unsupported_llm_raises_type_error() -> None:
    builder = PromoVideoPromptBuilder(llm=object())  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="with_structured_output"):
        builder.build_prompt(_sample_profile())
