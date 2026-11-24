"""Offline integration tests for PromoVideoService, video tools, MarketingAgent, and `closr video` CLI."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import httpx
import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda
from pydantic import Field
from typer.testing import CliRunner

import closr
from closr import (
    MarketingAgent,
    PromoVideoPromptBuilder,
    PromoVideoService,
    SeedanceClient,
    VideoGenerationResult,
    VideoPromptSpec,
)
from closr.audience import OfflineHeuristicScanner
from closr.cli import app
from closr.config import ClosrSettings
from closr.schemas import (
    AudienceSegment,
    OrganizationProfile,
    TargetAudienceReport,
)
from closr.video import create_video_tools

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_DOCS_DIR = REPO_ROOT / "examples" / "sample_org_docs"


# ---------------------------------------------------------------------------
# Deterministic offline fakes
# ---------------------------------------------------------------------------


def _sample_profile() -> OrganizationProfile:
    return OrganizationProfile(
        organization_name="NovaFlow Analytics",
        industry="B2B SaaS / Revenue Intelligence",
        summary=(
            "AI-powered RevOps and predictive churn intelligence platform unifying CRM, "
            "billing, and product telemetry in 48 hours."
        ),
        products_or_services=["NovaFlow Pulse", "NovaFlow Retain", "NovaFlow Sync"],
        value_proposition=(
            "Predictable net revenue retention in 48 hours via explainable 90-day churn alerts."
        ),
        business_goals=["Grow qualified sales pipeline by 65% YoY"],
        marketing_needs=["High-intent LinkedIn ABM and short-form product video"],
        pain_points_addressed=["Preventable ARR loss from fragmented CRM and billing data"],
        brand_tone="Authoritative, pragmatic, analytical, and approachable",
        differentiators=["Explainable 90-day churn scoring", "48-hour zero-ETL onboarding"],
        geographic_focus=["North America", "EMEA"],
        existing_audience_hints=["VP of Revenue Operations", "VP of Customer Success"],
    )


def _sample_audience_report() -> TargetAudienceReport:
    return TargetAudienceReport(
        organization_name="NovaFlow Analytics",
        scan_queries=["RevOps churn prediction B2B SaaS"],
        signals_analyzed=3,
        segments=[
            AudienceSegment(
                name="Mid-Market B2B SaaS RevOps Leaders",
                description="RevOps and Revenue Leaders at Series B-D B2B SaaS companies.",
                demographics_or_firmographics="50-500 employees, B2B SaaS in North America & EMEA",
                roles_or_personas=["VP of Revenue Operations", "Chief Revenue Officer"],
                pain_points=["Fragmented CRM and billing telemetry"],
                buying_triggers=["Missed NRR targets"],
                preferred_channels=["LinkedIn", "YouTube"],
                targeting_keywords=["revops churn intelligence"],
                communities_and_hashtags=["#RevOps"],
                fit_score=94,
                rationale="Direct alignment with 48-hour zero-ETL onboarding.",
                messaging_angle="Spot renewal risk 90 days early in 48 hours.",
            )
        ],
        recommended_primary_segment="Mid-Market B2B SaaS RevOps Leaders",
        channel_priority=["LinkedIn", "YouTube"],
    )


def _sample_prompt_spec(
    *,
    ratio: str = "16:9",
    duration: int = 5,
    language: str = "en",
) -> VideoPromptSpec:
    return VideoPromptSpec(
        title="NovaFlow 48-Hour RevOps Clarity",
        prompt=(
            "A revenue operations leader sits in a sunlit modern glass office reviewing a clean "
            "NovaFlow retention dashboard on a laptop. She taps a clear 90-day renewal alert card "
            "and smiles as team members nod in agreement beside her. Warm natural morning light "
            "fills the calm workspace. A smooth, slow dolly-in frames the screen and her reaction."
        ),
        target_message="Spot 90-day renewal risk in 48 hours with zero-ETL setup.",
        recommended_ratio=ratio,  # type: ignore[arg-type]
        recommended_duration=duration,
        visual_highlights=[
            "RevOps leader reviewing clean retention dashboard",
            "Clear 90-day churn alert card highlighted",
            "Warm natural office lighting with calm team collaboration",
            "Smooth slow dolly-in camera movement",
        ],
        language=language,
    )


class FakeStructuredLLM:
    """Deterministic fake LLM supporting `with_structured_output`."""

    def __init__(self, responses: dict[type[Any], Any]) -> None:
        self.responses = responses
        self.calls: list[tuple[type[Any], list[BaseMessage]]] = []

    def with_structured_output(self, schema: type[Any]) -> RunnableLambda[Any, Any]:
        def _invoke(messages: list[BaseMessage]) -> Any:
            self.calls.append((schema, list(messages)))
            if schema not in self.responses:
                raise KeyError(f"Unexpected schema requested: {schema}")
            return self.responses[schema]

        return RunnableLambda(_invoke)


class VideoToolChainingChatModel(BaseChatModel):
    """Fake tool-calling chat model chaining `analyze_organization_documents` -> `generate_promo_video`."""

    docs_path: str
    executed_tools: list[str] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "fake-video-chaining-chat"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> VideoToolChainingChatModel:
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        last = messages[-1]
        if isinstance(last, ToolMessage):
            tool_name = last.name or "unknown"
            self.executed_tools.append(tool_name)
            if tool_name == "analyze_organization_documents":
                return ChatResult(
                    generations=[
                        ChatGeneration(
                            message=AIMessage(
                                content="",
                                tool_calls=[
                                    {
                                        "id": "call-video-1",
                                        "name": "generate_promo_video",
                                        "args": {
                                            "campaign_goal": "EMEA 48-hour RevOps launch",
                                            "platform": "linkedin",
                                            "duration": 5,
                                            "dry_run": False,
                                        },
                                    }
                                ],
                            )
                        )
                    ]
                )
            return ChatResult(
                generations=[
                    ChatGeneration(
                        message=AIMessage(
                            content=f"Completed {tool_name}: {last.content}"
                        )
                    )
                ]
            )

        if isinstance(last, HumanMessage):
            return ChatResult(
                generations=[
                    ChatGeneration(
                        message=AIMessage(
                            content="",
                            tool_calls=[
                                {
                                    "id": "call-docs-1",
                                    "name": "analyze_organization_documents",
                                    "args": {"path": self.docs_path},
                                }
                            ],
                        )
                    )
                ]
            )

        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="Ready."))])


def _make_seedance_mock_client(
    *,
    api_key: str | None = "seedance-test-key",
    video_bytes: bytes = b"\x00\x00\x00\x18ftypmp42fake-video-data",
    seen_requests: list[tuple[str, str, dict[str, Any] | None]] | None = None,
) -> SeedanceClient:
    """Create a `SeedanceClient` backed by `httpx.MockTransport` (POST -> queued -> succeeded -> mp4)."""
    poll_count = 0

    def _handler(request: httpx.Request) -> httpx.Response:
        nonlocal poll_count
        body: dict[str, Any] | None = None
        if request.content:
            body = json.loads(request.content.decode("utf-8"))
        if seen_requests is not None:
            seen_requests.append((request.method, str(request.url), body))

        if request.method == "POST" and request.url.path.endswith("/contents/generations/tasks"):
            return httpx.Response(
                200,
                json={"id": "cgt-task-123", "status": "queued", "model": "seedance-1-0-pro-250528"},
            )

        if request.method == "GET" and request.url.path.endswith("/contents/generations/tasks/cgt-task-123"):
            poll_count += 1
            if poll_count == 1:
                return httpx.Response(200, json={"id": "cgt-task-123", "status": "queued"})
            return httpx.Response(
                200,
                json={
                    "id": "cgt-task-123",
                    "status": "succeeded",
                    "model": "seedance-1-0-pro-250528",
                    "duration": 5,
                    "ratio": "16:9",
                    "resolution": "720p",
                    "content": {"video_url": "https://cdn.seedance.test/videos/promo.mp4"},
                },
            )

        if request.method == "GET" and str(request.url) == "https://cdn.seedance.test/videos/promo.mp4":
            return httpx.Response(200, content=video_bytes, headers={"Content-Type": "video/mp4"})

        return httpx.Response(404, json={"error": {"message": f"Unexpected request: {request.method} {request.url}"}})

    transport = httpx.MockTransport(_handler)
    return SeedanceClient(
        api_key=api_key,
        base_url="https://seedance.test/api/v3",
        poll_interval_seconds=0.001,
        poll_timeout_seconds=5.0,
        transport=transport,
    )


# ---------------------------------------------------------------------------
# 1. PromoVideoService tests
# ---------------------------------------------------------------------------


async def test_promo_video_service_dry_run_makes_zero_seedance_http_calls() -> None:
    seen_requests: list[tuple[str, str, dict[str, Any] | None]] = []
    client = _make_seedance_mock_client(api_key=None, seen_requests=seen_requests)
    prompt_llm = FakeStructuredLLM({VideoPromptSpec: _sample_prompt_spec(ratio="9:16", duration=8)})
    builder = PromoVideoPromptBuilder(llm=prompt_llm)  # type: ignore[arg-type]
    settings = ClosrSettings(_env_file=None, openai_api_key=None, google_api_key=None)  # type: ignore[call-arg]
    service = PromoVideoService(prompt_builder=builder, seedance_client=client, settings=settings)

    result = await service.create_promo_video(
        _sample_profile(),
        audience_report=_sample_audience_report(),
        campaign_goal="Short-form Reels demo",
        platform="reels",
        duration=8,
        dry_run=True,
    )

    assert seen_requests == []
    assert result.task_id == "dry-run"
    assert result.status == "dry_run"
    assert result.ratio == "9:16"
    assert result.duration == 8
    assert result.resolution == "720p"
    assert result.prompt_spec is not None
    assert result.prompt_spec.title == "NovaFlow 48-Hour RevOps Clarity"


async def test_promo_video_service_full_generation_polls_and_downloads_mp4(tmp_path: Path) -> None:
    seen_requests: list[tuple[str, str, dict[str, Any] | None]] = []
    expected_bytes = b"\x00\x00\x00\x18ftypmp42novaflow-promo"
    client = _make_seedance_mock_client(video_bytes=expected_bytes, seen_requests=seen_requests)
    prompt_llm = FakeStructuredLLM({VideoPromptSpec: _sample_prompt_spec()})
    builder = PromoVideoPromptBuilder(llm=prompt_llm)  # type: ignore[arg-type]
    settings = ClosrSettings(_env_file=None, openai_api_key=None, google_api_key=None)  # type: ignore[call-arg]
    service = PromoVideoService(prompt_builder=builder, seedance_client=client, settings=settings)

    mp4_path = tmp_path / "videos" / "novaflow.mp4"
    result = await service.create_promo_video(
        _sample_profile(),
        audience_report=_sample_audience_report(),
        campaign_goal="LinkedIn RevOps awareness",
        platform="linkedin",
        duration=5,
        resolution="720p",
        download_to=mp4_path,
    )

    assert result.task_id == "cgt-task-123"
    assert result.status == "succeeded"
    assert result.video_url == "https://cdn.seedance.test/videos/promo.mp4"
    assert result.local_path == str(mp4_path)
    assert mp4_path.read_bytes() == expected_bytes
    assert result.prompt_spec is not None
    assert [method for method, _, _ in seen_requests] == ["POST", "GET", "GET", "GET"]

    # Check get_video_status delegation
    status_result = await service.get_video_status("cgt-task-123")
    assert status_result.task_id == "cgt-task-123"
    assert status_result.status == "succeeded"


# ---------------------------------------------------------------------------
# 2. LangChain video tools tests
# ---------------------------------------------------------------------------


async def test_video_tools_happy_paths_and_error_handling(tmp_path: Path) -> None:
    client = _make_seedance_mock_client()
    prompt_llm = FakeStructuredLLM({VideoPromptSpec: _sample_prompt_spec()})
    builder = PromoVideoPromptBuilder(llm=prompt_llm)  # type: ignore[arg-type]
    settings = ClosrSettings(_env_file=None, openai_api_key=None, google_api_key=None)  # type: ignore[call-arg]
    service = PromoVideoService(prompt_builder=builder, seedance_client=client, settings=settings)

    cached_profile: OrganizationProfile | None = _sample_profile()
    cached_audience: TargetAudienceReport | None = _sample_audience_report()

    tools = {
        t.name: t
        for t in create_video_tools(
            service,
            profile_getter=lambda: cached_profile,
            audience_getter=lambda: cached_audience,
        )
    }
    assert set(tools.keys()) == {
        "draft_promo_video_prompt",
        "generate_promo_video",
        "check_video_generation_status",
    }

    # 1) draft_promo_video_prompt
    draft_json = await tools["draft_promo_video_prompt"].ainvoke(
        {"campaign_goal": "RevOps demo", "platform": "linkedin", "duration": 5}
    )
    spec = VideoPromptSpec.model_validate_json(draft_json)
    assert spec.title == "NovaFlow 48-Hour RevOps Clarity"
    assert spec.recommended_ratio == "16:9"

    # 2) generate_promo_video
    saved_mp4 = tmp_path / "tool_promo.mp4"
    gen_json = await tools["generate_promo_video"].ainvoke(
        {
            "campaign_goal": "RevOps demo",
            "platform": "linkedin",
            "duration": 5,
            "save_path": str(saved_mp4),
        }
    )
    gen_res = VideoGenerationResult.model_validate_json(gen_json)
    assert gen_res.task_id == "cgt-task-123"
    assert gen_res.status == "succeeded"
    assert saved_mp4.is_file()

    # 3) check_video_generation_status
    status_json = await tools["check_video_generation_status"].ainvoke(
        {"task_id": "cgt-task-123"}
    )
    status_res = VideoGenerationResult.model_validate_json(status_json)
    assert status_res.status == "succeeded"

    # 4) Missing profile error handling
    cached_profile = None
    missing_profile_err = await tools["draft_promo_video_prompt"].ainvoke(
        {"campaign_goal": "Test"}
    )
    assert missing_profile_err.startswith("Error:")
    assert "organization profile" in missing_profile_err.lower()

    # 5) Missing Seedance API key error handling in generate_promo_video & check_status
    no_key_client = _make_seedance_mock_client(api_key=None)
    no_key_service = PromoVideoService(
        prompt_builder=builder,
        seedance_client=no_key_client,
        settings=settings,
    )
    no_key_tools = {
        t.name: t
        for t in create_video_tools(no_key_service, profile_getter=_sample_profile)
    }
    no_key_gen_err = await no_key_tools["generate_promo_video"].ainvoke(
        {"campaign_goal": "Test", "dry_run": False}
    )
    assert no_key_gen_err.startswith("Error:")
    assert "SEEDANCE_API_KEY" in no_key_gen_err

    no_key_status_err = await no_key_tools["check_video_generation_status"].ainvoke(
        {"task_id": "cgt-task-123"}
    )
    assert no_key_status_err.startswith("Error:")
    assert "SEEDANCE_API_KEY" in no_key_status_err


# ---------------------------------------------------------------------------
# 3. MarketingAgent.generate_promo_video & chat() chaining tests
# ---------------------------------------------------------------------------


async def test_marketing_agent_generate_promo_video_from_docs_end_to_end(tmp_path: Path) -> None:
    doc_llm = FakeStructuredLLM({OrganizationProfile: _sample_profile()})
    video_prompt_llm = FakeStructuredLLM({VideoPromptSpec: _sample_prompt_spec()})
    seedance_client = _make_seedance_mock_client()
    settings = ClosrSettings(
        _env_file=None,  # type: ignore[call-arg]
        openai_api_key=None,
        google_api_key=None,
        mcp_config_path=tmp_path / "missing_mcp.json",
        vectorstore_dir=tmp_path / "vectorstore",
    )
    agent = MarketingAgent(
        settings=settings,
        doc_llm=doc_llm,  # type: ignore[arg-type]
        video_prompt_llm=video_prompt_llm,  # type: ignore[arg-type]
        seedance_client=seedance_client,
        embeddings=DeterministicFakeEmbedding(size=16),
        scanner=OfflineHeuristicScanner(),
        include_builtin_mcp=False,
    )

    out_mp4 = tmp_path / "agent_promo.mp4"
    result = await agent.generate_promo_video(
        docs_path=SAMPLE_DOCS_DIR,
        campaign_goal="EMEA 48-hour RevOps launch",
        platform="linkedin",
        duration=5,
        download_to=out_mp4,
    )

    assert result.status == "succeeded"
    assert result.local_path == str(out_mp4)
    assert out_mp4.is_file()
    assert agent.current_profile is not None
    assert agent.current_profile.organization_name == "NovaFlow Analytics"
    assert result.prompt_spec is not None
    assert result.prompt_spec.title == "NovaFlow 48-Hour RevOps Clarity"


async def test_marketing_agent_chat_chains_doc_analysis_to_generate_promo_video(tmp_path: Path) -> None:
    doc_llm = FakeStructuredLLM({OrganizationProfile: _sample_profile()})
    video_prompt_llm = FakeStructuredLLM({VideoPromptSpec: _sample_prompt_spec()})
    seedance_client = _make_seedance_mock_client()
    chat_model = VideoToolChainingChatModel(docs_path=str(SAMPLE_DOCS_DIR))

    settings = ClosrSettings(
        _env_file=None,  # type: ignore[call-arg]
        openai_api_key=None,
        google_api_key=None,
        mcp_config_path=tmp_path / "missing_mcp.json",
        vectorstore_dir=tmp_path / "vectorstore",
    )
    agent = MarketingAgent(
        settings=settings,
        doc_llm=doc_llm,  # type: ignore[arg-type]
        video_prompt_llm=video_prompt_llm,  # type: ignore[arg-type]
        seedance_client=seedance_client,
        agent_llm=chat_model,
        embeddings=DeterministicFakeEmbedding(size=16),
        scanner=OfflineHeuristicScanner(),
        include_builtin_mcp=False,
    )

    assert agent.current_profile is None
    reply = await agent.chat(
        f"Analyze docs in {SAMPLE_DOCS_DIR} and generate a LinkedIn promo video.",
        thread_id="video-chain-thread",
    )

    assert agent.current_profile is not None
    assert agent.current_profile.organization_name == "NovaFlow Analytics"
    assert chat_model.executed_tools == [
        "analyze_organization_documents",
        "generate_promo_video",
    ]
    assert "cgt-task-123" in reply
    assert "succeeded" in reply


# ---------------------------------------------------------------------------
# 4. `closr video` CLI & subprocess smoke tests
# ---------------------------------------------------------------------------


def test_cli_video_dry_run_and_full_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = CliRunner()

    def _fake_create_agent(
        *,
        provider: Any = None,
        offline: bool = False,
        no_mcp: bool = False,
    ) -> MarketingAgent:
        doc_llm = FakeStructuredLLM({OrganizationProfile: _sample_profile()})
        video_prompt_llm = FakeStructuredLLM({VideoPromptSpec: _sample_prompt_spec()})
        seedance_client = _make_seedance_mock_client()
        settings = ClosrSettings(
            _env_file=None,  # type: ignore[call-arg]
            openai_api_key=None,
            google_api_key=None,
            mcp_config_path=tmp_path / "missing_mcp.json",
            vectorstore_dir=tmp_path / "vectorstore",
        )
        return MarketingAgent(
            settings=settings,
            doc_llm=doc_llm,  # type: ignore[arg-type]
            video_prompt_llm=video_prompt_llm,  # type: ignore[arg-type]
            seedance_client=seedance_client,
            embeddings=DeterministicFakeEmbedding(size=16),
            scanner=OfflineHeuristicScanner(),
            include_builtin_mcp=not no_mcp,
        )

    monkeypatch.setattr("closr.cli._create_agent", _fake_create_agent)

    # 1) --dry-run
    dry_res = runner.invoke(
        app,
        [
            "video",
            str(SAMPLE_DOCS_DIR),
            "--goal",
            "EMEA launch",
            "--platform",
            "linkedin",
            "--dry-run",
        ],
        env={"COLUMNS": "140"},
    )
    assert dry_res.exit_code == 0, dry_res.output
    assert "NovaFlow 48-Hour RevOps Clarity" in dry_res.output
    assert "dry_run" in dry_res.output

    # 2) Full generation with --save-video and --output
    video_file = tmp_path / "out" / "promo.mp4"
    json_file = tmp_path / "out" / "video_result.json"
    full_res = runner.invoke(
        app,
        [
            "video",
            str(SAMPLE_DOCS_DIR),
            "--goal",
            "EMEA launch",
            "--platform",
            "linkedin",
            "--duration",
            "5",
            "--resolution",
            "720p",
            "--save-video",
            str(video_file),
            "--output",
            str(json_file),
        ],
        env={"COLUMNS": "140"},
    )
    assert full_res.exit_code == 0, full_res.output
    assert "cgt-task-123" in full_res.output
    assert "succeeded" in full_res.output
    assert video_file.is_file()
    assert json_file.is_file()
    saved_result = VideoGenerationResult.model_validate_json(
        json_file.read_text(encoding="utf-8")
    )
    assert saved_result.task_id == "cgt-task-123"
    assert saved_result.local_path == str(video_file)

    # 3) Validation error exit code
    bad_ratio = runner.invoke(
        app,
        ["video", str(SAMPLE_DOCS_DIR), "--ratio", "5:4"],
    )
    assert bad_ratio.exit_code == 1
    assert "Error:" in bad_ratio.output


def test_subprocess_cli_video_help_and_clean_import(tmp_path: Path) -> None:
    closr_bin = REPO_ROOT / ".venv" / "bin" / "closr"
    if not closr_bin.is_file():
        closr_bin = Path(sys.executable).parent / "closr"
    assert closr_bin.is_file(), f"CLI executable not found at {closr_bin}"

    clean_env = {
        k: v
        for k, v in os.environ.items()
        if k not in {"OPENAI_API_KEY", "GOOGLE_API_KEY", "SEEDANCE_API_KEY", "ARK_API_KEY"}
        and not k.startswith("CLOSR_")
    }
    clean_env["COLUMNS"] = "160"

    help_proc = subprocess.run(
        [str(closr_bin), "video", "--help"],
        cwd=str(tmp_path),
        env=clean_env,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert help_proc.returncode == 0, f"stderr: {help_proc.stderr}"
    for flag in ("--goal", "--platform", "--duration", "--ratio", "--resolution", "--dry-run", "--save-video"):
        assert flag in help_proc.stdout

    import_proc = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import closr; "
                "assert hasattr(closr, 'PromoVideoService'); "
                "assert hasattr(closr, 'PromoVideoPromptBuilder'); "
                "assert hasattr(closr, 'SeedanceClient'); "
                "assert hasattr(closr, 'VideoPromptSpec'); "
                "assert hasattr(closr, 'VideoGenerationResult'); "
                "assert closr.get_settings().get_available_providers() == []"
            ),
        ],
        cwd=str(tmp_path),
        env=clean_env,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert import_proc.returncode == 0, f"stderr: {import_proc.stderr}"
    assert hasattr(closr, "PromoVideoService")
