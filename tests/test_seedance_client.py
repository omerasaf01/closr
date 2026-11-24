"""Unit tests for the async Seedance / BytePlus ModelArk v3 video client."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from closr.config import ClosrSettings
from closr.schemas import VideoGenerationRequest
from closr.video.seedance_client import SeedanceAPIError, SeedanceClient


def _make_settings(**overrides: Any) -> ClosrSettings:
    """Create deterministic settings without reading `.env`."""
    defaults: dict[str, Any] = {
        "_env_file": None,
        "seedance_api_key": "test-seedance-key",
        "seedance_base_url": "https://ark.example.com/api/v3/",
        "seedance_model": "seedance-1-0-pro-250528",
        "seedance_poll_interval_seconds": 0.01,
        "seedance_poll_timeout_seconds": 0.5,
    }
    defaults.update(overrides)
    return ClosrSettings(**defaults)


# ---------------------------------------------------------------------------
# Initialization & missing API key
# ---------------------------------------------------------------------------


def test_init_does_not_raise_when_api_key_missing() -> None:
    settings = _make_settings(seedance_api_key=None)
    client = SeedanceClient(settings=settings)
    assert client.api_key is None
    assert client.base_url == "https://ark.example.com/api/v3"
    assert client.default_model == "seedance-1-0-pro-250528"


@pytest.mark.asyncio
async def test_missing_api_key_raises_value_error_on_api_calls() -> None:
    settings = _make_settings(seedance_api_key=None)
    client = SeedanceClient(settings=settings)
    req = VideoGenerationRequest(prompt="A bright studio showcase of our app")

    expected_msg = "Seedance API key (SEEDANCE_API_KEY or ARK_API_KEY) is not configured."
    with pytest.raises(ValueError, match="Seedance API key"):
        await client.create_task(req)

    with pytest.raises(ValueError, match="Seedance API key") as exc_info:
        await client.get_task("task-123")
    assert str(exc_info.value) == expected_msg

    with pytest.raises(ValueError, match="Seedance API key"):
        await client.cancel_task("task-123")


# ---------------------------------------------------------------------------
# Payload construction
# ---------------------------------------------------------------------------


def test_build_task_payload_text_only() -> None:
    client = SeedanceClient(settings=_make_settings())
    req = VideoGenerationRequest(
        prompt="Clean minimalist desk with laptop showing analytics dashboard",
        ratio="16:9",
        resolution="720p",
        duration=5,
        watermark=False,
    )

    payload = client.build_task_payload(req)

    assert payload == {
        "model": "seedance-1-0-pro-250528",
        "content": [
            {
                "type": "text",
                "text": "Clean minimalist desk with laptop showing analytics dashboard",
            }
        ],
        "ratio": "16:9",
        "resolution": "720p",
        "duration": 5,
        "watermark": False,
    }
    assert "generate_audio" not in payload
    assert "seed" not in payload


def test_build_task_payload_with_optional_fields_and_model_override() -> None:
    client = SeedanceClient(settings=_make_settings())
    req = VideoGenerationRequest(
        prompt="Smooth dolly zoom on electric coffee grinder",
        model="seedance-1-0-lite-250528",
        ratio="9:16",
        resolution="1080p",
        duration=10,
        watermark=True,
        generate_audio=True,
        seed=42,
        reference_image_url="https://cdn.example.com/ref.png",
    )

    payload = client.build_task_payload(req)

    assert payload == {
        "model": "seedance-1-0-lite-250528",
        "content": [
            {
                "type": "text",
                "text": "Smooth dolly zoom on electric coffee grinder",
            },
            {
                "type": "image_url",
                "image_url": {"url": "https://cdn.example.com/ref.png"},
            },
        ],
        "ratio": "9:16",
        "resolution": "1080p",
        "duration": 10,
        "watermark": True,
        "generate_audio": True,
        "seed": 42,
    }


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("data", "expected_url"),
    [
        (
            {
                "id": "cgt-1",
                "status": "succeeded",
                "content": {"video_url": "https://cdn.example.com/byteplus.mp4"},
            },
            "https://cdn.example.com/byteplus.mp4",
        ),
        (
            {
                "id": "cgt-2",
                "status": "succeeded",
                "content": [{"video_url": "https://cdn.example.com/list_str.mp4"}],
            },
            "https://cdn.example.com/list_str.mp4",
        ),
        (
            {
                "id": "cgt-3",
                "status": "succeeded",
                "content": [
                    {
                        "type": "video_url",
                        "video_url": {"url": "https://cdn.example.com/list_obj.mp4"},
                    }
                ],
            },
            "https://cdn.example.com/list_obj.mp4",
        ),
        (
            {
                "id": "cgt-4",
                "status": "succeeded",
                "video_url": "https://cdn.example.com/flat.mp4",
            },
            "https://cdn.example.com/flat.mp4",
        ),
        (
            {
                "id": "cgt-5",
                "status": "succeeded",
                "output": {"video_url": "https://cdn.example.com/output.mp4"},
            },
            "https://cdn.example.com/output.mp4",
        ),
    ],
)
def test_parse_task_response_video_url_variants(
    data: dict[str, Any], expected_url: str
) -> None:
    client = SeedanceClient(settings=_make_settings())
    result = client.parse_task_response(
        data,
        fallback_prompt="Fallback prompt",
        fallback_duration=5,
        fallback_ratio="16:9",
        fallback_resolution="720p",
    )

    assert result.status == "succeeded"
    assert result.video_url == expected_url
    assert result.prompt == "Fallback prompt"
    assert result.model == "seedance-1-0-pro-250528"
    assert result.duration == 5
    assert result.ratio == "16:9"
    assert result.resolution == "720p"
    assert result.error is None


@pytest.mark.parametrize(
    ("data", "expected_error"),
    [
        (
            {
                "id": "cgt-err-1",
                "status": "failed",
                "error": {"code": "BadRequest", "message": "Prompt violated safety policy"},
            },
            "Prompt violated safety policy",
        ),
        (
            {
                "id": "cgt-err-2",
                "status": "failed",
                "error": "Upstream worker crashed",
            },
            "Upstream worker crashed",
        ),
        (
            {
                "id": "cgt-err-3",
                "status": "failed",
                "message": "Task quota exceeded",
            },
            "Task quota exceeded",
        ),
    ],
)
def test_parse_task_response_error_variants(
    data: dict[str, Any], expected_error: str
) -> None:
    client = SeedanceClient(settings=_make_settings())
    result = client.parse_task_response(data)

    assert result.status == "failed"
    assert result.error == expected_error
    assert result.video_url is None


# ---------------------------------------------------------------------------
# CRUD API methods: create_task, get_task, cancel_task
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_get_and_cancel_task_verify_requests() -> None:
    captured: list[tuple[str, str, dict[str, str], Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8")) if request.content else None
        captured.append(
            (request.method, str(request.url), dict(request.headers), body)
        )

        if request.method == "POST" and request.url.path == "/api/v3/contents/generations/tasks":
            return httpx.Response(200, json={"id": "cgt-abc-123"})

        if (
            request.method == "GET"
            and request.url.path == "/api/v3/contents/generations/tasks/cgt-abc-123"
        ):
            return httpx.Response(
                200,
                json={
                    "id": "cgt-abc-123",
                    "status": "succeeded",
                    "model": "seedance-1-0-pro-250528",
                    "content": {"video_url": "https://cdn.example.com/out.mp4"},
                    "duration": 5,
                    "ratio": "16:9",
                    "resolution": "720p",
                },
            )

        if (
            request.method == "DELETE"
            and request.url.path == "/api/v3/contents/generations/tasks/cgt-abc-123"
        ):
            return httpx.Response(204)

        return httpx.Response(404, json={"message": "Not found"})

    client = SeedanceClient(
        settings=_make_settings(),
        transport=httpx.MockTransport(handler),
    )

    req = VideoGenerationRequest(
        prompt="Sunlight streaming across a modern workspace",
        seed=7,
    )
    created = await client.create_task(req)
    assert created.task_id == "cgt-abc-123"
    assert created.status == "queued"
    assert created.prompt == "Sunlight streaming across a modern workspace"
    assert created.model == "seedance-1-0-pro-250528"

    fetched = await client.get_task("cgt-abc-123", fallback_prompt=req.prompt)
    assert fetched.task_id == "cgt-abc-123"
    assert fetched.status == "succeeded"
    assert fetched.video_url == "https://cdn.example.com/out.mp4"

    cancelled = await client.cancel_task("cgt-abc-123")
    assert cancelled is True

    assert len(captured) == 3

    post_method, post_url, post_headers, post_body = captured[0]
    assert post_method == "POST"
    assert post_url == "https://ark.example.com/api/v3/contents/generations/tasks"
    assert post_headers["authorization"] == "Bearer test-seedance-key"
    assert post_headers["content-type"] == "application/json"
    assert post_body == {
        "model": "seedance-1-0-pro-250528",
        "content": [
            {"type": "text", "text": "Sunlight streaming across a modern workspace"}
        ],
        "ratio": "16:9",
        "resolution": "720p",
        "duration": 5,
        "watermark": False,
        "seed": 7,
    }

    get_method, get_url, get_headers, _ = captured[1]
    assert get_method == "GET"
    assert get_url == "https://ark.example.com/api/v3/contents/generations/tasks/cgt-abc-123"
    assert get_headers["authorization"] == "Bearer test-seedance-key"

    del_method, del_url, del_headers, _ = captured[2]
    assert del_method == "DELETE"
    assert del_url == "https://ark.example.com/api/v3/contents/generations/tasks/cgt-abc-123"
    assert del_headers["authorization"] == "Bearer test-seedance-key"


# ---------------------------------------------------------------------------
# Polling: wait_for_completion
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_wait_for_completion_transitions_queued_running_succeeded() -> None:
    statuses = iter(["queued", "running", "succeeded"])

    def handler(request: httpx.Request) -> httpx.Response:
        status = next(statuses)
        payload: dict[str, Any] = {"id": "cgt-poll-1", "status": status}
        if status == "succeeded":
            payload["content"] = {"video_url": "https://cdn.example.com/final.mp4"}
        return httpx.Response(200, json=payload)

    client = SeedanceClient(
        settings=_make_settings(),
        transport=httpx.MockTransport(handler),
    )

    result = await client.wait_for_completion(
        "cgt-poll-1",
        poll_interval=0.01,
        timeout=0.5,
        fallback_prompt="Product demo reel",
    )
    assert result.status == "succeeded"
    assert result.video_url == "https://cdn.example.com/final.mp4"
    assert result.prompt == "Product demo reel"


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal_status", ["failed", "cancelled", "canceled", "expired"])
async def test_wait_for_completion_stops_on_terminal_failure_states(
    terminal_status: str,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "cgt-term-1",
                "status": terminal_status,
                "error": {"message": f"Task ended with {terminal_status}"},
            },
        )

    client = SeedanceClient(
        settings=_make_settings(),
        transport=httpx.MockTransport(handler),
    )

    result = await client.wait_for_completion(
        "cgt-term-1", poll_interval=0.01, timeout=0.2
    )
    assert result.status == terminal_status
    assert result.error == f"Task ended with {terminal_status}"


@pytest.mark.asyncio
async def test_wait_for_completion_timeout_raises_timeout_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"id": "cgt-slow-99", "status": "running"})

    client = SeedanceClient(
        settings=_make_settings(),
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(TimeoutError) as exc_info:
        await client.wait_for_completion(
            "cgt-slow-99",
            poll_interval=0.01,
            timeout=0.05,
        )

    message = str(exc_info.value)
    assert "cgt-slow-99" in message
    assert "running" in message


# ---------------------------------------------------------------------------
# End-to-end generate_video + download_video
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_video_end_to_end_with_download(tmp_path: Path) -> None:
    fake_mp4 = b"\x00\x00\x00\x18ftypmp42FAKEVIDEOBYTES"
    poll_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal poll_count
        if request.method == "POST" and request.url.path == "/api/v3/contents/generations/tasks":
            return httpx.Response(200, json={"id": "cgt-e2e-1", "status": "queued"})

        if request.method == "GET" and request.url.path == "/api/v3/contents/generations/tasks/cgt-e2e-1":
            poll_count += 1
            if poll_count == 1:
                return httpx.Response(200, json={"id": "cgt-e2e-1", "status": "running"})
            return httpx.Response(
                200,
                json={
                    "id": "cgt-e2e-1",
                    "status": "succeeded",
                    "content": {"video_url": "https://cdn.example.com/redirect-promo.mp4"},
                },
            )

        if request.method == "GET" and str(request.url) == "https://cdn.example.com/redirect-promo.mp4":
            return httpx.Response(
                302,
                headers={"Location": "https://storage.example.com/assets/promo.mp4"},
            )

        if request.method == "GET" and str(request.url) == "https://storage.example.com/assets/promo.mp4":
            assert "authorization" not in request.headers
            return httpx.Response(200, content=fake_mp4)

        return httpx.Response(404)

    output_file = tmp_path / "videos" / "promo.mp4"
    client = SeedanceClient(
        settings=_make_settings(),
        transport=httpx.MockTransport(handler),
    )

    req = VideoGenerationRequest(prompt="Dynamic product reveal in warm lighting")
    result = await client.generate_video(
        req,
        wait=True,
        download_to=output_file,
        poll_interval=0.01,
        timeout=0.5,
    )

    assert result.task_id == "cgt-e2e-1"
    assert result.status == "succeeded"
    assert result.video_url == "https://cdn.example.com/redirect-promo.mp4"
    assert result.local_path == str(output_file)
    assert output_file.exists()
    assert output_file.read_bytes() == fake_mp4


@pytest.mark.asyncio
async def test_generate_video_wait_false_skips_polling_and_download(tmp_path: Path) -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(f"{request.method} {request.url.path}")
        return httpx.Response(200, json={"id": "cgt-nowait"})

    client = SeedanceClient(
        settings=_make_settings(),
        transport=httpx.MockTransport(handler),
    )
    req = VideoGenerationRequest(prompt="Quick aerial shot of rooftop solar panels")
    result = await client.generate_video(req, wait=False, download_to=tmp_path / "out.mp4")

    assert result.task_id == "cgt-nowait"
    assert result.status == "queued"
    assert result.local_path is None
    assert calls == ["POST /api/v3/contents/generations/tasks"]
    assert not (tmp_path / "out.mp4").exists()


# ---------------------------------------------------------------------------
# HTTP error handling & injected AsyncClient
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status_code", "body", "expected_fragment"),
    [
        (401, {"error": {"message": "Invalid bearer token"}}, "Invalid bearer token"),
        (429, {"message": "Rate limit exceeded"}, "Rate limit exceeded"),
        (500, "Internal Server Error", "Internal Server Error"),
    ],
)
async def test_http_errors_raise_seedance_api_error(
    status_code: int, body: Any, expected_fragment: str
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if isinstance(body, dict):
            return httpx.Response(status_code, json=body)
        return httpx.Response(status_code, text=body)

    client = SeedanceClient(
        settings=_make_settings(),
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(SeedanceAPIError) as exc_info:
        await client.get_task("cgt-err")

    err = exc_info.value
    assert err.status_code == status_code
    assert err.response_body == body
    assert str(status_code) in str(err)
    assert expected_fragment in str(err)


@pytest.mark.asyncio
async def test_download_video_http_failure_raises_seedance_api_error(
    tmp_path: Path,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, text="Video expired")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as injected:
        client = SeedanceClient(settings=_make_settings(), client=injected)
        with pytest.raises(SeedanceAPIError) as exc_info:
            await client.download_video(
                "https://cdn.example.com/missing.mp4",
                tmp_path / "missing.mp4",
            )

    assert exc_info.value.status_code == 404
    assert "404" in str(exc_info.value)
    assert not (tmp_path / "missing.mp4").exists()
