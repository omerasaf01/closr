"""Async HTTP client for ByteDance Seedance / BytePlus ModelArk v3 video generation API."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

import httpx

from closr.config import ClosrSettings, get_settings
from closr.schemas import VideoGenerationRequest, VideoGenerationResult

TERMINAL_STATUSES: frozenset[str] = frozenset(
    {"succeeded", "failed", "cancelled", "canceled", "expired"}
)


class SeedanceAPIError(RuntimeError):
    """Error raised when the Seedance / BytePlus ModelArk API fails or returns an error status."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        response_body: Any = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body


def _extract_url_value(value: Any) -> str | None:
    """Extract a non-empty URL string from a string or ``{"url": ...}`` mapping."""
    if isinstance(value, str):
        cleaned = value.strip()
        return cleaned or None
    if isinstance(value, dict):
        nested_url = value.get("url")
        if isinstance(nested_url, str):
            cleaned = nested_url.strip()
            return cleaned or None
    return None


def _extract_video_url(data: dict[str, Any]) -> str | None:
    """Defensively locate a generated video URL across known Seedance/Ark response shapes."""
    content = data.get("content")
    if isinstance(content, dict):
        if url := _extract_url_value(content.get("video_url")):
            return url
    elif isinstance(content, list):
        for item in content:
            if isinstance(item, dict):
                if url := _extract_url_value(item.get("video_url")):
                    return url

    if url := _extract_url_value(data.get("video_url")):
        return url

    output = data.get("output")
    if isinstance(output, dict):
        if url := _extract_url_value(output.get("video_url")):
            return url

    return None


def _extract_prompt_from_content(content: Any) -> str | None:
    """Extract the text prompt from a list-formatted ``content`` payload if present."""
    if not isinstance(content, list):
        return None
    for item in content:
        if isinstance(item, dict) and item.get("type") == "text":
            text = item.get("text")
            if isinstance(text, str) and text.strip():
                return text.strip()
    return None


def _extract_error_message(data: dict[str, Any], *, status: str) -> str | None:
    """Extract a readable error message from ``error.message``, ``error``, or ``message``."""
    error = data.get("error")
    if isinstance(error, dict):
        nested_msg = error.get("message")
        if isinstance(nested_msg, str) and nested_msg.strip():
            return nested_msg.strip()
        if error:
            return json.dumps(error, ensure_ascii=False)
    elif isinstance(error, str) and error.strip():
        return error.strip()

    if status != "succeeded":
        message = data.get("message")
        if isinstance(message, str) and message.strip():
            return message.strip()

    return None


def _coerce_int(value: Any) -> int | None:
    """Safely coerce an integer field, ignoring booleans and non-numeric values."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _extract_error_detail(body: Any) -> str:
    """Extract a concise human-readable error detail from a parsed HTTP error body."""
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict):
            msg = error.get("message")
            if isinstance(msg, str) and msg.strip():
                return msg.strip()
            return json.dumps(error, ensure_ascii=False)
        if isinstance(error, str) and error.strip():
            return error.strip()
        for key in ("message", "detail"):
            val = body.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
        return json.dumps(body, ensure_ascii=False)
    if isinstance(body, str) and body.strip():
        return body.strip()
    return ""


class SeedanceClient:
    """Async client for creating, polling, cancelling, and downloading Seedance videos."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str | None = None,
        default_model: str | None = None,
        poll_interval_seconds: float | None = None,
        poll_timeout_seconds: float | None = None,
        timeout: float = 30.0,
        settings: ClosrSettings | None = None,
        client: httpx.AsyncClient | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        resolved = settings or get_settings()
        if api_key is not None:
            self.api_key: str | None = api_key.strip() or None
        else:
            self.api_key = resolved.seedance_api_key

        raw_base_url = base_url if base_url is not None else resolved.seedance_base_url
        self.base_url: str = raw_base_url.rstrip("/")
        self.default_model: str = (
            default_model if default_model is not None else resolved.seedance_model
        )
        self.poll_interval_seconds: float = (
            poll_interval_seconds
            if poll_interval_seconds is not None
            else resolved.seedance_poll_interval_seconds
        )
        self.poll_timeout_seconds: float = (
            poll_timeout_seconds
            if poll_timeout_seconds is not None
            else resolved.seedance_poll_timeout_seconds
        )
        self.timeout: float = timeout
        self._client = client
        self._transport = transport

    def _require_api_key(self) -> str:
        """Return the configured API key or raise ``ValueError`` if missing."""
        if not self.api_key:
            raise ValueError(
                "Seedance API key (SEEDANCE_API_KEY or ARK_API_KEY) is not configured."
            )
        return self.api_key

    def _auth_headers(self) -> dict[str, str]:
        """Build authenticated JSON headers for Seedance API calls."""
        token = self._require_api_key()
        return {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

    async def _request(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        json_body: dict[str, Any] | None = None,
        follow_redirects: bool = False,
    ) -> httpx.Response:
        """Execute an HTTP request using an injected client/transport or a fresh client."""
        try:
            if self._client is not None:
                response = await self._client.request(
                    method,
                    url,
                    headers=headers,
                    json=json_body,
                    follow_redirects=follow_redirects,
                )
            else:
                async with httpx.AsyncClient(
                    timeout=self.timeout,
                    transport=self._transport,
                    follow_redirects=follow_redirects,
                ) as client:
                    response = await client.request(
                        method,
                        url,
                        headers=headers,
                        json=json_body,
                    )
        except httpx.HTTPError as exc:
            raise SeedanceAPIError(
                f"Seedance API request failed for {method} {url}: {exc}",
                status_code=None,
                response_body=None,
            ) from exc

        if response.is_error:
            raise self._build_api_error(response)

        return response

    @staticmethod
    def _build_api_error(response: httpx.Response) -> SeedanceAPIError:
        """Construct a ``SeedanceAPIError`` from an error ``httpx.Response``."""
        try:
            body: Any = response.json()
        except ValueError:
            body = response.text

        detail = _extract_error_detail(body)
        message = f"Seedance API error (HTTP {response.status_code})"
        if detail:
            message = f"{message}: {detail}"
        return SeedanceAPIError(
            message,
            status_code=response.status_code,
            response_body=body,
        )

    @staticmethod
    def _decode_json_object(response: httpx.Response) -> dict[str, Any]:
        """Parse a 2xx response body as a JSON object or raise ``SeedanceAPIError``."""
        try:
            data = response.json()
        except ValueError as exc:
            raise SeedanceAPIError(
                f"Seedance API returned invalid JSON (HTTP {response.status_code}).",
                status_code=response.status_code,
                response_body=response.text,
            ) from exc
        if not isinstance(data, dict):
            raise SeedanceAPIError(
                f"Seedance API returned unexpected payload type {type(data).__name__}.",
                status_code=response.status_code,
                response_body=data,
            )
        return data

    def build_task_payload(self, request: VideoGenerationRequest) -> dict[str, Any]:
        """Convert a ``VideoGenerationRequest`` into a ModelArk v3 task creation payload."""
        content: list[dict[str, Any]] = [{"type": "text", "text": request.prompt}]
        if request.reference_image_url:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": request.reference_image_url},
                }
            )

        payload: dict[str, Any] = {
            "model": request.model or self.default_model,
            "content": content,
            "ratio": request.ratio,
            "resolution": request.resolution,
            "duration": request.duration,
            "watermark": request.watermark,
        }
        if request.generate_audio is not None:
            payload["generate_audio"] = request.generate_audio
        if request.seed is not None:
            payload["seed"] = request.seed

        return payload

    def parse_task_response(
        self,
        data: dict[str, Any],
        *,
        fallback_prompt: str = "",
        fallback_model: str = "",
        fallback_duration: int | None = None,
        fallback_ratio: str | None = None,
        fallback_resolution: str | None = None,
    ) -> VideoGenerationResult:
        """Normalize a Seedance task payload into a ``VideoGenerationResult``."""
        raw_task_id = data.get("id") or data.get("task_id") or ""
        task_id = str(raw_task_id).strip()

        raw_status = data.get("status")
        if isinstance(raw_status, str) and raw_status.strip():
            status = raw_status.strip().lower()
        elif data.get("error") is not None:
            status = "failed"
        else:
            status = "queued"

        error = _extract_error_message(data, status=status)
        video_url = _extract_video_url(data)

        raw_model = data.get("model")
        model = (
            str(raw_model).strip()
            if isinstance(raw_model, str) and raw_model.strip()
            else (fallback_model or self.default_model)
        )

        raw_prompt = data.get("prompt")
        if isinstance(raw_prompt, str) and raw_prompt.strip():
            prompt = raw_prompt.strip()
        else:
            prompt = _extract_prompt_from_content(data.get("content")) or fallback_prompt

        duration = (
            _coerce_int(data.get("duration"))
            if data.get("duration") is not None
            else fallback_duration
        )

        raw_ratio = data.get("ratio")
        ratio = str(raw_ratio) if raw_ratio is not None else fallback_ratio

        raw_resolution = data.get("resolution")
        resolution = (
            str(raw_resolution) if raw_resolution is not None else fallback_resolution
        )

        return VideoGenerationResult(
            task_id=task_id,
            status=status,
            model=model,
            prompt=prompt,
            video_url=video_url,
            duration=duration,
            ratio=ratio,
            resolution=resolution,
            error=error,
        )

    async def create_task(
        self, request: VideoGenerationRequest
    ) -> VideoGenerationResult:
        """Submit a new video generation task to Seedance."""
        headers = self._auth_headers()
        payload = self.build_task_payload(request)
        url = f"{self.base_url}/contents/generations/tasks"

        response = await self._request("POST", url, headers=headers, json_body=payload)
        data = self._decode_json_object(response)
        result = self.parse_task_response(
            data,
            fallback_prompt=request.prompt,
            fallback_model=str(payload["model"]),
            fallback_duration=request.duration,
            fallback_ratio=request.ratio,
            fallback_resolution=request.resolution,
        )
        if not result.task_id:
            raise SeedanceAPIError(
                "Seedance create_task response did not include a task id.",
                status_code=response.status_code,
                response_body=data,
            )
        return result

    async def get_task(
        self,
        task_id: str,
        *,
        fallback_prompt: str = "",
        fallback_model: str = "",
        fallback_duration: int | None = None,
        fallback_ratio: str | None = None,
        fallback_resolution: str | None = None,
    ) -> VideoGenerationResult:
        """Fetch the current status and output metadata for an existing Seedance task."""
        cleaned_id = task_id.strip()
        if not cleaned_id:
            raise ValueError("task_id must not be empty.")

        headers = self._auth_headers()
        url = f"{self.base_url}/contents/generations/tasks/{cleaned_id}"

        response = await self._request("GET", url, headers=headers)
        data = self._decode_json_object(response)
        result = self.parse_task_response(
            data,
            fallback_prompt=fallback_prompt,
            fallback_model=fallback_model,
            fallback_duration=fallback_duration,
            fallback_ratio=fallback_ratio,
            fallback_resolution=fallback_resolution,
        )
        if not result.task_id:
            result = result.model_copy(update={"task_id": cleaned_id})
        return result

    async def cancel_task(self, task_id: str) -> bool:
        """Cancel or delete a Seedance task; returns ``True`` on a 2xx response."""
        cleaned_id = task_id.strip()
        if not cleaned_id:
            raise ValueError("task_id must not be empty.")

        headers = self._auth_headers()
        url = f"{self.base_url}/contents/generations/tasks/{cleaned_id}"
        await self._request("DELETE", url, headers=headers)
        return True

    async def wait_for_completion(
        self,
        task_id: str,
        *,
        poll_interval: float | None = None,
        timeout: float | None = None,
        fallback_prompt: str = "",
        fallback_model: str = "",
        fallback_duration: int | None = None,
        fallback_ratio: str | None = None,
        fallback_resolution: str | None = None,
    ) -> VideoGenerationResult:
        """Poll a Seedance task until it reaches a terminal state or times out."""
        interval = (
            poll_interval if poll_interval is not None else self.poll_interval_seconds
        )
        total_timeout = (
            timeout if timeout is not None else self.poll_timeout_seconds
        )
        if interval <= 0:
            raise ValueError("poll_interval must be positive.")
        if total_timeout <= 0:
            raise ValueError("timeout must be positive.")

        deadline = time.monotonic() + total_timeout
        last_status = "unknown"

        while True:
            result = await self.get_task(
                task_id,
                fallback_prompt=fallback_prompt,
                fallback_model=fallback_model,
                fallback_duration=fallback_duration,
                fallback_ratio=fallback_ratio,
                fallback_resolution=fallback_resolution,
            )
            last_status = result.status
            if result.status in TERMINAL_STATUSES:
                return result

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"Timed out waiting for Seedance task '{task_id}' after "
                    f"{total_timeout:.2f}s (last status: '{last_status}')."
                )
            await asyncio.sleep(min(interval, remaining))

    async def download_video(self, video_url: str, output_path: Path | str) -> Path:
        """Download a generated video from ``video_url`` and save it to ``output_path``."""
        cleaned_url = video_url.strip()
        if not cleaned_url:
            raise ValueError("video_url must not be empty.")

        destination = Path(output_path)
        response = await self._request(
            "GET",
            cleaned_url,
            follow_redirects=True,
        )

        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(response.content)
        return destination

    async def generate_video(
        self,
        request: VideoGenerationRequest,
        *,
        wait: bool = True,
        download_to: Path | str | None = None,
        poll_interval: float | None = None,
        timeout: float | None = None,
    ) -> VideoGenerationResult:
        """Create a Seedance task, optionally wait for completion, and optionally download."""
        result = await self.create_task(request)

        if wait and result.status not in TERMINAL_STATUSES:
            result = await self.wait_for_completion(
                result.task_id,
                poll_interval=poll_interval,
                timeout=timeout,
                fallback_prompt=request.prompt,
                fallback_model=result.model,
                fallback_duration=request.duration,
                fallback_ratio=request.ratio,
                fallback_resolution=request.resolution,
            )

        if download_to is not None and result.status == "succeeded" and result.video_url:
            saved_path = await self.download_video(result.video_url, download_to)
            result = result.model_copy(update={"local_path": str(saved_path)})

        return result
