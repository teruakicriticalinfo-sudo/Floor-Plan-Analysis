"""Model providers used by the floor-plan analyzer."""

from __future__ import annotations

import base64
import io
import json
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_OLLAMA_MODEL = "qwen3-vl:4b-instruct"
DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"


@dataclass(frozen=True)
class TextResponse:
    text: str


def _image_as_base64(image: Any) -> str:
    """Convert a PIL-compatible image into an Ollama image payload."""
    if not hasattr(image, "save"):
        raise TypeError("Ollamaへ渡す画像をPNGに変換できません。")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


class OllamaClient:
    """Small dependency-free client for Ollama's local chat API."""

    provider_name = "ollama"

    def __init__(self, host: str = DEFAULT_OLLAMA_HOST, timeout: int = 600, num_ctx: int = 16384, max_images: int = 1):
        self.host = host.rstrip("/")
        self.timeout = timeout
        self.num_ctx = num_ctx
        self.max_images = max_images

    def generate_content(
        self,
        *,
        model: str,
        contents: list[Any],
        config: dict[str, Any] | None = None,
    ) -> TextResponse:
        config = config or {}
        text_parts: list[str] = []
        images: list[str] = []
        for item in contents:
            if isinstance(item, str):
                text_parts.append(item)
            else:
                images.append(_image_as_base64(item))

        # Qwen 4B is more reliable when it sees one high-resolution whole plan.
        # Multiple overlapping crops can be mistaken for separate floor plans.
        if images:
            images = images[: self.max_images]
            text_parts = text_parts[:1]

        message: dict[str, Any] = {"role": "user", "content": "\n\n".join(text_parts)}
        if images:
            message["images"] = images
        payload: dict[str, Any] = {
            "model": model,
            "messages": [message],
            "stream": False,
            "options": {
                "temperature": config.get("temperature", 0),
                "num_ctx": self.num_ctx,
            },
            "keep_alive": "10m",
        }
        if config.get("response_json_schema"):
            payload["format"] = config["response_json_schema"]
        elif config.get("response_mime_type") == "application/json":
            payload["format"] = "json"

        request = Request(
            f"{self.host}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Ollama APIエラー ({exc.code}): {detail}") from exc
        except URLError as exc:
            raise RuntimeError(
                f"Ollamaへ接続できません ({self.host})。Ollamaが起動しているか確認してください: {exc.reason}"
            ) from exc
        except TimeoutError as exc:
            raise RuntimeError("Ollamaの応答がタイムアウトしました。") from exc

        try:
            content = result["message"]["content"]
        except (KeyError, TypeError) as exc:
            raise RuntimeError(f"Ollamaの応答形式が不正です: {result}") from exc
        return TextResponse(text=content)


class GeminiClient:
    """Adapter that gives google-genai the same interface as OllamaClient."""

    provider_name = "gemini"

    def __init__(self, api_key: str):
        from google import genai

        self._client = genai.Client(api_key=api_key)

    def generate_content(self, **kwargs: Any) -> Any:
        kwargs = dict(kwargs)
        config = dict(kwargs.get("config") or {})
        config.pop("response_json_schema", None)
        kwargs["config"] = config
        return self._client.models.generate_content(**kwargs)


class FallbackClient:
    """Use the secondary provider only when the primary request fails."""

    provider_name = "ollama→gemini"

    def __init__(self, primary: Any, fallback: Any, fallback_model: str = "gemini-2.5-flash"):
        self.primary = primary
        self.fallback = fallback
        self.fallback_model = fallback_model

    def generate_content(self, **kwargs: Any) -> Any:
        try:
            return self.primary.generate_content(**kwargs)
        except Exception as primary_error:
            try:
                fallback_kwargs = dict(kwargs)
                fallback_kwargs["model"] = self.fallback_model
                return self.fallback.generate_content(**fallback_kwargs)
            except Exception as fallback_error:
                raise RuntimeError(
                    f"OllamaとGeminiの両方で失敗しました。"
                    f" Ollama: {primary_error}; Gemini: {fallback_error}"
                ) from fallback_error


def create_analysis_client(
    provider: str,
    *,
    ollama_host: str = DEFAULT_OLLAMA_HOST,
    ollama_num_ctx: int = 16384,
    ollama_max_images: int = 1,
    gemini_api_key: str = "",
    enable_gemini_fallback: bool = False,
) -> Any:
    """Create the configured provider without requiring a Gemini key for local use."""
    normalized = provider.strip().lower()
    if normalized == "ollama":
        primary = OllamaClient(ollama_host, num_ctx=ollama_num_ctx, max_images=ollama_max_images)
        if enable_gemini_fallback and gemini_api_key:
            return FallbackClient(primary, GeminiClient(gemini_api_key))
        return primary
    if normalized == "gemini":
        if not gemini_api_key:
            raise ValueError("GEMINI_API_KEYが設定されていません。")
        return GeminiClient(gemini_api_key)
    raise ValueError("ANALYSIS_PROVIDERは ollama または gemini を指定してください。")
