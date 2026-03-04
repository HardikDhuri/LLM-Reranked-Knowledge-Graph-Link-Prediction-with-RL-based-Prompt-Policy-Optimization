"""LLM client for OpenAI-compatible AI Gateway."""

import json
import logging
import re
import time
from typing import Any

import requests

from src.config import get_settings

logger = logging.getLogger(__name__)


class LLMClient:
    """Client for OpenAI-compatible AI Gateway."""

    def __init__(
        self,
        api_key: str = None,
        base_url: str = None,
        model: str = None,
    ):
        settings = get_settings()
        self.api_key = api_key or settings.ai_gateway_api_key
        self.base_url = (base_url or settings.ai_gateway_base_url).rstrip("/")
        self.model = model or settings.ai_gateway_model
        self._session = requests.Session()
        self._session.headers.update(
            {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
        )
        # Stats
        self.total_calls = 0
        self.total_failures = 0
        self.total_tokens = 0
        self.total_latency = 0.0

    def list_models(self) -> list[str]:
        """Discover available models from the gateway."""
        try:
            resp = self._session.get(f"{self.base_url}/models", timeout=15)
            resp.raise_for_status()
            data = resp.json()
            models = [m["id"] for m in data.get("data", [])]
            return sorted(models)
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []

    def chat_completion(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 256,
        response_format: dict | None = None,
    ) -> dict[str, Any]:
        """Send a chat completion request.

        Returns parsed response dict with keys: content, usage, latency_s, raw.
        """
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format:
            payload["response_format"] = response_format

        for attempt in range(3):
            start = time.time()
            try:
                resp = self._session.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    timeout=60,
                )
                latency = time.time() - start

                if resp.status_code == 429:
                    wait = 2**attempt * 3
                    logger.warning(
                        f"Rate limited, waiting {wait}s (attempt {attempt + 1})"
                    )
                    time.sleep(wait)
                    continue

                resp.raise_for_status()
                self.total_calls += 1
                self.total_latency += latency
                data = resp.json()

                content = data["choices"][0]["message"]["content"]
                usage = data.get("usage", {})
                self.total_tokens += usage.get("total_tokens", 0)

                return {
                    "content": content,
                    "usage": usage,
                    "latency_s": latency,
                    "raw": data,
                }
            except Exception as e:
                self.total_failures += 1
                logger.warning(f"LLM call failed (attempt {attempt + 1}): {e}")
                if attempt < 2:
                    time.sleep(2**attempt)
                else:
                    raise

    def chat_completion_json(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 256,
    ) -> dict[str, Any]:
        """Chat completion with JSON output enforcement.

        Tries response_format={"type": "json_object"} first.
        Falls back to parsing JSON from content string.
        Returns dict with keys: parsed_json, content, usage, latency_s.
        """
        result = self.chat_completion(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        content = result["content"]
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            parsed = self._extract_json(content)
        return {
            "parsed_json": parsed,
            "content": content,
            "usage": result["usage"],
            "latency_s": result["latency_s"],
        }

    @staticmethod
    def _extract_json(text: str) -> dict | None:
        """Try to extract JSON object from text that may contain extra content."""
        # Try to find JSON block in markdown code fences
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
        # Try to find first { ... } block
        match = re.search(r"\{[^{}]*\}", text)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        logger.warning(f"Could not extract JSON from: {text[:200]}")
        return None

    def stats(self) -> dict[str, Any]:
        """Return a snapshot of usage statistics."""
        return {
            "total_calls": self.total_calls,
            "total_failures": self.total_failures,
            "total_tokens": self.total_tokens,
            "total_latency_s": round(self.total_latency, 2),
            "avg_latency_s": (
                round(self.total_latency / self.total_calls, 2)
                if self.total_calls > 0
                else 0.0
            ),
        }

    def reset_stats(self) -> None:
        """Reset all usage statistics to zero."""
        self.total_calls = 0
        self.total_failures = 0
        self.total_tokens = 0
        self.total_latency = 0.0
