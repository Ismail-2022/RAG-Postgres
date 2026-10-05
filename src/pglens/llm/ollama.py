"""Text generation through a local Ollama server.

Talks to Ollama's HTTP API directly, so no extra client package is needed.
Start the server with the Ollama app or `ollama serve`.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from typing import Any

import httpx

DEFAULT_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
DEFAULT_MODEL = "qwen3.5:latest"
# Retrieved chunks can run to several thousand tokens together. Ollama's
# default context window is smaller, and would silently drop the sources.
DEFAULT_CONTEXT = 8192


class OllamaLLM:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        host: str = DEFAULT_HOST,
        context: int = DEFAULT_CONTEXT,
        timeout: float = 300.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.model = model
        self.context = context
        self._client = httpx.Client(base_url=host, timeout=timeout, transport=transport)

    def _payload(self, prompt: str, stream: bool) -> dict[str, Any]:
        return {
            "model": self.model,
            "prompt": prompt,
            "stream": stream,
            "think": False,  # answer directly; no hidden reasoning tokens
            "options": {"temperature": 0, "num_ctx": self.context},
        }

    def complete(self, prompt: str) -> str:
        response = self._client.post(
            "/api/generate", json=self._payload(prompt, stream=False)
        )
        response.raise_for_status()
        return str(response.json()["response"]).strip()

    def stream(self, prompt: str) -> Iterator[str]:
        """Yield the model's text as Ollama produces it."""
        with self._client.stream(
            "POST", "/api/generate", json=self._payload(prompt, stream=True)
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line:
                    continue
                part = json.loads(line)
                if part.get("response"):
                    yield str(part["response"])
                if part.get("done"):
                    break
