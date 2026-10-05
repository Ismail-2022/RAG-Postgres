"""Text generation through a local Ollama server.

Talks to Ollama's HTTP API directly, so no extra client package is needed.
Start the server with the Ollama app or `ollama serve`.
"""

from __future__ import annotations

import os

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

    def complete(self, prompt: str) -> str:
        response = self._client.post(
            "/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "think": False,  # answer directly; no hidden reasoning tokens
                "options": {"temperature": 0, "num_ctx": self.context},
            },
        )
        response.raise_for_status()
        return str(response.json()["response"]).strip()
