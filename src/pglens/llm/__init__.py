"""Language model access.

Only the interface lives here, so the pipeline does not depend on one provider.
"""

from __future__ import annotations

from typing import Protocol


class LLM(Protocol):
    def complete(self, prompt: str) -> str: ...
