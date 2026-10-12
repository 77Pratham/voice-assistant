"""Streaming client for a local Ollama server (open-weight LLMs: Gemma, Llama, Qwen...)."""
from __future__ import annotations

import json
import logging
from contextlib import aclosing
from typing import AsyncIterator

import httpx

from .config import Settings

log = logging.getLogger("voice.llm")


class LLMError(RuntimeError):
    pass


class OllamaClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._http = httpx.AsyncClient(
            base_url=settings.ollama_url,
            timeout=httpx.Timeout(120.0, connect=5.0),
        )

    async def status(self) -> dict:
        """Reachability + whether the configured model has been pulled."""
        try:
            r = await self._http.get("/api/tags", timeout=5.0)
            r.raise_for_status()
            names = [m.get("name", "") for m in r.json().get("models", [])]
        except Exception as exc:
            return {"ok": False, "error": str(exc), "model": self.settings.llm_model}
        wanted = self.settings.llm_model
        present = any(n == wanted or n.split(":")[0] == wanted for n in names) or (
            ":" not in wanted and any(n.startswith(wanted + ":") for n in names)
        )
        return {"ok": True, "model": wanted, "model_pulled": present, "available": names}

    def _payload(self, messages: list[dict], max_tokens: int | None = None) -> dict:
        s = self.settings
        return {
            "model": s.llm_model,
            "messages": messages,
            "stream": True,
            "keep_alive": s.llm_keep_alive,
            "options": {
                "num_ctx": s.llm_num_ctx,
                "temperature": s.llm_temperature,
                "num_predict": max_tokens or s.llm_max_tokens,
            },
        }

    async def warmup(self) -> None:
        """Load the model into memory so the first real turn doesn't pay for it."""
        stream = self.stream_chat([{"role": "user", "content": "Hi"}], max_tokens=1)
        async with aclosing(stream):
            async for _ in stream:
                break

    async def stream_chat(
        self, messages: list[dict], max_tokens: int | None = None
    ) -> AsyncIterator[str]:
        """Yield text deltas. Closing/cancelling this generator closes the HTTP
        connection, which makes Ollama stop generating (this is what makes barge-in cheap)."""
        async with self._http.stream(
            "POST", "/api/chat", json=self._payload(messages, max_tokens)
        ) as resp:
            if resp.status_code != 200:
                body = (await resp.aread()).decode(errors="replace")
                raise LLMError(f"Ollama returned {resp.status_code}: {body[:300]}")
            async for line in resp.aiter_lines():
                if not line:
                    continue
                obj = json.loads(line)
                if "error" in obj:
                    raise LLMError(obj["error"])
                delta = obj.get("message", {}).get("content", "")
                if delta:
                    yield delta
                if obj.get("done"):
                    break

    async def aclose(self) -> None:
        await self._http.aclose()
