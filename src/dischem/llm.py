"""Async client for an OpenAI-compatible chat endpoint (vLLM), with a disk cache and one retry.

Why a cache: a pod can be reclaimed or rebooted. Every answer is written to disk as soon as it
arrives, so a restart continues where it stopped instead of spending the 3-hour budget again.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import json
import os
import time
from pathlib import Path
from typing import Any, Callable

import httpx


class LLMError(RuntimeError):
    pass


def encode_image(path: str | os.PathLike, max_side: int = 1024, quality: int = 90) -> str:
    """Return a base64 data URL, downscaled so the longest side is at most max_side."""
    from PIL import Image

    with Image.open(path) as im:
        im = im.convert("RGB")
        if max(im.size) > max_side:
            im.thumbnail((max_side, max_side))
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def extract_json(text: str) -> Any:
    """Parse a JSON object from a model reply, tolerating code fences and stray prose."""
    text = (text or "").strip()
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[4:] if text.lower().startswith("json") else text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            return json.loads(text[start:end + 1])
        raise


class DiskCache:
    def __init__(self, directory: str | os.PathLike):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def key(payload: Any) -> str:
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()

    def get(self, key: str) -> Any | None:
        p = self.dir / key[:2] / f"{key}.json"
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                return None
        return None

    def put(self, key: str, value: Any) -> None:
        p = self.dir / key[:2] / f"{key}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, p)


class ChatClient:
    """Sends chat requests with JSON-schema constrained output and returns parsed JSON."""

    def __init__(self, cfg: dict[str, Any], cache_dir: str | os.PathLike,
                 transport: httpx.AsyncBaseTransport | None = None):
        self.cfg = cfg
        self.cache = DiskCache(cache_dir)
        self.sem = asyncio.Semaphore(int(cfg["concurrency"]))
        self.stats = {"calls": 0, "cached": 0, "retries": 0, "failed": 0,
                      "prompt_tokens": 0, "completion_tokens": 0}
        self._client = httpx.AsyncClient(
            base_url=cfg["base_url"].rstrip("/"),
            headers={"Authorization": f"Bearer {cfg.get('api_key') or 'EMPTY'}"},
            timeout=httpx.Timeout(float(cfg["request_timeout_s"]), connect=10.0),
            transport=transport,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def wait_ready(self, timeout_s: int = 1800) -> None:
        deadline = time.time() + timeout_s
        last = ""
        while time.time() < deadline:
            try:
                r = await self._client.get("/models", timeout=5.0)
                if r.status_code == 200:
                    return
                last = f"HTTP {r.status_code}"
            except httpx.HTTPError as exc:
                last = repr(exc)
            await asyncio.sleep(5)
        raise LLMError(f"model server not ready after {timeout_s}s ({last})")

    async def json_chat(self, *, tag: str, system: str, user: str, schema: dict[str, Any],
                        images: list[str] | None = None, max_tokens: int = 800,
                        validate: Callable[[Any], Any] | None = None) -> Any | None:
        """Return parsed JSON, or None if the call failed after the single allowed retry."""
        content: list[dict[str, Any]] = [{"type": "image_url", "image_url": {"url": u}}
                                         for u in images or []]
        content.append({"type": "text", "text": user})
        body = {
            "model": self.cfg["model"],
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": content}],
            "temperature": float(self.cfg.get("temperature", 0.0)),
            "seed": int(self.cfg.get("seed", 42)),
            "max_tokens": int(max_tokens),
            "response_format": {"type": "json_schema",
                                "json_schema": {"name": tag, "schema": schema}},
        }
        body.update(self.cfg.get("extra_body") or {})
        key = DiskCache.key(body)
        hit = self.cache.get(key)
        if hit is not None:
            self.stats["cached"] += 1
            return hit

        attempts = 1 + int(self.cfg.get("max_retries", 1))
        async with self.sem:
            for attempt in range(attempts):
                if attempt:
                    self.stats["retries"] += 1
                self.stats["calls"] += 1
                try:
                    r = await self._client.post("/chat/completions", json=body)
                    r.raise_for_status()
                    data = r.json()
                    usage = data.get("usage") or {}
                    self.stats["prompt_tokens"] += int(usage.get("prompt_tokens") or 0)
                    self.stats["completion_tokens"] += int(usage.get("completion_tokens") or 0)
                    parsed = extract_json(data["choices"][0]["message"]["content"])
                    if validate:
                        parsed = validate(parsed)
                    self.cache.put(key, parsed)
                    return parsed
                except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError):
                    continue
        self.stats["failed"] += 1
        return None
