"""One place for every LLM / embedding call, with latency and token logging.

The key is read from the environment variable QBC13_API_KEY, or from the
git-ignored file .env in the project root. It is never written in code.
Every call is appended to logs/llm_calls.jsonl so Latency and Cost in the
evaluation come straight from real calls.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
from pathlib import Path

from openai import OpenAI

from .config import API_BASE_URL, ROOT, env

BASE_URL = API_BASE_URL
CHAT_MODEL = "openai/gpt-4o-mini"
EMBED_MODEL = "openai/text-embedding-3-small"
LOG_FILE = ROOT / "logs" / "llm_calls.jsonl"


def _api_key() -> str:
    key = env("QBC13_API_KEY")
    if not key:
        raise RuntimeError("set QBC13_API_KEY or put it in .env")
    return key


_client: OpenAI | None = None


def client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(base_url=BASE_URL, api_key=_api_key(), timeout=120, max_retries=2)
    return _client


# Calls since the last reset; the evaluation reads this per question.
CALLS: list[dict] = []


def _log(record: dict) -> None:
    CALLS.append(record)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ------------------------------------------------------------------ cache
# Week 12 (systems and efficiency): the same question with the same data
# gives the same prompt, so its answer can be reused instead of paid for
# again. Only deterministic calls (temperature 0) are cached. The cache is
# off by default so the evaluation always measures real calls; the app
# turns it on with enable_cache().

CACHE_FILE = ROOT / "cache" / "llm_cache.sqlite"
_cache_on = os.environ.get("LLM_CACHE", "0") == "1"


def enable_cache(on: bool = True) -> None:
    global _cache_on
    _cache_on = on


def _cache_key(payload: dict) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _cache_db() -> sqlite3.Connection:
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(CACHE_FILE)
    db.execute("CREATE TABLE IF NOT EXISTS answers (key TEXT PRIMARY KEY, answer TEXT, "
               "prompt_tokens INT, completion_tokens INT, latency_s REAL)")
    return db


def chat(messages: list[dict], *, model: str = CHAT_MODEL, task: str = "",
         temperature: float = 0.0, json_mode: bool = False, max_tokens: int = 1500) -> str:
    """Send a chat request and log its latency and token usage."""
    key = None
    if _cache_on and temperature == 0:
        key = _cache_key({"m": model, "msg": messages, "json": json_mode, "max": max_tokens})
        start = time.perf_counter()
        with _cache_db() as db:
            hit = db.execute("SELECT answer, prompt_tokens, completion_tokens, latency_s "
                             "FROM answers WHERE key = ?", (key,)).fetchone()
        if hit:
            _log({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "kind": "chat", "task": task,
                  "model": model, "cached": True,
                  "latency_s": round(time.perf_counter() - start, 3),
                  "prompt_tokens": 0, "completion_tokens": 0,
                  "saved_tokens": (hit[1] or 0) + (hit[2] or 0), "saved_latency_s": hit[3]})
            return hit[0]
    kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}
    start = time.perf_counter()
    response = client().chat.completions.create(
        model=model, messages=messages, temperature=temperature,
        max_tokens=max_tokens, **kwargs)
    latency = time.perf_counter() - start
    usage = response.usage
    _log({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "kind": "chat", "task": task,
          "model": model, "latency_s": round(latency, 3),
          "prompt_tokens": getattr(usage, "prompt_tokens", None),
          "completion_tokens": getattr(usage, "completion_tokens", None)})
    answer = response.choices[0].message.content or ""
    if key is not None and answer:
        with _cache_db() as db:
            db.execute("INSERT OR REPLACE INTO answers VALUES (?, ?, ?, ?, ?)",
                       (key, answer, getattr(usage, "prompt_tokens", None),
                        getattr(usage, "completion_tokens", None), round(latency, 3)))
    return answer


def chat_json(messages: list[dict], **kwargs) -> dict:
    """Chat with a JSON contract; the reply must parse or this raises."""
    return json.loads(chat(messages, json_mode=True, **kwargs))


def embed(texts: list[str], *, model: str = EMBED_MODEL, task: str = "",
          batch_size: int = 256) -> list[list[float]]:
    """Embed texts in batches and log each batch."""
    vectors: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        start = time.perf_counter()
        response = client().embeddings.create(model=model, input=batch)
        _log({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "kind": "embed", "task": task,
              "model": model, "latency_s": round(time.perf_counter() - start, 3),
              "n_texts": len(batch),
              "prompt_tokens": getattr(response.usage, "prompt_tokens", None)})
        vectors.extend(item.embedding for item in response.data)
    return vectors
