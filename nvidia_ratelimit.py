#!/usr/bin/env python3
"""
Cross-process rate limiter for the NVIDIA NIM API (free tier: 40 req/min).
Shared by all agents: build_client() wraps NVIDIA clients with RateLimitedClient,
which throttles chat.completions.create() via a file-locked sliding window keyed
by API key, so concurrent agent processes never collectively exceed the limit.
"""

import fcntl
import hashlib
import os
import time
from pathlib import Path

from openai import OpenAI

RPM_LIMIT = int(os.getenv("NVIDIA_RPM_LIMIT", "38"))  # margine di sicurezza sotto i 40 rpm
_WINDOW_SECONDS = 60.0
_RATE_DIR = Path(__file__).parent / ".cache"


def _acquire(api_key: str) -> None:
    """Block until a call slot is free under RPM_LIMIT for this API key (cross-process)."""
    _RATE_DIR.mkdir(exist_ok=True)
    key_hash = hashlib.sha256(api_key.encode()).hexdigest()[:16]
    lock_path = _RATE_DIR / f"nvidia_rpm_{key_hash}.lock"
    while True:
        with open(lock_path, "a+") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                f.seek(0)
                raw = f.read().strip()
                try:
                    timestamps = [float(x) for x in raw.split(",") if x]
                except ValueError:
                    timestamps = []
                now = time.time()
                timestamps = [t for t in timestamps if now - t < _WINDOW_SECONDS]
                if len(timestamps) < RPM_LIMIT:
                    timestamps.append(now)
                    f.seek(0)
                    f.truncate()
                    f.write(",".join(f"{t:.3f}" for t in timestamps))
                    return
                wait = _WINDOW_SECONDS - (now - timestamps[0]) + 0.05
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)
        time.sleep(max(wait, 0.05))


class _RateLimitedCompletions:
    def __init__(self, completions, api_key: str):
        self._completions = completions
        self._api_key = api_key

    def create(self, *args, **kwargs):
        _acquire(self._api_key)
        return self._completions.create(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._completions, name)


class _RateLimitedChat:
    def __init__(self, chat, api_key: str):
        self._chat = chat
        self._api_key = api_key

    @property
    def completions(self):
        return _RateLimitedCompletions(self._chat.completions, self._api_key)

    def __getattr__(self, name):
        return getattr(self._chat, name)


class RateLimitedClient:
    """Proxy around an OpenAI client that throttles chat.completions.create()."""

    def __init__(self, client: OpenAI, api_key: str):
        self._client = client
        self._api_key = api_key

    @property
    def chat(self):
        return _RateLimitedChat(self._client.chat, self._api_key)

    def __getattr__(self, name):
        return getattr(self._client, name)


def wrap_if_nvidia(provider: str, client: OpenAI, api_key: str) -> OpenAI:
    """Wrap client in a RateLimitedClient if provider == 'nvidia', else return it unchanged."""
    if provider == "nvidia":
        return RateLimitedClient(client, api_key)  # type: ignore[return-value]
    return client
