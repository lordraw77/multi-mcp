#!/usr/bin/env python3
"""Shared MCP plumbing for sub-agents.

A new sub-agent dropped into ``agents.d/`` (a ``.py`` next to its ``.json``)
only needs to expose the start launcher and the MCP helpers the orchestrator
calls (``MCPClient``, ``tools_to_openai``, ``mcp_result_to_text``,
``assistant_msg``). Import them from here so the boilerplate is not copied::

    # agents.d/myagent_agent.py
    import os
    from mcp_common import (
        MCPClient, tools_to_openai, mcp_result_to_text, assistant_msg, docker_start,
    )

    def start_docker():
        return docker_start(
            os.getenv("MYAGENT_DOCKER_IMAGE", "me/myagent-mcp:latest"),
            env={"FOO": os.getenv("MYAGENT_FOO", "")},
        )

These are the canonical copies shared by proxmox/synology/linux/homeassistant.
Two sub-agents intentionally keep their own variant (watchyourlan cleans tool
schemas for strict providers; homeassistant tweaks ``assistant_msg``) — an
external agent that needs that behaviour can override the function locally.
"""

import json
import uuid
import threading
import subprocess
from typing import Any, Optional


# ── MCP stdio client ───────────────────────────────────────────────────────────
class MCPClient:
    def __init__(self, proc: subprocess.Popen, client_name: str = "mcp-agent") -> None:
        self._proc = proc
        self._client_name = client_name
        self._lock = threading.Lock()
        self._responses: dict[str, Any] = {}
        self._events: dict[str, threading.Event] = {}
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()

    def _read_loop(self) -> None:
        if not self._proc.stdout:
            return
        for raw in self._proc.stdout:
            line = raw.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            msg_id = msg.get("id")
            if msg_id is None:
                continue
            with self._lock:
                self._responses[msg_id] = msg
                ev = self._events.get(msg_id)
            if ev:
                ev.set()

    def _rpc(self, method: str, params: Optional[dict] = None, timeout: float = 60) -> Any:
        req_id = str(uuid.uuid4())
        req: dict = {"jsonrpc": "2.0", "id": req_id, "method": method}
        if params:
            req["params"] = params

        ev = threading.Event()
        with self._lock:
            self._events[req_id] = ev

        if not self._proc.stdin:
            raise RuntimeError("MCP process stdin is closed")
        self._proc.stdin.write(json.dumps(req) + "\n")
        self._proc.stdin.flush()

        if not ev.wait(timeout):
            raise TimeoutError(f"MCP timeout waiting for '{method}'")

        with self._lock:
            resp = self._responses.pop(req_id)
            del self._events[req_id]

        if "error" in resp:
            raise RuntimeError(f"MCP error: {resp['error']}")
        return resp.get("result")

    def _notify(self, method: str, params: Optional[dict] = None) -> None:
        msg: dict = {"jsonrpc": "2.0", "method": method}
        if params:
            msg["params"] = params
        if not self._proc.stdin:
            raise RuntimeError("MCP process stdin is closed")
        self._proc.stdin.write(json.dumps(msg) + "\n")
        self._proc.stdin.flush()

    def initialize(self) -> dict:
        result = self._rpc("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "clientInfo": {"name": self._client_name, "version": "1.0.0"},
        })
        self._notify("notifications/initialized")
        return result

    def list_tools(self) -> list[dict]:
        result = self._rpc("tools/list")
        return result.get("tools", []) if result else []

    def call_tool(self, name: str, arguments: dict) -> Any:
        return self._rpc("tools/call", {"name": name, "arguments": arguments}, timeout=120)

    def close(self) -> None:
        try:
            if self._proc.stdin:
                self._proc.stdin.close()
            self._proc.wait(timeout=5)
        except Exception:
            self._proc.kill()


# ── Helpers ───────────────────────────────────────────────────────────────────
def tools_to_openai(tools: list[dict]) -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t.get("inputSchema", {"type": "object", "properties": {}}),
            },
        }
        for t in tools
    ]


def mcp_result_to_text(result: Any) -> str:
    if not isinstance(result, dict):
        return str(result)
    content = result.get("content", [])
    if isinstance(content, list):
        parts = [item["text"] for item in content if isinstance(item, dict) and item.get("type") == "text"]
        return "\n".join(parts) if parts else json.dumps(result)
    return json.dumps(result)


def assistant_msg(msg: Any) -> dict:
    d: dict = {"role": "assistant", "content": msg.content or ""}
    if msg.tool_calls:
        d["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            }
            for tc in msg.tool_calls
        ]
    return d


# ── Launcher helpers ──────────────────────────────────────────────────────────
def popen_stdio(cmd: list[str]) -> subprocess.Popen:
    """Spawn ``cmd`` wired for line-buffered JSON-RPC over stdio."""
    return subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )


def docker_start(
    image: str,
    env: Optional[dict[str, str]] = None,
    extra_args: Optional[list[str]] = None,
) -> subprocess.Popen:
    """``docker run --rm -i [extra_args] [-e K=V …] image`` over stdio.

    ``env`` keys with empty/None values are skipped (same as the built-in agents)."""
    cmd = ["docker", "run", "--rm", "-i"]
    cmd += list(extra_args or [])
    for key, val in (env or {}).items():
        if val:
            cmd += ["-e", f"{key}={val}"]
    cmd.append(image)
    return popen_stdio(cmd)
