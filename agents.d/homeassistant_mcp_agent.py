#!/usr/bin/env python3
"""
Home Assistant MCP Agent
Connects to the native MCP server exposed by Home Assistant (HAOS) via HTTP.
Uses mcp-proxy to bridge HTTP→stdio so the same MCPClient pattern works.
Provider: openrouter | groq | gemini | cloudflare | cerebras | mistral | nvidia | ollama
"""

import os
import sys
import json
import subprocess
import threading
import uuid
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from openai import OpenAI

# This sub-agent may run from agents.d/ or standalone — make the repo root
# (shared modules like nvidia_ratelimit, and the shared .env) importable.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from nvidia_ratelimit import wrap_if_nvidia

# ── Env ───────────────────────────────────────────────────────────────────────
ENV_FILE = _REPO_ROOT / ".env"
load_dotenv(ENV_FILE)
P = "HAOS_MCP"


def _e(key: str, default: str = "") -> str:
    return os.getenv(f"{P}_{key}", default)


def _key(var: str) -> str:
    """Read var, falling back to the MAIN_AGENT_ equivalent if unset."""
    val = os.getenv(var, "")
    if val:
        return val
    main_var = var.replace(f"{P}_", "MAIN_AGENT_", 1)
    return os.getenv(main_var, "")


# ── Provider registry ─────────────────────────────────────────────────────────
PROVIDERS: dict[str, dict] = {
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_var": f"{P}_OPENROUTER_API_KEY",
        "model_var": f"{P}_OPENROUTER_MODEL",
        "default_model": "openrouter/auto",
    },
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "api_key_var": f"{P}_GROQ_API_KEY",
        "model_var": f"{P}_GROQ_MODEL",
        "default_model": "llama-3.3-70b-versatile",
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "api_key_var": f"{P}_GEMINI_API_KEY",
        "model_var": f"{P}_GEMINI_MODEL",
        "default_model": "gemini-2.0-flash",
    },
    "cloudflare": {
        "base_url": None,
        "api_key_var": f"{P}_CLOUDFLARE_API_KEY",
        "model_var": f"{P}_CLOUDFLARE_MODEL",
        "default_model": "@cf/meta/llama-3.3-70b-instruct-fp8-fast",
    },
    "cerebras": {
        "base_url": "https://api.cerebras.ai/v1",
        "api_key_var": f"{P}_CEREBRAS_API_KEY",
        "model_var": f"{P}_CEREBRAS_MODEL",
        "default_model": "llama-3.3-70b",
    },
    "mistral": {
        "base_url": "https://api.mistral.ai/v1",
        "api_key_var": f"{P}_MISTRAL_API_KEY",
        "model_var": f"{P}_MISTRAL_MODEL",
        "default_model": "mistral-small-latest",
    },
    "nvidia": {
        "base_url": "https://integrate.api.nvidia.com/v1",
        "api_key_var": f"{P}_NVIDIA_API_KEY",
        "model_var": f"{P}_NVIDIA_MODEL",
        "default_model": "meta/llama-3.3-70b-instruct",
    },
    "ollama": {
        "base_url": None,
        "api_key_var": None,
        "model_var": f"{P}_OLLAMA_MODEL",
        "default_model": "llama3.2:1b",
    },
}


def build_client(provider: str) -> tuple[OpenAI, str]:
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown provider '{provider}'. Valid: {', '.join(PROVIDERS)}")

    cfg = PROVIDERS[provider]
    model = _e("MODEL") or os.getenv("MAIN_AGENT_MODEL", "") or os.getenv(cfg["model_var"], "") or cfg["default_model"]

    if provider == "cloudflare":
        account_id = _key(f"{P}_CLOUDFLARE_ACCOUNT_ID")
        if not account_id:
            raise ValueError(f"{P}_CLOUDFLARE_ACCOUNT_ID (or MAIN_AGENT_CLOUDFLARE_ACCOUNT_ID) is required")
        base_url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1"
    elif provider == "ollama":
        base_url = _e("OLLAMA_HOST", "") or os.getenv("MAIN_AGENT_OLLAMA_HOST", "http://localhost:11434")
        if not base_url.rstrip("/").endswith("/v1"):
            base_url = base_url.rstrip("/") + "/v1"
    else:
        base_url = cfg["base_url"]

    api_key = _key(cfg["api_key_var"]) if cfg["api_key_var"] else "ollama"
    if not api_key:
        raise ValueError(f"Missing API key: {cfg['api_key_var']} (or MAIN_AGENT equivalent)")

    client = OpenAI(base_url=base_url, api_key=api_key)
    return wrap_if_nvidia(provider, client, api_key), model


# ── MCP stdio client ───────────────────────────────────────────────────────────
class MCPClient:
    def __init__(self, proc: subprocess.Popen) -> None:
        self._proc = proc
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
            "clientInfo": {"name": "homeassistant-mcp-agent", "version": "1.0.0"},
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


# ── mcp-proxy launcher ────────────────────────────────────────────────────────
def start_proxy() -> subprocess.Popen:
    """Spawn mcp-proxy bridging the HA HTTP MCP endpoint to stdio."""
    ha_url = _e("URL", "")
    if not ha_url:
        raise ValueError("HAOS_MCP_URL is required (e.g. http://192.168.1.x:8123)")
    token = _e("TOKEN", "")
    if not token:
        raise ValueError("HAOS_MCP_TOKEN is required (Home Assistant long-lived access token)")

    mcp_endpoint = ha_url.rstrip("/") + "/api/mcp"

    env = os.environ.copy()
    env["API_ACCESS_TOKEN"] = token

    # Use uvx so mcp-proxy runs under its own Python env (requires Python >= 3.10)
    cmd = [
        "uvx", "mcp-proxy",
        "--transport=streamablehttp",
        "--stateless",
        mcp_endpoint,
    ]

    return subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        env=env,
    )


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    provider = _e("PROVIDER", "openrouter")

    try:
        llm, model = build_client(provider)
    except ValueError as exc:
        print(f"[error] {exc}")
        sys.exit(1)

    ha_url = _e("URL", "(not configured)")
    print(f"[*] Home Assistant MCP Agent — provider: {provider} — model: {model}")
    print(f"[*] Home Assistant URL: {ha_url}")
    print("[*] Starting mcp-proxy …")

    try:
        proc = start_proxy()
    except ValueError as exc:
        print(f"[error] {exc}")
        sys.exit(1)

    def _stderr() -> None:
        if not proc.stderr:
            return
        for line in proc.stderr:
            line = line.strip()
            if line:
                print(f"[mcp-proxy] {line}", file=sys.stderr)

    threading.Thread(target=_stderr, daemon=True).start()

    mcp = MCPClient(proc)

    try:
        mcp.initialize()
        tools = mcp.list_tools()
        print(f"[*] Ready — {len(tools)} tool(s) available")
        for t in tools:
            print(f"    • {t['name']}: {t.get('description', '')[:70]}")

        openai_tools = tools_to_openai(tools)
        system_prompt = (
            "You are a Home Assistant smart home assistant. Use the available tools to help "
            f"the user control, monitor, and automate their home at {ha_url}. "
            "You can control lights, switches, climate, covers, media players, and other entities. "
            "Be concise and precise. "
            "For irreversible automations or bulk changes always confirm with the user first."
        )
        messages: list[dict] = [{"role": "system", "content": system_prompt}]

        print("\nType your request (or 'exit' / 'quit' to stop).\n")

        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye.")
                break

            if user_input.lower() in ("exit", "quit", "q", ""):
                if not user_input:
                    continue
                print("Bye.")
                break

            messages.append({"role": "user", "content": user_input})

            while True:
                response = llm.chat.completions.create(
                    model=model,
                    messages=messages,
                    tools=openai_tools,
                    tool_choice="auto",
                )
                msg = response.choices[0].message
                messages.append(assistant_msg(msg))

                if not msg.tool_calls:
                    print(f"\nAssistant: {msg.content}\n")
                    break

                for tc in msg.tool_calls:
                    fn = tc.function.name
                    try:
                        args = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        args = {}

                    preview = json.dumps(args)
                    if len(preview) > 80:
                        preview = preview[:77] + "…"
                    print(f"  → {fn}({preview})")

                    try:
                        result = mcp.call_tool(fn, args)
                        result_text = mcp_result_to_text(result)
                    except Exception as exc:
                        result_text = f"Tool error: {exc}"

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result_text,
                    })

    finally:
        mcp.close()


if __name__ == "__main__":
    main()
