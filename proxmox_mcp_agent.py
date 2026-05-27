#!/usr/bin/env python3
"""
Proxmox MCP Agent
Provider: openrouter | groq | gemini | cloudflare | cerebras | mistral | ollama
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

# ── Env ───────────────────────────────────────────────────────────────────────
ENV_FILE = Path(__file__).parent / ".env"
load_dotenv(ENV_FILE)
P = "PROXMOX_MCP"


def _e(key: str, default: str = "") -> str:
    return os.getenv(f"{P}_{key}", default)


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
        "base_url": None,  # built at runtime
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
    "ollama": {
        "base_url": None,  # read from env
        "api_key_var": None,
        "model_var": f"{P}_OLLAMA_MODEL",
        "default_model": "llama3.2:1b",
    },
}


def build_client(provider: str) -> tuple[OpenAI, str]:
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown provider '{provider}'. Valid: {', '.join(PROVIDERS)}")

    cfg = PROVIDERS[provider]

    # Model: global override > per-provider env > hardcoded default
    model = _e("MODEL") or os.getenv(cfg["model_var"], "") or cfg["default_model"]

    if provider == "cloudflare":
        account_id = _e("CLOUDFLARE_ACCOUNT_ID")
        if not account_id:
            raise ValueError(f"{P}_CLOUDFLARE_ACCOUNT_ID is required for cloudflare")
        base_url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1"
    elif provider == "ollama":
        base_url = _e("OLLAMA_HOST", "http://localhost:11434")
        if not base_url.rstrip("/").endswith("/v1"):
            base_url = base_url.rstrip("/") + "/v1"
    else:
        base_url = cfg["base_url"]

    api_key = os.getenv(cfg["api_key_var"]) if cfg["api_key_var"] else "ollama"
    if not api_key:
        raise ValueError(f"Missing API key: {cfg['api_key_var']}")

    return OpenAI(base_url=base_url, api_key=api_key), model


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
            "clientInfo": {"name": "proxmox-mcp-agent", "version": "1.0.0"},
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


# ── Docker launcher ───────────────────────────────────────────────────────────
def start_docker() -> subprocess.Popen:
    image = _e("DOCKER_IMAGE", "lordraw/proxmox-mcp:latest")
    cmd = ["docker", "run", "--rm", "-i"]

    # Pass all PROXMOX_MCP_* connection vars directly — the container reads them as-is
    for var in ("HOST", "PORT", "USER", "PASSWORD", "VERIFY_SSL", "TOKEN_ID", "TOKEN_SECRET"):
        val = os.getenv(f"{P}_{var}")
        if val:
            cmd += ["-e", f"{P}_{var}={val}"]

    cmd.append(image)

    return subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    provider = _e("PROVIDER", "openrouter")

    try:
        llm, model = build_client(provider)
    except ValueError as exc:
        print(f"[error] {exc}")
        sys.exit(1)

    host = _e("HOST", "unknown")
    print(f"[*] Proxmox MCP Agent — provider: {provider} — model: {model}")
    print(f"[*] Proxmox host: {host}:{_e('PORT', '8006')}")
    print("[*] Starting MCP container …")

    proc = start_docker()

    def _stderr() -> None:
        if not proc.stderr:
            return
        for line in proc.stderr:
            line = line.strip()
            if line:
                print(f"[docker] {line}", file=sys.stderr)

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
            "You are a Proxmox VE administrator assistant. Use the available tools to help "
            f"the user manage, monitor, and operate their Proxmox cluster at {host}. "
            "Be concise and precise. "
            "For destructive or irreversible operations (delete VM, rollback snapshot, "
            "node reboot/shutdown, format disk) always confirm with the user first."
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

            # Agentic tool-call loop
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
