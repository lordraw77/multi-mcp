#!/usr/bin/env python3
"""
Multi-MCP Orchestrator
Routes requests to Proxmox, Synology NAS, or Linux SSH sub-agents.
Provider: openrouter | groq | gemini | cloudflare | cerebras | mistral | ollama
"""

import os
import sys
import json
import itertools
from itertools import zip_longest
import threading
from pathlib import Path
from typing import Any, Callable, Optional

from openai import BadRequestError, RateLimitError

from dotenv import load_dotenv
from openai import OpenAI

# ── Sub-agent modules (reuse Docker launchers, MCPClient, helpers) ─────────────
sys.path.insert(0, str(Path(__file__).parent))
import proxmox_mcp_agent as _proxmox   # P = "PROXMOX_MCP"
import synology_mcp_agent as _synology  # P = "SYNOLOGY_MCP"
import linux_mcp_agent as _linux        # P = "UXMCP"

# ── Env ───────────────────────────────────────────────────────────────────────
load_dotenv(Path(__file__).parent / ".env")
P = "MAIN_AGENT"


def _e(key: str, default: str = "") -> str:
    return os.getenv(f"{P}_{key}", default)


# ── Provider registry (orchestrator LLM) ──────────────────────────────────────
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
    "ollama": {
        "base_url": None,
        "api_key_var": None,
        "model_var": f"{P}_OLLAMA_MODEL",
        "default_model": "llama3.2:1b",
    },
}


# Providers eligible for rotation (ollama escluso: non ha API key cloud)
ROTATE_PROVIDERS = ["openrouter", "groq", "gemini", "cloudflare", "cerebras", "mistral"]


def get_provider_keys(provider: str) -> list[str]:
    """Return all configured API keys for a provider (primary + _2, _3, ...)."""
    cfg = PROVIDERS.get(provider, {})
    key_var = cfg.get("api_key_var")
    if not key_var:
        return []
    keys = []
    primary = os.getenv(key_var, "")
    if primary:
        keys.append(primary)
    n = 2
    while True:
        val = os.getenv(f"{key_var}_{n}", "")
        if not val:
            break
        keys.append(val)
        n += 1
    return keys


def get_cloudflare_entries() -> list[tuple[str, str]]:
    """Return list of (api_key, account_id) pairs for Cloudflare (primary + _2, _3, ...).
    Entries missing either key or account_id are skipped with a warning."""
    entries = []
    pairs = [
        (os.getenv(f"{P}_CLOUDFLARE_API_KEY", ""), os.getenv(f"{P}_CLOUDFLARE_ACCOUNT_ID", ""))
    ]
    n = 2
    while True:
        key = os.getenv(f"{P}_CLOUDFLARE_API_KEY_{n}", "")
        acct = os.getenv(f"{P}_CLOUDFLARE_ACCOUNT_ID_{n}", "")
        if not key and not acct:
            break
        pairs.append((key, acct))
        n += 1
    for i, (key, acct) in enumerate(pairs):
        label = "cloudflare" if i == 0 else f"cloudflare#{i + 1}"
        if key and acct:
            entries.append((key, acct))
        elif key or acct:
            print(f"[warn] {label}: richiede sia API_KEY che ACCOUNT_ID — ignorato")
    return entries


def available_providers() -> list[str]:
    """Return providers from ROTATE_PROVIDERS that have at least one usable entry."""
    result = []
    for name in ROTATE_PROVIDERS:
        if name == "cloudflare":
            if get_cloudflare_entries():
                result.append(name)
        elif get_provider_keys(name):
            result.append(name)
    return result


def build_client(
    provider: str,
    api_key: Optional[str] = None,
    account_id: Optional[str] = None,
) -> tuple[OpenAI, str]:
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown provider '{provider}'. Valid: {', '.join(PROVIDERS)}")
    cfg = PROVIDERS[provider]
    model = _e("MODEL") or os.getenv(cfg["model_var"], "") or cfg["default_model"]
    if provider == "cloudflare":
        resolved_account_id = account_id or _e("CLOUDFLARE_ACCOUNT_ID")
        if not resolved_account_id:
            raise ValueError(f"{P}_CLOUDFLARE_ACCOUNT_ID is required for cloudflare")
        base_url = f"https://api.cloudflare.com/client/v4/accounts/{resolved_account_id}/ai/v1"
    elif provider == "ollama":
        base_url = _e("OLLAMA_HOST", "http://localhost:11434")
        if not base_url.rstrip("/").endswith("/v1"):
            base_url = base_url.rstrip("/") + "/v1"
    else:
        base_url = cfg["base_url"]
    resolved_key = api_key or (os.getenv(cfg["api_key_var"]) if cfg["api_key_var"] else "ollama")
    if not resolved_key:
        raise ValueError(f"Missing API key: {cfg['api_key_var']}")
    return OpenAI(base_url=base_url, api_key=resolved_key), model


def build_pool() -> list[tuple[OpenAI, str, str]]:
    """Build rotation pool interleaved across providers (round-robin by key index).
    Result order: openrouter, groq, gemini, ..., openrouter#2, groq#2, ...
    Cloudflare: each (api_key, account_id) pair is one entry."""
    per_provider: list[list[tuple[OpenAI, str, str]]] = []
    for p in available_providers():
        entries: list[tuple[OpenAI, str, str]] = []
        if p == "cloudflare":
            for i, (cf_key, cf_acct) in enumerate(get_cloudflare_entries()):
                label = "cloudflare" if i == 0 else f"cloudflare#{i + 1}"
                try:
                    client, model = build_client(p, api_key=cf_key, account_id=cf_acct)
                    entries.append((client, model, label))
                except ValueError as exc:
                    print(f"[warn] {label} ignorato: {exc}")
        else:
            for i, key in enumerate(get_provider_keys(p)):
                label = p if i == 0 else f"{p}#{i + 1}"
                try:
                    client, model = build_client(p, api_key=key)
                    entries.append((client, model, label))
                except ValueError as exc:
                    print(f"[warn] {label} ignorato: {exc}")
        if entries:
            per_provider.append(entries)
    # Round-robin: slot 0 of each provider, then slot 1, etc.
    result = []
    for slot in zip_longest(*per_provider):
        result.extend(e for e in slot if e is not None)
    return result


# ── Generic sub-agent query runner ────────────────────────────────────────────
NextClient = Callable[[], tuple[OpenAI, str, str]]


def run_mcp_query(
    domain: str,
    start_docker_fn,
    next_client: NextClient,
    system_prompt: str,
    query: str,
) -> str:
    """Spawn a sub-agent Docker container, run one query through its MCP tools,
    return the final text answer. next_client() is called before each LLM call
    so the rotation cycle is shared with the orchestrator."""
    proc = start_docker_fn()

    def _fwd_stderr() -> None:
        if not proc.stderr:
            return
        for line in proc.stderr:
            line = line.strip()
            if line:
                print(f"  [{domain}][docker] {line}", file=sys.stderr)

    threading.Thread(target=_fwd_stderr, daemon=True).start()

    # MCPClient is identical across all three modules — reuse the proxmox copy
    mcp = _proxmox.MCPClient(proc)
    try:
        mcp.initialize()
        tools = mcp.list_tools()
        openai_tools = _proxmox.tools_to_openai(tools)
        messages: list[dict] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]

        while True:
            llm, model, _ = next_client()
            response = llm.chat.completions.create(
                model=model,
                messages=messages,
                tools=openai_tools,
                tool_choice="auto",
            )
            msg = response.choices[0].message
            messages.append(_proxmox.assistant_msg(msg))

            if not msg.tool_calls:
                return msg.content or "(no response)"

            for tc in msg.tool_calls:
                fn = tc.function.name
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}

                preview = json.dumps(args)
                if len(preview) > 80:
                    preview = preview[:77] + "…"
                print(f"  [{domain}] → {fn}({preview})")

                try:
                    result = mcp.call_tool(fn, args)
                    result_text = _proxmox.mcp_result_to_text(result)
                except Exception as exc:
                    result_text = f"Tool error: {exc}"

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result_text,
                })
    finally:
        mcp.close()


# ── Sub-agent handlers ────────────────────────────────────────────────────────
def ask_proxmox(query: str, next_client: NextClient) -> str:
    host = os.getenv("PROXMOX_MCP_HOST", "proxmox")
    return run_mcp_query(
        "proxmox",
        _proxmox.start_docker,
        next_client,
        (
            "You are a Proxmox VE administrator assistant. Use the available tools to manage, "
            f"monitor, and operate the Proxmox cluster at {host}. "
            "Be concise and precise. Confirm destructive or irreversible operations before executing."
        ),
        query,
    )


def ask_synology(query: str, next_client: NextClient) -> str:
    return run_mcp_query(
        "synology",
        _synology.start_docker,
        next_client,
        (
            "You are a Synology NAS assistant. Use the available tools to manage, monitor, "
            "and configure the Synology NAS device(s). "
            "Be concise and precise. Confirm destructive operations before executing."
        ),
        query,
    )


def ask_linux(query: str, next_client: NextClient) -> str:
    servers = _linux.list_configured_servers()
    server_list = ", ".join(s.split("(")[0].strip() for s in servers) or "none configured"
    return run_mcp_query(
        "linux",
        _linux.start_docker,
        next_client,
        (
            "You are a Linux systems administrator assistant with SSH access to a fleet of servers. "
            f"Configured servers: {server_list}. "
            "Use the available tools to manage, monitor, and troubleshoot the Linux servers. "
            "Confirm destructive operations before executing. "
            "Prefer parallel execution when running commands on multiple servers."
        ),
        query,
    )

# ── Orchestrator tool definitions ─────────────────────────────────────────────
ORCHESTRATOR_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "ask_proxmox",
            "description": (
                "Delegate a task to the Proxmox VE agent. Use for anything related to "
                "virtual machines, LXC containers, Proxmox nodes, cluster management, "
                "backups, snapshots, storage pools, firewall rules, HA, Ceph, or any "
                "Proxmox API operation."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The full task or question for the Proxmox agent.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ask_synology",
            "description": (
                "Delegate a task to the Synology NAS agent. Use for anything related to "
                "Synology DSM, shared folders, file management, Docker on Synology, "
                "scheduled tasks, packages, RAID/volume status, or NAS configuration."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The full task or question for the Synology NAS agent.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ask_linux",
            "description": (
                "Delegate a task to the Linux SSH fleet agent. Use for anything that "
                "requires SSH access to Linux servers: running shell commands, checking "
                "services, reading logs, managing processes, installing packages, or "
                "monitoring system resources on any of the configured Linux hosts."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The full task or question for the Linux SSH fleet agent.",
                    }
                },
                "required": ["query"],
            },
        },
    },
]


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    provider = _e("PROVIDER", "")
    verbose = _e("VERBOSE", "true").lower() in ("true", "1", "yes")

    rotating = provider.lower() in ("", "rotate", "auto")

    if rotating:
        active_pool: list[tuple[OpenAI, str, str]] = build_pool()
        if not active_pool:
            print("[error] Nessun provider disponibile per la rotazione (API key mancanti).")
            sys.exit(1)
        _cycle = itertools.cycle(active_pool)
        llm, model, cur_provider = next(_cycle)
    else:
        try:
            llm, model = build_client(provider)
            cur_provider = provider
        except ValueError as exc:
            print(f"[error] {exc}")
            sys.exit(1)
        active_pool = []
        _cycle = None

    def next_client() -> tuple[OpenAI, str, str]:
        nonlocal llm, model, cur_provider
        if rotating and _cycle:
            llm, model, cur_provider = next(_cycle)
        if verbose:
            print(f"  [llm: {cur_provider} / {model}]")
        return llm, model, cur_provider

    def _drop_provider(reason: str) -> bool:
        """Remove cur_provider from active_pool and rebuild cycle (no advance).
        Returns True → caller should continue; False → pool exhausted."""
        nonlocal _cycle
        if not (rotating and active_pool):
            return False
        active_pool[:] = [e for e in active_pool if e[2] != cur_provider]
        if not active_pool:
            print(f"\n[error] Tutti i provider sono stati rimossi ({reason}). Sessione terminata.\n")
            return False
        _cycle = itertools.cycle(active_pool)
        remaining = ", ".join(p for _, _, p in active_pool)
        print(f"  [{reason}] Provider '{cur_provider}' rimosso dalla rotazione. Attivi: {remaining}")
        return True

    dispatch = {
        "ask_proxmox": ask_proxmox,
        "ask_synology": ask_synology,
        "ask_linux": ask_linux,
    }

    linux_servers = _linux.list_configured_servers()
    if rotating:
        names = ", ".join(label for _, _, label in active_pool)
        print(f"[*] Multi-MCP Orchestrator — modalità rotazione: {names}")
    else:
        print(f"[*] Multi-MCP Orchestrator — provider: {cur_provider} — model: {model}")
    print(f"[*] Proxmox host : {os.getenv('PROXMOX_MCP_HOST', '(not configured)')}")
    print(f"[*] Synology NAS : {os.getenv('SYNOLOGY_MCP_NAS_CONFIG', '(not configured)')}")
    print(f"[*] Linux servers: {len(linux_servers)} configured")
    for s in linux_servers:
        print(f"    • {s}")

    system_prompt = (
        "You are an infrastructure orchestrator with access to three specialized agents:\n"
        "• ask_proxmox  — manages the Proxmox VE cluster (VMs, containers, nodes, "
        "storage, backups, snapshots, firewall, HA, Ceph)\n"
        "• ask_synology — manages Synology NAS devices (files, shares, Docker, packages, "
        "RAID/volume status, DSM configuration)\n"
        "• ask_linux    — manages Linux servers via SSH (shell commands, services, logs, "
        "processes, packages, system monitoring)\n\n"
        "Analyze each user request and delegate to the appropriate agent(s). "
        "You may call multiple agents in sequence when a task spans domains. "
        "Present the agents' responses clearly and concisely."
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

        # Orchestrator tool-call loop
        while True:
            llm, model, cur_provider = next_client()

            try:
                response = llm.chat.completions.create(
                    model=model,
                    messages=messages,
                    tools=ORCHESTRATOR_TOOLS,
                    tool_choice="auto",
                )
            except RateLimitError:
                if _drop_provider("429"):
                    continue
                print(f"\n[error] Provider '{cur_provider}' ha restituito 429. Riprova più tardi.\n")
                break
            except BadRequestError as exc:
                body = getattr(exc, "body", {}) or {}
                if isinstance(body, dict) and body.get("error", {}).get("code") == "tool_use_failed":
                    if rotating and _cycle:
                        print(f"  [tool_use_failed] Provider '{cur_provider}' ha fallito, riprovo con il prossimo...")
                        continue
                    print(f"\n[error] Provider '{cur_provider}' ha fallito la generazione del tool call.\n")
                else:
                    print(f"\n[error] {cur_provider} bad request: {exc}\n")
                break

            msg = response.choices[0].message
            messages.append(_proxmox.assistant_msg(msg))

            if not msg.tool_calls:
                print(f"\nAssistant: {msg.content}\n")
                break

            for tc in msg.tool_calls:
                fn = tc.function.name
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}

                query = args.get("query", "")
                short = query[:80] + ("…" if len(query) > 80 else "")
                print(f"  → [{fn}] {short}")

                handler = dispatch.get(fn)
                if handler:
                    try:
                        result_text = handler(query, next_client)
                    except Exception as exc:
                        result_text = f"Sub-agent error: {exc}"
                else:
                    result_text = f"Unknown tool: {fn}"

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result_text,
                })


if __name__ == "__main__":
    main()
