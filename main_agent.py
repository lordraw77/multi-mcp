#!/usr/bin/env python3
"""
Multi-MCP Orchestrator
Routes requests to Proxmox, Synology NAS, or Linux SSH sub-agents.
Provider: openrouter | groq | gemini | cloudflare | cerebras | mistral | nvidia | ollama | puter
"""

import os
import sys
import json
import itertools
from itertools import zip_longest
import threading
from pathlib import Path
from typing import Any, Callable, Optional

from openai import BadRequestError, RateLimitError, APIStatusError

from dotenv import load_dotenv
from openai import OpenAI

from nvidia_ratelimit import wrap_if_nvidia

# ── Sub-agent modules (reuse Docker launchers, MCPClient, helpers) ─────────────
sys.path.insert(0, str(Path(__file__).parent))
import proxmox_mcp_agent as _proxmox               # P = "PROXMOX_MCP"
import synology_mcp_agent as _synology              # P = "SYNOLOGY_MCP"
import linux_mcp_agent as _linux                    # P = "UXMCP"
import homeassistant_mcp_agent as _homeassistant    # P = "HAOS_MCP"
import watchyourlan_mcp_agent as _watchyourlan      # P = "WYLA_MCP"

# ── Env ───────────────────────────────────────────────────────────────────────
load_dotenv(Path(__file__).parent / ".env", override=True)
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
    "nvidia": {
        "base_url": "https://integrate.api.nvidia.com/v1",
        "api_key_var": f"{P}_NVIDIA_API_KEY",
        "model_var": f"{P}_NVIDIA_MODEL",
        "default_model": "meta/llama-3.3-70b-instruct",
    },
    "puter": {
        "base_url": "https://api.puter.com/v1",
        "api_key_var": f"{P}_PUTER_API_KEY",
        "model_var": f"{P}_PUTER_MODEL",
        "default_model": "gpt-4o-mini",
    },
    "ollama": {
        "base_url": None,
        "api_key_var": None,
        "model_var": f"{P}_OLLAMA_MODEL",
        "default_model": "llama3.2:1b",
        "chat_model_var": f"{P}_OLLAMA_CHAT_MODEL",
    },
}


# Providers eligible for rotation (ollama escluso: non ha API key cloud)
ROTATE_PROVIDERS = ["openrouter", "groq", "gemini", "cloudflare", "cerebras", "mistral", "nvidia", "puter"]


def _parse_rotate_slots() -> Optional[set[int]]:
    """Parse MAIN_AGENT_ROTATE_KEYS into a set of 1-based slot indices, or None for all.
    E.g. "1" → {1}, "1,3" → {1, 3}, "" → None (use all)."""
    raw = _e("ROTATE_KEYS", "").strip()
    if "#" in raw:
        raw = raw[:raw.index("#")].strip()
    if not raw:
        return None
    slots = {int(p.strip()) for p in raw.split(",") if p.strip().isdigit()}
    return slots if slots else None


def get_provider_keys(provider: str, slots: Optional[set[int]] = None) -> list[tuple[int, str]]:
    """Return (slot, key) pairs for a provider (primary + _2, _3, ...).
    Slot is 1-based: 1 = primary key, 2 = _2, etc.
    If slots is given, only those positions are returned; missing slots are skipped."""
    cfg = PROVIDERS.get(provider, {})
    key_var = cfg.get("api_key_var")
    if not key_var:
        return []
    all_keys: dict[int, str] = {}
    primary = os.getenv(key_var, "")
    if primary:
        all_keys[1] = primary
    n = 2
    while True:
        val = os.getenv(f"{key_var}_{n}", "")
        if not val:
            break
        all_keys[n] = val
        n += 1
    if slots is None:
        return list(all_keys.items())
    return [(s, all_keys[s]) for s in sorted(slots) if s in all_keys]


def get_cloudflare_entries(slots: Optional[set[int]] = None) -> list[tuple[int, str, str]]:
    """Return (slot, api_key, account_id) triples for Cloudflare (primary + _2, _3, ...).
    If slots is given (1-based indices), only those positions are returned; missing slots are skipped.
    Entries missing either key or account_id are skipped with a warning."""
    all_pairs: dict[int, tuple[str, str]] = {}
    key0 = os.getenv(f"{P}_CLOUDFLARE_API_KEY", "")
    acct0 = os.getenv(f"{P}_CLOUDFLARE_ACCOUNT_ID", "")
    if key0 or acct0:
        all_pairs[1] = (key0, acct0)
    n = 2
    while True:
        key = os.getenv(f"{P}_CLOUDFLARE_API_KEY_{n}", "")
        acct = os.getenv(f"{P}_CLOUDFLARE_ACCOUNT_ID_{n}", "")
        if not key and not acct:
            break
        all_pairs[n] = (key, acct)
        n += 1
    selected = all_pairs if slots is None else {s: all_pairs[s] for s in sorted(slots) if s in all_pairs}
    entries = []
    for slot, (key, acct) in selected.items():
        label = "cloudflare" if slot == 1 else f"cloudflare#{slot}"
        if key and acct:
            entries.append((slot, key, acct))
        elif key or acct:
            print(f"[warn] {label}: richiede sia API_KEY che ACCOUNT_ID — ignorato")
    return entries


def available_providers(slots: Optional[set[int]] = None) -> list[str]:
    """Return providers from ROTATE_PROVIDERS that have at least one usable entry."""
    result = []
    for name in ROTATE_PROVIDERS:
        if name == "cloudflare":
            if get_cloudflare_entries(slots):
                result.append(name)
        elif get_provider_keys(name, slots):
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
    client = OpenAI(base_url=base_url, api_key=resolved_key)
    return wrap_if_nvidia(provider, client, resolved_key), model


def _provider_entries(provider: str, slots: Optional[set[int]] = None) -> list[tuple[OpenAI, str, str]]:
    """Build all (client, model, label) entries for a single provider across its key slots.
    Cloudflare: each (api_key, account_id) pair is one entry.
    Respects the given slot filter (from MAIN_AGENT_ROTATE_KEYS)."""
    entries: list[tuple[OpenAI, str, str]] = []
    if provider == "cloudflare":
        for slot, cf_key, cf_acct in get_cloudflare_entries(slots):
            label = "cloudflare" if slot == 1 else f"cloudflare#{slot}"
            try:
                client, model = build_client(provider, api_key=cf_key, account_id=cf_acct)
                entries.append((client, model, label))
            except ValueError as exc:
                print(f"[warn] {label} ignorato: {exc}")
    else:
        for slot, key in get_provider_keys(provider, slots):
            label = provider if slot == 1 else f"{provider}#{slot}"
            try:
                client, model = build_client(provider, api_key=key)
                entries.append((client, model, label))
            except ValueError as exc:
                print(f"[warn] {label} ignorato: {exc}")
    return entries


def build_pool() -> list[tuple[OpenAI, str, str]]:
    """Build rotation pool interleaved across providers (round-robin by key index).
    Result order: openrouter, groq, gemini, ..., openrouter#2, groq#2, ...
    Cloudflare: each (api_key, account_id) pair is one entry.
    Respects MAIN_AGENT_ROTATE_KEYS to restrict which key slots are included."""
    slots = _parse_rotate_slots()
    per_provider: list[list[tuple[OpenAI, str, str]]] = []
    for p in available_providers(slots):
        entries = _provider_entries(p, slots)
        if entries:
            per_provider.append(entries)
    # Round-robin: slot 0 of each provider, then slot 1, etc.
    result = []
    for slot in zip_longest(*per_provider):
        result.extend(e for e in slot if e is not None)
    return result


def build_single_pool(provider: str) -> list[tuple[OpenAI, str, str]]:
    """Build a pool restricted to one named provider (all its usable key slots).
    Used when an agent pins a specific provider: failover happens only between that
    provider's own keys — never across different providers. No rotation if it has one key.
    Respects MAIN_AGENT_ROTATE_KEYS to restrict which key slots are included."""
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown provider '{provider}'. Valid: {', '.join(PROVIDERS)}")
    return _provider_entries(provider, _parse_rotate_slots())


# ── Generic sub-agent query runner ────────────────────────────────────────────
NextClient = Callable[[], tuple[OpenAI, str, str]]
DropProvider = Callable[[str], bool]

_FATAL_CODES = {"organization_restricted", "account_suspended", "access_denied"}
_FATAL_PHRASES = ("organization has been restricted", "account suspended", "access denied")
_CTX_TOO_LARGE_PHRASES = (
    "request too large", "reduce your message size", "maximum context length",
    "prompt is too long", "context_length_exceeded", "context length exceeded",
    "tokens per minute", "input is too long",
)

def _is_context_too_large(exc: Exception) -> bool:
    msg = str(exc).lower()
    status = getattr(exc, "status_code", 0)
    return status == 413 or any(p in msg for p in _CTX_TOO_LARGE_PHRASES)


def _is_fatal_error(exc: BadRequestError) -> tuple[bool, str]:
    body = getattr(exc, "body", {}) or {}
    err = body.get("error", {}) if isinstance(body, dict) else {}
    code = err.get("code", "") or ""
    msg_text = str(err.get("message", "")).lower()
    fatal = code in _FATAL_CODES or any(p in msg_text for p in _FATAL_PHRASES)
    return fatal, code


def run_mcp_query(
    domain: str,
    start_docker_fn,
    next_client: NextClient,
    system_prompt: str,
    query: str,
    drop_provider: Optional[DropProvider] = None,
    max_tokens: Optional[int] = None,
    max_tool_result_chars: Optional[int] = None,
) -> str:
    """Spawn a sub-agent process, run one query through its MCP tools,
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
            try:
                kw: dict = dict(model=model, messages=messages, tools=openai_tools, tool_choice="auto")
                if max_tokens:
                    kw["max_tokens"] = max_tokens
                response = llm.chat.completions.create(**kw)
            except RateLimitError:
                if drop_provider and drop_provider("429"):
                    continue
                raise
            except BadRequestError as exc:
                fatal, code = _is_fatal_error(exc)
                if fatal and drop_provider and drop_provider(code or "restricted"):
                    continue
                raise
            except APIStatusError as exc:
                if _is_context_too_large(exc) and drop_provider and drop_provider("ctx-too-large"):
                    continue
                raise

            msg = response.choices[0].message
            messages.append(_proxmox.assistant_msg(msg))

            if not msg.tool_calls:
                return msg.content or "(no response)"

            for tc in msg.tool_calls:
                fn = tc.function.name
                raw_args = tc.function.arguments or "{}"
                try:
                    args = json.loads(raw_args)
                except json.JSONDecodeError:
                    try:
                        args = json.loads(raw_args.rstrip() + "}")
                    except json.JSONDecodeError:
                        args = {}
                if not isinstance(args, dict):
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

                if max_tool_result_chars and len(result_text) > max_tool_result_chars:
                    result_text = result_text[:max_tool_result_chars] + f"\n… [truncated to {max_tool_result_chars} chars]"

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result_text,
                })
    finally:
        mcp.close()


# ── Sub-agent handlers ────────────────────────────────────────────────────────
def ask_proxmox(query: str, next_client: NextClient, drop_provider: Optional[DropProvider] = None, max_tokens: Optional[int] = None, max_tool_result_chars: Optional[int] = None) -> str:
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
        drop_provider=drop_provider,
        max_tokens=max_tokens,
        max_tool_result_chars=max_tool_result_chars,
    )


def ask_synology(query: str, next_client: NextClient, drop_provider: Optional[DropProvider] = None, max_tokens: Optional[int] = None, max_tool_result_chars: Optional[int] = None) -> str:
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
        drop_provider=drop_provider,
        max_tokens=max_tokens,
        max_tool_result_chars=max_tool_result_chars,
    )


def ask_linux(query: str, next_client: NextClient, drop_provider: Optional[DropProvider] = None, max_tokens: Optional[int] = None, max_tool_result_chars: Optional[int] = None) -> str:
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
        drop_provider=drop_provider,
        max_tokens=max_tokens,
        max_tool_result_chars=max_tool_result_chars,
    )


def ask_homeassistant(query: str, next_client: NextClient, drop_provider: Optional[DropProvider] = None, max_tokens: Optional[int] = None, max_tool_result_chars: Optional[int] = None) -> str:
    ha_url = os.getenv("HAOS_MCP_URL", "(not configured)")
    return run_mcp_query(
        "homeassistant",
        _homeassistant.start_proxy,
        next_client,
        (
            "You are a Home Assistant smart home assistant. Use the available tools to control, "
            f"monitor, and automate the home at {ha_url}. "
            "You can manage lights, switches, climate, covers, media players, and other entities. "
            "Be concise and precise. Confirm irreversible or bulk automations before executing."
        ),
        query,
        drop_provider=drop_provider,
        max_tokens=max_tokens,
        max_tool_result_chars=max_tool_result_chars,
    )


def ask_watchyourlan(query: str, next_client: NextClient, drop_provider: Optional[DropProvider] = None, max_tokens: Optional[int] = None, max_tool_result_chars: Optional[int] = None) -> str:
    wyl_url = os.getenv("WYLA_MCP_URL", "(not configured)")
    return run_mcp_query(
        "watchyourlan",
        _watchyourlan.start_docker,
        next_client,
        (
            "You are a network monitoring assistant with access to WatchYourLAN. "
            f"Use the available tools to monitor the network at {wyl_url}. "
            "You can list known devices, check online/offline status, view device history, "
            "and manage device names and groups. Be concise and precise."
        ),
        query,
        drop_provider=drop_provider,
        max_tokens=max_tokens,
        max_tool_result_chars=max_tool_result_chars,
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
    {
        "type": "function",
        "function": {
            "name": "ask_homeassistant",
            "description": (
                "Delegate a task to the Home Assistant smart home agent. Use for anything "
                "related to home automation: controlling lights, switches, thermostats, climate, "
                "covers/blinds, media players, sensors, alarms, scenes, scripts, or automations. "
                "Also use for querying the state of any smart home device or entity."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The full task or question for the Home Assistant agent.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ask_watchyourlan",
            "description": (
                "Delegate a task to the WatchYourLAN network monitoring agent. Use for anything "
                "related to network device discovery and monitoring: listing known devices, "
                "checking which devices are online or offline, viewing device history, "
                "identifying unknown devices on the network, or managing device names and groups."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The full task or question for the WatchYourLAN agent.",
                    }
                },
                "required": ["query"],
            },
        },
    },
]


# ── Reusable orchestrator (shared by the CLI REPL and the HTTP sidecar) ───────
SYSTEM_PROMPT = (
    "You are an infrastructure and smart home orchestrator with access to five specialized agents:\n"
    "• ask_proxmox       — manages the Proxmox VE cluster (VMs, containers, nodes, "
    "storage, backups, snapshots, firewall, HA, Ceph)\n"
    "• ask_synology      — manages Synology NAS devices (files, shares, Docker, packages, "
    "RAID/volume status, DSM configuration)\n"
    "• ask_linux         — manages Linux servers via SSH (shell commands, services, logs, "
    "processes, packages, system monitoring)\n"
    "• ask_homeassistant — controls Home Assistant smart home (lights, switches, climate, "
    "covers, media players, sensors, scenes, scripts, automations)\n"
    "• ask_watchyourlan  — monitors the network via WatchYourLAN (device discovery, "
    "online/offline status, device history, unknown device detection)\n\n"
    "Analyze each user request and delegate to the appropriate agent(s). "
    "You may call multiple agents in sequence when a task spans domains. "
    "Present the agents' responses clearly and concisely."
)


class Orchestrator:
    """Reusable Multi-MCP orchestrator.

    Build once, then call :meth:`run_turn` per request. Holds the provider
    configuration (a rotation pool or a single pinned provider) read from the
    process ``.env`` (``MAIN_AGENT_*``) — exactly the same configuration the CLI
    uses. The HTTP sidecar (``agent_server.py``) drives one shared instance so
    that the web console and Telegram reach the orchestrator through the gateway.

    Thread-safe: :meth:`run_turn` is serialized with a lock because sub-agents
    spawn Docker containers and the rotation pool is mutated mid-turn.

    NOTE: :meth:`run_turn` mirrors the tool-call loop in :func:`main`; the REPL
    is intentionally left untouched so standalone usage is byte-for-byte the same.
    """

    def __init__(self) -> None:
        self.provider = _e("PROVIDER", "")
        self.verbose = _e("VERBOSE", "true").lower() in ("true", "1", "yes")
        self.rotating = self.provider.lower() in ("", "rotate", "auto")
        self.system_prompt = SYSTEM_PROMPT
        self._lock = threading.Lock()

        # Ollama dual-model: separate client for final chat synthesis
        self.chat_llm: Optional[OpenAI] = None
        self.chat_model_name: str = ""
        if self.provider.lower() == "ollama":
            chat_model_var = _e("OLLAMA_CHAT_MODEL", "")
            if chat_model_var:
                self.chat_model_name = chat_model_var
                ollama_host = _e("OLLAMA_HOST", "http://localhost:11434")
                if not ollama_host.rstrip("/").endswith("/v1"):
                    ollama_host = ollama_host.rstrip("/") + "/v1"
                self.chat_llm = OpenAI(base_url=ollama_host, api_key="ollama")

        # Non-rotating: a single fixed client (raises ValueError if misconfigured)
        self.tool_model: str = ""
        self._fixed_client: Optional[OpenAI] = None
        if not self.rotating:
            self._fixed_client, self.tool_model = build_client(self.provider)

        self.dispatch = {
            "ask_proxmox": ask_proxmox,
            "ask_synology": ask_synology,
            "ask_linux": ask_linux,
            "ask_homeassistant": ask_homeassistant,
            "ask_watchyourlan": ask_watchyourlan,
        }

        # Per-turn rotation state (reset at the start of every run_turn)
        self._pool: list[tuple[OpenAI, str, str]] = []
        self._cycle = None
        self._llm: Optional[OpenAI] = None
        self._model: str = ""
        self._cur_provider: str = ""

    # ── rotation helpers ──────────────────────────────────────────────────────
    def pool_labels(self) -> list[str]:
        """Provider labels currently eligible for rotation (for banners)."""
        return [label for _, _, label in build_pool()] if self.rotating else [self.provider]

    def _reset_rotation(self) -> None:
        """Rebuild the rotation pool so each turn starts from a full, healthy set
        (a transient 429 in one turn never permanently kills a provider)."""
        if self.rotating:
            self._pool = build_pool()
            if not self._pool:
                raise RuntimeError("Nessun provider disponibile per la rotazione (API key mancanti).")
            self._cycle = itertools.cycle(self._pool)
            self._llm, self._model, self._cur_provider = next(self._cycle)
        else:
            self._pool = []
            self._cycle = None
            self._llm, self._model, self._cur_provider = self._fixed_client, self.tool_model, self.provider

    def _next_client(self) -> tuple[OpenAI, str, str]:
        if self.rotating and self._cycle:
            self._llm, self._model, self._cur_provider = next(self._cycle)
        if self.verbose:
            print(f"  [llm: {self._cur_provider} / {self._model}]")
        return self._llm, self._model, self._cur_provider

    def _drop_provider(self, reason: str) -> bool:
        """Remove the current provider from this turn's pool and rebuild the cycle.
        Returns True → caller should retry; False → pool exhausted."""
        if not (self.rotating and self._pool):
            return False
        self._pool[:] = [e for e in self._pool if e[2] != self._cur_provider]
        if not self._pool:
            print(f"\n[error] Tutti i provider sono stati rimossi ({reason}). Turno terminato.\n")
            return False
        self._cycle = itertools.cycle(self._pool)
        remaining = ", ".join(p for _, _, p in self._pool)
        print(f"  [{reason}] Provider '{self._cur_provider}' rimosso dalla rotazione. Attivi: {remaining}")
        return True

    # ── one orchestrator turn ─────────────────────────────────────────────────
    def run_turn(self, messages: list[dict], on_step: Optional[Callable[[dict], None]] = None) -> str:
        """Run one orchestrator turn over ``messages`` (appended in place with the
        assistant/tool exchange). Returns the final assistant text.

        ``on_step`` — optional callback receiving ``{"type": "tool_call"|"tool_result", ...}``
        as each sub-agent is delegated, for progress streaming.
        """
        with self._lock:
            self._reset_rotation()

            while True:
                llm, model, cur_provider = self._next_client()

                try:
                    response = llm.chat.completions.create(
                        model=model,
                        messages=messages,
                        tools=ORCHESTRATOR_TOOLS,
                        tool_choice="auto",
                    )
                except RateLimitError:
                    if self._drop_provider("429"):
                        continue
                    return f"[error] Provider '{cur_provider}' ha restituito 429. Riprova più tardi."
                except BadRequestError as exc:
                    fatal, code = _is_fatal_error(exc)
                    if fatal:
                        if self._drop_provider(code or "restricted"):
                            continue
                        return f"[error] Provider '{cur_provider}' è bloccato ({code})."
                    if code == "tool_use_failed" and self.rotating and self._cycle:
                        print(f"  [tool_use_failed] Provider '{cur_provider}' ha fallito, riprovo con il prossimo...")
                        continue
                    return f"[error] {cur_provider} bad request: {exc}"
                except APIStatusError as exc:
                    if _is_context_too_large(exc):
                        if self._drop_provider("ctx-too-large"):
                            continue
                        return f"[error] Provider '{cur_provider}' context too large e nessun altro disponibile."
                    return f"[error] {cur_provider} API error {getattr(exc, 'status_code', '?')}: {exc}"

                msg = response.choices[0].message
                messages.append(_proxmox.assistant_msg(msg))

                if not msg.tool_calls:
                    if self.chat_llm:
                        if self.verbose:
                            print(f"  [llm: ollama / {self.chat_model_name}]")
                        final_resp = self.chat_llm.chat.completions.create(
                            model=self.chat_model_name,
                            messages=messages,
                        )
                        return final_resp.choices[0].message.content or "(no response)"
                    return msg.content or "(no response)"

                for tc in msg.tool_calls:
                    fn = tc.function.name
                    raw_args = tc.function.arguments or "{}"
                    try:
                        args = json.loads(raw_args)
                    except json.JSONDecodeError:
                        try:
                            args = json.loads(raw_args.rstrip() + "}")
                        except json.JSONDecodeError:
                            args = {}
                    if not isinstance(args, dict):
                        args = {}

                    query = args.get("query", "")
                    short = query[:80] + ("…" if len(query) > 80 else "")
                    print(f"  → [{fn}] {short}")
                    if on_step:
                        on_step({"type": "tool_call", "id": tc.id, "name": fn, "query": query})

                    handler = self.dispatch.get(fn)
                    if handler:
                        try:
                            result_text = handler(query, self._next_client, self._drop_provider)
                        except Exception as exc:
                            print(f"  [{fn}] errore: {exc}", file=sys.stderr)
                            result_text = f"Sub-agent error: {exc}"
                    else:
                        result_text = f"Unknown tool: {fn}"

                    if on_step:
                        on_step({"type": "tool_result", "id": tc.id, "name": fn, "result": result_text})

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result_text,
                    })


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    provider = _e("PROVIDER", "")
    verbose = _e("VERBOSE", "true").lower() in ("true", "1", "yes")

    rotating = provider.lower() in ("", "rotate", "auto")

    # Ollama dual-model: separate client for final chat synthesis
    chat_llm: Optional[OpenAI] = None
    chat_model_name: str = ""
    if provider.lower() == "ollama":
        chat_model_var = _e("OLLAMA_CHAT_MODEL", "")
        if chat_model_var:
            chat_model_name = chat_model_var
            ollama_host = _e("OLLAMA_HOST", "http://localhost:11434")
            if not ollama_host.rstrip("/").endswith("/v1"):
                ollama_host = ollama_host.rstrip("/") + "/v1"
            chat_llm = OpenAI(base_url=ollama_host, api_key="ollama")

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
        "ask_homeassistant": ask_homeassistant,
        "ask_watchyourlan": ask_watchyourlan,
    }

    linux_servers = _linux.list_configured_servers()
    if rotating:
        names = ", ".join(label for _, _, label in active_pool)
        print(f"[*] Multi-MCP Orchestrator — modalità rotazione: {names}")
    else:
        if chat_llm:
            print(f"[*] Multi-MCP Orchestrator — provider: {cur_provider} — tool model: {model} — chat model: {chat_model_name}")
        else:
            print(f"[*] Multi-MCP Orchestrator — provider: {cur_provider} — model: {model}")
    print(f"[*] Proxmox host      : {os.getenv('PROXMOX_MCP_HOST', '(not configured)')}")
    print(f"[*] Synology NAS      : {os.getenv('SYNOLOGY_MCP_NAS_CONFIG', '(not configured)')}")
    print(f"[*] Linux servers     : {len(linux_servers)} configured")
    for s in linux_servers:
        print(f"    • {s}")
    print(f"[*] Home Assistant    : {os.getenv('HAOS_MCP_URL', '(not configured)')}")
    print(f"[*] WatchYourLAN      : {os.getenv('WYLA_MCP_URL', '(not configured)')}")

    system_prompt = (
        "You are an infrastructure and smart home orchestrator with access to five specialized agents:\n"
        "• ask_proxmox       — manages the Proxmox VE cluster (VMs, containers, nodes, "
        "storage, backups, snapshots, firewall, HA, Ceph)\n"
        "• ask_synology      — manages Synology NAS devices (files, shares, Docker, packages, "
        "RAID/volume status, DSM configuration)\n"
        "• ask_linux         — manages Linux servers via SSH (shell commands, services, logs, "
        "processes, packages, system monitoring)\n"
        "• ask_homeassistant — controls Home Assistant smart home (lights, switches, climate, "
        "covers, media players, sensors, scenes, scripts, automations)\n"
        "• ask_watchyourlan  — monitors the network via WatchYourLAN (device discovery, "
        "online/offline status, device history, unknown device detection)\n\n"
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
                fatal, code = _is_fatal_error(exc)
                if fatal:
                    if _drop_provider(code or "restricted"):
                        continue
                    print(f"\n[error] Provider '{cur_provider}' è bloccato ({code}). Sessione terminata.\n")
                elif code == "tool_use_failed":
                    if rotating and _cycle:
                        print(f"  [tool_use_failed] Provider '{cur_provider}' ha fallito, riprovo con il prossimo...")
                        continue
                    print(f"\n[error] Provider '{cur_provider}' ha fallito la generazione del tool call.\n")
                else:
                    print(f"\n[error] {cur_provider} bad request: {exc}\n")
                break
            except APIStatusError as exc:
                if _is_context_too_large(exc):
                    if _drop_provider("ctx-too-large"):
                        continue
                    print(f"\n[error] Provider '{cur_provider}' context too large e nessun altro disponibile.\n")
                else:
                    print(f"\n[error] {cur_provider} API error {getattr(exc, 'status_code', '?')}: {exc}\n")
                break

            msg = response.choices[0].message
            messages.append(_proxmox.assistant_msg(msg))

            if not msg.tool_calls:
                if chat_llm:
                    if verbose:
                        print(f"  [llm: ollama / {chat_model_name}]")
                    final_resp = chat_llm.chat.completions.create(
                        model=chat_model_name,
                        messages=messages,
                    )
                    print(f"\nAssistant: {final_resp.choices[0].message.content}\n")
                else:
                    print(f"\nAssistant: {msg.content}\n")
                break

            for tc in msg.tool_calls:
                fn = tc.function.name
                raw_args = tc.function.arguments or "{}"
                try:
                    args = json.loads(raw_args)
                except json.JSONDecodeError:
                    try:
                        args = json.loads(raw_args.rstrip() + "}")
                    except json.JSONDecodeError:
                        args = {}
                if not isinstance(args, dict):
                    args = {}

                query = args.get("query", "")
                short = query[:80] + ("…" if len(query) > 80 else "")
                print(f"  → [{fn}] {short}")

                handler = dispatch.get(fn)
                if handler:
                    try:
                        result_text = handler(query, next_client, _drop_provider)
                    except Exception as exc:
                        print(f"  [{fn}] errore: {exc}", file=sys.stderr)
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
