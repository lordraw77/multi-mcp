#!/usr/bin/env python3
"""
Lab Health Agent — ReAct-style periodic health checker for the home lab.

Runs a full status check at startup and every LAB_HEALTH_INTERVAL_HOURS hours.
Collects data from all sub-agents (Proxmox, Synology, Linux, Home Assistant,
WatchYourLAN), synthesises a structured report, and sends it to Telegram.
This agent is READ-ONLY: it reports what should be fixed but never acts.
"""

import itertools
import os
import sys
import time
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import requests


def _fmt_duration(seconds: float) -> str:
    """Return a human-readable duration: seconds → minutes → hours → days."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = seconds / 60
    if minutes < 60:
        return f"{minutes:.1f}m"
    hours = minutes / 60
    if hours < 24:
        return f"{hours:.1f}h"
    return f"{hours / 24:.1f}d"


def _parse_schedule_times(raw: str) -> list[tuple[int, int]]:
    """Parse a comma-separated 'HH:MM' list into sorted (hour, minute) tuples."""
    times: list[tuple[int, int]] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            h, m = part.split(":")
            times.append((int(h), int(m)))
        except ValueError:
            print(f"[warn] Ignoring invalid schedule time: {part!r}")
    return sorted(times)


def _seconds_until_next_schedule(times: list[tuple[int, int]]) -> float:
    """Seconds from now until the next of the given daily (hour, minute) times."""
    now = datetime.now()
    candidates = []
    for h, m in times:
        candidate = now.replace(hour=h, minute=m, second=0, microsecond=0)
        if candidate <= now:
            candidate += timedelta(days=1)
        candidates.append(candidate)
    return (min(candidates) - now).total_seconds()


from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))
import main_agent as _main

load_dotenv(Path(__file__).parent / ".env", override=True)

# ── Configuration ─────────────────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_ALLOWED_USERS = [
    int(u.strip())
    for u in os.getenv("TELEGRAM_ALLOWED_USERS", "").split(",")
    if u.strip().isdigit()
]
CHECK_INTERVAL_HOURS = int(os.getenv("LAB_HEALTH_INTERVAL_HOURS", "12"))
# Comma-separated "HH:MM" times (local container time). When set (typically
# from docker-compose), the agent runs ONLY at these times instead of at
# startup + every CHECK_INTERVAL_HOURS.
SCHEDULE_TIMES_RAW = os.getenv("LAB_HEALTH_SCHEDULE_TIMES", "").strip()
# Unset (or non-numeric) → None → no limit
_max_tokens_raw = os.getenv("LAB_HEALTH_MAX_TOKENS", "").strip()
MAX_TOKENS = int(_max_tokens_raw) if _max_tokens_raw.isdigit() else None
_max_tool_raw = os.getenv("LAB_HEALTH_MAX_TOOL_RESULT_CHARS", "").strip()
MAX_TOOL_RESULT_CHARS = int(_max_tool_raw) if _max_tool_raw.isdigit() else None

# ── Telegram helpers ──────────────────────────────────────────────────────────
def _tg_send_raw(chat_id: int, text: str, parse_mode: Optional[str] = "Markdown") -> bool:
    payload: dict = {"chat_id": chat_id, "text": text}
    if parse_mode:
        payload["parse_mode"] = parse_mode
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            json=payload,
            timeout=15,
        )
        return r.ok
    except Exception as exc:
        print(f"[telegram] send error: {exc}", file=sys.stderr)
        return False


def tg_send(chat_id: int, text: str) -> None:
    """Send a message to Telegram in blocks, splitting on paragraph boundaries."""
    max_len = 4000
    paragraphs = text.split("\n\n")
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        candidate = (current + "\n\n" + para).lstrip("\n") if current else para
        if len(candidate) <= max_len:
            current = candidate
        else:
            if current:
                chunks.append(current)
            # paragraph itself exceeds limit: split on newlines
            if len(para) > max_len:
                for line in para.split("\n"):
                    if len((current + "\n" + line).lstrip("\n")) <= max_len:
                        current = (current + "\n" + line).lstrip("\n")
                    else:
                        if current:
                            chunks.append(current)
                        current = line
            else:
                current = para
    if current:
        chunks.append(current)

    for i, chunk in enumerate(chunks, 1):
        ok = _tg_send_raw(chat_id, chunk, parse_mode="Markdown")
        if not ok:
            _tg_send_raw(chat_id, chunk, parse_mode=None)
        if len(chunks) > 1 and i < len(chunks):
            time.sleep(0.3)  # avoid Telegram flood limits between blocks


def tg_broadcast(text: str) -> None:
    for uid in TELEGRAM_ALLOWED_USERS:
        tg_send(uid, text)


# ── Health check prompts ──────────────────────────────────────────────────────
_HEALTH_QUERIES: dict[str, str] = {
    "proxmox": (
        "Read-only health check of the Proxmox cluster. Report concisely:\n"
        "1. Node status: CPU %, RAM %, uptime for each node.\n"
        "2. Any VMs or LXC containers in stopped/error state (exclude intentionally stopped).\n"
        "3. Storage pool usage and health — flag any pool above 80 % or degraded.\n"
        "4. Most recent backup job results — any failures or missed schedules.\n"
        "5. Any HA, cluster, or Ceph alerts.\n"
        "Do NOT modify anything. Just report facts and flag issues."
    ),
    "synology": (
        "Read-only health check of the Synology NAS. Report concisely:\n"
        "1. Volume usage — flag if any volume is above 85 %.\n"
        "2. RAID / SHR health and individual disk SMART status.\n"
        "3. Available DSM or package updates.\n"
        "4. Active alerts or errors in the notification center.\n"
        "Do NOT modify anything. Just report facts and flag issues."
    ),
    "linux": (
        "Read-only health check of all configured Linux servers. For each host:\n"
        "1. Disk partition usage — flag any partition above 85 %.\n"
        "2. Memory and swap usage.\n"
        "3. Load average relative to CPU count (flag if load > 2× CPU count).\n"
        "4. Failed systemd services (systemctl --failed).\n"
        "5. Critical errors or OOM kills in journalctl in the last 24 h.\n"
        "Do NOT run destructive commands. Just report facts and flag issues."
    ),
    "homeassistant": (
        "Read-only health check of Home Assistant. Report concisely:\n"
        "1. Entities in 'unavailable' or 'unknown' state (focus on critical: security, climate, "
        "network, door/window sensors).\n"
        "2. Recent automation or script errors in the logbook.\n"
        "3. Any integration warnings or errors.\n"
        "Do NOT control any devices. Just report facts and flag issues."
    ),
    "watchyourlan": (
        "Read-only network health check via WatchYourLAN. Report concisely:\n"
        "1. Unknown or unrecognised devices currently online.\n"
        "2. Known critical devices that appear to be offline.\n"
        "3. Total online vs total known device count.\n"
        "Do NOT modify any device records. Just report facts and flag issues."
    ),
}

_SYNTHESIS_SYSTEM = (
    "You are a senior infrastructure engineer reviewing automated health-check reports "
    "from a home lab. Produce a concise, structured status report.\n\n"
    "Format:\n"
    "🔴 *CRITICAL* — issues needing immediate action\n"
    "🟡 *WARNING* — issues to address soon\n"
    "🟢 *OK* — healthy systems\n\n"
    "For every 🔴 and 🟡 item include: WHAT the issue is, WHICH system/component, "
    "and HOW to fix it (specific command or action).\n"
    "If everything is healthy, say so briefly under 🟢.\n"
    "This report is READ-ONLY — do not suggest that any automated action was taken.\n"
    "Do not truncate or abbreviate any item — include full details for every issue found."
)


# ── ReAct health check ────────────────────────────────────────────────────────
def run_health_check(
    next_client_fn,
    drop_provider_fn,
    synthesis_client: Optional[tuple] = None,
) -> str:
    """
    ReAct loop:
      Reason  → decide which domains to query
      Act     → call each sub-agent with a health-check query
      Observe → collect results
      Reason  → synthesise findings with LLM
      Report  → return formatted string ready for Telegram

    synthesis_client: optional (OpenAI, model_name, label) tuple to use a
    separate model for the final synthesis step (e.g. Ollama chat model).
    Falls back to next_client_fn() if not provided.
    """
    domain_handlers = {
        "proxmox":       _main.ask_proxmox,
        "synology":      _main.ask_synology,
        "linux":         _main.ask_linux,
        "homeassistant": _main.ask_homeassistant,
        "watchyourlan":  _main.ask_watchyourlan,
    }

    ts_start = datetime.now()
    print(f"[health] Starting lab health check at {ts_start:%Y-%m-%d %H:%M:%S}")

    results: dict[str, str] = {}
    timings: dict[str, float] = {}
    for domain, query in _HEALTH_QUERIES.items():
        print(f"[health] Checking {domain}...")
        t0 = time.monotonic()
        try:
            results[domain] = domain_handlers[domain](
                query, next_client_fn, drop_provider_fn,
                max_tokens=MAX_TOKENS,
                max_tool_result_chars=MAX_TOOL_RESULT_CHARS,
            )
            elapsed = time.monotonic() - t0
            timings[domain] = elapsed
            print(f"[health] {domain}: ok ({_fmt_duration(elapsed)})")
        except Exception as exc:
            elapsed = time.monotonic() - t0
            timings[domain] = elapsed
            results[domain] = f"ERROR collecting data: {exc}"
            print(f"[health] {domain}: error — {exc} ({_fmt_duration(elapsed)})", file=sys.stderr)

    # Synthesise with LLM (dedicated chat client if provided, else rotate pool)
    combined = "\n\n".join(
        f"=== {domain.upper()} ===\n{text}" for domain, text in results.items()
    )
    synthesis_user = (
        "Here are the raw health-check results from all lab systems:\n\n"
        f"{combined}\n\n"
        "Produce the structured status report."
    )

    total_agents = sum(timings.values())
    timing_lines = "  ".join(f"{d}: {_fmt_duration(t)}" for d, t in timings.items())
    print(f"[health] Sub-agent timings — {timing_lines}  |  total: {_fmt_duration(total_agents)}")

    if synthesis_client:
        syn_llm, syn_model, syn_label = synthesis_client
    else:
        syn_llm, syn_model, syn_label = next_client_fn()

    print(f"[health] Synthesising with {syn_label}/{syn_model}...")
    t0_syn = time.monotonic()
    try:
        syn_kw: dict = dict(
            model=syn_model,
            messages=[
                {"role": "system", "content": _SYNTHESIS_SYSTEM},
                {"role": "user",   "content": synthesis_user},
            ],
        )
        if MAX_TOKENS:
            syn_kw["max_tokens"] = MAX_TOKENS
        resp = syn_llm.chat.completions.create(**syn_kw)
        report_body = resp.choices[0].message.content or "(empty report)"
    except Exception as exc:
        # Fallback: emit raw results if synthesis fails
        report_body = (
            f"⚠️ Synthesis LLM error: {exc}\n\n"
            + "\n\n".join(f"*{d.upper()}*\n{t[:400]}" for d, t in results.items())
        )

    syn_elapsed = time.monotonic() - t0_syn
    total_elapsed = total_agents + syn_elapsed
    print(f"[health] Synthesis: {_fmt_duration(syn_elapsed)}  |  grand total: {_fmt_duration(total_elapsed)}")

    timing_summary = (
        "  ".join(f"{d}: {_fmt_duration(t)}" for d, t in timings.items())
        + f"  synthesis: {_fmt_duration(syn_elapsed)}  |  *total: {_fmt_duration(total_elapsed)}*"
    )

    ts_label = ts_start.strftime("%Y-%m-%d %H:%M")
    return (
        f"*🔬 Lab Health Report — {ts_label}*\n\n"
        f"{report_body}\n\n"
        f"⏱ _{timing_summary}_"
    )


# ── Scheduler ─────────────────────────────────────────────────────────────────
def _run_check_and_report(next_client_fn, drop_provider_fn, synthesis_client=None) -> None:
    try:
        report = run_health_check(next_client_fn, drop_provider_fn, synthesis_client)
        print(f"\n{'='*60}\n{report}\n{'='*60}\n")
        if TELEGRAM_ALLOWED_USERS:
            tg_broadcast(report)
            print(f"[health] Sent to {len(TELEGRAM_ALLOWED_USERS)} Telegram user(s).")
        else:
            print("[health] TELEGRAM_ALLOWED_USERS not set — skipping Telegram.")
    except Exception as exc:
        msg = f"⚠️ Lab Health Agent unhandled error: {exc}"
        print(msg, file=sys.stderr)
        try:
            tg_broadcast(msg)
        except Exception:
            pass


def health_loop(next_client_fn, drop_provider_fn, synthesis_client=None) -> None:
    schedule_times = _parse_schedule_times(SCHEDULE_TIMES_RAW)

    if schedule_times:
        times_label = ", ".join(f"{h:02d}:{m:02d}" for h, m in schedule_times)
        print(f"[health] Scheduled mode — running daily at: {times_label} (local time)")
        while True:
            wait_secs = _seconds_until_next_schedule(schedule_times)
            next_run = datetime.now() + timedelta(seconds=wait_secs)
            print(f"[health] Next check at {next_run:%Y-%m-%d %H:%M:%S} (in {_fmt_duration(wait_secs)}).")
            time.sleep(wait_secs)
            _run_check_and_report(next_client_fn, drop_provider_fn, synthesis_client)
    else:
        interval_secs = CHECK_INTERVAL_HOURS * 3600
        while True:
            _run_check_and_report(next_client_fn, drop_provider_fn, synthesis_client)
            next_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[health] Next check in {CHECK_INTERVAL_HOURS}h (at approximately {next_ts} + {CHECK_INTERVAL_HOURS}h).")
            time.sleep(interval_secs)


# ── Entry point ───────────────────────────────────────────────────────────────
def main() -> None:
    if not TELEGRAM_BOT_TOKEN:
        print("[warn] TELEGRAM_BOT_TOKEN not set — reports will only print to stdout.")
    if not TELEGRAM_ALLOWED_USERS:
        print("[warn] TELEGRAM_ALLOWED_USERS not set — no Telegram recipients.")

    verbose = os.getenv("MAIN_AGENT_VERBOSE", "true").lower() in ("true", "1", "yes")
    provider_override = os.getenv("LAB_HEALTH_PROVIDER", "").strip().lower()
    model_override = os.getenv("LAB_HEALTH_MODEL", "").strip()
    synthesis_client = None

    if provider_override == "ollama":
        # Fixed Ollama client — no pool, no rotation, no drop
        try:
            tool_llm, tool_model = _main.build_client("ollama")
        except ValueError as exc:
            print(f"[error] {exc}")
            sys.exit(1)
        if model_override:
            tool_model = model_override

        def next_client():
            if verbose:
                print(f"  [llm: ollama / {tool_model}]")
            return tool_llm, tool_model, "ollama"

        def drop_provider(reason: str) -> bool:
            print(f"  [{reason}] Ollama is the only provider — cannot drop.")
            return False

        # Separate chat model for synthesis (MAIN_AGENT_OLLAMA_CHAT_MODEL)
        chat_model_name = os.getenv("MAIN_AGENT_OLLAMA_CHAT_MODEL", "").strip()
        if chat_model_name:
            synthesis_client = (tool_llm, chat_model_name, "ollama-chat")
            print(f"[*] Ollama tool model : {tool_model}")
            print(f"[*] Ollama chat model : {chat_model_name} (synthesis)")
        else:
            print(f"[*] Ollama model      : {tool_model}")

        provider_label = "ollama"

    else:
        if provider_override and provider_override != "rotate":
            # Pinned provider — failover only between its own keys, no cross-provider rotation
            try:
                active_pool: list = _main.build_single_pool(provider_override)
            except ValueError as exc:
                print(f"[error] {exc}")
                sys.exit(1)
            if not active_pool:
                print(f"[error] Provider '{provider_override}' non ha API key utilizzabili. Controlla .env.")
                sys.exit(1)
            if model_override:
                active_pool = [(llm, model_override, lbl) for llm, _, lbl in active_pool]
        else:
            # Rotation pool (default behaviour: provider vuoto o "rotate")
            active_pool = _main.build_pool()
            if not active_pool:
                print("[error] No LLM providers available. Check API keys in .env.")
                sys.exit(1)

        _lock = threading.Lock()
        _state: dict = {"cycle": itertools.cycle(list(active_pool))}

        def next_client():
            with _lock:
                entry = next(_state["cycle"])
            llm, model, prov = entry
            if verbose:
                print(f"  [llm: {prov} / {model}]")
            _state["last_provider"] = prov
            return llm, model, prov

        def drop_provider(reason: str) -> bool:
            with _lock:
                cur = _state.get("last_provider", "")
                active_pool[:] = [e for e in active_pool if e[2] != cur]
                if not active_pool:
                    print(f"[error] All providers removed ({reason}).")
                    return False
                _state["cycle"] = itertools.cycle(list(active_pool))
                remaining = ", ".join(p for _, _, p in active_pool)
                print(f"  [{reason}] Provider '{cur}' removed. Active: {remaining}")
            return True

        provider_label = ", ".join(label for _, _, label in active_pool)

    schedule_times = _parse_schedule_times(SCHEDULE_TIMES_RAW)
    print(f"[*] Lab Health Agent")
    print(f"[*] Provider(s)      : {provider_label}")
    if schedule_times:
        print(f"[*] Schedule         : daily at {', '.join(f'{h:02d}:{m:02d}' for h, m in schedule_times)}")
    else:
        print(f"[*] Check interval   : every {CHECK_INTERVAL_HOURS}h (runs immediately at startup)")
    print(f"[*] Max tokens/call  : {MAX_TOKENS or '(unlimited)'}")
    print(f"[*] Max tool result  : {MAX_TOOL_RESULT_CHARS or '(unlimited)'} chars")
    print(f"[*] Telegram users   : {TELEGRAM_ALLOWED_USERS or '(none)'}")
    print(f"[*] Proxmox          : {os.getenv('PROXMOX_MCP_HOST', '(not configured)')}")
    print(f"[*] Synology         : {os.getenv('SYNOLOGY_MCP_NAS_CONFIG', '(not configured)')}")
    print(f"[*] Home Assistant   : {os.getenv('HAOS_MCP_URL', '(not configured)')}")
    print(f"[*] WatchYourLAN     : {os.getenv('WYLA_MCP_URL', '(not configured)')}")
    print()

    health_loop(next_client, drop_provider, synthesis_client)


if __name__ == "__main__":
    main()
