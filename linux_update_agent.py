#!/usr/bin/env python3
"""
Linux Update Agent — ReAct-style periodic updater for the Linux fleet.

Runs at startup and every LINUX_UPDATE_INTERVAL_HOURS hours.
Phase 1 (CHECK):  one read-only pass over all configured Linux servers
                  (via linux_mcp) to list pending package updates.
Phase 2 (UPDATE): one dedicated pass per server with pending updates,
                  applying a full non-interactive upgrade (never dist-upgrade,
                  never an automatic reboot).
Phase 3 (REPORT): LLM synthesis of what was updated where, sent to Telegram
                  with paragraph-boundary message pagination.

Set LINUX_UPDATE_DRY_RUN=true to stop after phase 1 and only report
available updates without installing anything.
"""

import itertools
import os
import re
import sys
import time
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import requests
from dotenv import load_dotenv

_ROOT = Path(__file__).parent
_AGENTS_DIR = _ROOT / "agents.d"
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_AGENTS_DIR))

import main_agent as _main
import agent_registry as _registry
import linux_mcp_agent as _linux

_linux_spec = next(
    (a for a in _registry.load_agents(_main.run_mcp_query, agents_dir=_AGENTS_DIR) if a.name == "linux"),
    None,
)
if _linux_spec is None:
    raise RuntimeError("linux agent not found in agents.d/linux.json — cannot start")
ask_linux = _linux_spec.handler

load_dotenv(Path(__file__).parent / ".env", override=True)

# ── Configuration ─────────────────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_ALLOWED_USERS = [
    int(u.strip())
    for u in os.getenv("TELEGRAM_ALLOWED_USERS", "").split(",")
    if u.strip().isdigit()
]
CHECK_INTERVAL_HOURS = int(os.getenv("LINUX_UPDATE_INTERVAL_HOURS", "24"))
# Comma-separated "HH:MM" times (local container time). When set (typically
# from docker-compose), the agent runs ONLY at these times instead of at
# startup + every CHECK_INTERVAL_HOURS.
SCHEDULE_TIMES_RAW = os.getenv("LINUX_UPDATE_SCHEDULE_TIMES", "").strip()
DRY_RUN = os.getenv("LINUX_UPDATE_DRY_RUN", "false").lower() in ("true", "1", "yes")
EXCLUDE_SERVERS = {
    s.strip().lower()
    for s in os.getenv("LINUX_UPDATE_EXCLUDE_SERVERS", "").split(",")
    if s.strip()
}
# Unset (or non-numeric) → None → no limit
_max_tokens_raw = os.getenv("LINUX_UPDATE_MAX_TOKENS", "").strip()
MAX_TOKENS = int(_max_tokens_raw) if _max_tokens_raw.isdigit() else None
_max_tool_raw = os.getenv("LINUX_UPDATE_MAX_TOOL_RESULT_CHARS", "").strip()
MAX_TOOL_RESULT_CHARS = int(_max_tool_raw) if _max_tool_raw.isdigit() else None


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


# ── Server list ───────────────────────────────────────────────────────────────
def target_servers() -> list[str]:
    """Configured server labels minus LINUX_UPDATE_EXCLUDE_SERVERS."""
    labels = [s.split("(")[0].strip() for s in _linux.list_configured_servers()]
    return [l for l in labels if l.lower() not in EXCLUDE_SERVERS]


# ── Prompts ───────────────────────────────────────────────────────────────────
def _check_query(servers: list[str]) -> str:
    server_list = ", ".join(servers)
    return (
        "Read-only check of pending package updates on these Linux servers: "
        f"{server_list}. Ignore any other configured server.\n"
        "For EACH server:\n"
        "1. Detect the package manager (apt, dnf, yum or zypper).\n"
        "2. Refresh the package metadata (e.g. 'sudo apt-get update', "
        "'dnf check-update', 'zypper refresh').\n"
        "3. List the packages that have updates available.\n"
        "Do NOT install, upgrade or remove anything in this phase.\n\n"
        "At the very END of your answer, output one line per server in EXACTLY "
        "this format (no markdown, no extra text on those lines):\n"
        "UPDATES: <server_label> | <number_of_pending_updates>"
    )


def _update_query(label: str) -> str:
    return (
        f"Apply ALL pending package updates on the server '{label}' ONLY. "
        "Do NOT touch any other server.\n"
        "1. Detect the package manager.\n"
        "2. Refresh metadata, then apply a FULL upgrade non-interactively:\n"
        "   - apt: 'sudo apt-get update' then "
        "'sudo DEBIAN_FRONTEND=noninteractive apt-get -y upgrade'\n"
        "   - dnf: 'sudo dnf -y upgrade'   - yum: 'sudo yum -y update'\n"
        "   - zypper: 'sudo zypper --non-interactive update'\n"
        "   Never use dist-upgrade or full-upgrade.\n"
        "3. Report the exact list of packages that were upgraded, with "
        "old → new versions (or state clearly that nothing was upgraded).\n"
        "4. Check whether a reboot is required ('/var/run/reboot-required' on "
        "Debian/Ubuntu, 'needs-restarting -r' on RHEL-family) and report it, "
        "but do NOT reboot the server.\n"
        "5. Report any error output verbatim."
    )


_SYNTHESIS_SYSTEM = (
    "You are a senior Linux systems engineer summarising the results of an "
    "automated package-update run across a fleet of servers. Produce a concise, "
    "structured report formatted for Telegram Markdown.\n\n"
    "For EACH server output one section:\n"
    "✅ *<server>* — <N> packages updated (or 'already up to date')\n"
    "   followed by the list of updated packages with versions.\n"
    "   Add '⚠️ reboot required' if the server needs a reboot.\n"
    "Use ❌ *<server>* for servers where the update failed, including the error.\n"
    "Use 📋 *<server>* for servers in dry-run mode, listing the available "
    "(not installed) updates.\n"
    "Do not truncate or abbreviate package lists — include every package reported.\n"
    "Do not invent results: only report what is present in the raw data."
)


# ── Check-phase parsing ───────────────────────────────────────────────────────
def parse_pending(check_text: str, servers: list[str]) -> dict[str, int]:
    """Parse 'UPDATES: label | N' marker lines. Fallback: assume every server
    is a candidate (the per-server update pass re-checks before installing)."""
    by_lower = {s.lower(): s for s in servers}
    pending: dict[str, int] = {}
    for raw_label, raw_count in re.findall(
        r"UPDATES:\s*([^|\n]+?)\s*\|\s*(\d+)", check_text, re.IGNORECASE
    ):
        label = by_lower.get(raw_label.strip().lower())
        if label:
            pending[label] = int(raw_count)
    if not pending:
        print("[update] No parseable UPDATES markers — treating all servers as candidates.")
        return {s: -1 for s in servers}  # -1 = unknown count
    return pending


# ── ReAct update run ──────────────────────────────────────────────────────────
def run_update_cycle(
    next_client_fn,
    drop_provider_fn,
    synthesis_client: Optional[tuple] = None,
) -> str:
    """
    ReAct loop:
      Reason  → which servers are in scope (env config minus exclusions)
      Act     → check pending updates on all servers (read-only)
      Observe → parse pending-update counts per server
      Act     → apply full upgrade, one isolated pass per server (unless dry-run)
      Observe → collect per-server results
      Reason  → synthesise findings with LLM
      Report  → return formatted string ready for Telegram
    """
    servers = target_servers()
    ts_start = datetime.now()
    print(f"[update] Starting update cycle at {ts_start:%Y-%m-%d %H:%M:%S}")
    print(f"[update] Target servers: {', '.join(servers) or '(none)'}"
          + (f"  (excluded: {', '.join(sorted(EXCLUDE_SERVERS))})" if EXCLUDE_SERVERS else ""))

    if not servers:
        return "*🔄 Linux Update Report*\n\nNo Linux servers configured (or all excluded)."

    timings: dict[str, float] = {}

    # ── Phase 1: CHECK ────────────────────────────────────────────────────────
    print("[update] Phase 1 — checking pending updates on all servers...")
    t0 = time.monotonic()
    try:
        check_text = ask_linux(
            _check_query(servers), next_client_fn, drop_provider_fn,
            max_tokens=MAX_TOKENS,
            max_tool_result_chars=MAX_TOOL_RESULT_CHARS,
        )
    except Exception as exc:
        timings["check"] = time.monotonic() - t0
        print(f"[update] check phase failed: {exc}", file=sys.stderr)
        return (
            f"*🔄 Linux Update Report — {ts_start:%Y-%m-%d %H:%M}*\n\n"
            f"❌ Update check failed, nothing was installed:\n{exc}"
        )
    timings["check"] = time.monotonic() - t0
    print(f"[update] check done ({_fmt_duration(timings['check'])})")

    pending = parse_pending(check_text, servers)
    to_update = [s for s, n in pending.items() if n != 0]
    summary = "  ".join(
        f"{s}: {'?' if pending.get(s, 0) < 0 else pending.get(s, 0)}" for s in servers
    )
    print(f"[update] Pending updates — {summary}")

    # ── Phase 2: UPDATE (skipped in dry-run) ──────────────────────────────────
    results: dict[str, str] = {}
    if DRY_RUN:
        print("[update] DRY RUN — skipping update phase.")
    elif not to_update:
        print("[update] All servers already up to date.")
    else:
        for label in to_update:
            print(f"[update] Phase 2 — updating {label}...")
            t0 = time.monotonic()
            try:
                results[label] = ask_linux(
                    _update_query(label), next_client_fn, drop_provider_fn,
                    max_tokens=MAX_TOKENS,
                    max_tool_result_chars=MAX_TOOL_RESULT_CHARS,
                )
                timings[label] = time.monotonic() - t0
                print(f"[update] {label}: ok ({_fmt_duration(timings[label])})")
            except Exception as exc:
                timings[label] = time.monotonic() - t0
                results[label] = f"ERROR applying updates: {exc}"
                print(f"[update] {label}: error — {exc} "
                      f"({_fmt_duration(timings[label])})", file=sys.stderr)

    # ── Phase 3: SYNTHESIS ────────────────────────────────────────────────────
    mode_note = (
        "DRY RUN mode: nothing was installed, report the AVAILABLE updates per server."
        if DRY_RUN else
        "Updates were applied where pending; report what was actually installed per server."
    )
    combined = f"=== CHECK PHASE (all servers) ===\n{check_text}"
    for label, text in results.items():
        combined += f"\n\n=== UPDATE PHASE: {label} ===\n{text}"
    synthesis_user = (
        f"{mode_note}\n"
        f"Servers in scope: {', '.join(servers)}.\n\n"
        f"Raw results of the update run:\n\n{combined}\n\n"
        "Produce the structured per-server update report."
    )

    total_agents = sum(timings.values())
    timing_lines = "  ".join(f"{d}: {_fmt_duration(t)}" for d, t in timings.items())
    print(f"[update] Phase timings — {timing_lines}  |  total: {_fmt_duration(total_agents)}")

    if synthesis_client:
        syn_llm, syn_model, syn_label = synthesis_client
    else:
        syn_llm, syn_model, syn_label = next_client_fn()

    print(f"[update] Synthesising with {syn_label}/{syn_model}...")
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
        raw_sections = [f"*CHECK*\n{check_text[:800]}"] + [
            f"*{label.upper()}*\n{text[:800]}" for label, text in results.items()
        ]
        report_body = f"⚠️ Synthesis LLM error: {exc}\n\n" + "\n\n".join(raw_sections)

    syn_elapsed = time.monotonic() - t0_syn
    total_elapsed = total_agents + syn_elapsed
    print(f"[update] Synthesis: {_fmt_duration(syn_elapsed)}  |  grand total: {_fmt_duration(total_elapsed)}")

    timing_summary = (
        "  ".join(f"{d}: {_fmt_duration(t)}" for d, t in timings.items())
        + f"  synthesis: {_fmt_duration(syn_elapsed)}  |  *total: {_fmt_duration(total_elapsed)}*"
    )

    ts_label = ts_start.strftime("%Y-%m-%d %H:%M")
    title = "🔄 Linux Update Report" + (" (dry run)" if DRY_RUN else "")
    return (
        f"*{title} — {ts_label}*\n\n"
        f"{report_body}\n\n"
        f"⏱ _{timing_summary}_"
    )


# ── Scheduler ─────────────────────────────────────────────────────────────────
def _run_update_and_report(next_client_fn, drop_provider_fn, synthesis_client=None) -> None:
    try:
        report = run_update_cycle(next_client_fn, drop_provider_fn, synthesis_client)
        print(f"\n{'='*60}\n{report}\n{'='*60}\n")
        if TELEGRAM_ALLOWED_USERS:
            tg_broadcast(report)
            print(f"[update] Sent to {len(TELEGRAM_ALLOWED_USERS)} Telegram user(s).")
        else:
            print("[update] TELEGRAM_ALLOWED_USERS not set — skipping Telegram.")
    except Exception as exc:
        msg = f"⚠️ Linux Update Agent unhandled error: {exc}"
        print(msg, file=sys.stderr)
        try:
            tg_broadcast(msg)
        except Exception:
            pass


def update_loop(next_client_fn, drop_provider_fn, synthesis_client=None) -> None:
    schedule_times = _parse_schedule_times(SCHEDULE_TIMES_RAW)

    if schedule_times:
        times_label = ", ".join(f"{h:02d}:{m:02d}" for h, m in schedule_times)
        print(f"[update] Scheduled mode — running daily at: {times_label} (local time)")
        while True:
            wait_secs = _seconds_until_next_schedule(schedule_times)
            next_run = datetime.now() + timedelta(seconds=wait_secs)
            print(f"[update] Next run at {next_run:%Y-%m-%d %H:%M:%S} (in {_fmt_duration(wait_secs)}).")
            time.sleep(wait_secs)
            _run_update_and_report(next_client_fn, drop_provider_fn, synthesis_client)
    else:
        interval_secs = CHECK_INTERVAL_HOURS * 3600
        while True:
            _run_update_and_report(next_client_fn, drop_provider_fn, synthesis_client)
            next_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[update] Next run in {CHECK_INTERVAL_HOURS}h (at approximately {next_ts} + {CHECK_INTERVAL_HOURS}h).")
            time.sleep(interval_secs)


# ── Entry point ───────────────────────────────────────────────────────────────
def main() -> None:
    if not TELEGRAM_BOT_TOKEN:
        print("[warn] TELEGRAM_BOT_TOKEN not set — reports will only print to stdout.")
    if not TELEGRAM_ALLOWED_USERS:
        print("[warn] TELEGRAM_ALLOWED_USERS not set — no Telegram recipients.")

    verbose = os.getenv("MAIN_AGENT_VERBOSE", "true").lower() in ("true", "1", "yes")
    provider_override = os.getenv("LINUX_UPDATE_PROVIDER", "").strip().lower()
    model_override = os.getenv("LINUX_UPDATE_MODEL", "").strip()
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

    servers = target_servers()
    schedule_times = _parse_schedule_times(SCHEDULE_TIMES_RAW)
    print(f"[*] Linux Update Agent")
    print(f"[*] Provider(s)      : {provider_label}")
    if schedule_times:
        print(f"[*] Schedule         : daily at {', '.join(f'{h:02d}:{m:02d}' for h, m in schedule_times)}")
    else:
        print(f"[*] Run interval     : every {CHECK_INTERVAL_HOURS}h (runs immediately at startup)")
    print(f"[*] Mode             : {'DRY RUN (report only)' if DRY_RUN else 'full upgrade'}")
    print(f"[*] Max tokens/call  : {MAX_TOKENS or '(unlimited)'}")
    print(f"[*] Max tool result  : {MAX_TOOL_RESULT_CHARS or '(unlimited)'} chars")
    print(f"[*] Telegram users   : {TELEGRAM_ALLOWED_USERS or '(none)'}")
    print(f"[*] Target servers   : {', '.join(servers) or '(none configured)'}")
    if EXCLUDE_SERVERS:
        print(f"[*] Excluded         : {', '.join(sorted(EXCLUDE_SERVERS))}")
    print()

    update_loop(next_client, drop_provider, synthesis_client)


if __name__ == "__main__":
    main()
