# Architecture

## Overview

The project is a **two-tier agentic system**:

- **Tier 1 — the orchestrator.** A single LLM with five delegation tools
  (`ask_proxmox`, `ask_synology`, `ask_linux`, `ask_homeassistant`,
  `ask_watchyourlan`). It never touches infrastructure directly; it only decides
  *which* sub-agent should handle a request and how to combine their answers.

- **Tier 2 — the MCP sub-agents.** Each `ask_*` handler is itself a small agent:
  it boots a domain-specific **MCP server**, discovers that server's tools, and
  runs an inner tool-call loop driven by the same rotating LLM pool until the
  model returns a final text answer.

Both tiers share one provider rotation pool, so rate limits and outages fail
over uniformly.

## Components

| File | Role |
|------|------|
| [`main_agent.py`](../main_agent.py) | Orchestrator: provider registry, rotation pool, the `ask_*` handlers, the [`Orchestrator`](CLASSES.md#orchestrator) class, and the CLI REPL (`main`). |
| [`agent_server.py`](../agent_server.py) | FastAPI sidecar exposing an OpenAI-compatible `POST /v1/chat/completions` backed by one shared `Orchestrator`. |
| [`proxmox_mcp_agent.py`](../proxmox_mcp_agent.py) | Proxmox VE sub-agent + standalone CLI. Hosts the canonical [`MCPClient`](CLASSES.md#mcpclient) and helper functions reused by the orchestrator. |
| [`synology_mcp_agent.py`](../synology_mcp_agent.py) | Synology NAS sub-agent + standalone CLI. |
| [`linux_mcp_agent.py`](../linux_mcp_agent.py) | Linux SSH fleet sub-agent + standalone CLI; also exposes `list_configured_servers()`. |
| [`homeassistant_mcp_agent.py`](../homeassistant_mcp_agent.py) | Home Assistant sub-agent; bridges HA's HTTP MCP endpoint to stdio with `uvx mcp-proxy`. |
| [`watchyourlan_mcp_agent.py`](../watchyourlan_mcp_agent.py) | WatchYourLAN network-monitoring sub-agent; adds schema cleaning for strict providers. |
| [`lab_health_agent.py`](../lab_health_agent.py) | Autonomous, read-only health checker (ReAct loop → Telegram report). |
| [`linux_update_agent.py`](../linux_update_agent.py) | Autonomous package updater for the Linux fleet (check → upgrade → report). |
| [`nvidia_ratelimit.py`](../nvidia_ratelimit.py) | Cross-process rate limiter for the NVIDIA NIM free tier. |
| [`nvidia_free_models.py`](../nvidia_free_models.py) | Utility to list & benchmark NVIDIA NIM free models. |

See [AGENTS.md](AGENTS.md) for a per-module deep dive and [CLASSES.md](CLASSES.md)
for the class reference.

## Request flow (orchestrator turn)

A single turn is [`Orchestrator.run_turn`](CLASSES.md#orchestrator) (mirrored by
the REPL loop in `main_agent.main`):

1. **Reset rotation** — rebuild a fresh provider pool so a transient failure in a
   previous turn never permanently kills a provider.
2. **LLM call** — send the conversation + `ORCHESTRATOR_TOOLS` to the next client
   in the cycle.
3. **No tool calls?** → return the assistant text (optionally re-synthesised by a
   separate Ollama chat model). Done.
4. **Tool calls?** → for each call, dispatch to the matching `ask_*` handler. The
   handler runs `run_mcp_query`, which:
   a. spawns the MCP server process (`start_docker` / `start_proxy`),
   b. initialises MCP and lists tools,
   c. runs an inner loop: LLM ↔ MCP `tools/call` until the sub-agent LLM stops
      calling tools,
   d. returns the final text.
5. Append each sub-agent result as a `tool` message and loop back to step 2.

`on_step` callbacks let `agent_server.py` stream `tool_call` / `tool_result`
progress frames over SSE.

## Provider rotation & failover

The rotation machinery lives in `main_agent.py`:

- **Provider registry** — `PROVIDERS` maps each provider name to its base URL,
  API-key env var, model env var, and default model. Supported:
  `openrouter, groq, gemini, cloudflare, cerebras, mistral, nvidia, puter,
  ollama`. `ROTATE_PROVIDERS` lists those eligible for rotation (Ollama is
  excluded — it has no cloud key).

- **Multiple keys per provider** — `get_provider_keys()` reads `…_API_KEY`,
  `…_API_KEY_2`, `…_API_KEY_3`, … Cloudflare uses `get_cloudflare_entries()`
  because each entry needs both an API key and an account ID.

- **Key-slot filtering** — `MAIN_AGENT_ROTATE_KEYS` (parsed by
  `_parse_rotate_slots`) restricts which 1-based key slots participate.

- **Pool construction** — `build_pool()` interleaves providers round-robin
  (`openrouter, groq, …, openrouter#2, groq#2, …`). `build_single_pool(p)`
  restricts failover to one provider's own keys (used when an agent pins a
  provider).

- **Failover** — during a turn, a `429` (`RateLimitError`), a fatal block
  (`organization_restricted`, `account_suspended`, `access_denied`), or a
  context-too-large `413` causes `_drop_provider()` to remove the current
  provider from the turn's pool and continue with the next. A `tool_use_failed`
  bad request simply advances to the next client without dropping.

### Rotating vs. pinned vs. Ollama

- **Rotating** (default; `MAIN_AGENT_PROVIDER` empty / `rotate` / `auto`) — full
  interleaved pool, cross-provider failover.
- **Pinned** (`MAIN_AGENT_PROVIDER=<name>`) — a single fixed client; the
  autonomous agents instead use `build_single_pool` so failover stays within
  that provider's keys.
- **Ollama** — a fixed local client with an optional **dual-model** split: a
  tool-capable model drives the agentic loop, and a separate
  `…_OLLAMA_CHAT_MODEL` synthesises the final natural-language answer.

## The MCP protocol layer

Each sub-agent module ships an identical [`MCPClient`](CLASSES.md#mcpclient) —
a minimal JSON-RPC 2.0 client speaking the MCP `2024-11-05` protocol over the
child process's stdio:

- A background reader thread parses one JSON message per line and dispatches
  responses to per-request `threading.Event`s keyed by request id.
- `initialize()` → `tools/list` → `tools/call` are the only methods used.
- `notifications/initialized` is sent fire-and-forget after init.

Three module-level helpers adapt MCP to the OpenAI Chat Completions shape:

- `tools_to_openai(tools)` — MCP tool descriptors → OpenAI `tools` array.
- `mcp_result_to_text(result)` — flatten an MCP tool result's `content` blocks
  to plain text.
- `assistant_msg(msg)` — serialise an OpenAI assistant message (with tool calls)
  back into a `messages`-list dict.

The orchestrator reuses the **proxmox** module's copies of these
(`run_mcp_query` calls `_proxmox.MCPClient`, `_proxmox.tools_to_openai`, etc.)
since they are byte-for-byte identical across modules.

> **WatchYourLAN exception:** `watchyourlan_mcp_agent.tools_to_openai` runs each
> schema through `_clean_schema()` first, stripping `title`/`default` keys and
> dropping optional empty-string-default properties that make strict providers
> (e.g. Groq) emit malformed tool calls.

## Transport per sub-agent

| Sub-agent | Transport | Launcher |
|-----------|-----------|----------|
| Proxmox | Docker container, MCP over stdio | `start_docker()` → `lordraw/proxmox-mcp` |
| Synology | Docker container, MCP over stdio | `start_docker()` → `lordraw/synology-mcp` (optionally mounts `nas_config.json`) |
| Linux | Docker container, MCP over stdio | `start_docker()` → `lordraw/linux-mcp` (mounts `~/.ssh` read-only) |
| Home Assistant | `uvx mcp-proxy` bridging HA HTTP → stdio | `start_proxy()` |
| WatchYourLAN | Docker container, MCP over stdio | `start_docker()` → `lordraw/watchyourlan-mcp` |

## Autonomous agents (ReAct)

`lab_health_agent.py` and `linux_update_agent.py` import `main_agent` to reuse
the `ask_*` handlers and pool builders, then run their own scheduler:

- **Reason → Act → Observe → Reason → Report** loops.
- A `next_client` / `drop_provider` pair (rotating pool, pinned single pool, or
  fixed Ollama) is passed into the `ask_*` handlers.
- Results are synthesised by an LLM (optionally a dedicated Ollama chat model)
  and delivered to Telegram with paragraph-boundary message pagination.
- Scheduling is either a fixed interval (`…_INTERVAL_HOURS`, runs at startup too)
  or fixed daily times (`…_SCHEDULE_TIMES=HH:MM,HH:MM`).
- The health agent is strictly **read-only**; the update agent applies upgrades
  (never `dist-upgrade`, never an automatic reboot) unless `…_DRY_RUN=true`.
