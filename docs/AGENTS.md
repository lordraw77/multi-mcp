# Modules & agents

A per-module reference. For classes see [CLASSES.md](CLASSES.md); for the big
picture see [ARCHITECTURE.md](ARCHITECTURE.md).

Every module uses the same conventions:

- A prefix constant `P` (e.g. `PROXMOX_MCP`) and a helper `_e(key, default)` that
  reads `os.getenv(f"{P}_{key}", default)`.
- A `PROVIDERS` registry and a `build_client(provider) -> (OpenAI, model)`.
- The shared MCP helpers `tools_to_openai`, `mcp_result_to_text`, `assistant_msg`.
- A `main()` REPL for standalone use, guarded by `if __name__ == "__main__"`.

---

## `main_agent.py` — Orchestrator

The hub. Routes requests to the five sub-agents and owns provider rotation.

**Provider machinery**

| Function | Purpose |
|----------|---------|
| `_parse_rotate_slots()` | Parse `MAIN_AGENT_ROTATE_KEYS` (`"1,3"`) into a set of 1-based key slots, or `None` for all. |
| `get_provider_keys(provider, slots)` | `(slot, key)` pairs for a provider (`…_API_KEY`, `…_API_KEY_2`, …). |
| `get_cloudflare_entries(slots)` | `(slot, api_key, account_id)` triples for Cloudflare (needs both per entry). |
| `available_providers(slots)` | Providers from `ROTATE_PROVIDERS` that have ≥1 usable entry. |
| `build_client(provider, api_key=None, account_id=None)` | Build one `OpenAI` client + model, NVIDIA-wrapped. |
| `_provider_entries(provider, slots)` | All `(client, model, label)` entries for one provider across its key slots. |
| `build_pool()` | The full round-robin interleaved rotation pool. |
| `build_single_pool(provider)` | A pool restricted to one provider's own keys (failover within that provider only). |

**Sub-agent execution**

- `run_mcp_query(domain, mcp_module, start_docker_fn, next_client, system_prompt, query, …)`
  — the generic sub-agent runner: spawns the MCP process, forwards its stderr,
  runs the inner agentic loop with shared rotation, returns the final text.
  Uses `mcp_module` (the sub-agent's own module) for the shared MCP helpers, and
  supports `max_tokens` and `max_tool_result_chars` (truncation).
- `AGENTS` — the registry, built at import by `agent_registry.load_agents()` from
  `agents.d/*.json` (see below). `ORCHESTRATOR_TOOLS`, `DISPATCH` and
  `SYSTEM_PROMPT` are all derived from it, so the set of agents is config-driven —
  no hardcoded `ask_*` functions.

**Sub-agent registry (`agent_registry.py` + `agents.d/`)**

Each sub-agent is one JSON file in `agents.d/` (override the dir with
`MAIN_AGENT_AGENTS_DIR`). Add/remove an agent = add/remove a file — no code change,
no image rebuild (the folder is bind-mountable in Docker).

| Field | Meaning |
|-------|---------|
| `name`, `order`, `enabled` | id, sort key, on/off (`*.json.disabled` also skips it). |
| `tool_name`, `module` / `module_path`, `start_fn` | OpenAI tool name; module to import **by name** (`module`) or **by file** (`module_path`, abs or relative to the agents dir); launcher (`start_docker` / `start_proxy`). |
| `summary`, `tool_description`, `query_description` | bullet for `SYSTEM_PROMPT` / tool description / `query` parameter description shown to the LLM. |
| `system_prompt`, `status_line` | sub-agent prompt and CLI banner line. |
| `max_tokens`, `max_tool_result_chars` | optional per-agent limits. |

Placeholders in `system_prompt` / `status_line`: `{env:VAR}`, `{env:VAR:default}`,
and `{ctx:key}` (resolved from the module's optional `prompt_context()` — Linux uses
it for the server list). `load_agents()` skips malformed files / unimportable
modules with a `[warn]`, never crashing the orchestrator.

**Externalized agents (code + config, no rebuild).** The agents dir is added to
`sys.path`, so a brand-new agent can ship as a `.py` dropped next to its `.json`
in the (bind-mounted) folder — both `module: my_agent` and
`module_path: my_agent.py` resolve there. The `.py` only needs the launcher plus
the four MCP helpers the orchestrator calls; import them from
[`mcp_common.py`](../mcp_common.py) (`MCPClient`, `tools_to_openai`,
`mcp_result_to_text`, `assistant_msg`, `docker_start`) instead of copying the
boilerplate. See [`agents.d/example_agent.py`](../agents.d/example_agent.py) +
[`agents.d/example.json.disabled`](../agents.d/example.json.disabled) for a template.

**Error classification** (shared by the sidecar and REPL):

- `_is_context_too_large(exc)` — 413 / "context length exceeded" family.
- `_is_fatal_error(exc)` — `organization_restricted`, `account_suspended`,
  `access_denied`.

**Entry points:** the [`Orchestrator`](CLASSES.md#orchestrator) class and
`main()`, the CLI REPL.

---

## `agent_server.py` — OpenAI-compatible HTTP sidecar

A thin FastAPI wrapper exposing the orchestrator as the `agent/multi-mcp` model.
Same `.env`, same rotation pool, same Docker MCP sub-agents as the CLI.

- `get_orch()` — lazily builds the one shared `Orchestrator`.
- `_prepare(messages)` — maps to plain dicts and **forces the orchestrator's own
  system prompt to lead** (any gateway system prompt is dropped).
- Endpoints:
  - `GET /health` → `{"status":"ok", …}`.
  - `GET /v1/models` → advertises the single `agent/multi-mcp` model.
  - `POST /v1/chat/completions` → non-streaming returns a full completion;
    streaming returns an SSE event stream.
- **Streaming protocol:** the turn runs in a worker thread; `on_step` pushes
  `tool_call` / `tool_result` control frames into an `asyncio.Queue` (mapped to
  named SSE events by SpiceSibyl / progressive Telegram edits), then the final
  answer arrives as a normal content chunk followed by a meta chunk and
  `[DONE]`.

Because the orchestrator is synchronous and slow (it spawns containers), every
turn runs via `asyncio.to_thread` / a worker thread and turns are serialized by
the `Orchestrator`'s own lock.

Request models: [`Message`](CLASSES.md#message),
[`ChatRequest`](CLASSES.md#chatrequest).

---

## `proxmox_mcp_agent.py` — Proxmox VE sub-agent

Manages a Proxmox VE cluster (VMs, LXC, nodes, storage, backups, snapshots,
firewall, HA, Ceph). Prefix `PROXMOX_MCP`.

- `start_docker()` launches `lordraw/proxmox-mcp:latest`, mapping
  `PROXMOX_MCP_*` env vars (`HOST`, `PORT`, `USER`, `PASSWORD`, `VERIFY_SSL`,
  `TOKEN_ID`, `TOKEN_SECRET`) to the `PROXMOX_*` names the container expects.
- **Canonical home of the shared helpers** — `run_mcp_query` in the orchestrator
  imports `MCPClient`, `tools_to_openai`, `mcp_result_to_text`, and
  `assistant_msg` from this module.

---

## `synology_mcp_agent.py` — Synology NAS sub-agent

Manages Synology DSM (files, shares, Docker, packages, RAID/volume status).
Prefix `SYNOLOGY_MCP`.

- `start_docker()` launches `lordraw/synology-mcp:latest`. If
  `SYNOLOGY_MCP_NAS_CONFIG` points to an existing file it is bind-mounted at
  `/config/nas_config.json`; otherwise individual `SYNOLOGY_MCP_*` credentials
  are passed as env vars (warns about missing `HOST`/`USER`/`PASSWORD`).
- `_ollama_base()` normalises the Ollama base URL to end in `/v1`.
- Uses `SYNOLOGY_MCP_AI_PROVIDER` / `…_AI_MODEL` for standalone runs.

---

## `linux_mcp_agent.py` — Linux SSH fleet sub-agent

SSH access to a fleet of Linux servers (commands, services, logs, processes,
packages, monitoring). Prefix `UXMCP`.

- `start_docker()` launches `lordraw/linux-mcp:latest`, maps each configured
  `UXMCP_SERVER_N_*` to `SERVER_N_*` (the names the image expects), and mounts an
  SSH key dir (`UXMCP_MCP_SSH_KEY_DIR` or `~/.ssh`) read-only at `/root/.ssh`.
- `list_configured_servers() -> list[str]` — `"label (host)"` strings for every
  configured `UXMCP_SERVER_N_HOST`; used by the orchestrator banner and by the
  autonomous agents to scope work.

---

## `homeassistant_mcp_agent.py` — Home Assistant sub-agent

Controls a Home Assistant smart home (lights, switches, climate, covers, media,
scenes, scripts, automations). Prefix `HAOS_MCP`.

- **No sibling container.** `start_proxy()` runs `uvx mcp-proxy
  --transport=streamablehttp --stateless <HA_URL>/api/mcp`, bridging HA's native
  HTTP MCP endpoint to stdio so the standard `MCPClient` works. Requires
  `HAOS_MCP_URL` and `HAOS_MCP_TOKEN` (a long-lived access token passed as
  `API_ACCESS_TOKEN`).
- `_key(var)` — reads `var`, falling back to its `MAIN_AGENT_` equivalent, so the
  HA agent can share the orchestrator's provider keys.

---

## `watchyourlan_mcp_agent.py` — WatchYourLAN sub-agent

Network device discovery & monitoring via WatchYourLAN (online/offline status,
history, unknown-device detection, rename/delete). Prefix `WYLA_MCP`.

- `start_docker()` launches `lordraw/watchyourlan-mcp:latest`, passing
  `WYL_BASE_URL` (from `WYLA_MCP_URL`, required) and optional `WYL_TIMEOUT`.
- `_clean_schema(schema)` — **the one structural difference from the other
  sub-agents.** It normalises MCP tool schemas for strict OpenAI-compatible
  providers: strips `title`/`default`, and drops optional properties whose
  default is an empty string (Groq and others emit malformed tool calls when
  forced to fill an empty default; the MCP server applies the default itself when
  the property is omitted). `tools_to_openai` here pipes every schema through it.
- `_key(var)` — same `MAIN_AGENT_` fallback as the HA agent.
- Its `main()` REPL also catches `tool_use_failed` bad requests and recovers
  truncated tool-call JSON by appending a closing brace.

---

## `lab_health_agent.py` — Autonomous health checker (read-only)

Runs a full lab status check at startup and on a schedule, then reports to
Telegram. **Never modifies anything** — it reports what should be fixed.

- `run_health_check(next_client_fn, drop_provider_fn, synthesis_client=None)` —
  the ReAct loop: queries each domain (`_HEALTH_QUERIES`) via the `ask_*`
  handlers, then synthesises a 🔴/🟡/🟢 report (`_SYNTHESIS_SYSTEM`) with WHAT /
  WHICH / HOW for each issue.
- `health_loop(…)` — scheduler: fixed daily `LAB_HEALTH_SCHEDULE_TIMES`, or every
  `LAB_HEALTH_INTERVAL_HOURS` (default 12, runs immediately at startup).
- `main()` — builds the client strategy (rotating pool / pinned single pool /
  fixed Ollama with optional dual chat model), prints a banner, runs the loop.
- Telegram helpers `_tg_send_raw`, `tg_send` (paragraph-boundary pagination
  under Telegram's 4000-char limit), `tg_broadcast`.
- Time helpers `_fmt_duration`, `_parse_schedule_times`,
  `_seconds_until_next_schedule`.

Tunables: `LAB_HEALTH_PROVIDER`, `LAB_HEALTH_MODEL`, `LAB_HEALTH_MAX_TOKENS`,
`LAB_HEALTH_MAX_TOOL_RESULT_CHARS`.

---

## `linux_update_agent.py` — Autonomous fleet updater

Updates the Linux fleet in three phases, then reports to Telegram. Reuses the
Linux sub-agent and `main_agent` pools.

- **Phase 1 (CHECK):** one read-only `ask_linux` pass listing pending updates;
  `parse_pending()` reads the `UPDATES: <server> | <N>` marker lines (falling
  back to "treat every server as a candidate").
- **Phase 2 (UPDATE):** one isolated `ask_linux` pass per server with pending
  updates, applying a full non-interactive upgrade — **never `dist-upgrade`,
  never an automatic reboot** (it reports if a reboot is required). Skipped under
  `LINUX_UPDATE_DRY_RUN=true`.
- **Phase 3 (REPORT):** LLM synthesis (`_SYNTHESIS_SYSTEM`) → Telegram.
- `target_servers()` — configured server labels minus
  `LINUX_UPDATE_EXCLUDE_SERVERS`.
- `run_update_cycle(…)`, `update_loop(…)`, `main()`, plus the same Telegram /
  time helpers and prompt builders (`_check_query`, `_update_query`).

Tunables mirror the health agent: `LINUX_UPDATE_PROVIDER`, `…_MODEL`,
`…_INTERVAL_HOURS` (default 24), `…_SCHEDULE_TIMES`, `…_DRY_RUN`,
`…_EXCLUDE_SERVERS`, `…_MAX_TOKENS`, `…_MAX_TOOL_RESULT_CHARS`.

---

## `nvidia_ratelimit.py` — NVIDIA NIM rate limiter

Cross-process throttle for the NVIDIA NIM free tier. See the
[class reference](CLASSES.md#nvidia-rate-limiter-classes).

- `_acquire(api_key)` — blocks until a call slot is free under `NVIDIA_RPM_LIMIT`
  for that key, using an `fcntl`-locked sliding-window file in `.cache/` keyed by
  a hash of the API key (so concurrent processes coordinate).
- `wrap_if_nvidia(provider, client, api_key)` — wraps NVIDIA clients only.

---

## `nvidia_free_models.py` — NVIDIA model benchmark utility

A standalone CLI to **list and benchmark** the models on the NVIDIA NIM free
endpoint, sorted fastest-first (time-to-first-token and tokens/sec).

- `list_models`, `looks_like_chat_model` (skips embedding/rerank/guardrail/etc.),
  `benchmark_model` (streaming TTFT + tok/s), and a `main()` argparse CLI.
- Persists failing models to `.cache/` (`nvidia_404_models.txt`,
  `nvidia_timeout_models.txt`, `nvidia_other_errors.txt`) so they aren't retested
  every run; `--retest-*` flags re-include them.
- Reads `MAIN_AGENT_NVIDIA_API_KEY` (or `NVIDIA_API_KEY`).

Usage examples are in the module docstring (`--list`, `--limit`, `--filter`,
`--max-tokens`, …).
