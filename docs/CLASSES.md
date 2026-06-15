# Class reference

Every class in the codebase, what it does, its public surface, and where it
lives. Module-level functions are covered in [AGENTS.md](AGENTS.md).

---

## `Orchestrator`

**File:** [`main_agent.py`](../main_agent.py)

The reusable Multi-MCP orchestrator. Build it once, then call `run_turn()` per
request. It holds the provider configuration (a rotation pool or a single pinned
provider) read from the process `.env` (`MAIN_AGENT_*`) — exactly the same
configuration the CLI uses. `agent_server.py` drives one shared instance so the
web console and Telegram reach the orchestrator through the same gateway.

**Thread-safety:** `run_turn` is serialized with an internal `threading.Lock`,
because sub-agents spawn Docker containers and the rotation pool is mutated
mid-turn.

### Construction

`__init__(self)` reads configuration and prepares state:

- `provider`, `verbose`, `rotating` — derived from `MAIN_AGENT_PROVIDER` /
  `MAIN_AGENT_VERBOSE`. `rotating` is true when the provider is empty, `rotate`,
  or `auto`.
- `chat_llm` / `chat_model_name` — for Ollama dual-model mode, a separate client
  used only for final chat synthesis (`MAIN_AGENT_OLLAMA_CHAT_MODEL`).
- `_fixed_client` / `tool_model` — a single fixed client when not rotating.
- `dispatch` — maps tool names (`ask_proxmox`, …) to their handler functions.
- Per-turn rotation state (`_pool`, `_cycle`, `_llm`, `_model`, `_cur_provider`)
  reset at the start of every `run_turn`.

### Key methods

| Method | Description |
|--------|-------------|
| `run_turn(messages, on_step=None) -> str` | Run one orchestrator turn over `messages` (appended in place). Returns the final assistant text. `on_step` is an optional callback receiving `{"type": "tool_call"\|"tool_result", …}` for progress streaming. |
| `pool_labels() -> list[str]` | Provider labels currently eligible for rotation (for banners). |
| `_reset_rotation()` | Rebuild the rotation pool so each turn starts from a full, healthy set. |
| `_next_client() -> (OpenAI, model, label)` | Advance the rotation cycle and return the next client (prints the active LLM when verbose). |
| `_drop_provider(reason) -> bool` | Remove the current provider from this turn's pool and rebuild the cycle. Returns `True` if the caller should retry, `False` if the pool is exhausted. |

> `run_turn` intentionally mirrors the tool-call loop in `main()`; the standalone
> REPL is left untouched so its behaviour is identical.

---

## `MCPClient`

**File:** present identically in
[`proxmox_mcp_agent.py`](../agents.d/proxmox_mcp_agent.py),
[`synology_mcp_agent.py`](../agents.d/synology_mcp_agent.py),
[`linux_mcp_agent.py`](../agents.d/linux_mcp_agent.py),
[`homeassistant_mcp_agent.py`](../agents.d/homeassistant_mcp_agent.py),
[`watchyourlan_mcp_agent.py`](../agents.d/watchyourlan_mcp_agent.py).

A minimal **JSON-RPC 2.0 client for the MCP `2024-11-05` protocol**, speaking
over a child process's stdio (`subprocess.Popen`). The orchestrator reuses the
Proxmox copy for all sub-agents (`_proxmox.MCPClient`) because every copy is
identical.

### Construction

`__init__(self, proc: subprocess.Popen)` — stores the process and starts a
daemon **reader thread** (`_read_loop`). Internal state: a `threading.Lock`, a
`_responses` dict (id → message), and an `_events` dict (id → `threading.Event`).

### Methods

| Method | Description |
|--------|-------------|
| `initialize() -> dict` | Send the MCP `initialize` handshake (`protocolVersion 2024-11-05`, client info) and the `notifications/initialized` notification. Returns the server's init result. |
| `list_tools() -> list[dict]` | Call `tools/list`; returns the `tools` array (or `[]`). |
| `call_tool(name, arguments) -> Any` | Call `tools/call` with a 120s timeout; returns the raw MCP result. |
| `close()` | Close stdin and wait up to 5s for the process; `kill()` on failure. |

### Internals

| Method | Description |
|--------|-------------|
| `_read_loop()` | Reader thread: parse one JSON message per stdout line, store it by id, and set the matching event. |
| `_rpc(method, params=None, timeout=60) -> Any` | Send a request with a fresh UUID id, block on its event until the response arrives, raise `TimeoutError` on timeout or `RuntimeError` on a JSON-RPC `error`. |
| `_notify(method, params=None)` | Send a fire-and-forget JSON-RPC notification (no id, no wait). |

---

## NVIDIA rate-limiter classes

**File:** [`nvidia_ratelimit.py`](../nvidia_ratelimit.py)

A cross-process throttle for the NVIDIA NIM free tier (≈40 req/min). NVIDIA
clients are wrapped so that `chat.completions.create()` blocks until a slot is
free under a file-locked sliding window keyed by API key — multiple concurrent
agent processes never collectively exceed the limit. The limit is
`NVIDIA_RPM_LIMIT` (default 38, a safety margin under 40).

### `RateLimitedClient`

A transparent proxy around an `OpenAI` client.

- `__init__(self, client, api_key)` — wraps the real client.
- `chat` (property) → returns a `_RateLimitedChat`.
- `__getattr__(name)` — delegates everything else to the wrapped client, so the
  proxy is a drop-in replacement.

### `_RateLimitedChat`

Proxy for the client's `.chat` namespace. Its `completions` property returns a
`_RateLimitedCompletions`; everything else delegates to the wrapped chat object.

### `_RateLimitedCompletions`

Proxy for `.chat.completions`. Its `create(*args, **kwargs)` calls the
module-level `_acquire(api_key)` (the blocking gate) **before** delegating to the
real `create()`. Other attributes delegate through.

> Helper: `wrap_if_nvidia(provider, client, api_key)` returns a
> `RateLimitedClient` when `provider == "nvidia"`, otherwise the client
> unchanged. Every `build_client()` in the project calls it.

---

## Pydantic request models (HTTP sidecar)

**File:** [`agent_server.py`](../agent_server.py)

OpenAI-compatible request shapes for the FastAPI endpoints. Both subclass
`pydantic.BaseModel`.

### `Message`

One chat message. Fields: `role: str`, `content: Optional[str] = None`.

### `ChatRequest`

The `POST /v1/chat/completions` body. Fields:

- `model: Optional[str]` — defaults to the `MODEL_ID` (`agent/multi-mcp`).
- `messages: list[Message]`.
- `stream: bool = False`.
- `model_config = {"extra": "ignore"}` — unknown OpenAI fields are silently
  dropped, so any client's extra parameters are tolerated.

---

## Note on "no other classes"

The remaining modules are deliberately **function-based**:

- The five sub-agent modules expose `build_client`, `tools_to_openai`,
  `mcp_result_to_text`, `assistant_msg`, `start_docker`/`start_proxy`, and
  `main` as module-level functions (plus `MCPClient` documented above).
- `lab_health_agent.py` and `linux_update_agent.py` are entirely functional
  (scheduler + ReAct loop + Telegram helpers).
- `nvidia_free_models.py` is a CLI script of module-level functions.

See [AGENTS.md](AGENTS.md) for those.
