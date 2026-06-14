# Configuration

All configuration is environment-based, loaded from the project `.env` (copy
[`.env.example`](../.env.example) and fill it in). Each component reads variables
under its own prefix; sub-agents fall back to `MAIN_AGENT_*` where noted.

## Provider selection & rotation (`MAIN_AGENT_*`)

| Variable | Meaning |
|----------|---------|
| `MAIN_AGENT_PROVIDER` | `` / `rotate` / `auto` → rotate across all providers (default). A provider name → pin to it. `ollama` → local Ollama. |
| `MAIN_AGENT_VERBOSE` | `true`/`false` — print the active LLM before each call. |
| `MAIN_AGENT_ROTATE_KEYS` | Restrict rotation to specific 1-based key slots, e.g. `1` or `1,3`. Empty → all slots. |
| `MAIN_AGENT_MODEL` | Global model override applied across providers (sub-agents also honour it). |

### Supported providers

`openrouter, groq, gemini, cloudflare, cerebras, mistral, nvidia, puter,
ollama`. For each cloud provider, set its API key (and model, optionally):

```ini
MAIN_AGENT_OPENROUTER_API_KEY=...
MAIN_AGENT_GROQ_API_KEY=...
MAIN_AGENT_GEMINI_API_KEY=...
MAIN_AGENT_CEREBRAS_API_KEY=...
MAIN_AGENT_MISTRAL_API_KEY=...
MAIN_AGENT_NVIDIA_API_KEY=...
MAIN_AGENT_PUTER_API_KEY=...
MAIN_AGENT_CLOUDFLARE_API_KEY=...
MAIN_AGENT_CLOUDFLARE_ACCOUNT_ID=...      # Cloudflare needs both
MAIN_AGENT_OLLAMA_HOST=http://localhost:11434
```

**Multiple keys per provider** for higher throughput / failover: append
`_2`, `_3`, … (e.g. `MAIN_AGENT_GROQ_API_KEY_2`). For Cloudflare also add the
matching `MAIN_AGENT_CLOUDFLARE_ACCOUNT_ID_2`.

**Ollama dual-model** (optional): `MAIN_AGENT_OLLAMA_CHAT_MODEL` runs final
answer synthesis on a separate (non-tool) model.

### Per-provider defaults

Each module's `PROVIDERS` registry carries sensible default models
(e.g. Groq → `llama-3.3-70b-versatile`, Gemini → `gemini-2.0-flash`, NVIDIA →
`meta/llama-3.3-70b-instruct`). Resolution order is:
**`MAIN_AGENT_MODEL` (global) → per-provider model var → hardcoded default.**

## Sub-agent configuration

### Proxmox (`PROXMOX_MCP_*`)

```ini
PROXMOX_MCP_HOST=proxmox
PROXMOX_MCP_PORT=8006
PROXMOX_MCP_USER=root@pam
PROXMOX_MCP_PASSWORD=...           # or PROXMOX_MCP_TOKEN_ID / _TOKEN_SECRET
PROXMOX_MCP_VERIFY_SSL=false
PROXMOX_MCP_DOCKER_IMAGE=lordraw/proxmox-mcp:latest
```

### Synology (`SYNOLOGY_MCP_*`)

Either point to a config file **or** set individual credentials:

```ini
SYNOLOGY_MCP_NAS_CONFIG=nas_config.json   # bind-mounted into the container
# — or —
SYNOLOGY_MCP_HOST=...
SYNOLOGY_MCP_PORT=5001
SYNOLOGY_MCP_USER=...
SYNOLOGY_MCP_PASSWORD=...
SYNOLOGY_MCP_DOCKER_IMAGE=lordraw/synology-mcp:latest
```

### Linux SSH fleet (`UXMCP_*`)

One numbered block per server (`SERVER_1`, `SERVER_2`, …):

```ini
UXMCP_SERVER_1_LABEL=web01
UXMCP_SERVER_1_HOST=192.168.1.10
UXMCP_SERVER_1_PORT=22
UXMCP_SERVER_1_USER=root
UXMCP_SERVER_1_PASSWORD=...               # or _KEY_PATH for key auth, _SUDO_PASSWORD
UXMCP_MCP_SSH_KEY_DIR=/opt/multi-mcp/.ssh # mounted read-only at /root/.ssh
UXMCP_MCP_DOCKER_IMAGE=lordraw/linux-mcp:latest
```

### Home Assistant (`HAOS_MCP_*`)

```ini
HAOS_MCP_URL=http://192.168.1.x:8123
HAOS_MCP_TOKEN=<long-lived access token>
```

### WatchYourLAN (`WYLA_MCP_*`)

```ini
WYLA_MCP_URL=http://192.168.0.x:8840
WYLA_MCP_TIMEOUT=                          # optional
WYLA_MCP_DOCKER_IMAGE=lordraw/watchyourlan-mcp:latest
```

## Autonomous agents

### Telegram (shared by both)

```ini
TELEGRAM_BOT_TOKEN=...
TELEGRAM_ALLOWED_USERS=123456789,987654321  # numeric chat ids, comma-separated
```

### Lab Health Agent (`LAB_HEALTH_*`)

| Variable | Default | Meaning |
|----------|---------|---------|
| `LAB_HEALTH_INTERVAL_HOURS` | `12` | Run every N hours (also at startup). |
| `LAB_HEALTH_SCHEDULE_TIMES` | — | `HH:MM,HH:MM` daily times (overrides interval). |
| `LAB_HEALTH_PROVIDER` | — | `ollama` / a provider name / `rotate`. |
| `LAB_HEALTH_MODEL` | — | Model override. |
| `LAB_HEALTH_MAX_TOKENS` | unlimited | Cap tokens per LLM call. |
| `LAB_HEALTH_MAX_TOOL_RESULT_CHARS` | unlimited | Truncate tool results. |

### Linux Update Agent (`LINUX_UPDATE_*`)

| Variable | Default | Meaning |
|----------|---------|---------|
| `LINUX_UPDATE_INTERVAL_HOURS` | `24` | Run every N hours (also at startup). |
| `LINUX_UPDATE_SCHEDULE_TIMES` | — | `HH:MM,…` daily times (overrides interval). |
| `LINUX_UPDATE_DRY_RUN` | `false` | Stop after CHECK; report only, install nothing. |
| `LINUX_UPDATE_EXCLUDE_SERVERS` | — | Comma-separated server labels to skip. |
| `LINUX_UPDATE_PROVIDER` / `_MODEL` | — | Provider / model override. |
| `LINUX_UPDATE_MAX_TOKENS` / `_MAX_TOOL_RESULT_CHARS` | unlimited | Same as above. |

## NVIDIA rate limiter

| Variable | Default | Meaning |
|----------|---------|---------|
| `NVIDIA_RPM_LIMIT` | `38` | Requests/min cap for the NVIDIA NIM free tier (safety margin under 40). |

## Secrets handling

`.env`, `nas_config.json`, and `.ssh/` are **never** baked into images
(`.dockerignore` excludes them); they are provided at runtime via `env_file` and
volumes. See [DOCKER.md](DOCKER.md).
