# Multi-MCP Orchestrator

A multi-agent orchestrator that routes natural-language requests to a fleet of
specialized **MCP (Model Context Protocol)** sub-agents — Proxmox VE, Synology
NAS, a Linux SSH fleet, Home Assistant, and WatchYourLAN — using any of nine
LLM providers with automatic key/provider rotation and failover.

It runs three ways from the **same** `.env` and the **same** provider pool:

| Mode | Entry point | Use |
|------|-------------|-----|
| **CLI REPL** | `python3 main_agent.py` | Interactive terminal chat |
| **HTTP sidecar** | `uvicorn agent_server:app --port 8910` | OpenAI-compatible API (web / Telegram via SpiceSibyl) |
| **Autonomous agents** | `lab_health_agent.py`, `linux_update_agent.py` | Scheduled, unattended health checks & fleet updates |

## How it works

```
                      ┌──────────────────────────────────────┐
   user / web / TG ──►│   Orchestrator LLM (rotation pool)    │
                      │   main_agent.Orchestrator             │
                      └───────────────┬──────────────────────┘
                                      │ tool-calls (ask_*)
        ┌──────────────┬──────────────┼──────────────┬──────────────┐
        ▼              ▼              ▼              ▼              ▼
   ask_proxmox   ask_synology    ask_linux   ask_homeassistant ask_watchyourlan
        │              │              │              │              │
        ▼              ▼              ▼              ▼              ▼
   proxmox-mcp   synology-mcp    linux-mcp   uvx mcp-proxy   watchyourlan-mcp
   (docker)      (docker)        (docker)    (HA HTTP→stdio) (docker)
        └──────────────┴──────────────┴──────────────┴──────────────┘
                        MCP servers — JSON-RPC over stdio
```

1. The **orchestrator LLM** receives the user request and the five `ask_*` tool
   definitions ([`ORCHESTRATOR_TOOLS`](main_agent.py)).
2. It decides which sub-agent(s) to delegate to and emits tool calls.
3. Each `ask_*` handler spawns the matching **MCP server** (a sibling Docker
   container, or `uvx mcp-proxy` for Home Assistant), runs an inner agentic
   tool-call loop against that server's MCP tools, and returns the final text.
4. The orchestrator synthesises the sub-agent answers into one reply.

Every LLM call — orchestrator and sub-agent alike — draws from a **shared
rotation pool**, so a `429` or a blocked provider transparently fails over to
the next available key.

## Documentation

- **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** — components, request flow, provider rotation, the MCP protocol layer.
- **[docs/AGENTS.md](docs/AGENTS.md)** — every module and what it does (orchestrator, each sub-agent, the autonomous agents, NVIDIA rate limiter).
- **[docs/CLASSES.md](docs/CLASSES.md)** — reference for every class in the codebase.
- **[docs/CONFIGURATION.md](docs/CONFIGURATION.md)** — environment variables and provider setup.
- **[docs/DOCKER.md](docs/DOCKER.md)** — Docker overview: the orchestrator image, Docker-out-of-Docker, and the autonomous-agent compose stacks.
- **[DEPLOY.md](DEPLOY.md)** — step-by-step deployment of the HTTP sidecar.

## Quick start

```bash
git clone <repo> /opt/multi-mcp && cd /opt/multi-mcp
cp .env.example .env && $EDITOR .env     # fill in provider keys + sub-agent hosts
pip install openai python-dotenv requests

python3 main_agent.py                     # interactive CLI
```

To run the OpenAI-compatible HTTP API:

```bash
pip install -r requirements-agent-server.txt
uvicorn agent_server:app --host 0.0.0.0 --port 8910
curl http://localhost:8910/health         # {"status":"ok","model":"agent/multi-mcp"}
```

Or run everything in Docker — see **[docs/DOCKER.md](docs/DOCKER.md)**.

## Requirements

- Python 3.10+ (the orchestrator image uses 3.12).
- Docker Engine — the sub-agents launch their MCP servers as containers.
- For Home Assistant: `uv`/`uvx` (preinstalled in the image) and Python ≥ 3.10.
- At least one LLM provider API key (see [docs/CONFIGURATION.md](docs/CONFIGURATION.md)).
