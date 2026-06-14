# Docker overview

The project ships three Docker deliverables, all built from the same source tree
and the same project `.env`:

| Image / stack | Source | Purpose |
|---------------|--------|---------|
| `lordraw/multi-mcp-orchestrator` | [`Dockerfile`](../Dockerfile) + [`docker-compose.yml`](../docker-compose.yml) | The OpenAI-compatible HTTP sidecar (`agent_server.py`), port **8910**. |
| `multi-mcp/lab-health-agent` | [`docker/lab_health_agent/`](../docker/lab_health_agent/) | The autonomous read-only health checker. |
| `multi-mcp/linux-update-agent` | [`docker/linux_update_agent/`](../docker/linux_update_agent/) | The autonomous fleet updater. |

> The MCP **servers** themselves (`lordraw/proxmox-mcp`, `synology-mcp`,
> `linux-mcp`, `watchyourlan-mcp`) are **not** built here — they are pulled from
> Docker Hub and launched at runtime by the agents (see DooD below).

## The core idea: Docker-out-of-Docker (DooD)

None of these containers bundle the MCP servers. Instead they mount the **host**
Docker socket and launch the MCP servers with `docker run --rm -i` on the host
daemon — so the MCP servers run as **sibling** containers, exactly as the
standalone CLI does on bare metal.

```
caller ──HTTP/schedule──► agent container ──/var/run/docker.sock──► host dockerd
                                                                        │
                                        ┌───────────────────────────────┤
                                        ▼          ▼          ▼          ▼
                                  proxmox-mcp  synology-mcp linux-mcp  watchyourlan-mcp
                                       (sibling containers, --rm -i, MCP over stdio)
```

**Consequences of DooD (important):**

- Bind-mounts in the spawned `docker run` are resolved by the **host** daemon,
  and the sub-agents compute the source path from the code dir `/opt/multi-mcp`.
  So files like `nas_config.json` or the `.ssh` dir must exist on the **host** at
  the **same absolute path** that the container uses, and the project must be
  mounted at `/opt/multi-mcp` inside the agent container.
- The agent container runs as **root** to access the Docker socket. There is no
  built-in auth — restrict who can reach port 8910 (front it with SpiceSibyl or a
  reverse proxy).

**Home Assistant is the exception:** it does *not* spawn a sibling container. It
runs `uvx mcp-proxy` **inside** the agent container to bridge HA's HTTP MCP
endpoint to stdio, so the container needs outbound network to the HA URL and to
fetch `mcp-proxy` on first use. `uv` is preinstalled in the orchestrator image.

## The orchestrator image

[`Dockerfile`](../Dockerfile) (`python:3.12-slim`):

- Installs the static **x86_64 Docker CLI** (change the arch for arm64 hosts),
  CA certs, tzdata, and `uv`/`uvx`.
- Installs `openai`, `python-dotenv`, `fastapi`, `uvicorn[standard]`.
- `WORKDIR /opt/multi-mcp`, copies `*.py` only — **secrets are never baked in**
  (`.dockerignore` excludes `.env`, `nas_config.json`, `.ssh`).
- `CMD uvicorn agent_server:app --host 0.0.0.0 --port 8910`.

[`docker-compose.yml`](../docker-compose.yml):

- `env_file: .env` — all `MAIN_AGENT_*`, `PROXMOX_MCP_*`, `SYNOLOGY_*`, `UXMCP_*`,
  `HAOS_*`, `WYLA_*`.
- Mounts `/var/run/docker.sock` (required for DooD).
- Optional, commented mounts for file-based sub-agent config (Synology
  `nas_config.json`, Linux `.ssh`) — uncomment only what you use, and keep host
  paths matching.
- A `/health` healthcheck.

### Run it

```bash
make docker-up        # docker compose up -d --build  → port 8910
make docker-logs      # follow logs
curl http://localhost:8910/health
```

Or publish/pull a prebuilt image:

```bash
make docker-release VERSION=v1.0.0   # build, tag latest+version, push
```

The [`Makefile`](../Makefile) also offers local targets: `cli` (run the REPL),
`install-agent-server`, and `agent-orchestrator` (run uvicorn on the host).

Full deployment walkthrough — including wiring SpiceSibyl via
`ORCHESTRATOR_BASE_URL` — is in [DEPLOY.md](../DEPLOY.md).

## The autonomous-agent stacks

Each lives under `docker/<agent>/` with its own `Dockerfile` + `docker-compose.yml`,
built from the **project root** as context (`context: ../..`). They:

- install `openai`, `requests`, `python-dotenv` and the Docker CLI,
- use the project `.env` (`env_file: ../../.env`),
- mount the Docker socket, the project at `/opt/multi-mcp`, and `~/.ssh`
  read-only (for the Linux MCP),
- set `TZ=Europe/Rome` and a `…_SCHEDULE_TIMES` override so they run at fixed
  daily times rather than at startup + interval.

```bash
# Health checker (default schedule 06:00 & 19:00 Rome time)
docker compose -f docker/lab_health_agent/docker-compose.yml up -d --build

# Fleet updater (default schedule 12:00 Rome time)
docker compose -f docker/linux_update_agent/docker-compose.yml up -d --build
```

Adjust the schedule, dry-run, exclusions, and provider via the env vars in
[CONFIGURATION.md](CONFIGURATION.md).

## Prerequisites on the host

- Docker Engine + Docker Compose.
- Outbound network to: the LLM providers, Docker Hub (to pull `lordraw/*-mcp`),
  and your Proxmox / Synology / HA / WatchYourLAN hosts.
- A filled-in `.env`.
- For arm64 hosts: change the Docker CLI download arch in the `Dockerfile`s.
