# Deploying the Multi-MCP Orchestrator (Docker)

The orchestrator runs as an OpenAI-compatible HTTP service (`agent_server.py`,
port **8910**) backed by `main_agent.py` — same `.env`, same provider rotation,
same MCP sub-agents as the standalone CLI. SpiceSibyl reaches it through
`ORCHESTRATOR_BASE_URL` so it's usable from the web console and Telegram.

## How it works: Docker-out-of-Docker (DooD)

The orchestrator does **not** bundle the MCP servers. It launches them on the
**host** Docker daemon with `docker run --rm -i lordraw/*-mcp` and talks to them
over stdio — they run as *sibling* containers, exactly like the CLI does today.

```
SpiceSibyl ──HTTP──► orchestrator container ──/var/run/docker.sock──► host dockerd
                                                                          │
                                          ┌───────────────────────────────┤
                                          ▼            ▼           ▼       ▼
                                    proxmox-mcp   synology-mcp  linux-mcp  watchyourlan-mcp
                                    (sibling containers, --rm -i, MCP over stdio)
```

Home Assistant is different: it uses `uvx mcp-proxy` **inside** the orchestrator
container (no sibling container), so the container needs outbound network to the
HA URL and to fetch `mcp-proxy` on first use. `uv` is preinstalled in the image.

## Prerequisites on the target server

- Docker Engine + Docker Compose.
- Outbound network to: the LLM providers (rotation pool), Docker Hub (to pull the
  `lordraw/*-mcp` images), and your Proxmox / Synology / HA / WatchYourLAN hosts.
- A `.env` (copy from `.env.example`, fill `MAIN_AGENT_*` keys + sub-agent config).

## Quick start

```bash
# on the target server
git clone <repo> /opt/multi-mcp && cd /opt/multi-mcp
cp .env.example .env && $EDITOR .env      # fill in keys + hosts

make docker-up                            # build + run, port 8910
make docker-logs                          # follow logs
curl http://localhost:8910/health         # {"status":"ok",...}
```

Or pull a prebuilt image instead of building:

```bash
# publisher (once)
make docker-release VERSION=v1.0.0
# target server: set `image:` in docker-compose.yml and `docker compose up -d`
```

## ⚠️ File-based sub-agent config (host paths must match)

DooD bind-mounts are resolved by the **host** daemon, and the sub-agents derive
the source path from the code dir `/opt/multi-mcp`. So these files must exist on
the **host** at the same absolute path, and be mounted into the container at that
same path. Uncomment the matching lines in `docker-compose.yml`:

| Sub-agent | `.env` setting | Host file (same path in container) |
|-----------|----------------|------------------------------------|
| Synology (nas_config) | `SYNOLOGY_MCP_NAS_CONFIG=nas_config.json` | `/opt/multi-mcp/nas_config.json` |
| Linux SSH keys        | `UXMCP_MCP_SSH_KEY_DIR=/opt/multi-mcp/.ssh` | `/opt/multi-mcp/.ssh/` |

Proxmox and WatchYourLAN pass only `-e` env vars (no bind-mounts) — nothing extra.
If you don't use a sub-agent, leave its mount commented out.

## Wiring SpiceSibyl to this deployment

In SpiceSibyl's `backend/.env`:

```ini
# same server as the orchestrator:
ORCHESTRATOR_BASE_URL=http://host.docker.internal:8910/v1
# orchestrator on a different host:
ORCHESTRATOR_BASE_URL=http://<orchestrator-server-ip>:8910/v1
```

Then select **agent/multi-mcp** in the web model picker, or `/model agent/multi-mcp`
in Telegram.

## Notes

- The image installs the **x86_64** static Docker CLI. For arm64 hosts, change the
  download URL/arch in the `Dockerfile`.
- The container runs as root to access the Docker socket. Restrict who can reach
  port 8910 (it has no auth of its own — front it with SpiceSibyl / a reverse proxy).
- Secrets are never baked into the image (`.dockerignore` excludes `.env`,
  `nas_config.json`, `.ssh`); they are provided at runtime via `env_file` + volumes.
