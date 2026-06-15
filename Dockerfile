# Multi-MCP Orchestrator — OpenAI-compatible HTTP sidecar (agent_server.py).
#
# Build context: the repo root (/opt/multi-mcp).
#   docker build -t lordraw/multi-mcp-orchestrator:latest .
#
# Runtime model: Docker-out-of-Docker. The sub-agents launch their MCP servers
# with `docker run --rm -i lordraw/*-mcp` on the *host* daemon (socket mounted),
# so they run as sibling containers — exactly like the standalone CLI does today.
#
# IMPORTANT: file-based sub-agent config is bind-mounted by the host daemon, so
# the paths must resolve on the host. WORKDIR is /opt/multi-mcp so that:
#   • synology  → -v /opt/multi-mcp/nas_config.json:/config/nas_config.json
#   • linux SSH → -v <UXMCP_MCP_SSH_KEY_DIR>:/root/.ssh
# line up between this container and the host (see docker-compose.yml volumes).
FROM python:3.12-slim

ARG DOCKER_VERSION=27.5.1

# Docker CLI (static) + CA certs + tzdata. `uv`/`uvx` is installed via pip below
# (the Home Assistant sub-agent runs `uvx mcp-proxy`).
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl tzdata \
 && curl -fsSL "https://download.docker.com/linux/static/stable/x86_64/docker-${DOCKER_VERSION}.tgz" \
    | tar -xz --strip-components=1 -C /usr/local/bin docker/docker \
 && apt-get purge -y curl \
 && apt-get autoremove -y \
 && rm -rf /var/lib/apt/lists/*

ENV TZ=Europe/Rome
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Orchestrator deps (openai, dotenv) + HTTP sidecar (fastapi, uvicorn) + uv (uvx).
RUN pip install --no-cache-dir \
        openai python-dotenv \
        fastapi "uvicorn[standard]" \
        uv

WORKDIR /opt/multi-mcp

# Source only — secrets (.env, nas_config.json, .ssh) are provided at runtime
# via docker-compose (env_file + volumes), never baked into the image.
COPY *.py ./

# Sub-agent registry: one JSON per agent (discovered at runtime by agent_registry).
# Baked in as a default; bind-mount agents.d/ via docker-compose to add/remove
# agents without rebuilding the image.
COPY agents.d/ ./agents.d/

ENV PYTHONUNBUFFERED=1
EXPOSE 8910

CMD ["uvicorn", "agent_server:app", "--host", "0.0.0.0", "--port", "8910"]
