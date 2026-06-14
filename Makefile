AGENT_HOST  ?= 0.0.0.0
AGENT_PORT  ?= 8910
IMAGE       ?= lordraw/multi-mcp-orchestrator
VERSION     ?= latest

.PHONY: cli agent-orchestrator install-agent-server \
        docker-build docker-up docker-down docker-logs docker-push docker-release

# ── Local (host) ──────────────────────────────────────────────────────────────
# Standalone CLI REPL (unchanged) — uses the local .env (MAIN_AGENT_*).
cli:
	python3 main_agent.py

# Install the extra deps the HTTP sidecar needs (FastAPI + uvicorn).
install-agent-server:
	pip install -r requirements-agent-server.txt

# OpenAI-compatible orchestrator sidecar — same .env, same rotation pool.
# SpiceSibyl reaches it via ORCHESTRATOR_BASE_URL=http://<host>:$(AGENT_PORT)/v1
agent-orchestrator:
	uvicorn agent_server:app --host $(AGENT_HOST) --port $(AGENT_PORT)

# ── Docker (deploy on any server with a Docker daemon) ────────────────────────
docker-build:
	docker build -t $(IMAGE):$(VERSION) .

docker-up:
	docker compose up -d --build

docker-down:
	docker compose down

docker-logs:
	docker compose logs -f

docker-push:
	docker push $(IMAGE):$(VERSION)

# Build, tag latest + version, push — usage: make docker-release VERSION=v1.0.0
docker-release: docker-build
	docker tag $(IMAGE):$(VERSION) $(IMAGE):latest
	docker push $(IMAGE):$(VERSION)
	docker push $(IMAGE):latest
