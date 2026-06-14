# Lab Health Agent — build context: la root del progetto (/opt/multi-mcp)
#   docker compose -f docker/lab_health_agent/docker-compose.yml build
FROM python:3.12-slim

# Docker CLI: i sub-agent MCP vengono lanciati con `docker run` sul daemon
# del host (socket montato dal compose).
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl tzdata \
 && curl -fsSL https://download.docker.com/linux/static/stable/x86_64/docker-27.5.1.tgz \
    | tar -xz --strip-components=1 -C /usr/local/bin docker/docker \
 && apt-get purge -y curl \
 && apt-get autoremove -y \
 && rm -rf /var/lib/apt/lists/*

ENV TZ=Europe/Rome
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN pip install --no-cache-dir openai requests python-dotenv

# Stesso path del host: i `-v` generati dai sub-agent (nas_config.json,
# chiavi SSH) devono esistere sul host con lo stesso percorso assoluto.
WORKDIR /opt/multi-mcp
COPY *.py nas_config.json ./

ENV PYTHONUNBUFFERED=1

CMD ["python", "lab_health_agent.py"]
