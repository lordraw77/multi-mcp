#!/usr/bin/env python3
"""Template for a fully externalized sub-agent.

Drop a ``.py`` like this next to a ``.json`` in the agents dir (bind-mounted in
Docker) to add an agent without touching the orchestrator or rebuilding the
image. Reference it from the JSON with ``"module": "example_agent"`` (the agents
dir is on sys.path) or ``"module_path": "example_agent.py"``.

The orchestrator only needs the launcher plus the four MCP helpers below — all
imported from mcp_common, so there is no boilerplate to copy.
"""

import os

from mcp_common import (  # noqa: F401 — re-exported so the orchestrator finds them
    MCPClient,
    tools_to_openai,
    mcp_result_to_text,
    assistant_msg,
    docker_start,
)


def start_docker():
    """Launch the agent's MCP server container over stdio.

    Map your own env vars to whatever the container expects. Keys with empty
    values are skipped automatically by docker_start()."""
    return docker_start(
        os.getenv("EXAMPLE_MCP_DOCKER_IMAGE", "me/example-mcp:latest"),
        env={
            "EXAMPLE_HOST": os.getenv("EXAMPLE_MCP_HOST", ""),
            "EXAMPLE_TOKEN": os.getenv("EXAMPLE_MCP_TOKEN", ""),
        },
    )
