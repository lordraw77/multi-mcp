#!/usr/bin/env python3
"""OpenAI-compatible HTTP wrapper around the Multi-MCP orchestrator.

Exposes ``POST /v1/chat/completions`` (and ``GET /v1/models``) backed by the
exact same :class:`main_agent.Orchestrator` — same ``.env`` (``MAIN_AGENT_*``),
same provider rotation pool, same Docker MCP sub-agents the CLI uses.

Run it on the host, next to the CLI::

    cd /opt/multi-mcp
    uvicorn agent_server:app --host 0.0.0.0 --port 8910

Then register it in SpiceSibyl as the ``agent/multi-mcp`` model
(``ORCHESTRATOR_BASE_URL=http://host.docker.internal:8910/v1``) so it is
reachable from both the web console and Telegram with no channel-specific code.

The orchestrator is synchronous and slow (it spawns Docker containers and may
chain several sub-agent calls), so every turn runs in a worker thread and turns
are serialized by the Orchestrator's own lock.
"""

import asyncio
import json
import threading
import time
import uuid
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

import main_agent

MODEL_ID = "agent/multi-mcp"

app = FastAPI(title="Multi-MCP Orchestrator", version="1.0")

_orch: Optional[main_agent.Orchestrator] = None


def get_orch() -> main_agent.Orchestrator:
    """Lazily build the shared Orchestrator (reads the process .env once)."""
    global _orch
    if _orch is None:
        _orch = main_agent.Orchestrator()
    return _orch


# ── OpenAI-compatible request shapes (extra fields ignored) ───────────────────
class Message(BaseModel):
    role: str
    content: Optional[str] = None


class ChatRequest(BaseModel):
    model: Optional[str] = MODEL_ID
    messages: list[Message]
    stream: bool = False

    model_config = {"extra": "ignore"}


def _prepare(messages: list[Message]) -> list[dict]:
    """Map to plain dicts and guarantee the orchestrator system prompt leads.

    The orchestrator decides which sub-agents to call, so its own system prompt
    must drive the turn; any system prompt sent by the gateway is dropped.
    """
    msgs = [
        {"role": m.role, "content": m.content or ""}
        for m in messages
        if m.role in ("user", "assistant", "tool")
    ]
    msgs.insert(0, {"role": "system", "content": get_orch().system_prompt})
    return msgs


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "model": MODEL_ID}


@app.get("/v1/models")
async def list_models() -> dict:
    return {
        "object": "list",
        "data": [{"id": MODEL_ID, "object": "model", "owned_by": "multi-mcp"}],
    }


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    orch = get_orch()
    messages = _prepare(req.messages)
    cid = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())
    model = req.model or MODEL_ID

    if not req.stream:
        try:
            answer = await asyncio.to_thread(orch.run_turn, messages)
        except Exception as exc:  # noqa: BLE001 — surface any orchestrator failure
            raise HTTPException(status_code=502, detail=str(exc))
        return {
            "id": cid,
            "object": "chat.completion",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": answer},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }

    async def event_stream():
        # Stream progress as the orchestrator delegates: on_step (called from the
        # worker thread) pushes control frames into an asyncio.Queue via the loop;
        # the final answer arrives as a normal content chunk.
        #
        # Frame protocol (all `data: <json>`):
        #   {"_sse_event": "tool_call",   "id", "name", "arguments": {...}}
        #   {"_sse_event": "tool_result", "id", "name", "result"}
        #   {"object": "chat.completion.chunk", "choices": [{"delta": {"content"}}]}
        # SpiceSibyl maps `_sse_event` frames to named SSE events (web bubbles);
        # the Telegram bot turns them into progressive status edits.
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()

        def on_step(ev: dict) -> None:
            if ev.get("type") == "tool_call":
                frame = {
                    "_sse_event": "tool_call",
                    "id": ev.get("id", ""),
                    "name": ev.get("name", ""),
                    "arguments": {"query": ev.get("query", "")},
                }
            else:
                frame = {
                    "_sse_event": "tool_result",
                    "id": ev.get("id", ""),
                    "name": ev.get("name", ""),
                    "result": ev.get("result", ""),
                }
            loop.call_soon_threadsafe(queue.put_nowait, ("step", frame))

        def run() -> None:
            try:
                answer = orch.run_turn(messages, on_step=on_step)
                loop.call_soon_threadsafe(queue.put_nowait, ("final", answer))
            except Exception as exc:  # noqa: BLE001
                loop.call_soon_threadsafe(queue.put_nowait, ("error", str(exc)))

        threading.Thread(target=run, daemon=True).start()

        while True:
            kind, payload = await queue.get()

            if kind == "step":
                yield f"data: {json.dumps(payload)}\n\n"
                continue

            if kind == "error":
                yield f"data: {json.dumps({'error': {'message': payload}})}\n\n"
                yield "data: [DONE]\n\n"
                return

            # kind == "final": emit the answer content, a light meta chunk, then DONE.
            content_chunk = {
                "id": cid,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": {"content": payload}, "finish_reason": None}],
            }
            meta_chunk = {
                "id": f"meta-{created}",
                "object": "chat.completion.meta",
                "created": created,
                "model": model,
                "choices": [{
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": payload,
                        "model": model,
                        "provider": "multi-mcp",
                        "finish_reason": "stop",
                        "created_at": created,
                        "capabilities": ["chat", "tools", "agent"],
                    },
                }],
                "usage": {},
                "metrics": {},
            }
            yield f"data: {json.dumps(content_chunk)}\n\n"
            yield f"data: {json.dumps(meta_chunk)}\n\n"
            yield "data: [DONE]\n\n"
            return

    return StreamingResponse(event_stream(), media_type="text/event-stream")
