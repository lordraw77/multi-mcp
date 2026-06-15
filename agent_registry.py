#!/usr/bin/env python3
"""Dynamic sub-agent registry for the Multi-MCP Orchestrator.

Each sub-agent is described by one JSON file in a directory (default ``agents.d/``
next to this module, overridable via ``MAIN_AGENT_AGENTS_DIR``). The orchestrator
discovers, imports and registers them at runtime, so agents can be added or
removed by dropping a file in — no code change, no image rebuild (bind-mount the
folder in Docker).

JSON schema (see ``agents.d/*.json``)::

    {
      "name": "proxmox",
      "order": 10,                 # sort key (then filename)
      "enabled": true,             # false → skipped (also: *.json.disabled)
      "tool_name": "ask_proxmox",  # OpenAI tool/function name
      "module": "proxmox_mcp_agent",
      "start_fn": "start_docker",  # launcher in that module (e.g. start_proxy)
      "summary": "...",            # one-liner for the orchestrator system prompt
      "tool_description": "...",   # tool description shown to the LLM
      "system_prompt": "...",      # sub-agent system prompt
      "status_line": "...",        # CLI banner line
      "max_tokens": null,
      "max_tool_result_chars": null
    }

Placeholders resolved in ``system_prompt`` and ``status_line``:
    {env:VAR}          → os.getenv("VAR", "")
    {env:VAR:default}  → os.getenv("VAR", "default")
    {ctx:key}          → module.prompt_context()[key] (if the module exposes it)
"""

import os
import re
import sys
import json
import importlib
import importlib.util
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

_PLACEHOLDER = re.compile(r"\{(env|ctx):([^}:]+)(?::([^}]*))?\}")


def _resolve(text: str, module: Any, _ctx_cache: dict) -> str:
    """Substitute {env:...} / {ctx:...} placeholders in ``text``."""
    def repl(m: re.Match) -> str:
        kind, key, default = m.group(1), m.group(2), m.group(3)
        if kind == "env":
            return os.getenv(key, default if default is not None else "")
        # ctx: pull from the module's prompt_context() (computed once per module)
        if "ctx" not in _ctx_cache:
            fn = getattr(module, "prompt_context", None)
            _ctx_cache["ctx"] = fn() if callable(fn) else {}
        return str(_ctx_cache["ctx"].get(key, default if default is not None else ""))

    return _PLACEHOLDER.sub(repl, text or "")


def _tool_def(tool_name: str, description: str, query_description: str) -> dict:
    """Build the OpenAI tool/function definition (single ``query`` parameter)."""
    return {
        "type": "function",
        "function": {
            "name": tool_name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": query_description,
                    }
                },
                "required": ["query"],
            },
        },
    }


@dataclass
class AgentSpec:
    name: str
    tool_name: str
    module: Any
    start_fn: Callable[[], Any]
    system_prompt: str
    tool_description: str
    summary: str
    status_line: str
    max_tokens: Optional[int]
    max_tool_result_chars: Optional[int]
    tool_def: dict = field(default_factory=dict)
    handler: Optional[Callable] = None


def make_handler(spec: AgentSpec, run_query: Callable) -> Callable:
    """Wrap ``run_query`` into the handler signature the orchestrator expects."""
    def handler(
        query: str,
        next_client: Callable,
        drop_provider: Optional[Callable] = None,
        max_tokens: Optional[int] = None,
        max_tool_result_chars: Optional[int] = None,
    ) -> str:
        return run_query(
            spec.name,
            spec.module,
            spec.start_fn,
            next_client,
            spec.system_prompt,
            query,
            drop_provider=drop_provider,
            max_tokens=max_tokens if max_tokens is not None else spec.max_tokens,
            max_tool_result_chars=(
                max_tool_result_chars
                if max_tool_result_chars is not None
                else spec.max_tool_result_chars
            ),
        )

    return handler


def _default_dir() -> Path:
    override = os.getenv("MAIN_AGENT_AGENTS_DIR", "").strip()
    if override:
        return Path(override)
    return Path(__file__).parent / "agents.d"


def _import_agent_module(cfg: dict, directory: Path):
    """Import the agent's Python module.

    ``module_path`` (a file, absolute or relative to the agents dir) loads an
    externalized ``.py`` that lives next to its JSON — so a whole new agent
    (code + config) can be dropped into the mounted folder with no rebuild.
    Otherwise ``module`` is imported by name; the agents dir is on ``sys.path``
    so a plain ``module: foo_agent`` also resolves to ``agents.d/foo_agent.py``.
    """
    module_path = cfg.get("module_path")
    mod_name = cfg.get("module")
    if module_path:
        path = Path(module_path)
        if not path.is_absolute():
            path = directory / path
        name = mod_name or path.stem
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"impossibile caricare il modulo da {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module
    return importlib.import_module(mod_name)


def load_agents(run_query: Callable, agents_dir: Optional[Path] = None) -> list[AgentSpec]:
    """Discover, import and register every enabled agent under ``agents_dir``.

    ``run_query`` is :func:`main_agent.run_mcp_query` (passed in to avoid a
    circular import). Malformed files or unresolvable modules are skipped with a
    warning so one bad agent never takes down the orchestrator.
    """
    directory = Path(agents_dir) if agents_dir else _default_dir()
    if not directory.is_dir():
        print(f"[warn] agents dir non trovata: {directory} — nessun agente caricato")
        return []

    # Make externalized agent .py files (dropped next to their JSON) importable.
    dir_str = str(directory.resolve())
    if dir_str not in sys.path:
        sys.path.insert(0, dir_str)

    raw: list[tuple[int, str, dict]] = []
    for path in directory.glob("*.json"):
        try:
            with path.open(encoding="utf-8") as fh:
                cfg = json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"[warn] {path.name} ignorato (JSON non valido): {exc}")
            continue
        if not cfg.get("enabled", True):
            continue
        raw.append((int(cfg.get("order", 100)), path.name, cfg))

    specs: list[AgentSpec] = []
    seen_tools: set[str] = set()
    for _, fname, cfg in sorted(raw, key=lambda t: (t[0], t[1])):
        name = cfg.get("name") or Path(fname).stem
        tool_name = cfg.get("tool_name")
        start_name = cfg.get("start_fn", "start_docker")
        if not tool_name or not (cfg.get("module") or cfg.get("module_path")):
            print(f"[warn] {fname} ignorato: servono 'tool_name' e 'module' (o 'module_path')")
            continue
        if tool_name in seen_tools:
            print(f"[warn] {fname} ignorato: tool_name duplicato '{tool_name}'")
            continue
        try:
            module = _import_agent_module(cfg, directory)
        except Exception as exc:
            ref = cfg.get("module_path") or cfg.get("module")
            print(f"[warn] {fname} ignorato: modulo '{ref}' non importabile: {exc}")
            continue
        start_fn = getattr(module, start_name, None)
        if not callable(start_fn):
            print(f"[warn] {fname} ignorato: '{module.__name__}.{start_name}' non trovato")
            continue

        ctx_cache: dict = {}
        spec = AgentSpec(
            name=name,
            tool_name=tool_name,
            module=module,
            start_fn=start_fn,
            system_prompt=_resolve(cfg.get("system_prompt", ""), module, ctx_cache),
            tool_description=cfg.get("tool_description", ""),
            summary=cfg.get("summary", ""),
            status_line=_resolve(cfg.get("status_line", f"{name} : configured"), module, ctx_cache),
            max_tokens=cfg.get("max_tokens"),
            max_tool_result_chars=cfg.get("max_tool_result_chars"),
        )
        query_desc = cfg.get(
            "query_description", f"The full task or question for the {name} agent."
        )
        spec.tool_def = _tool_def(tool_name, spec.tool_description, query_desc)
        spec.handler = make_handler(spec, run_query)
        specs.append(spec)
        seen_tools.add(tool_name)

    return specs


def build_system_prompt(agents: list[AgentSpec]) -> str:
    """Assemble the orchestrator system prompt from the registered agents."""
    width = max((len(a.tool_name) for a in agents), default=0) + 1
    bullets = "\n".join(f"• {a.tool_name:<{width}}— {a.summary}" for a in agents)
    return (
        f"You are an infrastructure and smart home orchestrator with access to "
        f"{len(agents)} specialized agents:\n"
        f"{bullets}\n\n"
        "Analyze each user request and delegate to the appropriate agent(s). "
        "You may call multiple agents in sequence when a task spans domains. "
        "Present the agents' responses clearly and concisely."
    )
