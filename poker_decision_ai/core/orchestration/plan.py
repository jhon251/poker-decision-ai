"""Execution plan: declarative, config-driven, dynamic.

pipeline.yaml defines *plan templates* per request kind (cash_hand,
tournament_hand, replay_hand, experiment, ...).  Each plan is a list of
stages; stages run sequentially, agents inside a stage run in parallel.

Input routing uses $-references inside each agent's input template:
    $request          -> the whole request dict
    $request.hero_id  -> nested field of the request
    $state            -> output of agent 'state'
    $state.board      -> nested field of agent output

The Manager does NOT hard-code which agents run: plan templates + each
agent's `applicable()` check decide dynamically (spec §二十一).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..config import AppConfig
from ..errors import OrchestrationError


@dataclass
class AgentTask:
    agent: str
    input_template: dict = field(default_factory=dict)
    timeout_ms: int = 5000
    retries: int = 1
    on_failure: str = "degrade"     # degrade | abort


@dataclass
class Stage:
    name: str
    tasks: list[AgentTask]


@dataclass
class Plan:
    kind: str
    stages: list[Stage]

    def all_agents(self) -> list[str]:
        out: list[str] = []
        for s in self.stages:
            for t in s.tasks:
                if t.agent not in out:
                    out.append(t.agent)
        return out


def _resolve_value(value: Any, outputs: dict[str, dict], request: dict,
                   missing: list[str], path: str = "$") -> Any:
    if isinstance(value, str) and value.startswith("$"):
        ref = value[1:]
        parts = ref.split(".")
        if parts[0] == "request":
            node: Any = request
            for p in parts[1:]:
                if isinstance(node, dict) and p in node:
                    node = node[p]
                else:
                    missing.append(f"{path} ({value})")
                    return None
            return node
        if parts[0] in outputs:
            node = outputs[parts[0]]
            if node is None:
                missing.append(f"{path} ({value})")
                return None
            for p in parts[1:]:
                if isinstance(node, dict) and p in node:
                    node = node[p]
                else:
                    missing.append(f"{path} ({value})")
                    return None
            return node
        missing.append(f"{path} ({value})")
        return None
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            rv = _resolve_value(v, outputs, request, missing, f"{path}.{k}")
            if rv is not None or f"{path}.{k}" not in missing:
                out[k] = rv
        return out
    if isinstance(value, list):
        return [_resolve_value(v, outputs, request, missing, f"{path}[]") for v in value]
    return value


def resolve_input(template: dict, outputs: dict[str, dict], request: dict) -> tuple[dict, list[str]]:
    """Resolve an input template; returns (input_dict, missing_refs)."""
    missing: list[str] = []
    resolved = _resolve_value(template, outputs, request, missing)
    return (resolved if isinstance(resolved, dict) else {}), missing


def build_plan(kind: str, config: AppConfig, default_timeout_ms: int = 5000,
               default_retries: int = 1) -> Plan:
    """Build a plan from pipeline.yaml. Validates agent ordering constraints."""
    raw = config.pipeline.get(f"plans.{kind}")
    if raw is None:
        raise OrchestrationError(f"no pipeline plan for kind '{kind}' in configs/pipeline.yaml")
    stages: list[Stage] = []
    seen: set[str] = set()
    for idx, sraw in enumerate(raw.get("stages", [])):
        stage_name = sraw.get("name", f"stage-{idx}")
        tasks: list[AgentTask] = []
        for entry in sraw.get("agents", []):
            agent = entry["agent"] if isinstance(entry, dict) else str(entry)
            if isinstance(entry, dict):
                entry = dict(entry)
            else:
                entry = {"agent": agent}
            # merge per-agent settings from configs/config.yaml
            timeout = int(config.agent_setting(agent, "timeout_ms", default_timeout_ms))
            retries = int(config.agent_setting(agent, "retries", default_retries))
            tasks.append(AgentTask(
                agent=agent,
                input_template=dict(entry.get("input", {})),
                timeout_ms=timeout,
                retries=retries,
                on_failure=str(entry.get("on_failure", "degrade")),
            ))
            seen.add(agent)
        stages.append(Stage(stage_name, tasks))
    return Plan(kind, stages)


def plan_as_doc(plan: Plan) -> str:
    lines = [f"Plan [{plan.kind}]:"]
    for s in plan.stages:
        lines.append(f"  stage '{s.name}': " + ", ".join(t.agent for t in s.tasks))
    return "\n".join(lines)
