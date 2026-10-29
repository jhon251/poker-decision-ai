"""OrchestrationEngine: parallel stage execution with timeout / retry / degrade.

The engine implements the Manager's runtime duties:
  - run stages sequentially, agents within a stage concurrently (asyncio)
  - per-agent timeout (asyncio.wait_for)
  - per-agent retry with backoff (retry_async)
  - graceful degradation: failures are recorded and downstream agents are
    told about missing dependencies (empty inputs + warnings) so the pipeline
    keeps running instead of dying (spec §十四 failure tests)
  - on_failure=abort raises OrchestrationError for critical agents
  - every outcome (ok/degraded/skipped/failed) is logged + metered
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from ..errors import AgentTimeoutError, OrchestrationError
from ..schemas import AgentEnvelope
from .context import AgentContext
from .plan import Plan, resolve_input
from .registry import AgentRegistry
from .retry import retry_async

_LOGGER = logging.getLogger("poker_ai")


@dataclass
class TaskResult:
    agent: str
    status: str = "pending"                 # ok | degraded | skipped | failed
    envelope: AgentEnvelope | None = None
    error: str | None = None
    skip_reason: str | None = None
    missing_deps: list = field(default_factory=list)
    attempts: int = 1

    def to_dict(self) -> dict:
        return {
            "agent": self.agent, "status": self.status,
            "error": self.error, "skip_reason": self.skip_reason,
            "missing_deps": self.missing_deps, "attempts": self.attempts,
            "envelope": self.envelope.to_dict() if self.envelope else None,
        }


@dataclass
class EngineResult:
    kind: str
    request_id: str
    results: dict[str, TaskResult] = field(default_factory=dict)
    order: list = field(default_factory=list)
    latency_ms: float = 0.0

    @property
    def status(self) -> str:
        if any(r.status == "failed" for r in self.results.values()):
            return "degraded"
        if any(r.status == "degraded" for r in self.results.values()):
            return "degraded"
        return "ok"

    def outputs(self) -> dict[str, dict | None]:
        return {name: (r.envelope.output if r.envelope else None) for name, r in self.results.items()}

    def envelopes(self) -> dict[str, dict]:
        return {name: r.envelope.to_dict() for name, r in self.results.items() if r.envelope}

    def degraded_agents(self) -> list[str]:
        return sorted(n for n, r in self.results.items() if r.status in ("degraded", "failed"))

    def skipped_agents(self) -> list[str]:
        return sorted(n for n, r in self.results.items() if r.status == "skipped")


class OrchestrationEngine:
    def __init__(self, registry: AgentRegistry, deadline_s: float | None = None) -> None:
        self.registry = registry
        self.deadline_s = deadline_s

    async def _run_task(self, task, inp: dict, missing: list[str], ctx: AgentContext) -> TaskResult:
        agent = self.registry.get(task.agent)
        result = TaskResult(agent=task.agent)

        # ---- V1.1 Agent Capability Registry：按 GameContext 声明式激活/跳过 ----
        from poker_decision_ai.core.gamemode import agent_supports_mode

        gc = getattr(ctx, "game_context", None)
        if gc is not None and not agent_supports_mode(getattr(agent, "supported_modes", ["*"]), gc):
            result.status = "skipped"
            result.skip_reason = (
                f"mode not supported: agent supports {list(getattr(agent, 'supported_modes', []))}, "
                f"current mode={gc.spec()}"
            )
            ctx.metrics.record_skipped(task.agent)
            _LOGGER.info("agent %s skipped: %s", task.agent, result.skip_reason)
            return result

        try:
            applicable, reason = await agent.applicable(inp, ctx)
            if not applicable:
                result.status = "skipped"
                result.skip_reason = reason
                ctx.metrics.record_skipped(task.agent)
                _LOGGER.info("agent %s skipped: %s", task.agent, reason)
                return result
        except Exception as e:  # applicable() failure must not crash the pipeline
            _LOGGER.warning("applicable() failed for %s: %s", task.agent, e)
        result.missing_deps = missing

        timeout_s = task.timeout_ms / 1000.0

        async def attempt() -> AgentEnvelope:
            try:
                return await asyncio.wait_for(agent.run(inp, ctx), timeout=timeout_s)
            except asyncio.TimeoutError:
                ctx.metrics.record_timeout(task.agent)
                raise AgentTimeoutError(f"{task.agent} timed out after {task.timeout_ms}ms") from None

        try:
            env = await retry_async(
                attempt, attempts=task.retries + 1,
                on_retry=lambda n, e: (ctx.metrics.record_retry(task.agent),
                                       ctx.logger.log_retry(ctx.request_id, task.agent, n, str(e))),
            )
            result.envelope = env
            result.attempts = task.retries + 1
            result.status = env.status if env.status in ("ok", "degraded") else "failed"
            if env.errors:
                result.error = "; ".join(str(e.get("message", "")) for e in env.errors)
            return result
        except Exception as e:
            result.status = "failed"
            result.error = str(e)
            ctx.logger.log_error(ctx.request_id, task.agent, getattr(e, "code", "AGENT_EXECUTION_ERROR"), str(e))
            _LOGGER.error("agent %s failed: %s", task.agent, e)
            return result

    async def execute(self, plan: Plan, ctx: AgentContext, request: dict) -> EngineResult:
        t0 = time.perf_counter()
        engine_result = EngineResult(kind=plan.kind, request_id=ctx.request_id)
        outputs: dict[str, dict | None] = {}

        for stage in plan.stages:
            if self.deadline_s and (time.perf_counter() - t0) > self.deadline_s:
                raise OrchestrationError(f"global deadline exceeded during stage '{stage.name}'")

            async def one(task) -> TaskResult:
                inp, missing = resolve_input(task.input_template, outputs, request)
                return await self._run_task(task, inp, missing, ctx)

            results: list[TaskResult] = await asyncio.gather(*(one(t) for t in stage.tasks))
            for task, res in zip(stage.tasks, results):
                engine_result.results[task.agent] = res
                engine_result.order.append(task.agent)
                outputs[task.agent] = res.envelope.output if res.envelope and res.status in ("ok", "degraded") else None
                if res.status == "failed" and task.on_failure == "abort":
                    raise OrchestrationError(
                        f"critical agent '{task.agent}' failed and its on_failure=abort policy stopped the pipeline: {res.error}"
                    )

        engine_result.latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        return engine_result
