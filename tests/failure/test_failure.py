"""Failure / degradation tests (spec §十四: graceful degradation).

Simulated failures: solver unconfigured, LLM absent, vision failure,
agent timeout, agent returning invalid output, repository failure.
In every case the system must keep running and report the problem
structurally (status/warnings) instead of crashing.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import ClassVar

import pytest

from poker_decision_ai.core.errors import RepositoryError
from poker_decision_ai.core.orchestration.base_agent import BaseAgent
from poker_decision_ai.core.orchestration.context import AgentContext
from poker_decision_ai.core.orchestration.engine import OrchestrationEngine
from poker_decision_ai.core.orchestration.plan import AgentTask, Plan, Stage
from poker_decision_ai.core.orchestration.registry import AgentRegistry
from poker_decision_ai.core.providers.solver import PostflopSolverCLIProvider
from poker_decision_ai.core.schemas import new_id


# --------------------------------------------------------------------- broken agents
@dataclass
class BadOutput:
    value: int = 0


class BrokenOutputAgent(BaseAgent):
    """Returns an object that fails output-schema validation."""
    name: ClassVar[str] = "broken_output"
    Input: ClassVar[type] = dict
    Output: ClassVar[type] = BadOutput

    async def handle(self, inp: dict, ctx: AgentContext):
        return {"not": "the right shape"}  # deliberately invalid


class SlowAgent(BaseAgent):
    name: ClassVar[str] = "slow"
    Input: ClassVar[type] = dict
    Output: ClassVar[type] = dict

    async def handle(self, inp: dict, ctx: AgentContext):
        await asyncio.sleep(1.0)
        return {}


class FailingRepo:
    """Repository stub that always raises RepositoryError."""

    def save(self, *a, **kw):
        raise RepositoryError("simulated storage failure")

    def get(self, *a, **kw):
        raise RepositoryError("simulated storage failure")


# --------------------------------------------------------------------- tests
async def test_solver_unconfigured_degrades(runtime, cash_hand_text):
    """Solver selected but not installed -> pipeline degrades, decision still made."""
    runtime.providers.solver = PostflopSolverCLIProvider("")  # not configured
    ctx = runtime.new_context(new_id("fail"), "fail-session")
    report = await runtime.runner.run({
        "kind": "cash_hand", "hand_history_text": cash_hand_text, "hero_id": "Hero",
    }, ctx)
    assert report.status in ("ok", "degraded")
    solver_env = report.envelopes.get("solver")
    if solver_env:
        assert solver_env["status"] in ("ok", "degraded")
        assert solver_env["output"]["is_approximate"] is True
    assert report.final_decision is not None


async def test_no_llm_still_decides(runtime, cash_hand_text):
    """No LLM provider at all -> deterministic fallbacks keep the pipeline alive."""
    runtime.providers.llm = None
    ctx = runtime.new_context(new_id("fail"), "fail-session", allow_llm=False)
    report = await runtime.runner.run({
        "kind": "cash_hand", "hand_history_text": cash_hand_text, "hero_id": "Hero",
    }, ctx)
    assert report.status in ("ok", "degraded")
    assert report.final_decision is not None
    judge = report.envelopes["judge"]["output"]
    assert judge["reasoning"]  # template reasoning present


async def test_vision_failure_with_image_only(runtime):
    """Image given but mock vision has no fixture -> degraded, no crash."""
    ctx = runtime.new_context(new_id("fail"), "fail-session")
    report = await runtime.runner.run({
        "kind": "cash_hand",
        "image": "does-not-exist.png",
        "hero_id": "Hero",
    }, ctx)
    # state gets nothing -> skipped; the pipeline still returns a report
    assert report.status in ("ok", "degraded")
    vision = report.envelopes.get("vision")
    if vision:
        assert vision["status"] in ("ok", "degraded")
    assert "state" in report.skipped or report.envelopes.get("state", {}).get("status") == "degraded"


async def test_agent_timeout_records_failure(ctx):
    registry = AgentRegistry()
    registry.register(SlowAgent())
    engine = OrchestrationEngine(registry)
    plan = Plan("t", [Stage("s", [AgentTask(agent="slow", timeout_ms=100, retries=0)])])
    result = await engine.execute(plan, ctx, {})
    assert result.results["slow"].status == "failed"
    assert "timed out" in (result.results["slow"].error or "").lower()
    assert result.status == "degraded"


async def test_agent_invalid_output(ctx):
    registry = AgentRegistry()
    registry.register(BrokenOutputAgent())
    engine = OrchestrationEngine(registry)
    plan = Plan("t", [Stage("s", [AgentTask(agent="broken_output", retries=0)])])
    result = await engine.execute(plan, ctx, {})
    assert result.results["broken_output"].status == "failed"
    assert result.status == "degraded"


async def test_decision_repo_failure_does_not_break_pipeline(runtime, cash_hand_text):
    """Decision persistence failure must not block the decision itself."""
    runtime.repos.decisions = FailingRepo()  # type: ignore[assignment]
    ctx = runtime.new_context(new_id("fail"), "fail-session")
    report = await runtime.runner.run({
        "kind": "cash_hand", "hand_history_text": cash_hand_text, "hero_id": "Hero",
    }, ctx)
    assert report.final_decision is not None


async def test_unknown_kind_degrades_cleanly(runtime):
    """V1.1：未知 kind 不再硬编码白名单——解析器回退默认 cash 模式并优雅降级。"""
    ctx = runtime.new_context(new_id("fail"), "fail-session")
    report = await runtime.runner.run({"kind": "no_such_plan"}, ctx)
    assert report.status in ("ok", "degraded")
    assert report.game_context is not None
    assert report.game_context["game_mode"] == "cash"
