"""Unit tests for the Experiment Agent."""
from __future__ import annotations

import pytest

from poker_decision_ai.agents.experiment.agent import ExperimentAgent, ExperimentOutput
from poker_decision_ai.core.schemas import json_schema_for, validate_instance


@pytest.fixture
def agent():
    return ExperimentAgent()


def _seed_decision(ctx, request_id, arm, score):
    ctx.repos.decisions.save({
        "request_id": request_id,
        "request": {"arm": arm},
        "evaluation": {"decision_quality": {"score": score}},
    })


async def test_schema_contract(agent):
    schema = json_schema_for(ExperimentOutput)
    assert schema["type"] == "object"
    assert "experiments" in schema["properties"]
    out = ExperimentOutput()
    validate_instance(ExperimentOutput, out)


async def test_applicable(agent, ctx):
    ok, _ = await agent.applicable({}, ctx)
    assert ok


async def test_two_arms_winner_and_significance(agent, ctx):
    for i in range(50):
        _seed_decision(ctx, f"a-{i}", "gto_only", 0.80)
    for i in range(50):
        _seed_decision(ctx, f"b-{i}", "gto_population", 0.30)
    env = await agent.run({}, ctx)
    out = env.output
    exp = out["experiments"][0]
    assert exp["winner"] == "gto_only"
    assert exp["significance"] > 0.95
    assert exp["status"] == "concluded"


async def test_insufficient_samples_running(agent, ctx):
    for i in range(10):
        _seed_decision(ctx, f"a-{i}", "gto_only", 0.80)
    for i in range(10):
        _seed_decision(ctx, f"b-{i}", "gto_population", 0.30)
    env = await agent.run({}, ctx)
    out = env.output
    exp = out["experiments"][0]
    assert exp["status"] == "running"
    assert exp["winner"] is None


async def test_empty_repo_degrades(agent, ctx):
    env = await agent.run({}, ctx)
    assert env.status in ("ok", "degraded")
    out = env.output
    assert out["experiments"] == []
    assert out["warnings"]


async def test_stores_to_research(agent, ctx):
    for i in range(50):
        _seed_decision(ctx, f"a-{i}", "gto_only", 0.80)
    for i in range(50):
        _seed_decision(ctx, f"b-{i}", "gto_population", 0.30)
    await agent.run({}, ctx)
    stored = ctx.repos.research.list("experiment")
    assert len(stored) == 1
    assert stored[0]["name"] == "strategy_arm_comparison"
