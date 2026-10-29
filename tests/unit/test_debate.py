"""Unit tests for the Debate Agent."""
from __future__ import annotations

import pytest

from poker_decision_ai.agents.debate.agent import DebateAgent, DebateOutput
from poker_decision_ai.core.schemas import json_schema_for, validate_instance


@pytest.fixture
def agent():
    return DebateAgent()


async def test_schema_contract(agent):
    schema = json_schema_for(DebateOutput)
    assert schema["type"] == "object"
    assert "conflict_detected" in schema["properties"]
    assert "positions" in schema["properties"]
    assert "recommendation" in schema["properties"]
    out = DebateOutput()
    validate_instance(DebateOutput, out)


async def test_applicable_true(agent, ctx):
    ok, _ = await agent.applicable({"state": {"pot": 5.0}, "solver": {"gto_action": "call"}}, ctx)
    assert ok


async def test_applicable_false(agent, ctx):
    ok, reason = await agent.applicable({}, ctx)
    assert not ok
    assert reason


async def test_conflict_solver_call_vs_exploit_raise(agent, ctx):
    env = await agent.run({
        "state": {"pot": 5.0, "to_call": 1.0, "bb": 0.1},
        "solver": {"gto_action": "call", "confidence": 0.8, "is_approximate": True},
        "exploit": {"best_exploit": {"suggested_action": "raise", "confidence": 0.9}},
    }, ctx)
    out = env.output
    assert out["conflict_detected"] is True
    srcs = [p["source"] for p in out["positions"]]
    assert "solver" in srcs and "exploit" in srcs
    assert out["recommendation"]


async def test_no_conflict_single_source(agent, ctx):
    env = await agent.run({
        "state": {"pot": 5.0, "to_call": 1.0, "bb": 0.1},
        "solver": {"gto_action": "call", "confidence": 0.8},
    }, ctx)
    out = env.output
    assert out["conflict_detected"] is False
    assert len(out["positions"]) == 1


async def test_empty_input_degrades(agent, ctx):
    env = await agent.run({}, ctx)
    assert env.status in ("ok", "degraded")
    out = env.output
    assert out["positions"] == []
    assert out["warnings"]


async def test_deterministic(agent, ctx):
    inp = {
        "state": {"pot": 5.0, "to_call": 1.0, "bb": 0.1},
        "solver": {"gto_action": "call", "confidence": 0.8},
        "exploit": {"best_exploit": {"suggested_action": "raise", "confidence": 0.9}},
    }
    o1 = (await agent.run(inp, ctx)).output
    o2 = (await agent.run(inp, ctx)).output
    assert o1 == o2
