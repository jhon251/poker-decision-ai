"""Unit tests for the Bluff Agent."""
from __future__ import annotations

import pytest

from poker_decision_ai.agents.bluff.agent import BluffAgent, BluffOutput
from poker_decision_ai.core.schemas import json_schema_for, validate_instance


@pytest.fixture
def agent():
    return BluffAgent()


def _overbet_river_state(sample_cash_state) -> dict:
    """A river state where the opponent overbets (~1.35 pot)."""
    s = dict(sample_cash_state)
    s["street"] = "river"
    s["pot"] = 10.45          # pot_before_bet = 4.45
    s["current_bet"] = 6.0
    s["to_call"] = 6.0
    return s


async def test_schema_contract(agent):
    schema = json_schema_for(BluffOutput)
    assert schema["type"] == "object"
    assert "value_probability" in schema["properties"]
    assert "bluff_probability" in schema["properties"]
    assert "implied_action" in schema["properties"]
    out = BluffOutput(value_probability=0.5, bluff_probability=0.5)
    validate_instance(BluffOutput, out)


async def test_applicable(agent, ctx):
    ok, _ = await agent.applicable({"state": {"current_bet": 1.0}}, ctx)
    assert ok
    ok2, reason = await agent.applicable({"state": {"current_bet": 0}}, ctx)
    assert not ok2
    assert reason


async def test_high_bluff_tendency(agent, ctx, sample_cash_state):
    state = _overbet_river_state(sample_cash_state)
    opponent = {"players": {"p2": {
        "feature_vector": {"bluff_tendency": 0.8, "tightness": 0.5, "calling_tendency": 0.5},
        "sample_size": 500,
    }}}
    env = await agent.run({"state": state, "opponent": opponent}, ctx)
    assert env.status in ("ok", "degraded")
    out = env.output
    assert out["bluff_probability"] > 0.5
    assert abs(out["value_probability"] + out["bluff_probability"] - 1.0) < 1e-6


async def test_tight_nit_value_heavy(agent, ctx, sample_cash_state):
    state = _overbet_river_state(sample_cash_state)
    opponent = {"players": {"p2": {
        "feature_vector": {"bluff_tendency": 0.5, "tightness": 0.8, "calling_tendency": 0.5},
        "sample_size": 500,
    }}}
    env = await agent.run({"state": state, "opponent": opponent}, ctx)
    out = env.output
    assert out["value_probability"] > 0.6


async def test_no_opponent_prior_only(agent, ctx, sample_cash_state):
    state = _overbet_river_state(sample_cash_state)
    env = await agent.run({"state": state}, ctx)
    out = env.output
    assert any(w["code"] == "NO_OPPONENT" for w in out["warnings"])
    assert out["bluff_probability"] == pytest.approx(0.5, abs=0.05)


async def test_deterministic(agent, ctx, sample_cash_state):
    state = _overbet_river_state(sample_cash_state)
    opponent = {"players": {"p2": {
        "feature_vector": {"bluff_tendency": 0.8, "tightness": 0.5, "calling_tendency": 0.5},
        "sample_size": 500,
    }}}
    a = await agent.run({"state": state, "opponent": opponent}, ctx)
    b = await agent.run({"state": state, "opponent": opponent}, ctx)
    assert a.output == b.output
