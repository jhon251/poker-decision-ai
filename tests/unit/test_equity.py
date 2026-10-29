"""Unit tests for the Equity Agent."""
from __future__ import annotations

import pytest

from poker_decision_ai.agents.equity.agent import EquityAgent, EquityOutput
from poker_decision_ai.core.schemas import json_schema_for, validate_instance


@pytest.fixture
def agent():
    return EquityAgent()


async def test_schema_contract(agent):
    schema = json_schema_for(EquityOutput)
    assert schema["type"] == "object"
    assert "hero_equity" in schema["properties"]
    assert "pot_odds" in schema["properties"]
    validate_instance(EquityOutput, EquityOutput())


async def test_applicable(agent, ctx):
    ok, _reason = await agent.applicable({}, ctx)
    assert not ok
    ok2, _reason2 = await agent.applicable({"state": {"players": []}}, ctx)
    assert ok2


async def test_ak_vs_random_no_range(agent, ctx, sample_cash_state):
    env = await agent.run({"state": sample_cash_state}, ctx)
    assert env.status in ("ok", "degraded")
    out = env.output
    assert 0.3 < out["hero_equity"] < 0.75
    assert out["villains"]["p2"]["equity"] > 0
    assert any(w.get("code") == "NO_RANGE" for w in out.get("warnings", []))


async def test_river_exact_method(agent, ctx, sample_cash_state):
    s = dict(sample_cash_state)
    s["street"] = "river"
    s["board"] = ["Qh", "7d", "2c", "3h", "9s"]
    env = await agent.run({"state": s}, ctx)
    assert env.output["villains"]["p2"]["method"] == "exact"


async def test_determinism(agent, ctx, sample_cash_state):
    a = await agent.run({"state": sample_cash_state}, ctx)
    b = await agent.run({"state": sample_cash_state}, ctx)
    assert a.output == b.output
