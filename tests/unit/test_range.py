"""Unit tests for the Range Agent."""
from __future__ import annotations

import pytest

from poker_decision_ai.agents.range.agent import RangeAgent, RangeOutput
from poker_decision_ai.core.schemas import json_schema_for, validate_instance


@pytest.fixture
def agent():
    return RangeAgent()


async def test_schema_contract(agent):
    schema = json_schema_for(RangeOutput)
    assert schema["type"] == "object"
    assert "villains" in schema["properties"]
    assert "hero_range" in schema["properties"]
    assert "positions" in schema["properties"]
    validate_instance(RangeOutput, RangeOutput(villains={}, hero_range={}, positions={}))


async def test_applicable(agent, ctx):
    ok, _reason = await agent.applicable({}, ctx)
    assert not ok
    ok2, _reason2 = await agent.applicable({"state": {"players": []}}, ctx)
    assert ok2


async def test_villain_bb_and_distribution(agent, ctx, sample_cash_state):
    env = await agent.run({"state": sample_cash_state}, ctx)
    assert env.status in ("ok", "degraded")
    out = env.output
    assert "p2" in out["villains"]
    v = out["villains"]["p2"]
    assert v["distribution"]
    assert v["combo_count"] > 0
    assert v["total_weight"] > 0
    assert "AA" in v["distribution"]
    assert out["hero_range"]["distribution"]


async def test_aa_card_removal(agent, ctx, sample_cash_state):
    """Hero holding AsKd must lower villain AA weight vs no hero-card removal."""
    env_with = await agent.run({"state": sample_cash_state}, ctx)
    aa_with = env_with.output["villains"]["p2"]["distribution"]["AA"]

    # same state but hero has no hole cards -> only board removal applies
    s2 = dict(sample_cash_state)
    s2["players"] = [
        dict(p, hole_cards=[] if p.get("is_hero") else p.get("hole_cards"))
        for p in sample_cash_state["players"]
    ]
    env_without = await agent.run({"state": s2}, ctx)
    aa_without = env_without.output["villains"]["p2"]["distribution"]["AA"]

    assert aa_with < aa_without


async def test_no_profile_warning(agent, ctx, sample_cash_state):
    env = await agent.run({"state": sample_cash_state}, ctx)
    assert any(w.get("code") == "NO_PROFILE_DATA" for w in env.output.get("warnings", []))


async def test_determinism(agent, ctx, sample_cash_state):
    a = await agent.run({"state": sample_cash_state}, ctx)
    b = await agent.run({"state": sample_cash_state}, ctx)
    assert a.output == b.output
