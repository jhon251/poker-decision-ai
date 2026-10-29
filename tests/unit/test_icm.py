"""Unit tests for the ICM Agent."""
from __future__ import annotations

import pytest

from poker_decision_ai.agents.icm.agent import IcmAgent, IcmOutput
from poker_decision_ai.core.schemas import json_schema_for, validate_instance


@pytest.fixture
def agent():
    return IcmAgent()


async def test_schema_contract(agent):
    schema = json_schema_for(IcmOutput)
    assert schema["type"] == "object"
    assert "icm_ev" in schema["properties"]
    assert "bubble_factor" in schema["properties"]
    out = IcmOutput()
    validate_instance(IcmOutput, out)


async def test_applicable_cash_and_tournament(agent, ctx, sample_cash_state, sample_tournament_state):
    ok, reason = await agent.applicable({"state": sample_cash_state}, ctx)
    assert not ok
    assert "cash" in reason.lower()
    ok2, _ = await agent.applicable({"state": sample_tournament_state}, ctx)
    assert ok2


async def test_tournament_icm(agent, ctx, sample_tournament_state):
    env = await agent.run({"state": sample_tournament_state}, ctx)
    assert env.status in ("ok", "degraded")
    out = env.output
    assert "hero" in out["icm_ev"]
    assert out["hero_icm_ev"] > 0
    assert out["bubble_factor"]["p2"] > 1.0
    assert out["method"] == "exact"


async def test_no_payouts_chip_share(agent, ctx, sample_tournament_state):
    s = dict(sample_tournament_state)
    s["tournament"] = dict(s["tournament"])
    s["tournament"]["payouts"] = []
    env = await agent.run({"state": s}, ctx)
    out = env.output
    assert out["method"] == "chip_share"
    assert "hero" in out["icm_ev"]
    assert any(w["code"] == "NO_PAYOUTS" for w in out["warnings"])


async def test_deterministic(agent, ctx, sample_tournament_state):
    a = await agent.run({"state": sample_tournament_state}, ctx)
    b = await agent.run({"state": sample_tournament_state}, ctx)
    assert a.output == b.output
