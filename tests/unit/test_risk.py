"""Unit tests for the Risk Agent."""
from __future__ import annotations

import pytest

from poker_decision_ai.agents.risk.agent import RiskAgent, RiskOutput
from poker_decision_ai.core.schemas import json_schema_for, validate_instance


@pytest.fixture
def agent():
    return RiskAgent()


async def test_schema_contract(agent):
    schema = json_schema_for(RiskOutput)
    assert schema["type"] == "object"
    assert "decision_risk" in schema["properties"]
    assert "high_risk_decision" in schema["properties"]
    validate_instance(RiskOutput, RiskOutput())


async def test_applicable(agent, ctx):
    ok, _reason = await agent.applicable({}, ctx)
    assert not ok
    ok2, _reason2 = await agent.applicable({"state": {"players": []}}, ctx)
    assert ok2


async def test_high_risk_decision(agent, ctx, sample_cash_state):
    s = dict(sample_cash_state)
    s["bb"] = 0.10            # pin bb so the bb conversion is explicit & robust
    s["pot"] = 20.0
    s["to_call"] = 5.0        # 50bb > 30bb
    env = await agent.run({"state": s, "equity": {"hero_equity": 0.2}}, ctx)
    out = env.output
    assert out["high_risk_decision"] is True


async def test_no_icm_nulls(agent, ctx, sample_cash_state):
    env = await agent.run({"state": sample_cash_state,
                           "equity": {"hero_equity": 0.5}}, ctx)
    out = env.output
    assert out["tournament_risk"] is None
    assert out["icm_risk"] is None


async def test_icm_risk_values(agent, ctx, sample_cash_state):
    icm = {"hero_icm_ev": 0.5, "hero_equity_share": 0.8,
           "risk_premium": {"p2": 0.6}}
    env = await agent.run({"state": sample_cash_state,
                           "equity": {"hero_equity": 0.5}, "icm": icm}, ctx)
    out = env.output
    assert out["icm_risk"] == pytest.approx(0.6)
    assert out["tournament_risk"] is not None


async def test_determinism(agent, ctx, sample_cash_state):
    a = await agent.run({"state": sample_cash_state, "equity": {"hero_equity": 0.5}}, ctx)
    b = await agent.run({"state": sample_cash_state, "equity": {"hero_equity": 0.5}}, ctx)
    assert a.output == b.output
