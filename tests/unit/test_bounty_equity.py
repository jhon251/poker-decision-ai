"""Unit tests for the Bounty Equity Agent (V1.1 §4b)."""
from __future__ import annotations

import pytest

from poker_decision_ai.agents.bounty_equity.agent import BountyEquityAgent, BountyEquityOutput
from poker_decision_ai.core.schemas import json_schema_for, validate_instance


def _state() -> dict:
    """pot=600, to_call=200, bb=100 -> pot_bb=6, to_call_bb=2。"""
    return {
        "game_type": "mtt", "street": "flop", "board": ["Qh", "7d", "2c"],
        "pot": 600.0, "to_call": 200.0, "bb": 100.0, "sb": 50.0,
        "hero_id": "hero",
        "players": [
            {"player_id": "p2", "stack": 3000.0, "status": "active", "is_hero": False, "bounty": 40.0},
            {"player_id": "hero", "stack": 4000.0, "status": "active", "is_hero": True, "bounty": 10.0},
        ],
        "tournament": {"bounty_type": "pko", "is_bounty": True, "progressive": True,
                       "total_bounty_pool": 5000.0, "prize_pool": 5000.0, "total_chips": 100000.0},
        "cash": None,
    }


@pytest.fixture
def agent():
    return BountyEquityAgent()


async def test_schema_contract(agent):
    schema = json_schema_for(BountyEquityOutput)
    assert schema["type"] == "object"
    for f in ("normal_required_equity", "bounty_adjusted_required_equity",
              "required_equity_delta", "bounty_chips_used", "villain_covered",
              "ev_call_normal_bb", "ev_call_with_bounty_bb",
              "effective_decision_equity", "warnings"):
        assert f in schema["properties"]
    validate_instance(BountyEquityOutput, BountyEquityOutput())


async def test_not_applicable_without_state(agent, ctx):
    ok, _ = await agent.applicable({}, ctx)
    assert not ok


async def test_bounty_adjusted_below_normal(agent, ctx):
    env = await agent.run({
        "state": _state(),
        "equity": {"hero_equity": 0.5},
        "bounty": {"is_bounty": True, "covered": True, "bounty_chips": 400.0},
    }, ctx)
    assert env.status in ("ok", "degraded")
    out = env.output
    # normal = 2/(6+4)=0.2; adjusted = 2/(6+4+4)=0.142857
    assert out["normal_required_equity"] == pytest.approx(0.2)
    assert out["bounty_adjusted_required_equity"] == pytest.approx(2 / 14)
    assert out["bounty_adjusted_required_equity"] < out["normal_required_equity"]
    assert out["required_equity_delta"] > 0
    assert out["bounty_chips_used"] == pytest.approx(4.0)
    assert out["villain_covered"] is True
    # ev_call_normal = 0.5*(6+2)-0.5*2 = 3.0; with bounty = 0.5*(6+2+4)-0.5*2 = 5.0
    assert out["ev_call_normal_bb"] == pytest.approx(3.0)
    assert out["ev_call_with_bounty_bb"] == pytest.approx(5.0)
    assert out["ev_call_with_bounty_bb"] > out["ev_call_normal_bb"]
    assert out["effective_decision_equity"] == pytest.approx(2 / 14)


async def test_no_bounty_adjusted_equals_normal(agent, ctx):
    env = await agent.run({"state": _state(), "equity": {"hero_equity": 0.5}}, ctx)
    out = env.output
    assert out["bounty_adjusted_required_equity"] == pytest.approx(out["normal_required_equity"])
    assert out["required_equity_delta"] == pytest.approx(0.0)
    assert out["effective_decision_equity"] == pytest.approx(out["normal_required_equity"])
    assert out["warnings"]


async def test_missing_equity_neutral(agent, ctx):
    env = await agent.run({
        "state": _state(),
        "bounty": {"is_bounty": True, "covered": True, "bounty_chips": 400.0},
    }, ctx)
    out = env.output
    # 0.5 中性：ev_call_normal = 0.5*(6+2)-0.5*2 = 3.0
    assert out["ev_call_normal_bb"] == pytest.approx(3.0)
    assert any(w.get("code") == "NO_EQUITY" for w in out["warnings"])


async def test_determinism(agent, ctx):
    inp = {"state": _state(), "equity": {"hero_equity": 0.6},
           "bounty": {"is_bounty": True, "covered": True, "bounty_chips": 400.0}}
    e1 = await agent.run(inp, ctx)
    e2 = await agent.run(inp, ctx)
    assert e1.output == e2.output
