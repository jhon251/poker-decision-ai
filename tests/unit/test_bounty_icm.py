"""Unit tests for the Bounty ICM Agent (V1.1 §4b)."""
from __future__ import annotations

import pytest

from poker_decision_ai.agents.bounty_icm.agent import BountyIcmAgent, BountyIcmOutput
from poker_decision_ai.core.schemas import json_schema_for, validate_instance


def _pko_state() -> dict:
    return {
        "hand_id": "h-bounty-1", "game_type": "mtt", "street": "flop",
        "board": ["Qh", "7d", "2c"], "pot": 600.0, "to_call": 200.0,
        "sb": 50.0, "bb": 100.0, "ante": 0.0,
        "hero_id": "hero",
        "players": [
            {"player_id": "p1", "stack": 1500.0, "status": "folded", "is_hero": False, "bounty": 20.0},
            {"player_id": "p2", "stack": 3000.0, "status": "active", "is_hero": False, "bounty": 40.0},
            {"player_id": "hero", "stack": 4000.0, "status": "active", "is_hero": True, "bounty": 10.0},
            {"player_id": "p4", "stack": 2000.0, "status": "folded", "is_hero": False, "bounty": 15.0},
        ],
        "cash": None,
        "tournament": {
            "tournament_id": "T-bounty", "total_players": 100, "players_remaining": 45,
            "total_chips": 100000.0,
            "payouts": [50.0, 30.0, 20.0, 15.0, 10.0, 8.0, 5.0, 4.0, 3.0, 2.0],
            "itm_players": 10, "stage": "bubble",
            "bounty_type": "pko", "is_bounty": True, "progressive": True,
            "bounty_rate": 0.5, "total_bounty_pool": 5000.0, "prize_pool": 5000.0,
        },
    }


@pytest.fixture
def agent():
    return BountyIcmAgent()


async def test_schema_contract(agent):
    schema = json_schema_for(BountyIcmOutput)
    assert schema["type"] == "object"
    for f in ("knockout_probability", "bounty_ev_by_villain", "icm_ev",
              "risk_premium", "combined_ev", "total_bounty_ev",
              "bubble_plus_bounty", "note", "warnings"):
        assert f in schema["properties"]
    validate_instance(BountyIcmOutput, BountyIcmOutput())


async def test_not_applicable_without_state(agent, ctx):
    ok, _ = await agent.applicable({}, ctx)
    assert not ok


async def test_ko_probability_and_combined_ev(agent, ctx):
    inp = {
        "state": _pko_state(),
        "icm": {"icm_ev": {"hero": 15.0, "p2": 10.0}, "risk_premium": {"p2": 0.2}},
        "equity": {"hero_equity": 0.6, "hero_equity_vs_each": {"p2": 0.6}},
        "bounty": {"is_bounty": True, "villain_bounty": 40.0, "progressive": True, "covered": True},
        "tournament_stage": {"stage": "bubble"},
    }
    env = await agent.run(inp, ctx)
    assert env.status in ("ok", "degraded")
    out = env.output
    assert 0.0 <= out["knockout_probability"]["p2"] <= 1.0
    assert out["knockout_probability"]["p2"] == pytest.approx(0.6)
    # bounty_ev = 0.6 * (0.5 * 40) = 12.0
    assert out["bounty_ev_by_villain"]["p2"] == pytest.approx(12.0)
    # combined = icm_ev + bounty_ev = 10 + 12
    assert out["combined_ev"]["p2"] == pytest.approx(22.0)
    assert out["icm_ev"]["p2"] == pytest.approx(10.0)
    assert out["total_bounty_ev"] == pytest.approx(12.0)
    assert out["bubble_plus_bounty"] is True
    assert out["note"]


async def test_missing_icm_warning(agent, ctx):
    env = await agent.run({
        "state": _pko_state(),
        "equity": {"hero_equity": 0.5},
        "bounty": {"is_bounty": True},
    }, ctx)
    assert env.status in ("ok", "degraded")
    out = env.output
    assert out["icm_ev"] is None
    assert out["warnings"]
    # 无 icm 时 combined_ev 仍由 bounty_ev 单独构成
    assert "p2" in out["combined_ev"]


async def test_no_bounty_empty_result(agent, ctx, sample_tournament_state):
    env = await agent.run({"state": sample_tournament_state,
                           "icm": {"icm_ev": {"p2": 10.0}}}, ctx)
    assert env.status in ("ok", "degraded")
    out = env.output
    assert out["bounty_ev_by_villain"] == {}
    assert out["knockout_probability"] == {}
    assert out["warnings"]


async def test_determinism(agent, ctx):
    inp = {
        "state": _pko_state(),
        "icm": {"icm_ev": {"hero": 15.0, "p2": 10.0}},
        "equity": {"hero_equity": 0.6, "hero_equity_vs_each": {"p2": 0.6}},
        "bounty": {"is_bounty": True},
        "tournament_stage": {"stage": "bubble"},
    }
    e1 = await agent.run(inp, ctx)
    e2 = await agent.run(inp, ctx)
    assert e1.output == e2.output
