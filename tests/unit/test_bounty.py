"""Unit tests for the Bounty Agent (V1.1 §4b)."""
from __future__ import annotations

import pytest

from poker_decision_ai.agents.bounty.agent import BountyAgent, BountyOutput
from poker_decision_ai.core.schemas import json_schema_for, validate_instance


def _pko_state() -> dict:
    """构造 PKO tournament 块 + 玩家 bounty 字段（hero 4000 码，active 对手 p2=3000/赏金40）。"""
    return {
        "hand_id": "h-bounty-1",
        "game_type": "mtt",
        "street": "flop",
        "board": ["Qh", "7d", "2c"],
        "pot": 600.0,
        "current_bet": 200.0,
        "to_call": 200.0,
        "min_raise_to": 400.0,
        "sb": 50.0,
        "bb": 100.0,
        "ante": 0.0,
        "players": [
            {"player_id": "p1", "name": "Alice", "seat": 1, "stack": 1500.0, "status": "folded",
             "hole_cards": [], "position": "MP", "is_hero": False, "bounty": 20.0},
            {"player_id": "p2", "name": "Bob", "seat": 2, "stack": 3000.0, "status": "active",
             "hole_cards": [], "position": "CO", "is_hero": False, "bounty": 40.0},
            {"player_id": "hero", "name": "Hero", "seat": 3, "stack": 4000.0, "status": "active",
             "hole_cards": ["As", "Kd"], "position": "BTN", "is_hero": True, "bounty": 10.0},
            {"player_id": "p4", "name": "Carol", "seat": 4, "stack": 2000.0, "status": "folded",
             "hole_cards": [], "position": "SB", "is_hero": False, "bounty": 15.0},
            {"player_id": "p5", "name": "Dave", "seat": 5, "stack": 1800.0, "status": "folded",
             "hole_cards": [], "position": "BB", "is_hero": False, "bounty": 25.0},
            {"player_id": "p6", "name": "Eve", "seat": 6, "stack": 1200.0, "status": "folded",
             "hole_cards": [], "position": "UTG", "is_hero": False, "bounty": 50.0},
        ],
        "button_seat": 3,
        "hero_id": "hero",
        "hero_seat": 3,
        "action_history": [],
        "street_actions": {},
        "available_actions": ["fold", "call", "raise"],
        "last_aggressor": "p2",
        "rake": 0.0,
        "source": "hand_history",
        "cash": None,
        "tournament": {
            "tournament_id": "T-bounty", "name": "PKO Test", "total_players": 100,
            "players_remaining": 45, "total_chips": 100000.0,
            "payouts": [50.0, 30.0, 20.0, 15.0, 10.0, 8.0, 5.0, 4.0, 3.0, 2.0],
            "next_payout": 2.0, "itm_players": 10, "satellite_seats": None,
            "stage": "bubble", "avg_stack": 2222.0,
            "bounty_type": "pko", "is_bounty": True, "progressive": True,
            "bounty_rate": 0.5, "total_bounty_pool": 5000.0, "prize_pool": 5000.0,
            "mystery_phase_active": False,
        },
    }


@pytest.fixture
def agent():
    return BountyAgent()


async def test_schema_contract(agent):
    schema = json_schema_for(BountyOutput)
    assert schema["type"] == "object"
    for f in ("bounty_type", "is_bounty", "progressive", "villain_bounty",
              "villain_bounties", "knockout_value", "progressive_growth",
              "bounty_chips", "chip_value", "bounty_pressure", "bounty_ev",
              "covered", "confidence", "evidence", "warnings"):
        assert f in schema["properties"]
    validate_instance(BountyOutput, BountyOutput())


async def test_not_applicable_without_state(agent, ctx):
    ok, _reason = await agent.applicable({}, ctx)
    assert not ok


async def test_applicable_cash_state(agent, ctx, sample_cash_state):
    # 无赏金也 applicable=True（模式过滤负责跳过）
    ok, _ = await agent.applicable({"state": sample_cash_state}, ctx)
    assert ok


async def test_pko_state_main_path(agent, ctx):
    env = await agent.run({"state": _pko_state()}, ctx)
    assert env.status in ("ok", "degraded")
    out = env.output
    assert out["bounty_type"] == "pko"
    assert out["is_bounty"] is True
    assert out["progressive"] is True
    assert out["covered"] is True
    assert out["villain_bounty"] == pytest.approx(40.0)
    assert out["knockout_value"] == pytest.approx(20.0)   # growth_rate(0.5) * 40
    assert out["bounty_pressure"] > 0
    assert out["chip_value"] == pytest.approx(0.1)        # (5000+5000)/100000
    assert out["hero_bounty"] == pytest.approx(10.0)
    assert out["progressive_growth"] == pytest.approx(20.0)


async def test_cash_state_all_zero(agent, ctx, sample_cash_state):
    env = await agent.run({"state": sample_cash_state}, ctx)
    assert env.status in ("ok", "degraded")
    out = env.output
    assert out["is_bounty"] is False
    assert out["bounty_type"] == ""
    assert out["villain_bounty"] == 0.0
    assert out["bounty_ev"] == 0.0
    assert out["knockout_value"] == 0.0
    assert out["warnings"]


async def test_villain_bounties_active_only(agent, ctx):
    # Eve(folded) 赏金 50 更高，但只统计 active villain（p2=40）
    env = await agent.run({"state": _pko_state()}, ctx)
    out = env.output
    assert out["villain_bounties"] == {"p2": 40.0}


async def test_determinism(agent, ctx):
    env1 = await agent.run({"state": _pko_state()}, ctx)
    env2 = await agent.run({"state": _pko_state()}, ctx)
    assert env1.output == env2.output
