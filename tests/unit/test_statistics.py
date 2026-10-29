"""Unit tests for the Statistics Agent."""
from __future__ import annotations

import pytest

from poker_decision_ai.agents.statistics.agent import StatisticsAgent, StatisticsOutput
from poker_decision_ai.core.schemas import json_schema_for, validate_instance


@pytest.fixture
def agent():
    return StatisticsAgent()


def _record(player_id, hand_id, vpip, pfr=True, game="cash", **kw):
    rec = {
        "hand_id": hand_id, "player_id": player_id, "game": game, "stakes": "NL10",
        "n_players": 6, "position": "BTN",
        "vpip": vpip, "pfr": pfr, "three_bet": False, "faced_3bet": False,
        "folded_to_3bet": False, "four_bet": False, "cbet_opportunity": False,
        "cbet": False, "faced_cbet": False, "folded_to_cbet": False,
        "saw_flop": True, "went_to_showdown": False, "won_at_showdown": False,
        "won_hand": False, "net_bb": -1.0, "bets": 0, "calls": 1, "folds": 0,
        "ts": 1_700_000_000,
    }
    rec.update(kw)
    return rec


async def test_schema_contract(agent):
    schema = json_schema_for(StatisticsOutput)
    assert schema["type"] == "object"
    for key in ("hero_stats", "villains", "hand_records_count", "normalized_hands",
                "confidence", "evidence", "warnings"):
        assert key in schema["properties"]
    validate_instance(StatisticsOutput, StatisticsOutput())


async def test_hero_vpip_from_records(agent, ctx, sample_cash_state):
    # 5 条 hero 记录：vpip 3 True / 2 False -> 0.6
    for i, v in enumerate([True, False, True, True, False]):
        ctx.repos.hands.save(_record("hero", f"h{i}", v))
    # 2 条 villain 记录
    ctx.repos.hands.save(_record("p2", "v0", True))
    ctx.repos.hands.save(_record("p2", "v1", False))

    env = await agent.run({"state": sample_cash_state}, ctx)
    out = env.output
    assert out["hand_records_count"] == 7
    assert out["normalized_hands"] == 5
    assert out["hero_stats"]["vpip"]["value"] == pytest.approx(0.6)
    assert out["hero_stats"]["vpip"]["samples"] == 5
    assert out["villains"]["p2"]["vpip"]["samples"] == 2


async def test_game_filter(agent, ctx, sample_cash_state):
    """cash 时剔除 game 不匹配的记录。"""
    ctx.repos.hands.save(_record("hero", "h_cash", True, game="cash"))
    ctx.repos.hands.save(_record("hero", "h_mtt", True, game="mtt"))
    env = await agent.run({"state": sample_cash_state}, ctx)
    out = env.output
    assert out["hero_stats"]["vpip"]["samples"] == 1


async def test_no_data_degraded(agent, ctx, sample_cash_state):
    env = await agent.run({"state": sample_cash_state}, ctx)
    out = env.output
    assert out["hero_stats"]["vpip"]["samples"] == 0
    assert out["hero_stats"]["vpip"]["value"] == 0.0
    assert env.status == "degraded"
    assert any(w["code"] == "NO_DATA" for w in out["warnings"])


async def test_deterministic(agent, ctx, sample_cash_state):
    for i in range(3):
        ctx.repos.hands.save(_record("hero", f"h{i}", i % 2 == 0))
    e1 = await agent.run({"state": sample_cash_state}, ctx)
    e2 = await agent.run({"state": sample_cash_state}, ctx)
    assert e1.output == e2.output
