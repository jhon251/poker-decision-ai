"""Unit tests for the Population Agent."""
from __future__ import annotations

import pytest

from poker_decision_ai.agents.population.agent import PopulationAgent, PopulationOutput
from poker_decision_ai.core.schemas import json_schema_for, validate_instance


@pytest.fixture
def agent():
    return PopulationAgent()


def _segment(vpip_value=0.30, samples=500):
    return {"vpip": {"value": vpip_value, "samples": samples},
            "pfr": {"value": 0.20, "samples": samples}, "sample_size": samples}


async def test_schema_contract(agent):
    schema = json_schema_for(PopulationOutput)
    assert schema["type"] == "object"
    for key in ("segment_keys", "primary_segment", "segments", "stats",
                "fallback_used", "missing", "confidence", "warnings"):
        assert key in schema["properties"]
    validate_instance(PopulationOutput, PopulationOutput())


async def test_segment_keys_construction(agent, ctx, sample_cash_state):
    """现金桌 6max hero BTN -> 5 级 fallback key。"""
    env = await agent.run({"state": sample_cash_state, "statistics": None,
                           "tournament_stage": None}, ctx)
    out = env.output
    assert out["segment_keys"] == [
        "NL10:cash:BTN:6max", "NL10:cash:6max", "NL10:cash",
        "cash:BTN:6max", "cash",
    ]


async def test_hit_most_specific(agent, ctx, sample_cash_state):
    ctx.repos.populations.upsert_segment("NL10:cash:BTN:6max", _segment(0.30))
    env = await agent.run({"state": sample_cash_state}, ctx)
    out = env.output
    assert out["primary_segment"] == "NL10:cash:BTN:6max"
    assert out["fallback_used"] is False
    assert out["missing"] is False
    assert out["stats"]["vpip"]["value"] == pytest.approx(0.30)
    assert out["segments"]["NL10:cash:BTN:6max"]["sample_size"] == 500


async def test_fallback_used(agent, ctx, sample_cash_state):
    ctx.repos.populations.upsert_segment("NL10:cash", _segment(0.25))
    env = await agent.run({"state": sample_cash_state}, ctx)
    out = env.output
    assert out["primary_segment"] == "NL10:cash"
    assert out["fallback_used"] is True
    assert out["missing"] is False
    assert out["stats"]["vpip"]["value"] == pytest.approx(0.25)


async def test_missing_no_exception(agent, ctx, sample_cash_state):
    env = await agent.run({"state": sample_cash_state}, ctx)
    out = env.output
    assert out["missing"] is True
    assert out["stats"] == {}
    assert env.status in ("ok", "degraded")


async def test_aggregate_from_hands(agent, ctx, sample_cash_state):
    """未预存段位时，从 repos.hands 聚合同段位（stakes/game 匹配）。"""
    for i in range(4):
        ctx.repos.hands.save({
            "hand_id": f"h{i}", "player_id": "hero", "game": "cash", "stakes": "NL10",
            "vpip": i % 2 == 0, "pfr": False, "three_bet": False, "faced_3bet": False,
            "folded_to_3bet": False, "four_bet": False, "cbet_opportunity": False,
            "cbet": False, "faced_cbet": False, "folded_to_cbet": False,
            "saw_flop": True, "went_to_showdown": False, "won_at_showdown": False,
            "won_hand": False, "net_bb": -1.0, "bets": 0, "calls": 1, "folds": 0, "ts": 1,
        })
    env = await agent.run({"state": sample_cash_state}, ctx)
    out = env.output
    assert out["missing"] is True
    assert out["aggregated_from_hands"] is True
    assert out["stats"]["vpip"]["samples"] == 4


async def test_mtt_keys(agent, ctx, sample_tournament_state):
    stage = {"stage": "bubble"}
    env = await agent.run({"state": sample_tournament_state, "statistics": None,
                           "tournament_stage": stage}, ctx)
    out = env.output
    assert out["segment_keys"][0] == "mtt:bubble:BTN:6max"
    assert out["segment_keys"][-1] == "mtt"
