"""Unit tests for the Opponent Agent."""
from __future__ import annotations

import pytest

from poker_decision_ai.agents.opponent.agent import OpponentAgent, OpponentOutput
from poker_decision_ai.core.schemas import json_schema_for, validate_instance

_FV_KEYS = ["tightness", "aggression", "bluff_tendency", "calling_tendency",
            "folding_tendency", "overbet_tendency", "check_raise_tendency",
            "tilt_risk", "icm_sensitivity", "stage_adaptivity"]


@pytest.fixture
def agent():
    return OpponentAgent()


def _record(player_id, hand_id, vpip=True, bets=1, calls=0, folds=0, **kw):
    rec = {
        "hand_id": hand_id, "player_id": player_id, "game": "cash", "stakes": "NL10",
        "n_players": 6, "position": "BTN", "vpip": vpip, "pfr": False,
        "three_bet": False, "faced_3bet": False, "folded_to_3bet": False,
        "four_bet": False, "cbet_opportunity": False, "cbet": False,
        "faced_cbet": False, "folded_to_cbet": False, "saw_flop": True,
        "went_to_showdown": False, "won_at_showdown": False, "won_hand": False,
        "net_bb": -1.0, "bets": bets, "calls": calls, "folds": folds,
        "ts": 1_700_000_000,
    }
    rec.update(kw)
    return rec


async def test_schema_contract(agent):
    schema = json_schema_for(OpponentOutput)
    assert schema["type"] == "object"
    assert "players" in schema["properties"]
    assert "confidence" in schema["properties"]
    validate_instance(OpponentOutput, OpponentOutput())


async def test_aggregate_from_hands(agent, ctx, sample_cash_state):
    """无档案：从 hands 聚合，feature_vector 在 0..1，样本量正确。"""
    for i in range(5):
        ctx.repos.hands.save(_record("hero", f"h{i}", vpip=(i % 2 == 0)))
    env = await agent.run({"state": sample_cash_state, "statistics": None}, ctx)
    out = env.output
    prof = out["players"]["hero"]
    assert prof["sample_size"] == 5
    assert prof["behavior_statistics"]["vpip"]["samples"] == 5
    for k in _FV_KEYS:
        assert 0.0 <= prof["feature_vector"][k] <= 1.0
    assert prof["player_id"] == "hero"
    assert prof["name"] == "Hero"


async def test_existing_profile_used(agent, ctx, sample_cash_state):
    ctx.repos.players.upsert({
        "player_id": "hero", "name": "Hero",
        "feature_vector": {"tightness": 0.8}, "sample_size": 5000,
        "behavior_statistics": {"vpip": {"value": 0.2, "samples": 5000}},
    })
    env = await agent.run({"state": sample_cash_state, "statistics": None}, ctx)
    out = env.output
    prof = out["players"]["hero"]
    assert prof["feature_vector"]["tightness"] == 0.8
    assert prof["feature_vector"]["aggression"] == 0.5      # 缺失维度补中性
    assert prof["sample_size"] == 5000


async def test_confidence_positive_correlation(agent, ctx, sample_cash_state):
    ctx.repos.players.upsert({"player_id": "hero", "sample_size": 100})
    ctx.repos.players.upsert({"player_id": "p2", "sample_size": 5000})
    env = await agent.run({"state": sample_cash_state, "statistics": None}, ctx)
    out = env.output
    assert out["players"]["p2"]["confidence"] > out["players"]["hero"]["confidence"]


async def test_no_data_neutral(agent, ctx, sample_cash_state):
    """无档案且无手牌记录 -> 中性画像 + warning，不抛异常。"""
    env = await agent.run({"state": sample_cash_state, "statistics": None}, ctx)
    out = env.output
    prof = out["players"]["hero"]
    assert prof["sample_size"] == 0
    assert prof["feature_vector"]["tightness"] == 0.5
    assert env.status == "degraded"
