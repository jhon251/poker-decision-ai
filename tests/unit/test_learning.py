"""Unit tests for the Learning Agent (over-fitting protection core)."""
from __future__ import annotations

import pytest

from poker_decision_ai.agents.learning.agent import LearningAgent, LearningOutput
from poker_decision_ai.core.schemas import json_schema_for, validate_instance


@pytest.fixture
def agent():
    return LearningAgent()


async def test_schema_contract(agent):
    schema = json_schema_for(LearningOutput)
    assert schema["type"] == "object"
    for f in ("updates", "rejected", "bad_beat_protection"):
        assert f in schema["properties"]
    out = LearningOutput(bad_beat_protection={"triggered": False, "note": ""})
    validate_instance(LearningOutput, out)


async def test_sample_insufficient_rejected(agent, ctx, sample_cash_state):
    stats = {"hero_stats": {"vpip": {"value": 0.4, "samples": 10}}, "villains": {}}
    env = await agent.run({"state": sample_cash_state, "statistics": stats}, ctx)
    out = env.output
    assert len(out["updates"]) >= 1
    assert all(u["decision"] == "reject" for u in out["updates"])
    assert any("样本不足" in u["reason"] for u in out["updates"])


async def test_significant_change_updates(agent, ctx, sample_cash_state):
    ctx.repos.players.upsert({
        "player_id": "hero",
        "behavior_statistics": {"vpip": {"value": 0.3, "samples": 500}},
        "sample_size": 500,
    })
    stats = {"hero_stats": {"vpip": {"value": 0.7, "samples": 200}}, "villains": {}}
    env = await agent.run({"state": sample_cash_state, "statistics": stats}, ctx)
    out = env.output
    assert any(u["decision"] == "update" for u in out["updates"])
    profile = ctx.repos.players.get("hero")
    merged = profile["behavior_statistics"]["vpip"]["value"]
    # merged value lies strictly between old (0.3) and new (0.7)
    assert 0.3 < merged < 0.7
    assert profile["behavior_statistics"]["vpip"]["samples"] == 700


async def test_skip_learning_rejects_all(agent, ctx, sample_cash_state):
    stats = {"hero_stats": {"vpip": {"value": 0.4, "samples": 100}}, "villains": {}}
    env = await agent.run({"state": sample_cash_state, "statistics": stats,
                           "request": {"skip_learning": True}}, ctx)
    out = env.output
    assert len(out["updates"]) >= 1
    assert all(u["decision"] == "reject" for u in out["updates"])
    assert all("skip_learning" in u["reason"] for u in out["updates"])


async def test_bad_beat_protection(agent, ctx, sample_cash_state):
    env = await agent.run({
        "state": sample_cash_state,
        "statistics": {"hero_stats": {}, "villains": {}},
        "request": {"outcome": {"result": "lost", "won": False}},
    }, ctx)
    assert env.output["bad_beat_protection"]["triggered"] is True
