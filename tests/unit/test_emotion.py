"""Unit tests for the Emotion Agent."""
from __future__ import annotations

import pytest

from poker_decision_ai.agents.emotion.agent import EmotionAgent, EmotionOutput
from poker_decision_ai.core.schemas import json_schema_for, validate_instance


@pytest.fixture
def agent():
    return EmotionAgent()


async def test_schema_contract(agent):
    schema = json_schema_for(EmotionOutput)
    assert schema["type"] == "object"
    assert "players" in schema["properties"]
    assert "hero" in schema["properties"]
    out = EmotionOutput()
    validate_instance(EmotionOutput, out)


async def test_applicable(agent, ctx):
    ok, _ = await agent.applicable({"state": {"game_type": "cash"}}, ctx)
    assert ok
    ok2, _ = await agent.applicable({}, ctx)
    assert not ok2


async def test_bad_beat_tilt(agent, ctx, sample_cash_state):
    # 5 losing hands; first is a bad beat (equity 0.8, lost) with a big-pot loss.
    for i in range(5):
        ctx.repos.hands.save({
            "hand_id": f"h-{i}", "player_id": "hero",
            "won_hand": False,
            "net_bb": -60.0 if i == 0 else -10.0,
            "equity_at_decision": 0.8 if i == 0 else 0.5,
            "vpip": True, "ts": 1000 + i,
        })
    env = await agent.run({"state": sample_cash_state}, ctx)
    assert env.status in ("ok", "degraded")
    out = env.output
    assert out["hero"]["tilt_probability"] > 0.5


async def test_empty_repo_baseline(agent, ctx, sample_cash_state):
    env = await agent.run({"state": sample_cash_state}, ctx)
    out = env.output
    assert out["hero"]["tilt_probability"] == pytest.approx(0.5)
    assert out["hero"]["overconfidence_probability"] == pytest.approx(0.5)
    assert out["hero"]["confidence"] < 0.3
    assert any(w["code"] == "NO_HERO_HANDS" for w in out["warnings"])


async def test_deterministic(agent, ctx, sample_cash_state):
    for i in range(3):
        ctx.repos.hands.save({
            "hand_id": f"h-{i}", "player_id": "hero",
            "won_hand": False, "net_bb": -20.0, "vpip": True, "ts": 100 + i,
        })
    a = await agent.run({"state": sample_cash_state}, ctx)
    b = await agent.run({"state": sample_cash_state}, ctx)
    assert a.output == b.output
