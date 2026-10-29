"""Unit tests for the Strategy Evolution Agent."""
from __future__ import annotations

import pytest

from poker_decision_ai.agents.strategy_evolution.agent import (
    StrategyEvolutionAgent,
    StrategyEvolutionOutput,
)
from poker_decision_ai.core.schemas import json_schema_for, validate_instance


@pytest.fixture
def agent():
    return StrategyEvolutionAgent()


async def test_schema_contract(agent):
    schema = json_schema_for(StrategyEvolutionOutput)
    assert schema["type"] == "object"
    assert "players" in schema["properties"]
    out = StrategyEvolutionOutput()
    validate_instance(StrategyEvolutionOutput, out)


async def test_applicable(agent, ctx):
    ok, _ = await agent.applicable({"state": {"game_type": "cash"}}, ctx)
    assert ok
    ok2, _ = await agent.applicable({}, ctx)
    assert not ok2


async def test_drift_detected(agent, ctx, sample_cash_state):
    # 30 old tight hands (vpip False), then 20 recent loose hands (vpip True).
    for i in range(30):
        ctx.repos.hands.save({
            "hand_id": f"old-{i}", "player_id": "p2",
            "vpip": False, "pfr": False, "ts": 1000 + i,
        })
    for i in range(20):
        ctx.repos.hands.save({
            "hand_id": f"new-{i}", "player_id": "p2",
            "vpip": True, "pfr": True, "ts": 2000 + i,
        })
    env = await agent.run({"state": sample_cash_state}, ctx)
    assert env.status in ("ok", "degraded")
    p2 = env.output["players"]["p2"]
    assert p2["drift_detected"] is True
    assert p2["sudden_shift"] is True
    assert p2["adaptation_direction"] == "looser"


async def test_no_records_all_false(agent, ctx, sample_cash_state):
    env = await agent.run({"state": sample_cash_state}, ctx)
    p2 = env.output["players"]["p2"]
    assert p2["drift_detected"] is False
    assert p2["sudden_shift"] is False
    assert p2["regime_change"] is False
    assert p2["change_score"] == 0.0
    assert any(w["code"] == "NO_HANDS" for w in env.output["warnings"])


async def test_deterministic(agent, ctx, sample_cash_state):
    for i in range(30):
        ctx.repos.hands.save({"hand_id": f"old-{i}", "player_id": "p2", "vpip": False, "ts": 1000 + i})
    for i in range(20):
        ctx.repos.hands.save({"hand_id": f"new-{i}", "player_id": "p2", "vpip": True, "ts": 2000 + i})
    a = await agent.run({"state": sample_cash_state}, ctx)
    b = await agent.run({"state": sample_cash_state}, ctx)
    assert a.output == b.output
