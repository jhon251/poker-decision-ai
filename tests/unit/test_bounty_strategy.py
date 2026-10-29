"""Unit tests for the Bounty Strategy Agent (V1.1 §4b)."""
from __future__ import annotations

import pytest

from poker_decision_ai.agents.bounty_strategy.agent import (
    BountyStrategyAgent,
    BountyStrategyOutput,
)
from poker_decision_ai.core.schemas import json_schema_for, validate_instance


def _state(hero_stack: float = 4000.0) -> dict:
    return {
        "game_type": "mtt", "street": "flop", "board": ["Qh", "7d", "2c"],
        "pot": 600.0, "to_call": 200.0, "bb": 100.0, "sb": 50.0,
        "hero_id": "hero",
        "players": [
            {"player_id": "p2", "stack": 3000.0, "status": "active", "is_hero": False, "bounty": 40.0},
            {"player_id": "hero", "stack": hero_stack, "status": "active", "is_hero": True, "bounty": 10.0},
        ],
        "tournament": {"bounty_type": "pko", "is_bounty": True, "progressive": True,
                       "players_remaining": 45, "total_chips": 100000.0},
        "cash": None,
    }


def _bounty(covered: bool = True, pressure: float = 0.6, villain_bounty: float = 40.0) -> dict:
    return {"is_bounty": True, "covered": covered, "bounty_pressure": pressure,
            "villain_bounty": villain_bounty}


@pytest.fixture
def agent():
    return BountyStrategyAgent()


async def test_schema_contract(agent):
    schema = json_schema_for(BountyStrategyOutput)
    assert schema["type"] == "object"
    for f in ("recommendations", "hunt_big_bounty", "avoid_big_bounty",
              "bounty_hunting_detected", "bubble_bounty_note",
              "final_table_bounty_note", "summary", "warnings"):
        assert f in schema["properties"]
    validate_instance(BountyStrategyOutput, BountyStrategyOutput())


async def test_not_applicable_without_state(agent, ctx):
    ok, _ = await agent.applicable({}, ctx)
    assert not ok


async def test_hunt_big_bounty(agent, ctx):
    env = await agent.run({
        "state": _state(hero_stack=4000.0),   # 40bb 深码
        "bounty": _bounty(covered=True, pressure=0.6),
        "tournament_stage": {"stage": "mid"},
    }, ctx)
    assert env.status in ("ok", "degraded")
    out = env.output
    assert out["hunt_big_bounty"] is True
    assert out["avoid_big_bounty"] is False
    types = [r["type"] for r in out["recommendations"]]
    assert "hunt_big_bounty" in types


async def test_avoid_protect_short_stack_bubble(agent, ctx):
    env = await agent.run({
        "state": _state(hero_stack=1500.0),   # 15bb 短码
        "bounty": _bounty(covered=False, pressure=0.1),
        "tournament_stage": {"stage": "bubble"},
    }, ctx)
    out = env.output
    assert out["avoid_big_bounty"] is True
    types = [r["type"] for r in out["recommendations"]]
    assert "avoid_big_bounty" in types
    assert "protect_stack" in types
    assert out["bubble_bounty_note"]


async def test_exploit_bounty_hunter(agent, ctx):
    env = await agent.run({
        "state": _state(),
        "bounty": _bounty(),
        "tournament_stage": {"stage": "mid"},
        "opponent": {"players": {"p2": {"feature_vector": {"bounty_hunting_tendency": 0.8}}}},
    }, ctx)
    out = env.output
    assert out["bounty_hunting_detected"].get("p2") is True
    types = [r["type"] for r in out["recommendations"]]
    assert "exploit_bounty_hunter" in types


async def test_no_bounty_empty_recommendations(agent, ctx):
    env = await agent.run({"state": _state()}, ctx)
    out = env.output
    assert out["recommendations"] == []
    assert out["hunt_big_bounty"] is False
    assert out["warnings"]


async def test_determinism_with_mock_llm(agent, ctx):
    inp = {
        "state": _state(),
        "bounty": _bounty(),
        "tournament_stage": {"stage": "mid"},
        "opponent": {"players": {"p2": {"feature_vector": {"bounty_hunting_tendency": 0.8}}}},
    }
    e1 = await agent.run(inp, ctx)
    e2 = await agent.run(inp, ctx)
    assert e1.output == e2.output
