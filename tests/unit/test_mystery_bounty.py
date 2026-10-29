"""Unit tests for the Mystery Bounty Agent (V1.1 §4b)."""
from __future__ import annotations

import pytest

from poker_decision_ai.agents.mystery_bounty.agent import (
    MysteryBountyAgent,
    MysteryBountyOutput,
)
from poker_decision_ai.core.schemas import json_schema_for, validate_instance


def _mystery_state(phase_active: bool = False, pool: float = 5000.0,
                   players_remaining: int = 25) -> dict:
    return {
        "game_type": "mtt", "street": "flop", "board": [], "pot": 0.0,
        "to_call": 0.0, "bb": 100.0, "sb": 50.0, "hero_id": "hero",
        "players": [
            {"player_id": "p2", "stack": 3000.0, "status": "active", "is_hero": False, "bounty": 0.0},
            {"player_id": "hero", "stack": 4000.0, "status": "active", "is_hero": True, "bounty": 0.0},
        ],
        "tournament": {
            "bounty_type": "mystery", "is_bounty": True,
            "total_bounty_pool": pool, "players_remaining": players_remaining,
            "mystery_phase_active": phase_active,
        },
        "cash": None,
    }


@pytest.fixture
def agent():
    return MysteryBountyAgent()


async def test_schema_contract(agent):
    schema = json_schema_for(MysteryBountyOutput)
    assert schema["type"] == "object"
    for f in ("phase", "mystery_pool_remaining", "expected_bounty_value",
              "bounty_ev", "strategy_note", "provider", "warnings"):
        assert f in schema["properties"]
    validate_instance(MysteryBountyOutput, MysteryBountyOutput())


async def test_not_applicable_without_state(agent, ctx):
    ok, _ = await agent.applicable({}, ctx)
    assert not ok


async def test_pending_phase_expected_value(agent, ctx):
    env = await agent.run({"state": _mystery_state(phase_active=False)}, ctx)
    assert env.status in ("ok", "degraded")
    out = env.output
    assert out["phase"] == "pending"
    assert out["provider"] == "deterministic"
    assert out["mystery_pool_remaining"] == pytest.approx(5000.0)
    # pending 期望值 = pool / players_remaining = 5000 / 25
    assert out["expected_bounty_value"] == pytest.approx(200.0)
    # covered -> bounty_ev = 200 * 0.5 = 100
    assert out["bounty_ev"] == pytest.approx(100.0)
    assert out["strategy_note"]


async def test_active_phase(agent, ctx):
    env = await agent.run({"state": _mystery_state(phase_active=True)}, ctx)
    out = env.output
    assert out["phase"] == "active"
    assert out["expected_bounty_value"] == pytest.approx(200.0)
    assert out["strategy_note"]


async def test_no_tournament_all_zero(agent, ctx, sample_cash_state):
    env = await agent.run({"state": sample_cash_state}, ctx)
    out = env.output
    assert out["phase"] == "pending"
    assert out["expected_bounty_value"] == 0.0
    assert out["bounty_ev"] == 0.0
    assert out["mystery_pool_remaining"] == 0.0
    assert out["warnings"]


async def test_determinism(agent, ctx):
    e1 = await agent.run({"state": _mystery_state(phase_active=False)}, ctx)
    e2 = await agent.run({"state": _mystery_state(phase_active=False)}, ctx)
    assert e1.output == e2.output
