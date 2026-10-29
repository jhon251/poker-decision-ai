"""Unit tests for the Board Agent (reference test pattern)."""
from __future__ import annotations

import pytest

from poker_decision_ai.agents.board.agent import BoardAgent, BoardFeatures
from poker_decision_ai.core.errors import AgentExecutionError
from poker_decision_ai.core.schemas import json_schema_for, validate_instance


@pytest.fixture
def agent():
    return BoardAgent()


async def test_schema_contract(agent):
    schema = json_schema_for(BoardFeatures)
    assert schema["type"] == "object"
    assert "paired" in schema["properties"]
    # output schema must validate cleanly against a constructed instance
    out = BoardFeatures(street="flop", board=["Qh", "7d", "2c"])
    validate_instance(BoardFeatures, out)


async def test_not_applicable_preflop(agent, ctx):
    ok, _reason = await agent.applicable({"state": {"board": [], "street": "preflop"}}, ctx)
    assert not ok


async def test_wet_flush_board(agent, ctx):
    env = await agent.run({"state": {
        "board": ["As", "9s", "3s"], "street": "flop",
    }}, ctx)
    assert env.status in ("ok", "degraded")
    out = env.output
    assert out["monotone"] is True
    assert out["flush_draw_present"] is True
    assert out["wetness_score"] > 0.5
    assert out["texture"] == "wet"


async def test_dry_paired_board(agent, ctx):
    env = await agent.run({"state": {"board": ["Kh", "Kd", "2c"], "street": "flop"}}, ctx)
    out = env.output
    assert out["paired"] is True
    assert out["texture"] in ("static", "dry")
    assert "trips_draw" in out["draw_structure"]


async def test_rainbow_static(agent, ctx):
    env = await agent.run({"state": {"board": ["Qh", "7d", "2c"], "street": "flop"}}, ctx)
    out = env.output
    assert out["rainbow"] is True
    assert out["range_interaction"]["board_favors"] == "aggressor"


async def test_invalid_board_card(agent, ctx):
    with pytest.raises(AgentExecutionError):
        await agent.run({"state": {"board": ["Xx"], "street": "flop"}}, ctx)
