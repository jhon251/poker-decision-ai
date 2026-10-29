"""Unit tests for the Image Agent."""
from __future__ import annotations

import pytest

from poker_decision_ai.agents.image.agent import ImageAgent, ImageOutput
from poker_decision_ai.core.schemas import json_schema_for, validate_instance


@pytest.fixture
def agent():
    return ImageAgent()


async def test_schema_contract(agent):
    schema = json_schema_for(ImageOutput)
    assert schema["type"] == "object"
    assert "hero_image" in schema["properties"]
    assert "villain_reactions" in schema["properties"]
    out = ImageOutput()
    validate_instance(ImageOutput, out)


async def test_applicable(agent, ctx):
    ok, _ = await agent.applicable({"state": {"game_type": "cash"}}, ctx)
    assert ok
    ok2, _ = await agent.applicable({}, ctx)
    assert not ok2


async def test_tight_aggressive_labels(agent, ctx, sample_cash_state):
    statistics = {"hero_stats": {
        "vpip": {"value": 0.1, "samples": 200},
        "aggression_frequency": {"value": 0.5, "samples": 200},
    }}
    env = await agent.run({"state": sample_cash_state, "statistics": statistics}, ctx)
    assert env.status in ("ok", "degraded")
    out = env.output
    assert "Tight" in out["hero_image"]["labels"]
    assert "Aggressive" in out["hero_image"]["labels"]


async def test_no_input_neutral(agent, ctx, sample_cash_state):
    env = await agent.run({"state": sample_cash_state}, ctx)
    out = env.output
    scores = out["hero_image"]["scores"]
    assert scores["tight"] == pytest.approx(0.5)
    assert scores["aggressive"] == pytest.approx(0.5)
    assert out["warnings"]


async def test_deterministic(agent, ctx, sample_cash_state):
    statistics = {"hero_stats": {
        "vpip": {"value": 0.1, "samples": 200},
        "aggression_frequency": {"value": 0.5, "samples": 200},
    }}
    a = await agent.run({"state": sample_cash_state, "statistics": statistics}, ctx)
    b = await agent.run({"state": sample_cash_state, "statistics": statistics}, ctx)
    assert a.output == b.output
