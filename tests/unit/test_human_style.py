"""Unit tests for the Human Style Agent."""
from __future__ import annotations

import pytest

from poker_decision_ai.agents.human_style.agent import HumanStyleAgent, HumanStyleOutput
from poker_decision_ai.core.schemas import json_schema_for, validate_instance


@pytest.fixture
def agent():
    return HumanStyleAgent()


def _fv(tightness, aggression, calling=0.5):
    return {"tightness": tightness, "aggression": aggression, "bluff_tendency": 0.5,
            "calling_tendency": calling, "folding_tendency": 0.5,
            "overbet_tendency": 0.0, "check_raise_tendency": 0.0, "tilt_risk": 0.0,
            "icm_sensitivity": 0.5, "stage_adaptivity": 0.5}


def _opponent(fv, sample_size=100):
    return {"players": {"p2": {"player_id": "p2", "name": "Bob",
                               "feature_vector": fv, "sample_size": sample_size,
                               "behavior_statistics": {}}}}


async def test_schema_contract(agent):
    schema = json_schema_for(HumanStyleOutput)
    assert schema["type"] == "object"
    assert "players" in schema["properties"]
    validate_instance(HumanStyleOutput, HumanStyleOutput())


async def test_applicable(agent, ctx):
    ok, _reason = await agent.applicable({}, ctx)
    assert not ok
    ok2, _reason2 = await agent.applicable({"state": {"players": []}}, ctx)
    assert ok2


async def test_nit(agent, ctx, sample_cash_state):
    env = await agent.run({"state": sample_cash_state,
                           "opponent": _opponent(_fv(0.1, 0.1))}, ctx)
    assert env.output["players"]["p2"]["primary_label"] == "Nit"


async def test_maniac(agent, ctx, sample_cash_state):
    env = await agent.run({"state": sample_cash_state,
                           "opponent": _opponent(_fv(0.8, 0.7))}, ctx)
    assert env.output["players"]["p2"]["primary_label"] == "Maniac"


async def test_calling_station(agent, ctx, sample_cash_state):
    env = await agent.run({"state": sample_cash_state,
                           "opponent": _opponent(_fv(0.5, 0.3, calling=0.9))}, ctx)
    assert "Calling Station" in env.output["players"]["p2"]["labels"]


async def test_no_opponent_default(agent, ctx, sample_cash_state):
    env = await agent.run({"state": sample_cash_state}, ctx)
    out = env.output
    assert out["players"]["p2"]["primary_label"] == "Regular"
    assert any(w for w in out.get("warnings", []))


async def test_determinism(agent, ctx, sample_cash_state):
    a = await agent.run({"state": sample_cash_state, "opponent": _opponent(_fv(0.8, 0.7))}, ctx)
    b = await agent.run({"state": sample_cash_state, "opponent": _opponent(_fv(0.8, 0.7))}, ctx)
    assert a.output == b.output
