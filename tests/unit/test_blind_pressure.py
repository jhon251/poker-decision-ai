"""Unit tests for the Blind Pressure Agent."""
from __future__ import annotations

import pytest

from poker_decision_ai.agents.blind_pressure.agent import BlindPressureAgent, BlindPressureOutput
from poker_decision_ai.core.schemas import json_schema_for, validate_instance


@pytest.fixture
def agent():
    return BlindPressureAgent()


def _set_hero_stack(state: dict, stack: float) -> dict:
    """Deep-enough copy of the state with the hero's stack replaced."""
    s = dict(state)
    players = [dict(p) for p in s["players"]]
    for p in players:
        if p.get("is_hero") or p.get("player_id") == s.get("hero_id"):
            p["stack"] = stack
    s["players"] = players
    return s


def _set_hero_stack_bb(state: dict, bb_multiple: float) -> dict:
    """Set the hero stack to ``bb_multiple`` big blinds (robust to fixture bb)."""
    bb = float(state.get("bb") or 1.0)
    return _set_hero_stack(state, bb * bb_multiple)


async def test_schema_contract(agent):
    schema = json_schema_for(BlindPressureOutput)
    assert schema["type"] == "object"
    for f in ("stack_bb", "m_score", "blind_pressure", "steal_pressure",
              "resteal_pressure", "open_shove_zone"):
        assert f in schema["properties"]
    out = BlindPressureOutput(stack_bb=10.0, blind_pressure=0.75, open_shove_zone=True)
    validate_instance(BlindPressureOutput, out)


async def test_applicable_cash(agent, ctx, sample_cash_state):
    ok, _reason = await agent.applicable({"state": sample_cash_state}, ctx)
    assert ok


async def test_short_stack(agent, ctx, sample_cash_state):
    state = _set_hero_stack_bb(sample_cash_state, 10.0)  # 10bb
    env = await agent.run({"state": state}, ctx)
    assert env.status in ("ok", "degraded")
    out = env.output
    assert out["open_shove_zone"] is True
    assert out["blind_pressure"] > 0.7
    assert out["stack_bb"] == pytest.approx(10.0, rel=0.01)


async def test_deep_stack(agent, ctx, sample_cash_state):
    state = _set_hero_stack_bb(sample_cash_state, 100.0)  # 100bb
    env = await agent.run({"state": state}, ctx)
    out = env.output
    assert out["blind_pressure"] == pytest.approx(0.0, abs=0.001)
    assert out["open_shove_zone"] is False


async def test_cash_deep_applies_not_skipped(agent, ctx, sample_cash_state):
    ok, _reason = await agent.applicable({"state": sample_cash_state}, ctx)
    assert ok
    state = _set_hero_stack_bb(sample_cash_state, 100.0)  # cash 深码也适用
    env = await agent.run({"state": state}, ctx)
    assert env.output["blind_pressure"] == pytest.approx(0.0, abs=0.001)


async def test_deterministic(agent, ctx, sample_cash_state):
    state = _set_hero_stack_bb(sample_cash_state, 10.0)
    a = await agent.run({"state": state}, ctx)
    b = await agent.run({"state": state}, ctx)
    assert a.output == b.output
