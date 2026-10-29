"""Unit tests for the Confidence Agent (dynamic weights)."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from poker_decision_ai.agents.confidence.agent import ConfidenceAgent, ConfidenceOutput
from poker_decision_ai.core.schemas import json_schema_for, validate_instance


@pytest.fixture
def agent():
    return ConfidenceAgent()


async def test_schema_contract(agent):
    schema = json_schema_for(ConfidenceOutput)
    assert schema["type"] == "object"
    for key in ("players", "hero_weights", "confidence", "evidence", "warnings"):
        assert key in schema["properties"]
    validate_instance(ConfidenceOutput, ConfidenceOutput())


def _opponent_profile(sample_size, *, full=False):
    """构造 opponent 画像输入。full=True 表示高质量（指标多/新/全近期）。"""
    profile = {"sample_size": sample_size, "behavior_statistics": {},
               "recent_behavior": {}, "historical_behavior": {}, "last_seen": ""}
    if full:
        profile["behavior_statistics"] = {
            f"m{i}": {"value": 0.5, "samples": sample_size} for i in range(10)
        }
        profile["recent_behavior"] = {"vpip": {"value": 0.5, "samples": sample_size}}
        profile["last_seen"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return profile


async def test_weight_direction(agent, ctx, sample_cash_state):
    low = {"players": {"hero": _opponent_profile(15)}}
    high = {"players": {"hero": _opponent_profile(5000, full=True)}}

    e_low = await agent.run({"state": sample_cash_state, "opponent": low}, ctx)
    e_high = await agent.run({"state": sample_cash_state, "opponent": high}, ctx)

    w_low = e_low.output["players"]["hero"]["weights"]
    w_high = e_high.output["players"]["hero"]["weights"]

    # 手数多 -> individual 权重大；手数少 -> population 权重大
    assert w_high["individual"] > w_low["individual"]
    assert w_low["population"] > w_high["population"]
    # 5000 手高质量 -> individual 接近上限 0.8
    assert w_high["individual"] >= 0.75
    # 15 手 -> individual_confidence ≈ 0.37
    assert e_low.output["players"]["hero"]["individual_confidence"] == pytest.approx(0.37, abs=0.01)


async def test_weights_sum_to_one(agent, ctx, sample_cash_state):
    low = {"players": {"hero": _opponent_profile(15)}}
    env = await agent.run({"state": sample_cash_state, "opponent": low}, ctx)
    w = env.output["players"]["hero"]["weights"]
    assert sum(w.values()) == pytest.approx(1.0, abs=1e-6)
    assert env.output["hero_weights"] == w


async def test_empty_input_no_exception(agent, ctx, sample_cash_state):
    env = await agent.run({"state": sample_cash_state}, ctx)
    out = env.output
    w = out["players"]["hero"]["weights"]
    assert sum(w.values()) == pytest.approx(1.0, abs=1e-6)
    # 无数据 -> 依赖 population
    assert w["population"] > w["individual"]


async def test_not_applicable_no_state(agent, ctx):
    ok, reason = await agent.applicable({}, ctx)
    assert not ok
