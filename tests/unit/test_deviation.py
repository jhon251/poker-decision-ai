"""Unit tests for the Deviation Agent."""
from __future__ import annotations

import pytest

from poker_decision_ai.agents.deviation.agent import ACTIONS, DeviationAgent, DeviationOutput
from poker_decision_ai.core.schemas import json_schema_for, validate_instance


@pytest.fixture
def agent():
    return DeviationAgent()


def _set_dev(monkeypatch, ctx, **overrides):
    cfg = {"enabled": True, "probability": 0.15, "min_probability": 0.0,
           "max_probability": 0.2, "seed": None, "experiment_mode": False}
    cfg.update(overrides)
    monkeypatch.setattr(ctx.config, "deviation_config", lambda: dict(cfg))


async def test_schema_contract(agent):
    schema = json_schema_for(DeviationOutput)
    assert schema["type"] == "object"
    for field in ("deviated", "deviation_probability", "original_action",
                  "final_action", "reason", "experiment_mode"):
        assert field in schema["properties"]
    out = DeviationOutput()
    validate_instance(DeviationOutput, out)


async def test_applicable(agent, ctx):
    ok, _ = await agent.applicable({}, ctx)
    assert ok


async def test_probability_zero_never_deviates(agent, ctx, monkeypatch):
    _set_dev(monkeypatch, ctx, probability=0.0, seed=42)
    env = await agent.run({"state": {}, "judge": {"action": "call"}}, ctx)
    out = env.output
    assert out["deviated"] is False
    assert out["final_action"] == "call"


async def test_probability_one_seeded_deviates(agent, ctx, monkeypatch):
    # experiment_mode allows p=1.0 beyond the default [0, 0.2] window
    _set_dev(monkeypatch, ctx, probability=1.0, seed=42, experiment_mode=True)
    env = await agent.run({"state": {}, "judge": {"action": "call"}}, ctx)
    out = env.output
    assert out["deviated"] is True
    assert out["final_action"] != "call"
    assert out["final_action"] in ACTIONS


async def test_disabled_passthrough(agent, ctx, monkeypatch):
    _set_dev(monkeypatch, ctx, enabled=False)
    env = await agent.run({"state": {}, "judge": {"action": "raise"}}, ctx)
    out = env.output
    assert out["deviated"] is False
    assert out["original_action"] == "raise"
    assert out["final_action"] == "raise"


async def test_judge_missing(agent, ctx, monkeypatch):
    _set_dev(monkeypatch, ctx, probability=1.0, seed=42)
    env = await agent.run({"state": {}}, ctx)
    out = env.output
    assert out["deviated"] is False
    assert out["original_action"] == "fold"
    assert out["final_action"] == "fold"


async def test_triggered_warning_mentions_deviation(agent, ctx, monkeypatch):
    _set_dev(monkeypatch, ctx, probability=1.0, seed=42, experiment_mode=True)
    env = await agent.run({"state": {}, "judge": {"action": "call"}}, ctx)
    out = env.output
    assert out["warnings"]
    assert any("偏离" in w["message"] for w in out["warnings"])


async def test_probability_clipped_out_of_range(agent, ctx, monkeypatch):
    _set_dev(monkeypatch, ctx, probability=0.9, seed=42)  # beyond max 0.2
    env = await agent.run({"state": {}, "judge": {"action": "call"}}, ctx)
    out = env.output
    assert out["deviation_probability"] <= 0.2
    assert any(w["code"] == "DEVIATION_PROB_CLIPPED" for w in out["warnings"])
