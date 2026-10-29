"""Unit tests for the Evaluation Agent."""
from __future__ import annotations

import pytest

from poker_decision_ai.agents.evaluation.agent import EvaluationAgent, EvaluationOutput
from poker_decision_ai.core.schemas import json_schema_for, validate_instance


@pytest.fixture
def agent():
    return EvaluationAgent()


def _good_judge() -> dict:
    return {
        "action": "raise",
        "confidence": 0.95,
        "key_evidence": ["e1", "e2", "e3", "e4", "e5"],
        "final_inputs_summary": {
            "equity": "ok", "solver": "ok", "exploit": "ok", "risk": "ok",
        },
    }


async def test_schema_contract(agent):
    schema = json_schema_for(EvaluationOutput)
    assert schema["type"] == "object"
    assert "decision_quality" in schema["properties"]
    assert "outcome" in schema["properties"]
    assert "decision_vs_outcome" in schema["properties"]
    out = EvaluationOutput(decision_quality={"score": 0.5, "factors": []})
    validate_instance(EvaluationOutput, out)


async def test_applicable_true(agent, ctx):
    ok, _ = await agent.applicable({"judge": _good_judge()}, ctx)
    assert ok


async def test_applicable_false(agent, ctx):
    ok, reason = await agent.applicable({}, ctx)
    assert not ok
    assert reason


async def test_high_confidence_scores_high(agent, ctx):
    env = await agent.run({"state": {"pot": 5.0}, "judge": _good_judge()}, ctx)
    assert env.status in ("ok", "degraded")
    out = env.output
    assert out["decision_quality"]["score"] > 0.6


async def test_outcome_lost_but_good_decision_separation(agent, ctx):
    inp = {
        "state": {"pot": 5.0},
        "judge": _good_judge(),
        "request": {"outcome": {"result": "lost", "net_bb": -10.0, "won": False}},
    }
    env = await agent.run(inp, ctx)
    out = env.output
    assert out["decision_vs_outcome"]["decision_good"] is True
    assert out["decision_vs_outcome"]["outcome_good"] is False
    assert "不修改策略" in out["decision_vs_outcome"]["note"]


async def test_no_outcome_null(agent, ctx):
    env = await agent.run({"state": {"pot": 5.0}, "judge": _good_judge()}, ctx)
    out = env.output
    assert out["outcome"] is None
    assert out["decision_vs_outcome"]["outcome_good"] is None


async def test_calibration_brier(agent, ctx):
    inp = {
        "judge": _good_judge(),
        "request": {"outcome": {"result": "lost", "net_bb": -10.0, "won": False,
                                "expected_equity": 0.8}},
    }
    env = await agent.run(inp, ctx)
    out = env.output
    assert out["calibration"] is not None
    assert out["calibration"]["brier_score"] == pytest.approx((0.8 - 0.0) ** 2, abs=1e-3)


async def test_no_calibration_without_expected(agent, ctx):
    env = await agent.run({"judge": _good_judge()}, ctx)
    out = env.output
    assert out["calibration"] is None


async def test_persists_decision(agent, ctx):
    rid = ctx.request_id
    await agent.run({"state": {"pot": 5.0}, "judge": _good_judge()}, ctx)
    stored = ctx.repos.decisions.get(rid)
    assert stored is not None
    assert stored["evaluation"]["decision_quality"]["score"] > 0.0


async def test_degrade_no_input(agent, ctx):
    env = await agent.run({}, ctx)
    assert env.status in ("ok", "degraded")
    out = env.output
    assert out["decision_quality"]["score"] <= 0.5
