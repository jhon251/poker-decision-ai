"""Unit tests for the Hypothesis Agent."""
from __future__ import annotations

import pytest

from poker_decision_ai.agents.hypothesis.agent import HypothesisAgent, HypothesisOutput
from poker_decision_ai.core.schemas import json_schema_for, validate_instance

_REQUIRED_FIELDS = {"id", "hypothesis", "reason", "required_data", "variables",
                    "control_group", "test_method", "expected_result",
                    "status", "created_at"}


@pytest.fixture
def agent():
    return HypothesisAgent()


async def test_schema_contract(agent):
    schema = json_schema_for(HypothesisOutput)
    assert schema["type"] == "object"
    assert "hypotheses" in schema["properties"]
    out = HypothesisOutput(hypotheses=[])
    validate_instance(HypothesisOutput, out)


async def test_empty_input_produces_complete_hypotheses(agent, ctx):
    env = await agent.run({}, ctx)
    out = env.output
    assert len(out["hypotheses"]) >= 2
    for h in out["hypotheses"]:
        assert _REQUIRED_FIELDS.issubset(set(h.keys()))
        assert h["status"] == "proposed"
        assert h["required_data"], "required_data must be non-empty"
        assert h["variables"], "variables must be non-empty"


async def test_llm_mock_path_does_not_crash(agent, ctx):
    env = await agent.run({
        "statistics": {"hero_stats": {"vpip": {"value": 0.5, "samples": 100}}},
    }, ctx)
    out = env.output
    assert len(out["hypotheses"]) >= 2
    for h in out["hypotheses"]:
        assert _REQUIRED_FIELDS.issubset(set(h.keys()))


async def test_hypotheses_stored_in_research_repo(agent, ctx):
    env = await agent.run({}, ctx)
    out = env.output
    stored = ctx.repos.research.list(kind="hypothesis")
    assert len(stored) >= 2
    output_ids = {h["id"] for h in out["hypotheses"]}
    stored_ids = {r["id"] for r in stored}
    assert output_ids <= stored_ids
