"""Unit tests for the Research Agent (knowledge-base archiver)."""
from __future__ import annotations

import pytest

from poker_decision_ai.agents.research.agent import ResearchAgent, ResearchOutput
from poker_decision_ai.core.schemas import json_schema_for, validate_instance


@pytest.fixture
def agent():
    return ResearchAgent()


async def test_schema_contract(agent):
    schema = json_schema_for(ResearchOutput)
    assert schema["type"] == "object"
    for f in ("notes", "stored"):
        assert f in schema["properties"]
    out = ResearchOutput(notes=[], stored=0)
    validate_instance(ResearchOutput, out)


async def test_finding_and_counter_example_stored(agent, ctx):
    learning = {
        "updates": [
            {"decision": "update", "significance": 0.95, "metric": "vpip", "id": "hero",
             "old_value": 0.3, "new_value": 0.41, "samples": 700},
            {"decision": "reject", "significance": 0.1, "metric": "pfr", "id": "hero",
             "old_value": 0.2, "new_value": 0.25, "samples": 100},
        ],
        "bad_beat_protection": {"triggered": True, "note": "bad beat"},
    }
    env = await agent.run({"learning": learning}, ctx)
    out = env.output
    assert out["stored"] == 2
    kinds = {n["kind"] for n in out["notes"]}
    assert kinds == {"finding", "counter_example"}
    findings = ctx.repos.research.list(kind="finding")
    assert len(findings) == 1
    assert findings[0]["title"] == "vpip 显著变化: hero"


async def test_empty_input_stored_zero(agent, ctx):
    env = await agent.run({}, ctx)
    out = env.output
    assert out["stored"] == 0
    assert out["notes"] == []
