"""Unit tests for the Tournament Stage Agent."""
from __future__ import annotations

import pytest

from poker_decision_ai.agents.tournament_stage.agent import (
    TournamentStageAgent,
    TournamentStageOutput,
)
from poker_decision_ai.core.schemas import json_schema_for, validate_instance

_VALID_STAGES = ("early", "mid", "late", "bubble", "itm", "final_table", "heads_up")


@pytest.fixture
def agent():
    return TournamentStageAgent()


async def test_schema_contract(agent):
    schema = json_schema_for(TournamentStageOutput)
    assert schema["type"] == "object"
    assert "stage" in schema["properties"]
    assert "bb_avg" in schema["properties"]
    out = TournamentStageOutput()
    validate_instance(TournamentStageOutput, out)


async def test_applicable_cash_and_tournament(agent, ctx, sample_cash_state, sample_tournament_state):
    ok, reason = await agent.applicable({"state": sample_cash_state}, ctx)
    assert not ok
    ok2, _ = await agent.applicable({"state": sample_tournament_state}, ctx)
    assert ok2


async def test_heads_up_and_final_table(agent, ctx, sample_tournament_state):
    s = dict(sample_tournament_state)
    s["tournament"] = dict(s["tournament"])

    s["tournament"]["players_remaining"] = 2
    env = await agent.run({"state": s}, ctx)
    assert env.output["stage"] == "heads_up"

    s["tournament"]["players_remaining"] = 9
    env = await agent.run({"state": s}, ctx)
    assert env.output["stage"] == "final_table"


async def test_sample_stage_valid(agent, ctx, sample_tournament_state):
    env = await agent.run({"state": sample_tournament_state}, ctx)
    out = env.output
    assert out["stage"] in _VALID_STAGES
    assert out["players_remaining"] == 45
    assert out["bb_avg"] > 0


async def test_deterministic(agent, ctx, sample_tournament_state):
    a = await agent.run({"state": sample_tournament_state}, ctx)
    b = await agent.run({"state": sample_tournament_state}, ctx)
    assert a.output == b.output
