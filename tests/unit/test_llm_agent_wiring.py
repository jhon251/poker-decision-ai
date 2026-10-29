"""V2 回归：非 Mock LLM 的 ctx.llm_json 解包路径（防 tuple/dict 解包回归）。

V1 测试全部走 Mock（LLM 润色块被 `!= "mock"` 跳过），真实 Provider 下
Judge/Debate 曾把 (data, usage) 元组当 dict 解包导致崩溃——本测试用
非 Mock 假 Provider 覆盖全部 6 个 LLM 消费 Agent。
"""
from __future__ import annotations

import json

from poker_decision_ai.core.providers.llm import LLMProvider, LLMResponse, LLMUsage


class FakeRealLLM(LLMProvider):
    """非 Mock 假 Provider：返回与请求 schema 匹配的合法 JSON。"""

    name = "fake_real"
    is_mock = False
    model = "fake"

    def __init__(self, payload: dict):
        self.payload = payload

    async def complete(self, messages, *, json_schema=None, temperature=None,
                       max_tokens=None, tier="fast") -> LLMResponse:
        return LLMResponse(text=json.dumps(self.payload, ensure_ascii=False),
                           usage=LLMUsage(10, 10, 20, 0.0), model="fake")


async def test_judge_real_llm_polish_path(ctx, sample_cash_state):
    from poker_decision_ai.agents.judge.agent import JudgeAgent

    ctx.providers.llm = FakeRealLLM({"reasoning": "真实 LLM 润色后的决策说明。"})
    env = await JudgeAgent().run({
        "state": sample_cash_state,
        "equity": {"hero_equity": 0.6, "pot_odds": {"required_equity": 0.3}},
    }, ctx)
    assert env.status in ("ok", "degraded")
    assert "真实 LLM 润色" in env.output["reasoning"]


async def test_debate_real_llm_polish_path(ctx, sample_cash_state):
    from poker_decision_ai.agents.debate.agent import DebateAgent

    ctx.providers.llm = FakeRealLLM({
        "strongest_argument": "LLM 润色论点", "counterargument": "LLM 润色反方",
        "recommendation": "LLM 润色建议",
    })
    env = await DebateAgent().run({
        "state": sample_cash_state,
        "solver": {"gto_action": "call", "confidence": 0.5},
        "exploit": {"best_exploit": {"suggested_action": "fold", "confidence": 0.5}},
    }, ctx)
    assert env.status in ("ok", "degraded")
    out = env.output
    assert out["strongest_argument"] == "LLM 润色论点"
    assert out["recommendation"] == "LLM 润色建议"


async def test_exploit_real_llm_path(ctx, sample_cash_state):
    from poker_decision_ai.agents.exploit.agent import ExploitAgent

    ctx.providers.llm = FakeRealLLM({"summary": "LLM 剥削总结。"})
    env = await ExploitAgent().run({
        "state": sample_cash_state,
        "opponent": {"players": {"Bob": {"feature_vector": {"fold_to_cbet": 0.9},
                                         "behavior_statistics": {"fold_to_cbet": {"value": 0.8, "samples": 100}}}}},
    }, ctx)
    assert env.status in ("ok", "degraded")


async def test_bounty_strategy_real_llm_path(ctx):
    from poker_decision_ai.agents.bounty_strategy.agent import BountyStrategyAgent

    ctx.providers.llm = FakeRealLLM({"summary": "LLM 赏金总结。"})
    env = await BountyStrategyAgent().run({
        "state": {"game_type": "mtt", "players": [], "bb": 100.0},
        "bounty": {"is_bounty": True, "bounty_pressure": 0.6, "hero_bounty": 10.0,
                   "villain_bounty": 100.0},
    }, ctx)
    assert env.status in ("ok", "degraded")


async def test_hypothesis_real_llm_path(ctx):
    from poker_decision_ai.agents.hypothesis.agent import HypothesisAgent

    ctx.providers.llm = FakeRealLLM({
        "hypothesis": "LLM 假设", "reason": "r",
        "required_data": ["hands"], "variables": ["x"], "control_group": "c",
        "test_method": "t", "expected_result": "e",
    })
    env = await HypothesisAgent().run({}, ctx)
    assert env.status in ("ok", "degraded")
    assert any(h["hypothesis"] == "LLM 假设" for h in env.output["hypotheses"])


async def test_research_real_llm_path(ctx):
    from poker_decision_ai.agents.research.agent import ResearchAgent

    ctx.providers.llm = FakeRealLLM({"summary": "LLM 归档总结。"})
    env = await ResearchAgent().run({
        "learning": {"updates": [{"target": "player", "id": "Bob", "metric": "vpip",
                                  "old_value": 0.2, "new_value": 0.4, "samples": 500,
                                  "significance": 0.99, "decision": "update",
                                  "reason": "x"}]},
    }, ctx)
    assert env.status in ("ok", "degraded")
