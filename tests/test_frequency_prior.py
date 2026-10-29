"""MC1-R5 — frequency_prior 确定性引擎 + 生产接线测试。

验收面：纯函数层（频率/Wilson/衰减/回退/无数据不编造）、索引层（聚合/
桶级衰减/回退）、agent（villain 选择/来源标注）、judge 消费（config 权重/
provenance/影子字段/诚实降级）、端到端 plan（frequency_prior 进 judge 输入）。
"""
from __future__ import annotations

import asyncio
import json

import pytest

from poker_decision_ai.domain.poker import frequency_prior as fp
from poker_decision_ai.domain.poker.frequency_prior import (
    ActionFreq, FrequencyObservation, FrequencyPrior, FrequencyPriorConfig,
    action_frequencies, observations_from_street_actions)


def _obs(action: str, n: int, player_id: str = "V1", street: str = "flop",
         facing_bet: bool = True, position: str | None = "BTN",
         ts: float | None = 1000.0) -> list[FrequencyObservation]:
    return [FrequencyObservation(player_id=player_id, street=street,
                                 facing_bet=facing_bet, action=action,
                                 position=position, ts=ts)
            for _ in range(n)]


# ---------------------------------------------------------------- 纯函数层

def test_rates_wilson_and_samples():
    obs = (_obs("fold", 6) + _obs("call", 3) + _obs("raise", 1))
    prior = action_frequencies(obs, player_id="V1", facing_bet=True,
                               position="BTN", street="flop",
                               config=FrequencyPriorConfig(min_sample_size=5))
    assert prior.source == "player×position×street"
    assert prior.p_continue == pytest.approx(0.4)
    assert prior.p_fold == pytest.approx(0.6)
    fold = prior.frequencies["fold"]
    assert fold["n"] == 6 and fold["n_eff"] == pytest.approx(10.0)
    assert 0.0 <= fold["wilson_low"] < 0.6 < fold["wilson_high"] <= 1.0
    # n=10 的 60% Wilson 上界应 < 1（区间非平凡）
    assert fold["wilson_high"] < 0.95


def test_time_decay_prefers_recent():
    old = _obs("fold", 5, ts=0.0)
    recent = _obs("call", 5, ts=10_000_000.0)
    prior = action_frequencies(old + recent, player_id="V1", facing_bet=True)
    assert prior.p_fold < 0.5           # 近期 call 权重更高
    no_decay = action_frequencies(
        [FrequencyObservation(player_id="V", street="flop", facing_bet=True,
                              action=a, ts=None)
         for a in ["fold"] * 5 + ["call"] * 5],
        player_id="V", facing_bet=True)
    assert no_decay.p_fold == pytest.approx(0.5)   # 无 ts → 等权（声明语义）


def test_fallback_chain_player_to_population():
    obs = _obs("fold", 3) + _obs("call", 2)      # n=5 < 默认阈值 20
    prior = action_frequencies(obs, player_id="V1", facing_bet=True,
                               config=FrequencyPriorConfig(min_sample_size=20))
    assert prior.source.startswith("population")
    codes = [w["code"] for w in prior.warnings]
    assert "INSUFFICIENT_SAMPLE" in codes        # 群体层也不足 → 显式标注
    assert prior.p_continue is not None          # 但仍给出真实观测的频率


def test_no_data_never_fabricates():
    prior = action_frequencies([], player_id="X", facing_bet=True)
    assert prior.source == "UNAVAILABLE"
    assert prior.frequencies is None
    assert prior.p_continue is None
    codes = [w["code"] for w in prior.warnings]
    assert "NO_DATA" in codes


def test_context_action_legality():
    obs = _obs("check", 5, facing_bet=False) + _obs("bet", 5, facing_bet=False)
    prior = action_frequencies(obs, player_id="V1", facing_bet=False)
    assert set(prior.frequencies) == {"check", "bet"}
    assert prior.p_continue == pytest.approx(0.5)
    assert prior.p_fold is None


def test_street_replay_derives_facing_bet():
    rows = [{"player_id": "P1", "action": "check"},
            {"player_id": "P2", "action": "bet", "amount": 4.0},
            {"player_id": "P1", "action": "call", "amount": 4.0}]
    obs = observations_from_street_actions(rows, street="flop")
    by = {(o.player_id, o.action): o.facing_bet for o in obs}
    assert by[("P1", "check")] is False
    assert by[("P2", "bet")] is False            # 首次进攻 → 未面对下注
    assert by[("P1", "call")] is True            # 面对下注跟注
    # 未知动作/post 不产生观测（不编造）
    rows2 = [{"player_id": "P1", "action": "post", "amount": 1.0},
             {"player_id": "P1", "action": "muck"}]
    assert observations_from_street_actions(rows2, street="flop") == []


# ---------------------------------------------------------------- 索引层

def _write_chunk(tmp_path: object) -> object:
    import pathlib
    d = pathlib.Path(tmp_path)
    rows_pf = [{"player_id": "P1", "action": "post", "amount": 0.5},
               {"player_id": "P2", "action": "post", "amount": 1.0},
               {"player_id": "P1", "action": "raise", "to": 3.0},
               {"player_id": "P2", "action": "call", "amount": 2.0}]
    rows_flop = [{"player_id": "P1", "action": "check"},
                 {"player_id": "P2", "action": "bet", "amount": 4.0}]
    (d / "chunk_000.jsonl").write_text(json.dumps({"canonical": {
        "hand_id": "t1", "started_at": "2025-01-01T00:00:00",
        "players": [{"player_id": "P1", "position": "BTN"},
                    {"player_id": "P2", "position": "BB"}],
        "preflop_actions": rows_pf, "flop_actions": rows_flop}}),
        encoding="utf-8")
    return d


def test_index_query_and_fallback(tmp_path):
    from poker_decision_ai.agents.frequency_prior.agent import (
        _query_index, build_index_from_chunks)
    d = _write_chunk(tmp_path)
    index = build_index_from_chunks([d / "chunk_000.jsonl"])
    assert index, "index should not be empty"
    cfg = FrequencyPriorConfig(min_sample_size=1)
    q = _query_index(index, player_id="P2", position="BB", street="preflop",
                     facing_bet=True, cfg=cfg)
    assert q.source.startswith("player")
    assert q.p_continue == pytest.approx(1.0)    # P2 面对加注 → call
    assert q.frequencies["fold"]["n"] == 0
    # 无此玩家 → 群体回退 + 显式 warning
    q2 = _query_index(index, player_id="NOPE", position="BB", street="flop",
                      facing_bet=True, cfg=cfg)
    assert q2.source.startswith("population")
    assert any(w["code"] == "FALLBACK_POPULATION" for w in q2.warnings)
    # 群体样本也不足阈值 → INSUFFICIENT_SAMPLE（仍给真实观测频率，不编造）
    q3 = _query_index(index, player_id="NOPE", position="BB", street="flop",
                      facing_bet=True,
                      cfg=FrequencyPriorConfig(min_sample_size=20))
    assert q3.source.startswith("population")
    assert any(w["code"] == "INSUFFICIENT_SAMPLE" for w in q3.warnings)
    assert q3.frequencies is not None and q3.n_total >= 1


# ---------------------------------------------------------------- agent 层

def test_agent_selects_villain_and_labels_provenance(tmp_path, monkeypatch):
    from poker_decision_ai.agents.frequency_prior import agent as fpa
    d = _write_chunk(tmp_path)
    monkeypatch.setattr(fpa, "_load_index", lambda cfg, ctx, warnings: (
        fpa.build_index_from_chunks([d / "chunk_000.jsonl"]),
        {"source_dir": str(d), "decay_granularity": "bucket_ts_max"}))

    class Ctx:
        class config:
            @staticmethod
            def frequency_prior_config():
                return {"min_sample_size": 1, "half_life_days": 90,
                        "wilson_z": 1.96}

        def llm(self):
            raise AssertionError("LLM must not be called")

    agent = fpa.FrequencyPriorAgent()
    state = {"street": "preflop", "to_call": 2.0,
             "players": [
                 {"player_id": "Hero", "is_hero": True, "status": "active"},
                 {"player_id": "P1", "status": "active", "position": "BTN"},
                 {"player_id": "P2", "status": "active", "position": "BB"}]}
    out = asyncio.run(agent.handle({"state": state}, Ctx()))
    assert out.available
    assert out.selection["player_id"] in ("P1", "P2")
    assert out.selection["rule"] in ("last_aggressor", "first_active")
    prior = out.villains[out.selection["player_id"]]
    assert prior["provenance"]["source_type"] == "frequency_baseline"
    assert prior["n_total"] >= 1
    # 统计 agent 不可用 → fail-closed（无数据不编造）
    monkeypatch.setattr(fpa, "_load_index", lambda cfg, ctx, w: ({}, {}))
    out2 = asyncio.run(agent.handle({"state": state}, Ctx()))
    assert not out2.available
    assert any(w["code"] == "NO_DATA" for w in out2.warnings)


# ---------------------------------------------------------------- judge 层

class _FakeLLM:
    name = "mock"


class _FakeConfig:
    def __init__(self, d):
        self._d = d

    def frequency_prior_config(self):
        return self._d


class _FakeCtx:
    def __init__(self, d=None):
        self.config = _FakeConfig(d or {"enabled": True, "weight": 0.15,
                                        "fold_equity_gain": 0.3,
                                        "call_channel_gain": 0.2})

    def llm(self):
        return _FakeLLM()


def _freq_src(source: str = "player×position×street") -> dict:
    return {"available": True, "confidence": 0.6,
            "selection": {"player_id": "V1", "rule": "first_active"},
            "villains": {"V1": {
                "player_id": "V1", "position": "BB", "street": "flop",
                "facing_bet": True, "source": source,
                "context_actions": ["fold", "call", "raise"],
                "frequencies": {"fold": {"rate": 0.6, "wilson_low": 0.31,
                                         "wilson_high": 0.83, "n": 6,
                                         "n_eff": 10.0},
                                "call": {"rate": 0.3, "wilson_low": 0.1,
                                         "wilson_high": 0.6, "n": 3,
                                         "n_eff": 10.0},
                                "raise": {"rate": 0.1, "wilson_low": 0.0,
                                          "wilson_high": 0.4, "n": 1,
                                          "n_eff": 10.0}},
                "p_continue": 0.4, "p_fold": 0.6, "n_total": 10,
                "warnings": []}}}


def _judge_inp(freq_src):
    return {"state": {"to_call": 10.0, "pot": 40.0, "bb": 1.0,
                      "players": [{"player_id": "V1", "status": "active",
                                   "is_hero": False}]},
            "equity": {"hero_equity": 0.45,
                       "pot_odds": {"required_equity": 0.21}},
            "exploit": {"best_exploit": {"suggested_action": "fold",
                                         "confidence": 0.4}},
            "villain_frequency_prior": freq_src}


def test_judge_consumes_frequency_prior_with_provenance():
    from poker_decision_ai.agents.judge.agent import JudgeAgent
    out = asyncio.run(JudgeAgent().handle(_judge_inp(_freq_src()), _FakeCtx()))
    assert out.weights_used.get("villain_frequency_prior") == 0.15
    assert out.final_inputs_summary.get("villain_frequency_prior") == "ok"
    ev = [e for e in out.evidence
          if isinstance(e, dict) and e.get("source") == "frequency_baseline"]
    assert ev, "frequency prior evidence entry required"
    prov = ev[0]["provenance"]
    assert prov["source"] == "frequency_baseline"
    assert prov["n"] == 10
    assert prov["interval"]["fold_wilson_low"] == pytest.approx(0.31)
    # 影子字段：互不覆盖、分开记录
    assert out.shadow_comparison["frequency_baseline"]["suggested_action"] == "fold"
    assert out.shadow_comparison["frequency_baseline"]["provenance"] == \
        "frequency_baseline"
    assert out.shadow_comparison["legacy_response_model"]["provenance"] == \
        "opponent_conditioned_chain"


def test_judge_population_fallback_marked_degraded():
    from poker_decision_ai.agents.judge.agent import JudgeAgent
    out = asyncio.run(JudgeAgent().handle(
        _judge_inp(_freq_src(source="population×street")), _FakeCtx()))
    assert out.final_inputs_summary.get("villain_frequency_prior") == "degraded"


def test_judge_disabled_config_degrades_honestly():
    from poker_decision_ai.agents.judge.agent import JudgeAgent
    ctx = _FakeCtx({"enabled": False, "weight": 0.15})
    out = asyncio.run(JudgeAgent().handle(_judge_inp(_freq_src()), ctx))
    assert out.final_inputs_summary.get("villain_frequency_prior") == "disabled"
    assert "villain_frequency_prior" not in out.weights_used


def test_judge_without_prior_is_unchanged():
    from poker_decision_ai.agents.judge.agent import JudgeAgent
    inp = _judge_inp(None)
    inp.pop("villain_frequency_prior")
    out = asyncio.run(JudgeAgent().handle(inp, _FakeCtx()))
    assert "villain_frequency_prior" not in out.weights_used
    assert out.shadow_comparison == {}


# ------------------------------------------------------- 端到端 plan 接线

def test_plan_wires_frequency_prior_into_judge():
    from poker_decision_ai.domain.poker.gamemodes.base import judge_stage
    from poker_decision_ai.domain.poker.gamemodes.cash import CashGamePlugin
    stage = judge_stage(CashGamePlugin())
    task = next(t for t in stage.tasks if t.agent == "judge")
    assert task.input_template.get("villain_frequency_prior") == "$frequency_prior"


def test_registry_and_realtime_whitelist_include_agent():
    from poker_decision_ai.core.orchestration.registry import register_all_agents
    registry = register_all_agents()
    assert registry.get("frequency_prior") is not None
    from poker_decision_ai.realtime import FAST_AGENTS
    assert "frequency_prior" in FAST_AGENTS
