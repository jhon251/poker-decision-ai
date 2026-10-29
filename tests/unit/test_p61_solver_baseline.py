"""V3F-P61 — Solver Baseline Integration 测试（环境无真实 solver 路线）。

覆盖：四通道环境发现（确定性、不可用诚实）；unavailable 语义
（ProviderNotConfiguredError）；malformed 输出拒绝（非 JSON/缺 strategy/
频率越界/频率和 >1）；provenance（BLOCKED 原因 + solver_label 纪律）；
P60 集成（solver_source 注入 + 候选共存 + 不参与双计 + 不可用时决策
语义不变）；future fixture（schema 契约草案，明确非 solver ground truth）。
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from poker_decision_ai.core.errors import ProviderError
from poker_decision_ai.core.providers.solver import (
    PostflopSolverCLIProvider,
    discover_solver_environment,
    parse_solver_strategy_output,
)
from poker_decision_ai.strategy_intel import evaluate_intel
from poker_decision_ai.strategy_robustness import make_state

AUDIT = Path("docs/V3F_P61_SOLVER_ENVIRONMENT_AUDIT.json")


# ------------------------------------------------------------ 环境发现
def test_discovery_channels_reported():
    r = discover_solver_environment()
    assert set(r["channels"]) == {"executable_path", "path_search",
                                  "pip_package", "sample_outputs"}
    assert r["contract"] == "P61"
    # 本环境（P61 审计）：全部通道未命中 → BLOCKED 诚实
    if not r["available"]:
        assert "no real solver found" in r["blocked_reason"]


def test_discovery_deterministic():
    assert discover_solver_environment() == discover_solver_environment()


# ------------------------------------------------------------ unavailable
def test_provider_unavailable_semantics():
    """未配置 → 使用即 ProviderNotConfiguredError（不伪造 solve）。"""
    prov = PostflopSolverCLIProvider(executable="")
    assert prov.is_configured() is False
    assert "path not set" in prov.status_reason()
    import asyncio
    with pytest.raises(Exception) as ei:
        asyncio.run(prov.solve({"board": ["Ah", "Kd", "7c"]}))
    assert "not configured" in str(ei.value)


def test_provider_missing_executable():
    prov = PostflopSolverCLIProvider(executable="Z:/nope/postflop-solver.exe")
    assert prov.is_configured() is False
    assert "not found" in prov.status_reason()


# ------------------------------------------------------------ malformed 输出
def test_parse_rejects_non_json():
    with pytest.raises(ProviderError, match="not JSON"):
        parse_solver_strategy_output("not json at all")


def test_parse_rejects_missing_strategy():
    with pytest.raises(ProviderError, match="missing 'strategy'"):
        parse_solver_strategy_output('{"ev": {"bet": 1.0}}')


def test_parse_rejects_frequency_out_of_range():
    with pytest.raises(ProviderError, match="out of range"):
        parse_solver_strategy_output('{"strategy": {"bet": 1.5}}')
    with pytest.raises(ProviderError, match="out of range"):
        parse_solver_strategy_output('{"strategy": {"bet": -0.1}}')


def test_parse_rejects_frequencies_sum_above_one():
    with pytest.raises(ProviderError, match="sum"):
        parse_solver_strategy_output('{"strategy": {"bet": 0.7, "call": 0.7}}')


def test_parse_accepts_valid_draft_contract():
    ok = parse_solver_strategy_output(
        '{"strategy": {"bet": 0.3, "check": 0.7}, "ev": {"bet": 0.4}}')
    assert ok["strategy"] == {"bet": 0.3, "check": 0.7}
    assert ok["contract"] == "p61-draft"      # 草案契约显式标注


# ------------------------------------------------------------ P60 集成
def test_intel_solver_source_default_blocked():
    r = evaluate_intel(make_state("flop", ["9c", "4c"], ["Ah", "7c", "2c"],
                                  "IP", 10.0, "check"),
                       {"hero_equity": 0.45})
    assert r["solver_source"]["available"] is False
    assert "BLOCKED" in r["solver_source"]["status"]


def test_intel_solver_source_discovery_injection():
    """discovery 结果可注入 intel（BLOCKED 原因来自真实审计通道）。"""
    disc = discover_solver_environment()
    r = evaluate_intel(make_state("flop", ["9c", "4c"], ["Ah", "7c", "2c"],
                                  "IP", 10.0, "check"),
                       {"hero_equity": 0.45},
                       solver_source_override={
                           "available": disc["available"],
                           "status": disc["blocked_reason"]})
    assert r["solver_source"]["available"] is False
    assert "no real solver found" in r["solver_source"]["status"]


def test_heuristic_and_solver_candidate_coexist():
    """solver_candidate（未来接口）与 heuristic 候选共存：独立来源、
    不参与 heuristic 的 base/adjustments 恒等式、solver_label 纪律。"""
    st = make_state("flop", ["9c", "4c"], ["Ah", "7c", "2c"], "IP", 10.0,
                    "check")
    r = evaluate_intel(
        st, {"hero_equity": 0.45}, range_src={"villains": {"v2": {
            "distribution": {"AA": 0.1, "KK": 0.1, "AKs": 0.1,
                             "QQ": 0.1, "AQo": 0.2, "KQo": 0.4}}}},
        solver_candidate={"action": "bet", "utility_bb": 0.9,
                          "frequency": 0.65})
    # heuristic 候选仍按自身恒等式（base+cv==final）
    for c in r["candidates"]:
        assert c["base_ev_bb"] + c["adjustments"]["saca_cv_bb"] == \
            pytest.approx(c["final_ev_bb"], abs=3e-6)
    # solver 候选独立存在且标注纪律
    ss = r["solver_source"]
    assert ss["available"] is True
    assert ss["candidate"]["provenance"] == "solver_label"
    assert ss["candidate"]["is_human_ground_truth"] is False
    # 不与 heuristic 候选混合（独立字段）
    assert all(c["action"] != "solver" for c in r["candidates"])


def test_solver_unavailable_keeps_decision_semantics():
    """solver 不可用 → CASH 决策语义与 P60 完全一致（基线不因 solver
    缺失而改变）。"""
    st = make_state("flop", ["9c", "4c"], ["Ah", "7c", "2c"], "IP", 10.0,
                    "check")
    kw = {"range_src": {"villains": {"v2": {"distribution": {
        "AA": 0.1, "KK": 0.1, "AKs": 0.1, "QQ": 0.1, "AQo": 0.2,
        "KQo": 0.4}}}}}
    r_default = evaluate_intel(st, {"hero_equity": 0.45}, **kw)
    r_override = evaluate_intel(
        st, {"hero_equity": 0.45}, solver_source_override={
            "available": False, "status": "no solver"},
        **{k: copy.deepcopy(v) for k, v in kw.items()})
    a = {k: v for k, v in r_default.items() if k != "solver_source"}
    b = {k: v for k, v in r_override.items() if k != "solver_source"}
    assert a == b                                  # 决策语义不变


# ------------------------------------------------------------ future fixture
def test_future_fixture_is_schema_contract_not_ground_truth():
    """fixture 文档明确：schema 契约草案 ≠ solver ground truth。"""
    fixture = {
        "fixture_type": "schema_contract_draft",
        "is_solver_ground_truth": False,
        "example": {"strategy": {"bet": 0.4, "check": 0.6},
                    "note": "aligned with real solver output in a future "
                            "phase"},
    }
    assert fixture["is_solver_ground_truth"] is False
    ok = parse_solver_strategy_output(json.dumps(fixture["example"]))
    assert ok["contract"] == "p61-draft"


def test_audit_snapshot_declares_blocked():
    if not AUDIT.exists():
        pytest.skip("audit snapshot not generated")
    d = json.loads(AUDIT.read_text(encoding="utf-8"))
    assert d["conclusion"] == "NO REAL SOLVER PRESENT — BLOCKED persists (P35)"
    assert d["available"] is False
    assert d["honesty"]["no_fake"]


# ------------------------------------------------------------ 确定性
def test_intel_with_solver_override_deterministic():
    st = make_state("flop", ["9c", "4c"], ["Ah", "7c", "2c"], "IP", 10.0,
                    "check")
    kw = {"solver_source_override": {"available": False, "status": "x"}}
    a = evaluate_intel(st, {"hero_equity": 0.45}, **kw)
    b = evaluate_intel(st, {"hero_equity": 0.45}, **kw)
    assert a == b
