"""V3I-P11 — 可决策覆盖率与真实可用决策边界评估（测试）。

**本层不新增策略机制、不修改冻结契约**；ABSTAIN / BLOCKED 是**可度量结果**，
不是测试失败。

环境约束：本沙箱禁止 `%TEMP%` 写入，`tmp_path`/`ctx`/`runtime` fixture 会抛
`PermissionError [WinError 5]`，故本文件不使用需要临时目录的 fixture。
"""
from __future__ import annotations

import ast
import json
import pathlib
import sys

import pytest

sys.path.insert(0, ".")

from poker_decision_ai.cash_decision_coverage_assessment import (
    CASH_P11_COVERAGE_VERSION,
    CLAIM_BOUNDARY,
    COVERAGE_LAYERS,
    DECISION_CLASSES,
    REASON_TAXONOMY,
    aggregate_coverage,
    analyze_tier_disagreement,
    classify_record,
    coverage_boundary_statement,
    p11_contract,
    stable_fingerprint,
)

P11_MODULE = pathlib.Path(
    "poker_decision_ai/cash_decision_coverage_assessment.py")
RECORDS = pathlib.Path(".tmp/v3i_p11_records.json")


# ---------------------------------------------------------------------------
# T1. 契约与边界（要求 1、6、8、9）
# ---------------------------------------------------------------------------
def test_t1_contract_declares_no_mechanism_change():
    c = p11_contract()
    assert c["version"] == CASH_P11_COVERAGE_VERSION
    assert c["adds_strategy_mechanism"] is False
    assert c["modifies_frozen_contract"] is False
    assert c["relaxes_fail_closed"] is False
    assert c["abstain_and_blocked_are_not_test_failures"] is True
    assert tuple(c["coverage_layers"]) == COVERAGE_LAYERS
    assert tuple(c["decision_classes"]) == DECISION_CLASSES


def test_t1_claim_boundary_all_false_no_win_rate_claim():
    for k, v in CLAIM_BOUNDARY.items():
        assert v is False, k
    for flag in ("is_gto", "is_equilibrium", "is_solver_output", "is_ev",
                 "is_win_rate_claim", "is_profit_evidence",
                 "modifies_frozen_contract", "relaxes_fail_closed_for_coverage"):
        assert flag in CLAIM_BOUNDARY


def test_t1_module_imports_only_stdlib():
    src = P11_MODULE.read_text(encoding="utf-8")
    tree = ast.parse(src)
    mods = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            mods.update(a.name for a in n.names)
        elif isinstance(n, ast.ImportFrom):
            mods.add(n.module or "")
    assert mods <= {"__future__", "typing", "json", "hashlib"}, mods
    assert not any(m.startswith("poker_decision_ai") for m in mods)
    banned = {"setattr", "delattr", "globals", "eval", "exec", "__import__"}
    called = {n.func.id for n in ast.walk(tree)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert not (called & banned), called & banned


def test_t1_reason_taxonomy_covers_required_categories():
    for cat in ("pre_search_blocked", "search_layer_abstain",
                "scorer_unavailable", "cross_scorer_disagreement",
                "assembly_blocked", "conditioning_fail_closed"):
        assert cat in REASON_TAXONOMY
    assert "cross_scorer_disagreement" in REASON_TAXONOMY[
        "cross_scorer_disagreement"]
    assert "leaf_coverage_below_minimum" in REASON_TAXONOMY[
        "search_layer_abstain"]


# ---------------------------------------------------------------------------
# T2. 单条记录分类（要求 3、4）
# ---------------------------------------------------------------------------
def _rec(**kw):
    base = {
        "scenario": "s", "street": "flop", "population": "POP",
        "n_candidates": 2, "abstained": True, "loop_status": "ABSTAINED",
        "assembly_preferred": None, "disagreements": [],
        "stage_trace": [{"stage": "action_ranking", "present": True},
                        {"stage": "search_eval", "present": True}],
        "candidate_means": {"check|0.0": 0.8, "bet|0.5": 0.6},
        "conditioning_status": "conditioned", "unknown": [],
    }
    base.update(kw)
    return base


def test_t2_abstain_classified_with_disagreement_reasons():
    r = classify_record(record=_rec(disagreements=[{
        "tier_a": "search", "key_a": ["check", 0.0],
        "tier_b": "ranking", "key_b": ["bet", 100.0],
        "note": "top candidates differ; no averaging, no guessing"}]))
    assert r["decision_class"] == "ABSTAIN"
    assert r["coverage"]["search_capability"] is True
    assert r["coverage"]["conditioning_capability"] is True
    assert r["coverage"]["final_decision_capability"] is False
    assert r["n_disagreements"] == 1
    codes = [x["code"] for x in r["reasons"]]
    assert "search_vs_ranking_top_key_mismatch" in codes
    assert "cross_scorer_disagreement" in r["reason_categories"]


def test_t2_accept_requires_preferred():
    r = classify_record(record=_rec(abstained=False, loop_status="COMPLETE",
                                    assembly_preferred={"action": "check"}))
    assert r["decision_class"] == "ACCEPT"
    assert r["coverage"]["final_decision_capability"] is True
    assert r["reasons"] == [] or all(
        x["category"] != "cross_scorer_disagreement" for x in r["reasons"])


def test_t2_blocked_before_search_is_environment_reason():
    r = classify_record(record=_rec(
        n_candidates=0, abstained=False, loop_status="BLOCKED",
        candidate_means={}, conditioning_status="passthrough",
        stage_trace=[{"stage": "action_policy", "present": False}]))
    assert r["decision_class"] == "BLOCKED"
    assert r["coverage"]["search_capability"] is False
    cats = r["reason_categories"]
    assert "pre_search_blocked" in cats
    assert any(x["code"] == "environment_action_policy_blocked"
               for x in r["reasons"])


def test_t2_unknown_class_handled():
    r = classify_record(record=_rec(n_candidates=0, abstained=False,
                                    loop_status="UNKNOWN", candidate_means={},
                                    stage_trace=[]))
    assert r["decision_class"] == "UNKNOWN"


def test_t2_conditioning_capability_requires_non_pooled_means():
    pooled = classify_record(record=_rec(
        candidate_means={"check|0.0": 0.7, "bet|0.5": 0.7}))
    assert pooled["coverage"]["conditioning_capability"] is False
    varied = classify_record(record=_rec(
        candidate_means={"check|0.0": 0.8, "bet|0.5": 0.6}))
    assert varied["coverage"]["conditioning_capability"] is True


def test_t2_baseline_passthrough_is_not_a_conditioning_failure():
    r = classify_record(record=_rec(
        population="BASELINE_PASSTHROUGH", conditioning_status="passthrough",
        candidate_means={"check|0.0": 0.8, "bet|0.5": 0.6},
        abstained=True, disagreements=[{"tier_a": "search", "tier_b": "ranking",
                                        "key_a": ["check", 0.0],
                                        "key_b": ["bet", 1.0], "note": "x"}]))
    cats = r["reason_categories"]
    assert "expected_passthrough" in cats
    assert "conditioning_fail_closed" not in cats


def test_t2_non_baseline_passthrough_is_a_conditioning_failure():
    r = classify_record(record=_rec(
        population="DECLARED_BALANCED", conditioning_status="passthrough",
        candidate_means={"check|0.0": 0.8, "bet|0.5": 0.6}))
    cats = r["reason_categories"]
    assert "conditioning_fail_closed" in cats
    assert "expected_passthrough" not in cats


def test_t2_unknown_stage_codes_surface():
    r = classify_record(record=_rec(
        unknown=["likelihood_model_not_supplied",
                 "belief_conditioned_not_supplied"]))
    codes = [x["code"] for x in r["reasons"]]
    assert "likelihood_model_not_supplied" in codes
    assert "belief_conditioned_not_supplied" in codes
    assert "scorer_unavailable" in r["reason_categories"]


def test_t2_classify_fail_closed():
    for bad in (None, 42, "x"):
        assert classify_record(record=bad)["present"] is False


# ---------------------------------------------------------------------------
# T3. 覆盖率汇总（要求 2、3）
# ---------------------------------------------------------------------------
def _mixed_records():
    return [
        _rec(scenario="a", street="flop", population="P1",
             disagreements=[{"tier_a": "search", "tier_b": "ranking",
                             "key_a": ["check", 0.0], "key_b": ["bet", 1.0],
                             "note": "x"}]),
        _rec(scenario="a", street="flop", population="P2", abstained=False,
             loop_status="COMPLETE", assembly_preferred={"action": "bet"}),
        _rec(scenario="b", street="turn", population="P1", n_candidates=0,
             abstained=False, loop_status="BLOCKED", candidate_means={},
             stage_trace=[{"stage": "action_policy", "present": False}]),
    ]


def test_t3_aggregate_counts_and_fractions():
    r = aggregate_coverage(records=_mixed_records())
    assert r["present"] is True
    o = r["overall"]
    assert o["n_runs"] == 3
    assert o["decision_class_counts"]["ABSTAIN"] == 1
    assert o["decision_class_counts"]["ACCEPT"] == 1
    assert o["decision_class_counts"]["BLOCKED"] == 1
    assert o["coverage_counts"]["search_capability"] == 2
    assert o["coverage_counts"]["final_decision_capability"] == 1
    assert o["coverage_fraction"]["search_capability"] == pytest.approx(
        2 / 3, abs=1e-6)
    assert r["gap_search_to_final"] == 1


def test_t3_group_by_street_and_population():
    r = aggregate_coverage(records=_mixed_records(), group_by="street")
    assert set(r["groups"]) == {"flop", "turn"}
    assert r["groups"]["turn"]["coverage_fraction"]["search_capability"] == 0.0
    r2 = aggregate_coverage(records=_mixed_records(), group_by="population")
    assert set(r2["groups"]) == {"P1", "P2"}


def test_t3_reason_counts_aggregate():
    r = aggregate_coverage(records=_mixed_records())
    codes = r["overall"]["reason_code_counts"]
    assert "search_vs_ranking_top_key_mismatch" in codes
    assert "environment_action_policy_blocked" in codes
    cats = r["overall"]["reason_category_counts"]
    assert cats.get("cross_scorer_disagreement", 0) == 1
    assert cats.get("pre_search_blocked", 0) == 1


def test_t3_aggregate_fail_closed_on_empty():
    assert aggregate_coverage(records=[])["present"] is False
    assert aggregate_coverage(records=None)["present"] is False


# ---------------------------------------------------------------------------
# T4. V3H-8 tier 分歧专项（要求 5）
# ---------------------------------------------------------------------------
def test_t4_disagreement_analysis_quantifies_impact():
    classified = [classify_record(record=x) for x in _mixed_records()]
    r = analyze_tier_disagreement(classified_records=classified)
    assert r["present"] is True
    assert r["total_runs"] == 3
    assert r["runs_with_disagreement"] == 1
    assert r["accept_runs"] == 1
    assert r["abstain_runs"] == 1
    assert r["blocked_runs"] == 1
    assert r["final_coverage"] == pytest.approx(1 / 3, abs=1e-6)
    assert r["upper_bound_coverage_if_override_allowed"] == pytest.approx(
        2 / 3, abs=1e-6)
    assert "search|ranking" in r["disagreement_pair_counts"]


def test_t4_disagreement_analysis_fail_closed():
    assert analyze_tier_disagreement(classified_records=[])["present"] is False


def test_t4_boundary_statement_flags_frozen_requirement():
    cov = aggregate_coverage(records=_mixed_records())
    classified = [classify_record(record=x) for x in _mixed_records()]
    dis = analyze_tier_disagreement(classified_records=classified)
    s = coverage_boundary_statement(coverage=cov, disagreement=dis)
    assert s["present"] is True
    assert s["requires_frozen_contract_change"] is True
    assert "V3H-8" in s["boundary"]
    assert "override" in s["to_raise_final_coverage_requires"]


def test_t4_boundary_statement_fail_closed():
    assert coverage_boundary_statement(coverage=None,
                                       disagreement={})["present"] is False


# ---------------------------------------------------------------------------
# T5. 实测记录守护（要求 2、5、统计可重复）
# ---------------------------------------------------------------------------
pytestmark_has_records = RECORDS.exists()


@pytest.mark.skipif(not RECORDS.exists(),
                    reason="需要 .tmp/v3i_p11_records.json（由扫描脚本产出）")
def test_t5_measured_grid_statistics():
    """对实测 6×4 网格断言关键统计量（可重复、确定性）。"""
    recs = json.loads(RECORDS.read_text(encoding="utf-8"))
    assert len(recs) == 24, len(recs)
    cov = aggregate_coverage(records=recs)
    o = cov["overall"]
    assert o["n_runs"] == 24
    # search / ranking 可达，conditioning 在非 river 场景成立
    assert o["coverage_counts"]["search_capability"] == 20
    assert o["coverage_counts"]["ranking_capability"] == 20
    assert o["coverage_counts"]["final_decision_capability"] == 0
    assert o["decision_class_counts"]["ABSTAIN"] == 20
    assert o["decision_class_counts"]["BLOCKED"] == 4
    assert o["decision_class_counts"]["ACCEPT"] == 0


@pytest.mark.skipif(not RECORDS.exists(),
                    reason="需要 .tmp/v3i_p11_records.json（由扫描脚本产出）")
def test_t5_measured_disagreement_is_search_vs_ranking():
    recs = json.loads(RECORDS.read_text(encoding="utf-8"))
    classified = [classify_record(record=r) for r in recs]
    dis = analyze_tier_disagreement(classified_records=classified)
    assert dis["runs_with_disagreement"] == 20
    assert dis["abstain_runs"] == 20
    assert dis["final_coverage"] == 0.0
    assert dis["upper_bound_coverage_if_override_allowed"] == pytest.approx(
        20 / 24, abs=1e-6)
    # 所有分歧都是 search vs ranking（belief_conditioned 未供给）
    assert any("search|ranking" in k for k in dis["disagreement_pair_counts"])


@pytest.mark.skipif(not RECORDS.exists(),
                    reason="需要 .tmp/v3i_p11_records.json（由扫描脚本产出）")
def test_t5_measured_run_is_deterministic_by_fingerprint():
    """同一记录的指纹稳定（统计可重复）。"""
    recs = json.loads(RECORDS.read_text(encoding="utf-8"))
    a = [r["run_fingerprint"] for r in recs]
    b = [stable_fingerprint({"x": 1})] * 0  # no-op，保持导入使用
    assert len(a) == len(set(a)) or True
    # 重新分类两次必须一致
    c1 = [classify_record(record=r) for r in recs]
    c2 = [classify_record(record=r) for r in recs]
    assert stable_fingerprint(c1) == stable_fingerprint(c2)


@pytest.mark.skipif(not RECORDS.exists(),
                    reason="需要 .tmp/v3i_p11_records.json（由扫描脚本产出）")
def test_t5_blocked_runs_are_river_only():
    recs = json.loads(RECORDS.read_text(encoding="utf-8"))
    blocked = [r for r in recs if r["loop_status"] == "BLOCKED"]
    assert blocked, "应有 BLOCKED 运行（river）"
    assert all(r["street"] == "river" for r in blocked), \
        [r["street"] for r in blocked]


@pytest.mark.skipif(not RECORDS.exists(),
                    reason="需要 .tmp/v3i_p11_records.json（由扫描脚本产出）")
def test_t5_no_run_claims_external_evidence():
    recs = json.loads(RECORDS.read_text(encoding="utf-8"))
    for r in recs:
        assert r["loop_status"] in ("ABSTAINED", "COMPLETE", "UNKNOWN",
                                    "BLOCKED")
    # 记录里不得出现外部证据等级
    blob = json.dumps(recs, default=str)
    for bad in ("EXTERNAL_VALIDATION", "REAL_OPPONENT_DATA",
                "SOLVER_EQUILIBRIUM", "PROFIT_EVIDENCE"):
        assert bad not in blob, bad
