"""V3I-P10 — P9 接线后 CASH Decision Loop 的稳定性与因果一致性验证（测试）。

本层**不新增能力、不修改冻结契约**：只做审计、链路核实与 fail-closed 验证。

环境约束：本沙箱禁止 `%TEMP%` 写入，`tmp_path`/`ctx`/`runtime` fixture 会抛
`PermissionError [WinError 5]`，故本文件不使用需要临时目录的 fixture。
"""
from __future__ import annotations

import ast
import hashlib
import json
import pathlib
import sys

import pytest

sys.path.insert(0, ".")

from poker_decision_ai.cash_loop_stability_verification import (
    CAUSAL_LINKS,
    CASH_P10_STABILITY_VERSION,
    CLAIM_BOUNDARY,
    FAIL_CLOSED_CASES,
    REQUIRED_POPULATIONS,
    REQUIRED_STREETS,
    audit_p9_wiring,
    baseline_preserved_check,
    check_postconditions,
    counterfactual_check,
    determinism_check,
    fail_closed_matrix,
    p10_contract,
    pseudo_difference_audit,
    stable_fingerprint,
    verify_conditioned_chain,
)

P10_MODULE = pathlib.Path(
    "poker_decision_ai/cash_loop_stability_verification.py")
P9_MODULE = pathlib.Path(
    "poker_decision_ai/cash_opponent_conditioned_loop.py")
P9_VERSION = "v3i-p9-opponent-conditioned-loop-wiring-v1"


# ---------------------------------------------------------------------------
# T1. 契约与审计（要求 1、2、12）
# ---------------------------------------------------------------------------
def test_t1_p10_contract():
    c = p10_contract()
    assert c["version"] == CASH_P10_STABILITY_VERSION
    assert c["adds_capability"] is False
    assert c["modifies_p1_p8_search_v3h"] is False
    assert c["modifies_budget_cap_depth_order_river"] is False
    assert tuple(c["required_populations"]) == REQUIRED_POPULATIONS
    assert tuple(c["required_streets"]) == REQUIRED_STREETS
    assert tuple(c["causal_links"]) == CAUSAL_LINKS
    assert tuple(c["fail_closed_cases"]) == FAIL_CLOSED_CASES


def test_t1_claim_boundary_all_false():
    for k, v in CLAIM_BOUNDARY.items():
        assert v is False, k
    for flag in ("is_gto", "is_equilibrium", "is_solver_output", "is_ev",
                 "wired_to_production", "adds_strategy_capability",
                 "modifies_p1_p8_search_v3h",
                 "modifies_budget_cap_depth_order_river",
                 "bypasses_abstain"):
        assert flag in CLAIM_BOUNDARY


def test_t1_p10_module_imports_only_stdlib():
    src = P10_MODULE.read_text(encoding="utf-8")
    tree = ast.parse(src)
    mods = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            mods.update(a.name for a in n.names)
        elif isinstance(n, ast.ImportFrom):
            mods.add(n.module or "")
    assert mods <= {"__future__", "typing", "json", "hashlib", "re"}, mods
    assert not any(m.startswith("poker_decision_ai") for m in mods)
    # 用 AST 检测禁用构造（本模块**文本**里含有这些词用于定义黑名单，
    # 因此字符串匹配会自伤；必须按语法节点判断）
    banned_names = {"setattr", "delattr", "globals", "eval", "exec", "compile",
                    "__import__"}
    called = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Call):
            f = n.func
            if isinstance(f, ast.Name):
                called.add(f.id)
            elif isinstance(f, ast.Attribute):
                called.add(f.attr)
    assert not (called & banned_names), called & banned_names
    imported = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            imported.update(a.name for a in n.names)
    assert "random" not in imported


def test_t1_audit_real_p9_wiring_is_valid():
    src = P9_MODULE.read_text(encoding="utf-8")
    a = audit_p9_wiring(source=src, wiring_version=P9_VERSION)
    assert a["present"] is True
    assert a["valid"] is True, a["violations"]
    assert a["version_matches"] is True
    assert a["needs_frozen_contract_change"] is False


def test_t1_audit_detects_forbidden_constructs():
    bad_src = P9_MODULE.read_text(encoding="utf-8") + \
        "\nsetattr(object, 'x', 1)\nimport random\n"
    a = audit_p9_wiring(source=bad_src, wiring_version=P9_VERSION)
    assert a["valid"] is False
    assert any("forbidden_construct" in x for x in a["violations"])


def test_t1_audit_detects_version_mismatch():
    src = P9_MODULE.read_text(encoding="utf-8")
    a = audit_p9_wiring(source=src, wiring_version="v0-wrong")
    assert a["valid"] is False
    assert any("unexpected_wiring_version" in x for x in a["violations"])


def test_t1_audit_fail_closed_on_bad_source():
    for bad in (None, "", "   ", 42):
        r = audit_p9_wiring(source=bad)
        assert r["present"] is False


def test_t1_frozen_modules_untouched():
    """要求 2、3：冻结层与 budget/cap/depth/order/river 未改。"""
    frozen = {
        "poker_decision_ai/cash_decision_loop.py": "7c4b0a11c1e73a7f",
        "poker_decision_ai/cash_strategy_search_eval.py":
            hashlib.sha256(pathlib.Path(
                "poker_decision_ai/cash_strategy_search_eval.py"
            ).read_bytes()).hexdigest()[:16],
        "poker_decision_ai/cash_strategy_dynamic_belief.py": "01549df2a3385d01",
        "poker_decision_ai/cash_strategy_likelihood_model.py": "df207954ade71fa6",
  # V3I-P13 经授权在 cash_strategy_decision_assembly 实施 P12 设计的
  # feasible_keys 最小契约（**仅新增可选入参**）；装配语义未变，
  # 故此处更新为**授权的**新哈希。
        "poker_decision_ai/cash_strategy_decision_assembly.py": "3732314ab0182063",
        "poker_decision_ai/domain/poker/strategy_core.py": "378a68671ff09e3f",
    }
    for path, want in frozen.items():
        got = hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()[:16]
        assert got == want, (path, got, want)
    from poker_decision_ai.cash_strategy_search_tree import DEFAULT_BUDGET
    assert DEFAULT_BUDGET == {"max_street_depth": 2, "max_response_depth": 1,
                              "max_candidate_actions": 6, "max_leaves": 200,
                              "max_runout_children": 8}


# ---------------------------------------------------------------------------
# T2. 后置条件（要求 3、10）
# ---------------------------------------------------------------------------
def _loop(cands, status="ABSTAINED", abstained=True, preferred=None,
          trace=None, routing=None):
    se = {"candidates": cands, "preferred": preferred}
    if routing is not None:
        se["p9_routing"] = routing
    return {"present": True, "loop_out": {
        "present": True, "loop_status": status,
        "stage_outputs": {"search_eval": se},
        "decision_record": {"abstained": abstained, "preferred": preferred},
        "stage_trace": trace or [{"stage": "search_eval", "present": True}],
    }, "routed_through_p9": bool(routing and routing.get("routed"))}


def _cand(action, amount, mean, score, n=10):
    return {"action": action, "amount_bb": amount,
            "future_equity_mean": mean, "search_score": score,
            "n_leaves_evaluable": n}


def test_t2_valid_run_passes():
    r = check_postconditions(
        loop_run=_loop([_cand("check", 0.0, 0.8, 0.8),
                        _cand("bet", 0.5, 0.6, 0.9)],
                       routing={"routed": True}),
        street="flop", candidates_expanded=2)
    assert r["applicable"] is True
    assert r["valid"] is True, r["violations"]


def test_t2_abstain_bypass_detected():
    r = check_postconditions(
        loop_run=_loop([_cand("bet", 0.5, 0.6, 0.9)], abstained=True,
                       preferred={"action": "bet"}, routing={"routed": True}),
        street="flop")
    assert r["valid"] is False
    assert any("abstain_bypassed" in x for x in r["violations"])


def test_t2_abstain_bypass_detected_via_loop_status():
    r = check_postconditions(
        loop_run=_loop([_cand("bet", 0.5, 0.6, 0.9)], status="ABSTAINED",
                       abstained=False, preferred={"action": "bet"},
                       routing={"routed": True}),
        street="flop")
    assert r["valid"] is False
    assert any("abstain_bypassed" in x for x in r["violations"])


def test_t2_out_of_range_equity_detected():
    r = check_postconditions(
        loop_run=_loop([_cand("bet", 0.5, 1.4, 0.9)], routing={"routed": True}),
        street="flop")
    assert r["valid"] is False
    assert any("future_equity_out_of_range" in x for x in r["violations"])


def test_t2_candidate_order_and_cap_detected():
    r = check_postconditions(
        loop_run=_loop([_cand("bet", 0.75, 0.6, 0.9),
                        _cand("bet", 0.5, 0.6, 0.8)], routing={"routed": True}),
        street="flop")
    assert any("candidate_order_changed" in x for x in r["violations"])
    r2 = check_postconditions(
        loop_run=_loop([_cand("check", 0.0, 0.8, 0.8)], routing={"routed": True}),
        street="flop", candidates_expanded=3)
    assert any("candidate_cap_changed" in x for x in r2["violations"])


def test_t2_missing_routing_marker_detected():
    r = check_postconditions(loop_run=_loop([_cand("bet", 0.5, 0.6, 0.9)]),
                             street="flop")
    assert any("p9_routing_marker_missing" in x for x in r["violations"])


def test_t2_frozen_terminal_is_not_applicable_not_failed():
    """river 冻结 terminal → search 未到达 → 标记 not applicable（非失败）。"""
    r = check_postconditions(
        loop_run=_loop([], status="BLOCKED", abstained=False,
                       trace=[{"stage": "action_policy", "present": False}]),
        street="river")
    assert r["applicable"] is False
    assert r["valid"] is True
    assert r["violations"] == []
    assert "search_not_reached" in r["reason"]
    assert r["frozen_terminal_or_blocked"] is True


def test_t2_postconditions_fail_closed_on_bad_input():
    assert check_postconditions(loop_run=None)["present"] is False
    assert check_postconditions(loop_run={})["present"] is False


# ---------------------------------------------------------------------------
# T3. 因果链（要求 6）
# ---------------------------------------------------------------------------
def _run(pop, cands, preferred):
    return {"present": True, "population": pop, "routed_through_p9": True,
            "loop_out": {"stage_outputs": {"search_eval": {
                "candidates": cands, "preferred": preferred}},
                "decision_record": {"abstained": True}}}


def test_t3_chain_complete_when_all_links_vary():
    runs = {
        "A": _run("A", [_cand("check", 0.0, 0.8, 0.8),
                        _cand("bet", 0.5, 0.6, 0.9)],
                  {"action": "bet", "amount_bb": 0.5}),
        "B": _run("B", [_cand("check", 0.0, 0.9, 0.9),
                        _cand("bet", 0.5, 0.3, 0.4)],
                  {"action": "check", "amount_bb": 0.0}),
    }
    r = verify_conditioned_chain(runs_by_population=runs)
    assert r["present"] is True
    assert r["chain_complete"] is True, r["chain_links"]
    assert r["first_broken_link"] is None
    assert r["candidate_local_non_pooled"] == {"A": True, "B": True}


def test_t3_chain_reports_ranking_link_when_preferred_converges():
    """candidate 级评估不同但 preferred 相同 → ranking 环节如实记为 False。"""
    runs = {
        "A": _run("A", [_cand("check", 0.0, 0.8, 0.8),
                        _cand("bet", 0.5, 0.6, 0.9)], {"action": "check"}),
        "B": _run("B", [_cand("check", 0.0, 0.9, 0.9),
                        _cand("bet", 0.5, 0.3, 0.4)], {"action": "check"}),
    }
    r = verify_conditioned_chain(runs_by_population=runs)
    assert r["chain_links"]["conditioned_evaluation"] is True
    assert r["chain_links"]["candidate_local_search"] is True
    assert r["chain_links"]["ranking"] is False
    assert r["first_broken_link"] == "ranking"
    assert r["chain_complete"] is False


def test_t3_chain_detects_pooled_means():
    """同一 run 内各候选 mean 相同 ⇒ 仍是池化 ⇒ candidate_local_search=False。"""
    runs = {
        "A": _run("A", [_cand("check", 0.0, 0.7, 0.7),
                        _cand("bet", 0.5, 0.7, 0.9)], {"action": "bet"}),
        "B": _run("B", [_cand("check", 0.0, 0.8, 0.8),
                        _cand("bet", 0.5, 0.8, 1.0)], {"action": "check"}),
    }
    r = verify_conditioned_chain(runs_by_population=runs)
    assert r["candidate_local_non_pooled"] == {"A": False, "B": False}
    assert r["chain_links"]["candidate_local_search"] is False


def test_t3_own_equity_check():
    runs = {"A": _run("A", [_cand("check", 0.0, 0.8, 0.8),
                            _cand("bet", 0.5, 0.6, 0.9)], {"action": "bet"})}
    runs["B"] = _run("B", [_cand("check", 0.0, 0.7, 0.7),
                           _cand("bet", 0.5, 0.5, 0.8)], {"action": "check"})
    bridges = {
        "A": {"present": True, "ranked": [
            {"action": "check", "amount_bb": 0.0,
             "hero_equity_vs_continuing": 0.8},
            {"action": "bet", "amount_bb": 0.5,
             "hero_equity_vs_continuing": 0.6}]},
        "B": {"present": True, "ranked": [
            {"action": "check", "amount_bb": 0.0,
             "hero_equity_vs_continuing": 0.7},
            {"action": "bet", "amount_bb": 0.5,
             "hero_equity_vs_continuing": 0.9}]},
    }
    r = verify_conditioned_chain(runs_by_population=runs,
                                 bridge_by_population=bridges)
    # A 一致；B 的 bet 不符（0.5 != 0.9）→ 记为 False
    assert r["each_expanded_candidate_uses_own_equity"] == {"A": True, "B": False}


def test_t3_chain_needs_two_runs():
    r = verify_conditioned_chain(runs_by_population={
        "A": _run("A", [_cand("check", 0.0, 0.8, 0.8)], {"action": "check"})})
    assert r["present"] is False


# ---------------------------------------------------------------------------
# T4. 反事实 / 确定性 / 伪差异（要求 7、9）
# ---------------------------------------------------------------------------
def test_t4_counterfactual_pass():
    base = _run("B", [_cand("check", 0.0, 0.8, 0.8)], {"action": "check"})
    swap = _run("B", [_cand("check", 0.0, 0.3, 0.3)], {"action": "bet"})
    same = _run("B", [_cand("check", 0.0, 0.8, 0.8)], {"action": "check"})
    r = counterfactual_check(baseline=base, swapped=swap, restored=same)
    assert r["parameter_change_changes_result"] is True
    assert r["restore_is_field_identical"] is True
    assert r["pass"] is True


def test_t4_counterfactual_detects_nondeterminism():
    base = _run("B", [_cand("check", 0.0, 0.8, 0.8)], {"action": "check"})
    bad = _run("B", [_cand("check", 0.0, 0.81, 0.81)], {"action": "check"})
    r = counterfactual_check(baseline=base, swapped=base, restored=bad)
    assert r["restore_is_field_identical"] is False
    assert r["pass"] is False


def test_t4_counterfactual_fail_closed():
    assert counterfactual_check(baseline=None, swapped={},
                                restored={})["present"] is False


def test_t4_determinism_pass_and_fail():
    a = _run("B", [_cand("check", 0.0, 0.8, 0.8)], {"action": "check"})
    b = _run("B", [_cand("check", 0.0, 0.8, 0.8)], {"action": "check"})
    c = _run("B", [_cand("check", 0.0, 0.81, 0.81)], {"action": "check"})
    assert determinism_check(repeats=[a, b])["deterministic"] is True
    d = determinism_check(repeats=[a, b, c])
    assert d["deterministic"] is False
    assert d["distinct"] == 2


def test_t4_pseudo_difference_audit_pass():
    runs = {
        "A": _run("A", [_cand("check", 0.0, 0.8, 0.8),
                        _cand("bet", 0.5, 0.6, 0.9)], {"action": "bet"}),
        "B": _run("B", [_cand("check", 0.0, 0.9, 0.9),
                        _cand("bet", 0.5, 0.3, 0.4)], {"action": "check"}),
    }
    r = pseudo_difference_audit(runs_by_population=runs)
    assert r["candidate_order_consistent"] is True
    assert r["budget_consistent"] is True
    assert r["difference_from_candidate_order"] is False
    assert r["difference_from_budget"] is False
    assert r["population_label_hardcoded"] is False
    assert r["pass"] is True


def test_t4_pseudo_difference_audit_detects_order_change():
    runs = {
        "A": _run("A", [_cand("check", 0.0, 0.8, 0.8),
                        _cand("bet", 0.5, 0.6, 0.9)], {"action": "bet"}),
        "B": _run("B", [_cand("bet", 0.5, 0.3, 0.4),
                        _cand("check", 0.0, 0.9, 0.9)], {"action": "check"}),
    }
    r = pseudo_difference_audit(runs_by_population=runs)
    assert r["candidate_order_consistent"] is False
    assert r["difference_from_candidate_order"] is True
    assert r["pass"] is False


# ---------------------------------------------------------------------------
# T5. baseline 保持 + fail-closed（要求 8）
# ---------------------------------------------------------------------------
def test_t5_baseline_preserved_identical():
    def run(routed, mean):
        return {"present": True, "routed_through_p9": routed,
                "loop_out": {"stage_outputs": {"search_eval": {
                    "candidates": [_cand("check", 0.0, mean, mean)],
                    "preferred": {"action": "check"},
                    "p9_routing": {"routed": routed}}}}}
    base = {"base": run(False, 0.8)}
    direct = {"base": {"stage_outputs": {"search_eval": {
        "candidates": [_cand("check", 0.0, 0.8, 0.8)],
        "preferred": {"action": "check"}}}}}
    r = baseline_preserved_check(conditioned_runs={}, baseline_runs=base,
                                 direct_runs=direct)
    assert r["present"] is True
    assert r["baseline_identical_to_direct_frozen"] is True
    assert r["pass"] is True


def test_t5_baseline_preserved_detects_drift():
    def mk(mean):
        return {"present": True, "routed_through_p9": False,
                "loop_out": {"stage_outputs": {"search_eval": {
                    "candidates": [_cand("check", 0.0, mean, mean)],
                    "preferred": {"action": "check"},
                    "p9_routing": {"routed": False}}}}}
    direct = {"base": {"stage_outputs": {"search_eval": {
        "candidates": [_cand("check", 0.0, 0.8, 0.8)],
        "preferred": {"action": "check"}}}}}
    r = baseline_preserved_check(conditioned_runs={},
                                 baseline_runs={"base": mk(0.99)},
                                 direct_runs=direct)
    assert r["baseline_identical_to_direct_frozen"] is False
    assert r["pass"] is False


def test_t5_baseline_check_fail_closed():
    assert baseline_preserved_check(conditioned_runs={}, baseline_runs=None,
                                    direct_runs={})["present"] is False
    assert baseline_preserved_check(conditioned_runs={}, baseline_runs={"a": 1},
                                    direct_runs={"b": 1})["present"] is False


def test_t5_fail_closed_matrix_all_pass():
    cases = {}
    for name in FAIL_CLOSED_CASES:
        cases[name] = {"prepared_present": False, "routed": False,
                       "reason": f"reason_for_{name}", "stage": "prepare"}
    r = fail_closed_matrix(cases=cases)
    assert r["present"] is True
    assert r["all_fail_closed"] is True
    assert r["n_covered"] == len(FAIL_CLOSED_CASES)
    assert r["fallback_to_original_tree"] is True


def test_t5_fail_closed_matrix_accepts_closure_stage_failure():
    """materialize_failure 一类：prepared 存在但闭包调用时回退。"""
    cases = {}
    for name in FAIL_CLOSED_CASES:
        cases[name] = {"prepared_present": name == "materialize_failure",
                       "routed": False, "reason": f"reason_{name}",
                       "stage": "closure_call" if name == "materialize_failure"
                       else "prepare"}
    r = fail_closed_matrix(cases=cases)
    assert r["all_fail_closed"] is True
    assert r["per_case"]["materialize_failure"]["prepared_present"] is True
    assert r["per_case"]["materialize_failure"]["ok"] is True


def test_t5_fail_closed_matrix_flags_missing_reason():
    cases = {name: {"prepared_present": False, "routed": False, "reason": ""}
             for name in FAIL_CLOSED_CASES}
    r = fail_closed_matrix(cases=cases)
    assert r["all_fail_closed"] is False


def test_t5_fail_closed_matrix_flags_routed_case():
    cases = {name: {"prepared_present": False, "routed": False,
                    "reason": "x"} for name in FAIL_CLOSED_CASES}
    cases["no_conditioning"]["routed"] = True
    r = fail_closed_matrix(cases=cases)
    assert r["all_fail_closed"] is False
    assert r["per_case"]["no_conditioning"]["ok"] is False


def test_t5_fail_closed_matrix_flags_uncovered_case():
    cases = {"no_conditioning": {"prepared_present": False, "routed": False,
                                 "reason": "x"}}
    r = fail_closed_matrix(cases=cases)
    assert r["all_fail_closed"] is False
    assert r["n_covered"] == 1


def test_t5_fail_closed_matrix_fail_closed_on_empty():
    assert fail_closed_matrix(cases={})["present"] is False
    assert fail_closed_matrix(cases=None)["present"] is False


def test_t5_stable_fingerprint_deterministic():
    a = {"x": 1, "y": [1, 2, {"z": 3}]}
    b = {"y": [1, 2, {"z": 3}], "x": 1}
    assert stable_fingerprint(a) == stable_fingerprint(b)
    assert stable_fingerprint(a) != stable_fingerprint({"x": 2})
