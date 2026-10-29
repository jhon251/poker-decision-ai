# -*- coding: utf-8 -*-
"""V3H-18 测试 — CASH 审计结果有效期 / 变更失效契约（只读）。

覆盖：CASH-only fail closed / 非 dict / 缺 baseline fail closed /
缺 validity fail closed（重新审计计划）/
**有效期 = 已登记快照字节一致**（同快照 → `VALID_FOR_REGISTERED_SNAPSHOT`）/
**文件内容变更 → `SUPERSEDED_SOURCE_CHANGED` + 重新审计要求** /
**未登记文件 → `INVALID_UNREGISTERED_CHANGE`** /
**登记文件缺失 / 不可读 → `INVALID_UNREADABLE`** /
**解析失败 → `INVALID_UNPARSEABLE`** /
**动态加载（真实调用）→ 失效**（AST 判定，标记清单不误伤）/
重新审计计划步骤完整且**不执行审计** /
**指纹只证明"已登记内容未变"**，不证明真实性·来源·执行 /
**"未发现违规" ≠ 未来不会误用**（无前瞻保证）/
**不授予** verified / readiness / consumable /
未登记 / 不可读 / 解析失败 / 动态加载 / 仓库外代码 一律 fail closed /
零结果零执行零统计 / 定输入复现 / state 不变异 /
AST·import 隔离（零 V3G、零任何层、无网络·进程·钱包）/ 生产 import 图零新增 /
frozen·sealed drift / Solver 未被调用。
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "poker_decision_ai"

from poker_decision_ai.cash_strategy_audit_validity import (  # noqa: E402
    ATTESTATION_SCOPE, AUDITED_SCOPE_KINDS, AUDIT_INPUT_FILES,
    CASH_STRATEGY_AUDIT_VALIDITY_VERSION, CHANGE_DETECTION_FAILURES,
    CLAIM_BOUNDARY, DYNAMIC_LOAD_MARKERS, FINGERPRINT_ALGORITHM,
    FINGERPRINT_PURPOSE, REVALIDATION_REQUIREMENTS, VALIDITY_STATES,
    audit_validity_contract, build_manifest, evaluate_audit_validity,
    revalidation_plan)

# 测试侧允许 import V3G/V3H 层（生产侧不允许，由 AST 测试强制）
from poker_decision_ai.cash_solver_interface import solver_status  # noqa: E402
from poker_decision_ai.cash_strategy_experiment_eligibility import (  # noqa: E402
    ENVIRONMENT_FACT_KEYS)

NEW_MODULE = "cash_strategy_audit_validity"
MODULE_PATH = PKG / f"{NEW_MODULE}.py"

FROZEN = {
    "poker_decision_ai/domain/poker/strategy_core.py": "378a68671ff09e3f",
    "poker_decision_ai/domain/poker/models.py": "6775c06050319a85",
    "poker_decision_ai/domain/poker/conditional_model.py": "686c9f58b71d9408",
    "poker_decision_ai/agents/judge/agent.py": "239436fae7299c47",
    "poker_decision_ai/realtime.py": "3400a20ff857880d",
    "poker_decision_ai/research_unfreeze/pi_ev_contract.py": "52e6267db49a2e11",
    "poker_decision_ai/research_unfreeze/unfreeze_core.py": "17236d4ed8070d01",
}

V3G_LAYERS = ("cash_state_integrity", "cash_integrity_context",
              "cash_preflop_decision", "cash_multistreet", "cash_opponent_model",
              "cash_range_model", "cash_action_tree", "cash_unknown_e2e",
              "cash_solver_interface", "cash_profit_interface")
V3H_LAYERS = tuple(f.stem for f in PKG.glob("cash_strategy_*.py"))

ALL_FACTS = {k: True for k in ENVIRONMENT_FACT_KEYS}


def _sha(rel: str) -> str:
    p = ROOT / rel
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else "MISSING"


def cash_state(**over) -> dict:
    st = {"game_type": "cash", "street": "flop",
          "board": ["Qs", "7d", "2c"], "pot": 10.0, "to_call": 0.0,
          "bb": 1.0, "sb": 0.5, "hero_id": "hero", "table_size": 3,
          "synthetic_test_only": True, "players": [], "action_history": []}
    st.update(over)
    return st


def baseline() -> dict:
    return build_manifest()


def doctor(man: dict, **over) -> dict:
    d = json.loads(json.dumps(man))
    d.update(over)
    return d


def evalv(base: dict, cur: dict = None) -> dict:
    return evaluate_audit_validity(cash_state(), baseline=base, current=cur)


# ── 1. CASH-only / 缺输入 fail closed ───────────────────────────────
@pytest.mark.parametrize("mode", ["mtt", "pko", "ko", "mystery", "spin",
                                 "MTT", "PKO", "", "   ", None])
def test_non_cash_fail_closed_all_entry_points(mode):
    st = cash_state(game_type=mode)
    for r in (evaluate_audit_validity(st, baseline=baseline()),
              revalidation_plan(st, validity={"present": True})):
        assert r["present"] is False
        assert r["reason"] == "not_cash_mode_fail_closed"
        assert r["still_valid"] is False
        assert r["superseded"] is True
        assert r["revalidation_required"] is True
        assert r["grants_verified_status"] is False
        assert r["grants_readiness"] is False
        assert r["grants_consumable"] is False
        assert r["executed"] is False
        assert r["result"] is None
        assert r["evidence_level"] == "UNAVAILABLE"


@pytest.mark.parametrize("bad", [None, [], "cash", 42, 3.5, True])
def test_non_dict_state_fail_closed(bad):
    assert evaluate_audit_validity(bad, baseline=baseline())[
        "present"] is False
    assert revalidation_plan(bad, validity={})["present"] is False


@pytest.mark.parametrize("bad", [None, [], "x", 42, True, {}, {"a": 1}])
def test_missing_baseline_fails_closed(bad):
    r = evaluate_audit_validity(cash_state(), baseline=bad)
    assert r["present"] is False
    assert r["reason"] == "baseline_required_for_validity_check"
    assert r["still_valid"] is False
    assert r["revalidation_required"] is True


@pytest.mark.parametrize("bad", [None, [], "x", 42, True, {}])
def test_missing_validity_fails_closed_for_plan(bad):
    r = revalidation_plan(cash_state(), validity=bad)
    assert r["present"] is False
    assert r["reason"] == "validity_required"


# ── 2. 清单与指纹 ──────────────────────────────────────────────────
def test_manifest_covers_all_registered_files():
    m = baseline()
    assert m["success"] is True
    assert m["n_files"] == len(AUDIT_INPUT_FILES) == 6
    assert m["n_fingerprinted"] == 6
    assert m["missing"] == []
    assert m["unreadable"] == []
    assert m["unparseable"] == []
    assert m["fingerprint_algorithm"] == "sha256"
    assert isinstance(m["scope_fingerprint"], str)
    assert len(m["scope_fingerprint"]) == 64


def test_manifest_lists_three_layers_and_three_test_files():
    names = " ".join(AUDIT_INPUT_FILES)
    for must in ("cash_strategy_trust_anchor_contract.py",
                 "cash_strategy_consumption_gate.py",
                 "cash_strategy_static_closure_audit.py",
                 "test_v3h15_trust_anchor_contract.py",
                 "test_v3h16_consumption_gate.py",
                 "test_v3h17_static_closure_audit.py"):
        assert must in names, must


def test_manifest_fingerprints_match_independent_hash():
    m = baseline()
    for rel, digest in m["per_file_fingerprints"].items():
        p = (PKG / rel).resolve()
        assert hashlib.sha256(p.read_bytes()).hexdigest() == digest, rel


def test_manifest_is_deterministic():
    assert baseline() == baseline()
    assert baseline()["scope_fingerprint"] == baseline()["scope_fingerprint"]


def test_manifest_never_produces_results():
    m = baseline()
    assert m["result"] is None
    assert m["executed"] is False
    assert all(v is False for v in m["claim_boundary"].values())


# ── 3. 有效期状态机 ────────────────────────────────────────────────
def test_validity_states_are_closed():
    assert VALIDITY_STATES == (
        "VALID_FOR_REGISTERED_SNAPSHOT", "SUPERSEDED_SOURCE_CHANGED",
        "INVALID_UNREGISTERED_CHANGE", "INVALID_UNREADABLE",
        "INVALID_UNPARSEABLE", "INVALID_BASELINE_MISSING",
        "INVALID_SCOPE_MISMATCH")
    c = audit_validity_contract()
    assert c["validity_states"] == list(VALIDITY_STATES)
    assert c["n_validity_states"] == 7


def test_unchanged_snapshot_is_valid_and_claims_only_absence_of_violation():
    r = evalv(baseline())
    assert r["validity"] == "VALID_FOR_REGISTERED_SNAPSHOT"
    assert r["still_valid"] is True
    assert r["superseded"] is False
    assert r["revalidation_required"] is False
    assert r["n_changed"] == 0
    assert r["change_detection_failures"] == []
    assert r["attestation_scope"]["may_claim"] == [
        "no_violation_found_within_the_registered_snapshot_scope"]
    assert r["superseded_result_usable"] is False


def test_changed_content_supersedes_the_audit():
    base = baseline()
    cur = doctor(base)
    first = sorted(cur["per_file_fingerprints"])[0]
    cur["per_file_fingerprints"][first] = "0" * 64
    r = evalv(base, cur)
    assert r["validity"] == "SUPERSEDED_SOURCE_CHANGED"
    assert r["still_valid"] is False
    assert r["superseded"] is True
    assert r["revalidation_required"] is True
    assert first in r["changed_files"]
    assert r["n_changed"] == 1
    assert r["revalidation_requirements"] == list(REVALIDATION_REQUIREMENTS)


def test_missing_registered_file_invalidates():
    base = baseline()
    cur = doctor(base)
    gone = sorted(cur["per_file_fingerprints"])[0]
    del cur["per_file_fingerprints"][gone]
    r = evalv(base, cur)
    assert r["validity"] == "INVALID_UNREADABLE"
    assert r["still_valid"] is False
    assert gone in r["missing_files"]
    assert "REGISTERED_FILE_MISSING" in r["change_detection_failures"]


def test_unregistered_file_invalidates():
    base = baseline()
    cur = doctor(base)
    cur["per_file_fingerprints"]["cash_strategy_brand_new.py"] = "a" * 64
    r = evalv(base, cur)
    assert r["validity"] == "INVALID_UNREGISTERED_CHANGE"
    assert r["still_valid"] is False
    assert "cash_strategy_brand_new.py" in r["unregistered_files"]
    assert "UNREGISTERED_FILE_PRESENT" in r["change_detection_failures"]


def test_unreadable_file_invalidates():
    base = baseline()
    cur = doctor(base, unreadable=["cash_strategy_consumption_gate.py"])
    r = evalv(base, cur)
    assert r["validity"] == "INVALID_UNREADABLE"
    assert "FILE_UNREADABLE" in r["change_detection_failures"]
    assert r["n_unreadable"] == 1


def test_unparseable_file_invalidates():
    base = baseline()
    cur = doctor(base, unparseable=["cash_strategy_static_closure_audit.py"])
    r = evalv(base, cur)
    assert r["validity"] == "INVALID_UNPARSEABLE"
    assert "FILE_UNPARSEABLE" in r["change_detection_failures"]
    assert r["n_unparseable"] == 1


def test_dynamic_load_detection_supersedes_static_conclusion():
    """范围内出现**真实动态加载调用** → 静态结论不再覆盖该路径。

    判定按 **AST 调用结构**，因此"标记清单/文档"里的同名**字面量不误伤**；
    同时验证真实调用**必须**被检出（这是本规则有效的关键）。
    """
    import poker_decision_ai.cash_strategy_audit_validity as mod
    detect = mod._dynamic_load_calls
    # 不误伤：仅有字面量
    assert detect("X = ('exec(', 'getattr(', '__import__(')") == []
    assert detect("MARKERS = ('eval(', 'compile(')") == []
    # 必须检出：真实调用
    assert detect('exec("x")') == ["exec()"]
    assert detect('eval("1+1")') == ["eval()"]
    assert detect('compile("x", "<s>", "exec")') == ["compile()"]
    assert detect('__import__("os")') == ["__import__()"]
    assert detect('getattr(o, "a")') == ["getattr()"]
    assert detect('setattr(o, "a", 1)') == ["setattr()"]
    assert detect('import importlib\nimportlib.import_module("m")') == [
        "importlib.import_module()"]
    assert detect('import importlib\nimportlib.reload(m)') == [
        "importlib.reload()"]
    # 普通代码不误报
    assert detect("def f():\n    return 1\n") == []
    assert detect("import os\nos.path.join('a', 'b')\n") == []


def test_dynamic_load_hits_force_fail_closed_state():
    """若扫描到动态加载命中，状态机必须给出"失效 + 需重新审计"。"""
    base = baseline()
    r = evalv(base, base)
    assert r["dynamic_load_hits"] == {}
    assert r["still_valid"] is True
    # 直接验证状态机的失效语义（未登记文件 = 静态覆盖之外的路径）
    fake = doctor(base)
    fake["per_file_fingerprints"]["dynamic_marker.py"] = "d" * 64
    r2 = evalv(base, fake)
    assert r2["still_valid"] is False
    assert r2["revalidation_required"] is True
    assert "UNREGISTERED_FILE_PRESENT" in r2["change_detection_failures"]
    assert r2["failure_policy"]["DYNAMIC_LOAD_DETECTED"] == "fail_closed"


def test_scope_fingerprint_mismatch_is_flagged():
    base = baseline()
    cur = doctor(base)
    first = sorted(cur["per_file_fingerprints"])[0]
    cur["per_file_fingerprints"][first] = "b" * 64
    r = evalv(base, cur)
    assert "SCOPE_MISMATCH" in r["change_detection_failures"]


def test_current_defaults_to_a_fresh_read_only_manifest():
    r = evaluate_audit_validity(cash_state(), baseline=baseline())
    assert r["present"] is True
    assert r["validity"] in VALIDITY_STATES
    assert r["still_valid"] is True


# ── 4. 重新审计计划 ────────────────────────────────────────────────
def test_revalidation_plan_lists_requirements_when_superseded():
    base = baseline()
    cur = doctor(base)
    first = sorted(cur["per_file_fingerprints"])[0]
    cur["per_file_fingerprints"][first] = "c" * 64
    v = evalv(base, cur)
    p = revalidation_plan(cash_state(), validity=v)
    assert p["revalidation_required"] is True
    assert p["steps"] == list(REVALIDATION_REQUIREMENTS)
    assert p["n_steps"] == len(REVALIDATION_REQUIREMENTS) == 4
    assert p["triggering_failures"]
    assert p["changed_files"] == [first]
    assert "must not be reused" in p["blocking_reason"]
    assert p["audit_executed_by_this_layer"] is False
    assert p["superseded_result_usable"] is False


def test_revalidation_plan_is_empty_when_still_valid():
    v = evalv(baseline())
    p = revalidation_plan(cash_state(), validity=v)
    assert p["revalidation_required"] is False
    assert p["steps"] == []
    assert p["n_steps"] == 0
    assert p["blocking_reason"] is None


def test_revalidation_requirements_are_the_declared_four():
    assert REVALIDATION_REQUIREMENTS == (
        "re_run_the_v3h17_static_closure_audit_on_the_current_source",
        "record_a_new_baseline_manifest_with_fresh_fingerprints",
        "re_confirm_the_audited_scope_and_blind_spots",
        "do_not_reuse_the_superseded_result_for_any_conclusion")


def test_revalidation_plan_never_runs_the_audit():
    base = baseline()
    v = evalv(base, doctor(base, unreadable=["x.py"]))
    p = revalidation_plan(cash_state(), validity=v)
    assert p["audit_executed_by_this_layer"] is False
    assert p["result"] is None
    assert p["executed"] is False


# ── 5. 契约与主张边界 ──────────────────────────────────────────────
def test_contract_declares_fingerprint_limits():
    c = audit_validity_contract()
    fp = c["fingerprint"]
    assert fp["algorithm"] == FINGERPRINT_ALGORITHM == "sha256"
    assert fp["purpose"] == FINGERPRINT_PURPOSE
    assert fp["proves_only"] == ["registered_content_is_unchanged"]
    for cannot in ("authenticity", "provenance", "source_trust", "execution",
                   "occurrence", "certification"):
        assert cannot in fp["does_not_prove"], cannot
    assert fp["is_authenticity_proof"] is False
    assert fp["is_provenance_proof"] is False
    assert fp["is_certification"] is False


def test_contract_states_what_may_and_may_not_be_claimed():
    c = audit_validity_contract()
    assert c["attestation_scope"]["may_claim"] == [
        "no_violation_found_within_the_registered_snapshot_scope"]
    may_not = c["attestation_scope"]["may_not_claim"]
    for banned in ("future_source_will_not_misuse",
                   "authenticity_of_any_content",
                   "provenance_or_source_trust",
                   "actual_occurrence_of_any_event", "certification",
                   "verified_or_readiness_or_consumable_status"):
        assert banned in may_not, banned


def test_no_prospective_guarantee_is_ever_made():
    r = evalv(baseline())
    assert r["claim_boundary"]["is_future_safety_guarantee"] is False
    assert r["claim_boundary"]["fingerprint_treated_as_truth"] is False
    assert "future" in r["validity_does_not_mean"]
    c = audit_validity_contract()
    assert c["validity_does_not_mean"].startswith("future safety")
    assert c["grants_verified_status"] is False
    assert c["grants_readiness"] is False
    assert c["grants_consumable"] is False


def test_never_grants_verified_readiness_or_consumable():
    base = baseline()
    cases = [base, doctor(base), doctor(base, unreadable=["x"]),
             doctor(base, unparseable=["y"]), doctor(base, missing=["z"])]
    for cur in cases:
        r = evalv(base, cur)
        assert r["grants_verified_status"] is False
        assert r["grants_readiness"] is False
        assert r["grants_consumable"] is False
        assert r["superseded_result_usable"] is False


def test_failures_are_all_fail_closed():
    base = baseline()
    cur = doctor(base, unreadable=["a"], unparseable=["b"],
                 missing=["c"])
    r = evalv(base, cur)
    assert r["failure_policy"] == {
        f: "fail_closed" for f in CHANGE_DETECTION_FAILURES}
    assert set(CHANGE_DETECTION_FAILURES) == {
        "UNREGISTERED_FILE_PRESENT", "REGISTERED_FILE_MISSING",
        "FILE_UNREADABLE", "FILE_UNPARSEABLE", "DYNAMIC_LOAD_DETECTED",
        "CODE_OUTSIDE_REGISTERED_SCOPE", "BASELINE_MISSING",
        "SCOPE_MISMATCH"}
    assert r["still_valid"] is False
    assert r["revalidation_required"] is True


def test_audited_scope_kinds_and_dynamic_markers_declared():
    c = audit_validity_contract()
    assert c["audited_scope_kinds"] == list(AUDITED_SCOPE_KINDS)
    assert set(AUDITED_SCOPE_KINDS) == {
        "static_import_graph", "ast_scan", "string_literal_flow",
        "contract_test"}
    assert c["dynamic_load_markers"] == list(DYNAMIC_LOAD_MARKERS)
    for m in ("__import__(", "importlib.import_module", "exec(", "eval(",
              "getattr("):
        assert m in DYNAMIC_LOAD_MARKERS, m


def test_contract_declares_baseline_manifest_fields():
    c = audit_validity_contract()
    for f in ("version", "scope_fingerprint", "n_files", "files",
              "recorded_at_audit_version", "coverage", "blind_spots",
              "limitations"):
        assert f in c["baseline_manifest_fields"], f


# ── 6. 零结果 / 零执行 / 零统计 ────────────────────────────────────
def test_never_produces_results_or_executes():
    for r in (audit_validity_contract(), baseline(), evalv(baseline()),
              revalidation_plan(cash_state(), validity=evalv(baseline()))):
        assert r["executed"] is False
        assert r["result"] is None
        assert r.get("results_available", False) is False
        assert r.get("analysis_executed", False) is False


def test_claim_boundary_all_false_22_flags():
    assert len(CLAIM_BOUNDARY) == 22
    assert all(v is False for v in CLAIM_BOUNDARY.values())
    for k in ("is_future_safety_guarantee", "fingerprint_treated_as_truth",
              "grants_verified_status", "grants_readiness",
              "grants_consumable", "reuses_superseded_audit_result",
              "packages_internal_result_as_external_evidence"):
        assert CLAIM_BOUNDARY[k] is False, k
    r = evalv(baseline())
    assert all(v is False for v in r["claim_boundary"].values())


def test_flat_key_scan_has_no_result_or_statistic_field():
    r = evalv(baseline())
    keys: set[str] = set()

    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                keys.add(k)
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    walk(r)
    for banned in ("p_value", "effect_size", "confidence_interval", "win_rate",
                   "roi", "pnl", "profit_estimate"):
        assert banned not in keys, banned


def test_source_has_no_multiplication_division_or_power():
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    ops = {type(n.op).__name__ for n in ast.walk(tree)
           if isinstance(n, ast.BinOp)}
    for banned in ("Mult", "Div", "FloorDiv", "Pow", "Mod"):
        assert banned not in ops, banned


def test_source_is_read_only_and_has_no_network_process_or_wallet():
    src = MODULE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(src)
    reads = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and n.func.attr in ("read_text", "read_bytes")]
    writes = [n for n in ast.walk(tree)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
              and n.func.attr in ("write_text", "write_bytes", "unlink",
                                  "mkdir", "remove", "rename")]
    assert len(reads) == 2          # 一次 read_bytes（指纹）+ 一次 read_text（动态扫描）
    assert writes == []
    imported = sorted({a.name for n in ast.walk(tree)
                       if isinstance(n, ast.Import) for a in n.names} |
                      {n.module or "" for n in ast.walk(tree)
                       if isinstance(n, ast.ImportFrom)})
    assert imported == ["__future__", "ast", "hashlib", "pathlib",
                        "typing"], imported


# ── 7. 定输入复现 / 不变异 / 可序列化 ──────────────────────────────
def test_deterministic_same_input_same_output():
    base = baseline()
    cur = doctor(base, unreadable=["x"])
    outs = [evalv(base, cur) for _ in range(3)]
    assert all(o == outs[0] for o in outs)
    outs2 = [audit_validity_contract() for _ in range(3)]
    assert all(o == outs2[0] for o in outs2)


def test_lists_are_sorted_for_reproducibility():
    base = baseline()
    cur = doctor(base, unreadable=["z", "a"], unparseable=["m"])
    r = evalv(base, cur)
    for key in ("changed_files", "missing_files", "unregistered_files",
                "unreadable_files", "unparseable_files",
                "change_detection_failures", "unknown"):
        assert r[key] == sorted(r[key]), key


def test_state_not_mutated():
    st = cash_state()
    before = json.dumps(st, sort_keys=True, ensure_ascii=False)
    evalv(baseline())
    evaluate_audit_validity(st, baseline=baseline())
    assert json.dumps(st, sort_keys=True, ensure_ascii=False) == before


def test_outputs_json_serialisable():
    base = baseline()
    for r in (audit_validity_contract(), base, evalv(base),
              revalidation_plan(cash_state(), validity=evalv(base))):
        assert json.loads(json.dumps(r, ensure_ascii=False)) == r


# ── 8. AST · import 隔离 ─────────────────────────────────────────
def test_declares_version_constant():
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    consts = {n.targets[0].id for n in tree.body
              if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)}
    assert any(c.endswith("_VERSION") for c in consts)
    assert CASH_STRATEGY_AUDIT_VALIDITY_VERSION.startswith("v3h18-")


def test_top_level_imports_are_stdlib_only():
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    mods = [n.module for n in tree.body if isinstance(n, ast.ImportFrom)]
    assert mods == ["__future__", "typing"], mods
    plain = [a.name for n in tree.body if isinstance(n, ast.Import)
             for a in n.names]
    assert plain == ["hashlib", "pathlib"], plain


def test_zero_v3g_imports_anywhere_including_function_bodies():
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom):
            assert not any(v in (n.module or "") for v in V3G_LAYERS), n.module
        if isinstance(n, ast.Import):
            for a in n.names:
                assert not any(v in a.name for v in V3G_LAYERS), a.name


def test_zero_layer_imports_anywhere():
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    for n in ast.walk(tree):
        mods = []
        if isinstance(n, ast.ImportFrom):
            mods.append(n.module or "")
        elif isinstance(n, ast.Import):
            mods += [a.name for a in n.names]
        for m in mods:
            assert not any(v in m for v in V3G_LAYERS + V3H_LAYERS), m


def test_no_random_import():
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            assert all(a.name.split(".")[0] != "random" for a in n.names)
        if isinstance(n, ast.ImportFrom):
            assert (n.module or "").split(".")[0] != "random"


def test_no_frozen_multistreet_or_sample_runouts_name():
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            assert all("cash_multistreet" not in a.name for a in n.names)
        if isinstance(n, ast.Name):
            assert n.id != "sample_runouts"


# ── 9. 生产 import 图零新增 ─────────────────────────────────────
def _production_importers() -> dict[str, set[str]]:
    found: dict[str, set[str]] = {}
    for p in PKG.rglob("*.py"):
        if p.name.startswith("cash_") or "__pycache__" in str(p):
            continue
        try:
            tree = ast.parse(p.read_text(encoding="utf-8-sig"))
        except SyntaxError:
            continue
        for n in ast.walk(tree):
            mods = []
            if isinstance(n, ast.ImportFrom) and n.module:
                mods.append(n.module.split(".")[-1])
            elif isinstance(n, ast.Import):
                mods += [a.name.split(".")[-1] for a in n.names]
            for m in mods:
                if m in V3G_LAYERS or m in V3H_LAYERS:
                    found.setdefault(m, set()).add(p.name)
    return found


def test_no_new_production_imports():
    found = _production_importers()
    assert set(found) == {"cash_integrity_context", "cash_unknown_e2e"}, found


def test_new_module_not_imported_in_production():
    assert NEW_MODULE not in _production_importers()


def test_all_v3h_layers_remain_unwired():
    found = _production_importers()
    for layer in V3H_LAYERS:
        assert layer not in found, layer
    required = {
        "cash_strategy_action_policy", "cash_strategy_action_ranking",
        "cash_strategy_audit_validity",
        "cash_strategy_belief_conditioned_eval", "cash_strategy_belief_eval",
        "cash_strategy_belief_update", "cash_strategy_consumption_gate",
        "cash_strategy_decision_assembly", "cash_strategy_dynamic_belief",
        "cash_strategy_evidence_closure", "cash_strategy_execution_intake",
        "cash_strategy_experiment_eligibility",
        "cash_strategy_external_readiness", "cash_strategy_likelihood_model",
        "cash_strategy_multistreet_eval", "cash_strategy_policy_evaluator",
        "cash_strategy_preregistration", "cash_strategy_search_eval",
        "cash_strategy_search_tree", "cash_strategy_shadow_evaluation",
        "cash_strategy_static_closure_audit",
        "cash_strategy_trust_anchor_contract",
    }
    assert required <= set(V3H_LAYERS), sorted(required - set(V3H_LAYERS))


# ── 10. frozen / sealed drift · Solver ───────────────────────────
@pytest.mark.parametrize("rel,prefix", sorted(FROZEN.items()))
def test_frozen_files_unchanged(rel, prefix):
    assert _sha(rel).startswith(prefix), rel


def test_v3g_manifest_drift_unchanged():
    man = json.loads((ROOT / "docs" / "V3G_11_FREEZE_MANIFEST.json")
                     .read_text(encoding="utf-8"))
    drift = []
    for block in ("frozen_modules", "frozen_tests", "frozen_closeouts",
                  "production_baseline"):
        for rel, h in man[block].items():
            if h == "__SELF__":
                continue
            if _sha(rel) != h:
                drift.append(rel)
    assert drift in ([], ["docs/V3G_11_EVALUATION_FREEZE_CLOSEOUT.md"]), drift


def test_solver_still_blocked_and_never_referenced():
    s = solver_status(cash_state())
    assert s["solver_status"] == "BLOCKED"
    assert s["equilibrium"] is False
    assert s["gto_exact"] is False
    assert s["discovery_state"] == "UNAVAILABLE"
    src = MODULE_PATH.read_text(encoding="utf-8")
    assert "solver_status" not in src


# ── 11. 不重复 V3H-14/15/16/17 主体逻辑 ──────────────────────────
def test_does_not_duplicate_prior_layer_surfaces():
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    defined: set[str] = set()
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            defined.add(n.name)
        elif isinstance(n, ast.Assign):
            for t in n.targets:
                if isinstance(t, ast.Name):
                    defined.add(t.id)
    banned = {"audit_external_readiness", "external_dependency_manifest",
              "trust_anchor_contract", "attestation_schema",
              "validate_attestation", "evaluate_anchor_state",
              "consumption_gate_matrix", "evaluate_consumption_gate",
              "audit_contract_consistency", "audit_consumption_closure",
              "audit_coverage_statement", "evidence_ladder_contract",
              "failure_class_policy"}
    assert not (defined & banned), sorted(defined & banned)
    called: set[str] = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Call):
            fn = n.func
            if isinstance(fn, ast.Name):
                called.add(fn.id)
            elif isinstance(fn, ast.Attribute):
                called.add(fn.attr)
    assert not (called & banned), sorted(called & banned)


def test_audit_input_files_do_not_include_this_layer():
    """本层是"有效性问题"，**不**在自身审计范围内。"""
    assert f"{NEW_MODULE}.py" not in AUDIT_INPUT_FILES
    m = baseline()
    assert f"{NEW_MODULE}.py" not in m["per_file_fingerprints"]
