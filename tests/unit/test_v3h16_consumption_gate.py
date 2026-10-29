# -*- coding: utf-8 -*-
"""V3H-16 测试 — CASH 契约一致性与消费边界审计（只读）。

覆盖：CASH-only fail closed / 非 dict / 缺或非法 anchor_state /
**消费门禁矩阵**（6 状态全部只读且只能"展示为未满足"；统计·收益·Solver·
生产·结论消费**恒 0**）/ **已验证状态在项目内不可达** /
`STRUCTURE_VALID` / `CONTENT_CONSISTENT` **不得**被当作 `readiness=True` /
truthy `readiness` / `signature_verified` / `authenticity_verified` 一律拒绝 /
未验证却声称已满足类别 → 冲突拒绝（疑似静默补全）/ 未知状态 fail closed /
12 条禁止推断 + 7 类误用风险登记 / **契约一致性审计 16 项不变量** /
不消费 V3H-14/V3H-15 判定 / 零结果零执行 / 定输入复现 / state 不变异 /
AST·import 隔离（零 V3G、零任何层、无网络/进程/钱包）/ 生产 import 图零新增 /
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

from poker_decision_ai.cash_strategy_consumption_gate import (  # noqa: E402
    ALWAYS_FORBIDDEN_CLASSES, CASH_STRATEGY_CONSUMPTION_GATE_VERSION,
    CLAIM_BOUNDARY, CONSUMPTION_CLASSES, CONTRACT_INVARIANTS,
    DISPLAY_AS_UNSATISFIED_ONLY_STATES, FORBIDDEN_INFERENCES, MISUSE_RISKS,
    READABLE_ANCHOR_STATES, REFUSAL_REASONS, V3H15_MODULE_STEM,
    audit_contract_consistency, consumption_gate_matrix,
    evaluate_consumption_gate)

# 测试侧允许 import V3G/V3H 层（生产侧不允许，由 AST 测试强制）
from poker_decision_ai.cash_solver_interface import solver_status  # noqa: E402
from poker_decision_ai.cash_strategy_experiment_eligibility import (  # noqa: E402
    ENVIRONMENT_FACT_KEYS)
from poker_decision_ai.cash_strategy_trust_anchor_contract import (  # noqa: E402
    ANCHOR_STATES, ASSERTION_CLASSES, evaluate_anchor_state,
    trust_anchor_contract)

NEW_MODULE = "cash_strategy_consumption_gate"
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


def anchor(state: str = "ANCHOR_UNSATISFIED", **over) -> dict:
    a = {"present": True, "state": state, "classes_satisfied": [],
         "classes_missing": list(ASSERTION_CLASSES), "readiness": False,
         "signature_verified": False, "authenticity_verified": False,
         "structural_fully_anchored": state == "ANCHOR_CONTENT_CONSISTENT"}
    a.update(over)
    return a


# ── 1. CASH-only fail closed ─────────────────────────────────────────
@pytest.mark.parametrize("mode", ["mtt", "pko", "ko", "mystery", "spin",
                                 "MTT", "PKO", "", "   ", None])
def test_non_cash_fail_closed_all_entry_points(mode):
    st = cash_state(game_type=mode)
    for r in (consumption_gate_matrix() if False else
              evaluate_consumption_gate(st, anchor_state=anchor()),
              audit_contract_consistency(st)):
        assert r["present"] is False
        assert r["reason"] == "not_cash_mode_fail_closed"
        assert r["consumption_allowed"] is False
        assert r["consumption_class"] == "REQUIRES_EXTERNAL_VERIFICATION"
        assert r["verified_status_granted"] is False
        assert r["executed"] is False
        assert r["result"] is None
        assert r["evidence_level"] == "UNAVAILABLE"


@pytest.mark.parametrize("bad", [None, [], "cash", 42, 3.5, True])
def test_non_dict_state_fail_closed(bad):
    assert evaluate_consumption_gate(bad, anchor_state=anchor())[
        "present"] is False
    assert audit_contract_consistency(bad)["present"] is False


@pytest.mark.parametrize("bad", [None, [], "x", 42, True, {}])
def test_missing_or_empty_anchor_state_fail_closed(bad):
    r = evaluate_consumption_gate(cash_state(), anchor_state=bad)
    assert r["present"] is False
    assert r["reason"] == "anchor_state_required"
    assert r["consumption_allowed"] is False
    assert r["verified_status_granted"] is False
    assert r["statistics_consumption_allowed"] is False
    assert r["profit_or_solver_consumption_allowed"] is False
    assert r["production_consumption_allowed"] is False
    assert r["claim_consumption_allowed"] is False


def test_anchor_state_present_false_fail_closed():
    r = evaluate_consumption_gate(cash_state(),
                                  anchor_state={"present": False,
                                                "reason": "x"})
    assert r["present"] is False
    assert r["reason"] == "anchor_state_present_false"


# ── 2. 消费门禁矩阵 ──────────────────────────────────────────────────
def test_matrix_covers_six_states_and_denies_all_downstream_consumption():
    m = consumption_gate_matrix()
    assert m["success"] is True
    assert m["n_readable_states"] == 6
    assert m["n_states_consumable_for_statistics"] == 0
    assert m["n_states_consumable_for_profit_or_solver"] == 0
    assert m["n_states_consumable_for_production"] == 0
    assert m["verified_status_reachable_internally"] is False
    assert m["verified_status_source"] == "external_only"
    for e in m["per_state"]:
        assert e["consumable_for_statistics"] is False, e["state"]
        assert e["consumable_for_profit_or_solver"] is False, e["state"]
        assert e["consumable_for_production"] is False, e["state"]
        assert e["consumable_for_claims"] is False, e["state"]
        assert e["verified_status"] is False, e["state"]
        assert e["display_as_unsatisfied_only"] is True, e["state"]


def test_matrix_states_match_the_v3h15_closed_enum():
    m = consumption_gate_matrix()
    assert m["readable_anchor_states"] == list(ANCHOR_STATES)
    assert READABLE_ANCHOR_STATES == ANCHOR_STATES
    assert DISPLAY_AS_UNSATISFIED_ONLY_STATES == ANCHOR_STATES


def test_consumption_class_enum_is_closed():
    assert CONSUMPTION_CLASSES == (
        "READ_AND_DISPLAY_ALLOWED", "DISPLAY_AS_UNSATISFIED_ONLY",
        "REFUSED_FOR_STATISTICS", "REFUSED_FOR_PROFIT_OR_SOLVER",
        "REFUSED_FOR_PRODUCTION", "REQUIRES_EXTERNAL_VERIFICATION",
        "CONSUMABLE_VERIFIED")
    assert set(ALWAYS_FORBIDDEN_CLASSES) == {
        "REFUSED_FOR_STATISTICS", "REFUSED_FOR_PROFIT_OR_SOLVER",
        "REFUSED_FOR_PRODUCTION", "REQUIRES_EXTERNAL_VERIFICATION",
        "CONSUMABLE_VERIFIED"}


def test_matrix_invariants_are_declared():
    m = consumption_gate_matrix()
    assert len(m["invariants"]) == 3
    joined = " ".join(m["invariants"])
    assert "never verifies authenticity" in joined
    assert "always denied" in joined
    assert "unreachable internally" in joined


def test_verified_status_is_unreachable_for_every_state():
    for s in ANCHOR_STATES:
        r = evaluate_consumption_gate(cash_state(), anchor_state=anchor(s))
        assert r["consumption_class"] == "DISPLAY_AS_UNSATISFIED_ONLY"
        assert r["consumption_class"] != "CONSUMABLE_VERIFIED"
        assert r["verified_status_granted"] is False
        assert r["consumption_allowed"] is False


# ── 3. 结构齐备 / 内容一致 ≠ readiness ──────────────────────────────
@pytest.mark.parametrize("s", ["ANCHOR_STRUCTURE_VALID",
                               "ANCHOR_CONTENT_CONSISTENT"])
def test_structure_or_content_is_never_readiness(s):
    r = evaluate_consumption_gate(cash_state(), anchor_state=anchor(s))
    assert r["readable"] is True                 # 可读，但……
    assert r["display_as_unsatisfied_only"] is True
    assert r["statistics_consumption_allowed"] is False
    assert r["profit_or_solver_consumption_allowed"] is False
    assert r["production_consumption_allowed"] is False
    assert r["claim_consumption_allowed"] is False
    assert r["verified_status_granted"] is False
    assert "source_trust_not_established" in r["refusal_reasons"]


def test_truthy_readiness_is_rejected():
    r = evaluate_consumption_gate(
        cash_state(), anchor_state=anchor("ANCHOR_CONTENT_CONSISTENT",
                                          readiness=True))
    assert "anchor_readiness_truthy_rejected" in r["refusal_reasons"]
    assert r["readable"] is False
    assert r["consumption_allowed"] is False


@pytest.mark.parametrize("flag", ["signature_verified", "authenticity_verified"])
def test_truthy_verification_flags_are_rejected(flag):
    r = evaluate_consumption_gate(cash_state(),
                                  anchor_state=anchor(**{flag: True}))
    assert f"anchor_{flag}_truthy_rejected" in r["refusal_reasons"]
    assert r["readable"] is False
    assert r["consumption_allowed"] is False


def test_verified_classes_without_verification_is_a_conflict():
    """未验证却声称已满足类别 → 冲突（疑似静默补全），fail closed。"""
    r = evaluate_consumption_gate(
        cash_state(),
        anchor_state=anchor("ANCHOR_CONTENT_CONSISTENT",
                            classes_satisfied=list(ASSERTION_CLASSES)))
    assert "anchor_classes_satisfied_unverified_rejected" in \
        r["refusal_reasons"]
    assert r["readable"] is False
    assert r["silent_field_completion_performed"] is False
    assert r["classes_satisfied_grants_nothing"] is True


def test_observed_classes_never_grant_anything():
    r = evaluate_consumption_gate(
        cash_state(),
        anchor_state=anchor("ANCHOR_STRUCTURE_VALID",
                            classes_satisfied=["A1_EXTERNAL_RESOURCE_EXISTS"]))
    assert r["classes_satisfied_observed"] == ["A1_EXTERNAL_RESOURCE_EXISTS"]
    assert r["classes_satisfied_grants_nothing"] is True
    assert r["consumption_allowed"] is False


# ── 4. 未知状态 fail closed ─────────────────────────────────────────
@pytest.mark.parametrize("s", [None, "", "  ", "ANCHOR_VERIFIED",
                               "ready", 42, True, ["ANCHOR_UNSATISFIED"]])
def test_unknown_or_invalid_state_fails_closed(s):
    r = evaluate_consumption_gate(cash_state(), anchor_state=anchor(s))
    assert "anchor_state_not_in_closed_enum" in r["refusal_reasons"]
    assert "unknown_consumption_state_fails_closed" in r["refusal_reasons"]
    assert r["readable"] is False
    assert r["consumption_class"] == "REQUIRES_EXTERNAL_VERIFICATION"
    assert r["consumption_allowed"] is False


def test_missing_state_key_fails_closed():
    a = anchor()
    del a["state"]
    r = evaluate_consumption_gate(cash_state(), anchor_state=a)
    assert r["consumption_allowed"] is False
    assert r["readable"] is False


def test_real_v3h15_anchor_states_are_all_handled():
    """真实 V3H-15 输出在各种输入下都能被本门禁安全处理。"""
    for atts in ([], [None, "x"], [{"assertion_class": "A9"}]):
        real = evaluate_anchor_state(cash_state(), attestations=atts)
        r = evaluate_consumption_gate(cash_state(), anchor_state=real)
        assert r["present"] is True
        assert r["consumption_allowed"] is False
        assert r["verified_status_granted"] is False
        assert r["statistics_consumption_allowed"] is False


def test_no_real_anchor_state_can_be_consumed():
    """枚举 V3H-15 全部真实状态：无一可被消费。"""
    seen = set()
    for atts in ([], [{"assertion_class": ASSERTION_CLASSES[0]}]):
        real = evaluate_anchor_state(cash_state(), attestations=atts)
        seen.add(real["state"])
        r = evaluate_consumption_gate(cash_state(), anchor_state=real)
        assert r["consumption_allowed"] is False
    assert seen <= set(ANCHOR_STATES)


# ── 5. 禁止推断与误用风险登记 ───────────────────────────────────────
def test_forbidden_inferences_are_closed_and_include_the_key_ones():
    assert len(FORBIDDEN_INFERENCES) == 12
    for x in ("existence_implies_trust",
              "signature_implies_occurrence",
              "hash_implies_occurrence",
              "structure_valid_implies_readiness",
              "content_consistent_implies_readiness",
              "integrity_A5_implies_occurrence_A6",
              "self_approved_external_declaration",
              "missing_field_completed_silently",
              "unverified_record_consumed_as_evidence"):
        assert x in FORBIDDEN_INFERENCES, x


def test_misuse_risks_cover_all_seven_requested_checks():
    assert len(MISUSE_RISKS) == 7
    assert MISUSE_RISKS == (
        "RSK-1_EXISTENCE_AS_TRUST",
        "RSK-2_SIGNATURE_OR_HASH_AS_OCCURRENCE",
        "RSK-3_SELF_APPROVED_EXTERNAL_DECLARATION",
        "RSK-4_STRUCTURE_OR_CONTENT_AS_READINESS",
        "RSK-5_A5_INFERRED_AS_A6",
        "RSK-6_SILENT_FIELD_COMPLETION",
        "RSK-7_UNVERIFIED_RECORD_REACHES_DOWNSTREAM",
    )


def test_gate_output_declares_all_forbidden_inferences_and_risks():
    r = evaluate_consumption_gate(cash_state(), anchor_state=anchor())
    assert r["forbidden_inferences"] == list(FORBIDDEN_INFERENCES)
    assert r["misuse_risks_covered"] == list(MISUSE_RISKS)
    assert r["silent_field_completion_performed"] is False
    assert r["external_resources_accessed"] is False


def test_rs5_a5_does_not_imply_a6_is_enforced_by_v3h15_and_registered_here():
    """A5 存在而 A6 缺失时，本门禁仍不得放行。"""
    real = evaluate_anchor_state(
        cash_state(),
        attestations=[{"attestation_id": "x", "assertion_class":
                       "A5_RECORD_INTEGRITY",
                       "attestation_type": "integrity_attestation",
                       "attesting_party": "acme-external-auditor",
                       "attesting_party_kind": "external_auditor",
                       "subject_ref": "s", "subject_content_digest": "a" * 64,
                       "issued_at_epoch_ms": 1, "signature_ref": "r",
                       "signature_scheme": "ed25519-external",
                       "revocation_ref": None, "scope_note": "one class"}])
    assert "A5_RECORD_INTEGRITY" in real["classes_satisfied"]
    assert "A6_RECORD_ACTUALLY_OCCURRED" in real["classes_missing"]
    r = evaluate_consumption_gate(cash_state(), anchor_state=real)
    assert r["consumption_allowed"] is False
    assert "integrity_A5_implies_occurrence_A6" in r["forbidden_inferences"]


def test_refusal_reason_enum_is_closed_and_documented():
    assert len(REFUSAL_REASONS) == 12
    for x in ("anchor_readiness_truthy_rejected",
              "anchor_classes_satisfied_unverified_rejected",
              "anchor_state_not_in_closed_enum",
              "source_trust_not_established",
              "external_verification_never_performed_here",
              "unknown_consumption_state_fails_closed"):
        assert x in REFUSAL_REASONS


# ── 6. 契约一致性审计 ──────────────────────────────────────────────
def test_contract_consistency_audit_passes_16_of_16():
    r = audit_contract_consistency(cash_state())
    assert r["present"] is True
    assert r["audited_module"] == V3H15_MODULE_STEM
    assert r["n_checks"] == 16
    assert r["n_failed"] == 0, [c for c in r["checks"] if not c["ok"]]
    assert r["consistent"] is True
    assert r["unknown"] == []


def test_every_contract_invariant_is_checkable_and_reported():
    r = audit_contract_consistency(cash_state())
    ids = [c["invariant"] for c in r["checks"]]
    assert ids == list(CONTRACT_INVARIANTS)
    assert len(CONTRACT_INVARIANTS) == 16
    for c in r["checks"]:
        assert isinstance(c["ok"], bool)
        assert c["detail"]


def test_all_seven_risks_are_marked_covered_when_consistent():
    r = audit_contract_consistency(cash_state())
    assert r["consistent"] is True
    assert set(r["risk_status"]) == set(MISUSE_RISKS)
    assert all(v == "COVERED" for v in r["risk_status"].values())


def test_audit_does_not_consume_or_recompute_upstream_verdicts():
    r = audit_contract_consistency(cash_state())
    assert r["behavioral_verdict_consumed"] is False
    assert r["v3h14_or_v3h15_verdict_recomputed"] is False
    assert r["external_resources_accessed"] is False
    assert r["silent_field_completion_performed"] is False
    src = MODULE_PATH.read_text(encoding="utf-8")
    for banned in ("audit_external_readiness", "external_dependency_manifest",
                   "RESOLUTION_CLASSES", "READINESS_VERDICTS",
                   "EXTERNAL_REQUIREMENTS"):
        assert banned not in src, banned


def test_audit_fails_closed_when_source_unavailable(monkeypatch):
    import poker_decision_ai.cash_strategy_consumption_gate as mod
    monkeypatch.setattr(mod, "_module_source",
                        lambda: (None, "module_source_unavailable:OSError"))
    r = mod.audit_contract_consistency(cash_state())
    assert r["present"] is False
    assert r["reason"].startswith("module_source_unavailable")
    assert r["consistent"] is False if "consistent" in r else True
    assert r["executed"] is False
    assert r["result"] is None


def test_audit_never_grants_consumption():
    r = audit_contract_consistency(cash_state())
    assert "consumption_allowed" not in r or r["consumption_allowed"] is False
    assert r["claim_boundary"]["permits_statistics_consumption"] is False
    assert r["claim_boundary"]["permits_production_consumption"] is False


def test_audited_module_does_not_import_layers_or_io():
    src = (PKG / f"{V3H15_MODULE_STEM}.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    for n in ast.walk(tree):
        mods = []
        if isinstance(n, ast.ImportFrom):
            mods.append(n.module or "")
        elif isinstance(n, ast.Import):
            mods += [a.name for a in n.names]
        for m in mods:
            assert not any(v in m for v in V3G_LAYERS + V3H_LAYERS), m
    lower = src.lower()
    for banned in ("socket", "requests", "urllib", "subprocess", "sqlite",
                   "pyautogui", "click"):
        assert banned not in lower, banned


# ── 7. 零结果 / 零执行 / 零统计 ────────────────────────────────────
def test_never_produces_results_or_executes():
    for r in (consumption_gate_matrix(),
              evaluate_consumption_gate(cash_state(), anchor_state=anchor()),
              audit_contract_consistency(cash_state())):
        assert r["executed"] is False
        assert r["result"] is None
        assert r.get("results_available", False) is False
        assert r.get("analysis_executed", False) is False


def test_claim_boundary_all_false_25_flags():
    assert len(CLAIM_BOUNDARY) == 25
    assert all(v is False for v in CLAIM_BOUNDARY.values())
    for k in ("structure_treated_as_readiness",
              "signature_treated_as_occurrence_proof",
              "existence_treated_as_trust",
              "self_approves_external_declarations",
              "silently_completes_missing_fields",
              "permits_statistics_consumption",
              "permits_profit_or_solver_consumption",
              "permits_production_consumption",
              "permits_claim_consumption",
              "grants_verified_status_internally"):
        assert CLAIM_BOUNDARY[k] is False, k
    r = evaluate_consumption_gate(cash_state(), anchor_state=anchor())
    assert all(v is False for v in r["claim_boundary"].values())


def test_flat_key_scan_has_no_result_or_statistic_field():
    r = evaluate_consumption_gate(cash_state(), anchor_state=anchor())
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
                   "roi", "pnl", "profit_estimate", "exploitability_value",
                   "equilibrium", "gto"):
        assert banned not in keys, banned


def test_source_has_no_multiplication_division_or_power():
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    ops = {type(n.op).__name__ for n in ast.walk(tree)
           if isinstance(n, ast.BinOp)}
    for banned in ("Mult", "Div", "FloorDiv", "Pow", "Mod"):
        assert banned not in ops, banned


def test_source_has_no_network_process_or_wallet_capability():
    """按**可执行结构**判定：不得 import / 调用任何网络·进程·钱包能力。

    （模块内出现 `"socket"` 等字样仅可能是"被禁止能力"的声明清单，
    不属于能力本身，因此按 AST 判定而非子串扫描。）
    """
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    banned = ("socket", "requests", "urllib", "http", "subprocess", "sqlite",
              "wallet", "click", "pyautogui", "advapi32", "win32")
    imported: list[str] = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            imported += [a.name for a in n.names]
        elif isinstance(n, ast.ImportFrom):
            imported.append(n.module or "")
        elif isinstance(n, ast.Attribute):
            pass
    for m in imported:
        assert not any(b in m for b in banned), m
    # 不得出现子进程 / 客户端自动化的调用结构
    for n in ast.walk(tree):
        if isinstance(n, ast.Call):
            fn = n.func
            name = ""
            if isinstance(fn, ast.Name):
                name = fn.id
            elif isinstance(fn, ast.Attribute):
                name = fn.attr
            assert name not in ("Popen", "run", "check_output", "call",
                                "system"), name


def test_only_readonly_source_inspection_performed():
    src = MODULE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(src)
    reads = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and n.func.attr == "read_text"]
    writes = [n for n in ast.walk(tree)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
              and n.func.attr in ("write_text", "write_bytes", "unlink",
                                  "mkdir", "remove", "rename", "rmdir")]
    assert len(reads) == 1
    assert writes == []
    imported = {n.names[0].name for n in ast.walk(tree)
                if isinstance(n, ast.Import)}
    assert imported == {"ast", "pathlib"}


# ── 8. 定输入复现 / 不变异 / 可序列化 ──────────────────────────────
def test_deterministic_same_input_same_output():
    st = cash_state()
    a = anchor("ANCHOR_STRUCTURE_VALID",
               classes_satisfied=["A1_EXTERNAL_RESOURCE_EXISTS"])
    outs = [evaluate_consumption_gate(st, anchor_state=a) for _ in range(5)]
    assert all(o == outs[0] for o in outs)
    outs2 = [audit_contract_consistency(st) for _ in range(3)]
    assert all(o == outs2[0] for o in outs2)
    outs3 = [consumption_gate_matrix() for _ in range(3)]
    assert all(o == outs3[0] for o in outs3)


def test_refusal_reasons_are_sorted_for_reproducibility():
    r = evaluate_consumption_gate(
        cash_state(), anchor_state=anchor("ANCHOR_CONTENT_CONSISTENT",
                                          readiness=True,
                                          classes_satisfied=["A1_"]))
    assert r["refusal_reasons"] == sorted(r["refusal_reasons"])
    assert r["unknown"] == sorted(r["unknown"])


def test_state_and_anchor_not_mutated():
    st = cash_state()
    a = anchor("ANCHOR_STRUCTURE_VALID",
               classes_satisfied=["A1_EXTERNAL_RESOURCE_EXISTS"])
    st_before = json.dumps(st, sort_keys=True, ensure_ascii=False)
    a_before = json.dumps(a, sort_keys=True, ensure_ascii=False)
    evaluate_consumption_gate(st, anchor_state=a)
    audit_contract_consistency(st)
    assert json.dumps(st, sort_keys=True, ensure_ascii=False) == st_before
    assert json.dumps(a, sort_keys=True, ensure_ascii=False) == a_before


def test_outputs_json_serialisable():
    st = cash_state()
    for r in (consumption_gate_matrix(),
              evaluate_consumption_gate(st, anchor_state=anchor()),
              audit_contract_consistency(st)):
        assert json.loads(json.dumps(r, ensure_ascii=False)) == r


# ── 9. AST · import 隔离 ─────────────────────────────────────────
def test_declares_version_constant():
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    consts = {n.targets[0].id for n in tree.body
              if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)}
    assert any(c.endswith("_VERSION") for c in consts)
    assert CASH_STRATEGY_CONSUMPTION_GATE_VERSION.startswith("v3h16-")


def test_top_level_imports_are_stdlib_only():
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    mods = [n.module for n in tree.body if isinstance(n, ast.ImportFrom)]
    assert mods == ["__future__", "typing"], mods
    plain = [a.name for n in tree.body if isinstance(n, ast.Import)
             for a in n.names]
    assert plain == ["ast", "pathlib"], plain


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


def test_no_client_automation_path_anywhere():
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    imported = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            imported += [a.name for a in n.names]
        elif isinstance(n, ast.ImportFrom):
            imported.append(n.module or "")
    for m in imported:
        for banned in ("pyautogui", "keyboard", "advapi32", "win32", "pynput"):
            assert banned not in m, m
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name):
            assert n.func.id != "input", "input() must not be called"


# ── 10. 生产 import 图零新增 ─────────────────────────────────────
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
        "cash_strategy_belief_conditioned_eval", "cash_strategy_belief_eval",
        "cash_strategy_belief_update", "cash_strategy_consumption_gate",
        "cash_strategy_decision_assembly", "cash_strategy_dynamic_belief",
        "cash_strategy_evidence_closure", "cash_strategy_execution_intake",
        "cash_strategy_experiment_eligibility",
        "cash_strategy_external_readiness", "cash_strategy_likelihood_model",
        "cash_strategy_multistreet_eval", "cash_strategy_policy_evaluator",
        "cash_strategy_preregistration", "cash_strategy_search_eval",
        "cash_strategy_search_tree", "cash_strategy_shadow_evaluation",
        "cash_strategy_trust_anchor_contract",
    }
    assert required <= set(V3H_LAYERS), sorted(required - set(V3H_LAYERS))


# ── 11. frozen / sealed drift · Solver ───────────────────────────
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


def test_solver_still_blocked_and_never_used_here():
    s = solver_status(cash_state())
    assert s["solver_status"] == "BLOCKED"
    assert s["available"] is False
    assert s["equilibrium"] is False
    assert s["gto_exact"] is False
    assert s["discovery_state"] == "UNAVAILABLE"
    src = MODULE_PATH.read_text(encoding="utf-8")
    assert "solver_status" not in src
    assert "solver_invoked" not in src


# ── 12. 与 V3H-14 / V3H-15 的关系 ────────────────────────────────
def test_does_not_duplicate_v3h15_assertion_classes():
    """本层**不**重新声明六类断言，只按调用方传入的状态工作。"""
    src = MODULE_PATH.read_text(encoding="utf-8")
    for cls in ASSERTION_CLASSES:
        assert f'"{cls}"' not in src, cls
    # 仅可在"上游结构令牌清单"与 docstring 说明中各出现一次，
    # 不得作为本层自己的常量重复实现。
    assert src.count("ASSERTION_CLASSES") == 2
    assert "ASSERTION_CLASS_SPECS" in src            # 仅在令牌清单内
    assert "ASSERTION_CLASS_SPECS: " not in src      # 不是本层声明
    assert "ASSERTION_CLASSES: " not in src


def test_gate_cannot_widen_permissions_from_caller_supplied_tokens():
    r = evaluate_consumption_gate(
        cash_state(), anchor_state=anchor("ANCHOR_CONTENT_CONSISTENT"),
        classes_satisfied=list(ASSERTION_CLASSES),
        requesting_consumer="statistics")
    assert r["requesting_consumer"] == "statistics"
    assert r["consumer_kind_recorded_only"] is True
    assert r["consumption_allowed"] is False
    assert r["statistics_consumption_allowed"] is False


def test_contract_contract_still_declares_no_verification():
    c = trust_anchor_contract()
    assert c["signature_verification"]["implemented_here"] is False
    assert c["signature_verification"]["performed_here"] is False
    assert c["attestations_issued_by_this_project"] is False
    assert c["sha256_role"]["sha256_alone_is_sufficient_for_any_class"] is False
    r = audit_contract_consistency(cash_state())
    assert r["consistent"] is True
