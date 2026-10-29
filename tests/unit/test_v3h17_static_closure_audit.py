# -*- coding: utf-8 -*-
"""V3H-17 测试 — CASH 外部证据消费边界静态闭环审计（只读）。

覆盖：CASH-only fail closed / 非 dict / 缺上游状态时 fail closed /
**六类受保护下游入口零可达**（收益·盈亏 / Solver·GTO / 策略评分·动作选择·
exploitability / 生产 Realtime·Judge / 外部复现结论 / 已验证·真实发生结论）/
锚状态字面量**不透出**到获准模块之外 / 生产链路零引用 /
结论性令牌**未**在锚状态流中**被消费**（仅允许否定声明）/
V3G-8·V3G-10 自有结论字段仅作**上下文记录**、不计违规 /
**七级证据阶梯严格递增**、跨级推断被禁止、本项目内上限 L3 /
`CONSUMABLE`（L7）不可达 / 七类失败类别全部 fail closed /
**盲区与限制如实声明** / **"未发现" ≠ "未来不会误用"** /
违规定位能力（monkeypatch 注入 + 移除生产 import 后误报复现）/
只读（一次 read_text、零写入）/ 定输入复现 / state 不变异 /
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

from poker_decision_ai.cash_strategy_static_closure_audit import (  # noqa: E402
    ALLOWED_ANCHOR_STATE_MODULES, CASH_STRATEGY_STATIC_CLOSURE_AUDIT_VERSION,
    CLAIM_BOUNDARY, CONCLUSION_TOKENS, EVIDENCE_LADDER, FAILURE_CLASSES,
    FORBIDDEN_LADDER_INFERENCES, LIMITATIONS,
    MAX_INTERNALLY_REACHABLE_LEVEL, NEGATIVE_DECLARATION_VALUES,
    PRODUCTION_CHAIN_MODULES, PROTECTED_SINKS, STATIC_AUDIT_BLIND_SPOTS,
    UNREACHABLE_INTERNALLY_LEVELS, audit_consumption_closure,
    audit_coverage_statement, evidence_ladder_contract, failure_class_policy)

# 测试侧允许 import V3G/V3H 层（生产侧不允许，由 AST 测试强制）
from poker_decision_ai.cash_solver_interface import solver_status  # noqa: E402
from poker_decision_ai.cash_strategy_experiment_eligibility import (  # noqa: E402
    ENVIRONMENT_FACT_KEYS)
from poker_decision_ai.cash_strategy_trust_anchor_contract import (  # noqa: E402
    ANCHOR_STATES)

NEW_MODULE = "cash_strategy_static_closure_audit"
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


# ── 1. CASH-only fail closed ─────────────────────────────────────────
@pytest.mark.parametrize("mode", ["mtt", "pko", "ko", "mystery", "spin",
                                 "MTT", "PKO", "", "   ", None])
def test_non_cash_fail_closed_all_entry_points(mode):
    st = cash_state(game_type=mode)
    for r in (audit_consumption_closure(st),
              audit_coverage_statement(st)):
        assert r["present"] is False
        assert r["reason"] == "not_cash_mode_fail_closed"
        assert r["audit_result"] == "FAIL_CLOSED"
        assert r["no_violation_found"] is False
        assert r["n_violations"] == 0
        assert r["n_files_scanned"] == 0
        assert r["consumable_reachable"] is False
        assert r["executed"] is False
        assert r["result"] is None
        assert r["evidence_level"] == "UNAVAILABLE"


@pytest.mark.parametrize("bad", [None, [], "cash", 42, 3.5, True])
def test_non_dict_state_fail_closed(bad):
    assert audit_consumption_closure(bad)["present"] is False
    assert audit_coverage_statement(bad)["present"] is False


def test_unconfirmable_anchor_states_fail_closed(monkeypatch):
    """若无法从契约模块确认六类锚状态 → fail closed（不猜）。"""
    import poker_decision_ai.cash_strategy_static_closure_audit as mod

    def fake_scan():
        return {"records": {"cash_strategy_trust_anchor_contract": {
            "file": "cash_strategy_trust_anchor_contract.py",
            "literals": {"NOT_AN_ANCHOR_STATE"},
            "imports": set(), "source": "", "tree": ast.parse("")}},
            "unparseable": [], "n_files": 1}

    monkeypatch.setattr(mod, "_scan_package", fake_scan)
    r = mod.audit_consumption_closure(cash_state())
    assert r["present"] is False
    assert r["reason"] == "anchor_states_not_confirmable_from_contract_module"
    assert r["audit_result"] == "FAIL_CLOSED"


# ── 2. 六类受保护下游入口：零可达 ────────────────────────────────────
def test_six_protected_sinks_declared():
    assert len(PROTECTED_SINKS) == 6
    names = [s["name"] for s in PROTECTED_SINKS]
    assert names == [
        "profit_or_pnl_computation",
        "solver_equilibrium_or_gto_conclusion",
        "strategy_scoring_action_selection_or_exploitability",
        "production_realtime_or_judge_chain",
        "external_replication_conclusion",
        "verified_or_actually_occurred_conclusion",
    ]
    for s in PROTECTED_SINKS:
        assert s["id"].startswith("SINK-")
        assert s["must_not_receive"]


def test_no_protected_sink_is_reachable_from_anchor_states():
    r = audit_consumption_closure(cash_state())
    assert r["present"] is True
    assert r["n_sinks"] == 6
    assert r["n_sinks_reachable"] == 0, r["sink_results"]
    for s in r["sink_results"]:
        assert s["reachable"] is False, s
        assert s["status"] == "NO_VIOLATION_FOUND", s
        assert s["evidence"] == []


def test_audit_reports_no_violation_and_zero_violations():
    r = audit_consumption_closure(cash_state())
    assert r["audit_result"] == "NO_VIOLATION_FOUND"
    assert r["no_violation_found"] is True
    assert r["n_violations"] == 0
    assert r["violations"] == []
    assert r["unknown"] == []


def test_anchor_state_literals_do_not_escape_the_allowed_modules():
    r = audit_consumption_closure(cash_state())
    assert r["unexpected_anchor_state_referencing_modules"] == []
    assert set(r["anchor_state_referencing_modules"]) <= set(
        ALLOWED_ANCHOR_STATE_MODULES)
    assert sorted(r["anchor_states"]) == sorted(ANCHOR_STATES)
    assert len(r["anchor_states"]) == 6


def test_production_chain_contains_no_anchor_reference():
    """生产链路（含 agents/judge）不得引用锚状态或三个锚相关层。"""
    r = audit_consumption_closure(cash_state())
    assert r["production_chain_references"] == {}
    assert r["production_conclusion_token_leaks"] == {}
    checked = 0
    for p in sorted(PKG.rglob("*.py")):
        if p.name.startswith("cash_") or "__pycache__" in str(p):
            continue
        src = p.read_text(encoding="utf-8-sig")
        for s in ANCHOR_STATES:
            assert s not in src, f"{p.name}:{s}"
        for m in ("cash_strategy_trust_anchor_contract",
                  "cash_strategy_consumption_gate",
                  "cash_strategy_static_closure_audit"):
            assert m not in src, f"{p.name}:{m}"
        checked += 1
    assert checked >= 2          # realtime.py 与 agents/judge/agent.py


def test_conclusion_tokens_are_not_consumed_in_the_anchor_flow():
    r = audit_consumption_closure(cash_state())
    assert r["conclusion_token_leaks"] == {}
    assert "anchor-state-referencing modules plus production modules only" in \
        r["conclusion_token_check_scope"]
    for tok in CONCLUSION_TOKENS:
        assert tok not in r["conclusion_token_leaks"]


def test_v3g_solver_and_profit_bindings_are_context_only_not_violations():
    """V3G-8/V3G-10 自有结论字段与锚状态流无关，仅作上下文记录。"""
    r = audit_consumption_closure(cash_state())
    ctx = r["conclusion_token_bindings_elsewhere_context_only"]
    assert set(ctx) <= {"cash_profit_interface.py", "cash_solver_interface.py"}
    assert "profit_identifiable" in ctx.get("cash_profit_interface.py", [])
    for f in ctx:
        assert f not in r["conclusion_token_leaks"]
        assert f not in r["production_conclusion_token_leaks"]
    assert r["n_violations"] == 0


def test_solver_frozen_facts_unchanged_by_this_audit():
    s = solver_status(cash_state())
    assert s["solver_status"] == "BLOCKED"
    assert s["equilibrium"] is False
    assert s["gto_exact"] is False
    assert s["available"] is False
    audit_consumption_closure(cash_state())
    assert solver_status(cash_state())["solver_status"] == "BLOCKED"


# ── 3. 检测能力：合成违规必须被定位 ─────────────────────────────────
def test_synthetic_production_reference_is_detected(monkeypatch):
    import poker_decision_ai.cash_strategy_static_closure_audit as mod
    real = mod._scan_package()
    doctored = {
        "records": dict(real["records"]),
        "unparseable": real["unparseable"],
        "n_files": real["n_files"],
    }
    doctored["records"]["realtime"] = {
        "file": "realtime.py",
        "literals": {ANCHOR_STATES[0]},
        "imports": set(),
        "source": "",
        "tree": ast.parse("x = 1"),
    }
    monkeypatch.setattr(mod, "_scan_package", lambda: doctored)
    r = mod.audit_consumption_closure(cash_state())
    assert r["no_violation_found"] is False
    assert r["audit_result"] == "VIOLATION_FOUND"
    assert r["n_sinks_reachable"] == 6
    assert "realtime.py" in r["production_chain_references"]
    assert "anchor_state_literal" in \
        r["production_chain_references"]["realtime.py"]
    assert any(v.startswith("production_chain_reference:") for v in
               r["violations"])


def test_synthetic_conclusion_token_consumption_is_detected(monkeypatch):
    import poker_decision_ai.cash_strategy_static_closure_audit as mod
    real = mod._scan_package()
    doctored = {
        "records": dict(real["records"]),
        "unparseable": real["unparseable"],
        "n_files": real["n_files"],
    }
    bad_src = "OUT = {'equilibrium': compute_value()}\n"
    doctored["records"]["cash_strategy_consumption_gate"] = {
        "file": "cash_strategy_consumption_gate.py",
        "literals": {ANCHOR_STATES[0]},
        "imports": set(),
        "source": bad_src,
        "tree": ast.parse(bad_src),
    }
    monkeypatch.setattr(mod, "_scan_package", lambda: doctored)
    r = mod.audit_consumption_closure(cash_state())
    assert r["no_violation_found"] is False
    assert any(t.startswith("conclusion_token_consumed:")
               for t in r["violations"])


def test_negative_declarations_are_not_counted_as_consumption():
    """`False` / `None` 形式的否定声明不构成"消费"。"""
    tree = ast.parse("{'equilibrium': False, 'gto_exact': None}\n")
    usage = mod_usage(tree)
    assert usage == {}


def mod_usage(tree):
    import poker_decision_ai.cash_strategy_static_closure_audit as mod
    return {t: u for t, u in
            mod._conclusion_token_usage(tree, CONCLUSION_TOKENS).items()
            if u["bindings"] or u["conditional"] or u["called"]}


def test_positive_binding_is_flagged():
    tree = ast.parse("{'equilibrium': compute()}\n")
    usage = mod_usage(tree)
    assert "equilibrium" in usage
    assert usage["equilibrium"]["bindings"]


@pytest.mark.parametrize("src", [
    "{'exploitability': 0.5}\n",
    "{'gto_exact': True}\n",
    "x = {} if equilibrium else {}\n",
    "y = f(equilibrium)\n",
])
def test_various_consumption_shapes_are_flagged(src):
    assert mod_usage(ast.parse(src)) != {}


def test_negative_declaration_values_are_only_false_and_none():
    assert NEGATIVE_DECLARATION_VALUES == (False, None)


def test_nested_conclusion_slots_are_inspected():
    """嵌套取值字典里的"结论槽位"同样必须被检查。"""
    safe = ast.parse("OUT = {'u': {'exploitability': None, 'gto': None}}\n")
    assert mod_usage(safe) == {}
    unsafe = ast.parse("OUT = {'u': {'exploitability': 0.42}}\n")
    usage = mod_usage(unsafe)
    assert "exploitability" in usage
    assert usage["exploitability"]["bindings"]


def test_nested_dict_binding_is_detected_end_to_end(monkeypatch):
    import poker_decision_ai.cash_strategy_static_closure_audit as mod
    real = mod._scan_package()
    doctored = {"records": dict(real["records"]),
                "unparseable": real["unparseable"],
                "n_files": real["n_files"]}
    bad_src = "OUT = {'uncertainty': {'exploitability': 0.42}}\n"
    doctored["records"]["cash_strategy_consumption_gate"] = {
        "file": "cash_strategy_consumption_gate.py",
        "literals": {ANCHOR_STATES[0]},
        "imports": set(), "source": bad_src, "tree": ast.parse(bad_src)}
    monkeypatch.setattr(mod, "_scan_package", lambda: doctored)
    r = mod.audit_consumption_closure(cash_state())
    assert r["no_violation_found"] is False
    assert any("conclusion_token_consumed" in v for v in r["violations"])


def test_real_gate_nested_slots_are_negations_not_consumption():
    """真实 V3H-16 门禁的 `exploitability/gto/equilibrium: None` 是**否定声明**。"""
    r = audit_consumption_closure(cash_state())
    assert "cash_strategy_consumption_gate.py" not in r["conclusion_token_leaks"]
    assert r["n_violations"] == 0


# ── 4. 七级证据阶梯 ─────────────────────────────────────────────────
def test_ladder_has_seven_ordered_levels():
    c = evidence_ladder_contract()
    assert c["levels"] == list(EVIDENCE_LADDER)
    assert c["n_levels"] == 7
    assert EVIDENCE_LADDER == (
        "L1_READABLE", "L2_STRUCTURALLY_VALID", "L3_CONTENT_CONSISTENT",
        "L4_SOURCE_TRUSTED", "L5_AUTHENTICITY_VERIFIED",
        "L6_ACTUALLY_OCCURRED", "L7_CONSUMABLE")
    assert c["strictly_increasing"] is True
    assert c["lower_level_never_implies_higher_level"] is True


def test_internal_ceiling_is_L3_and_L4_plus_are_unreachable():
    c = evidence_ladder_contract()
    assert c["max_internally_reachable_level"] == "L3_CONTENT_CONSISTENT"
    assert MAX_INTERNALLY_REACHABLE_LEVEL == "L3_CONTENT_CONSISTENT"
    assert c["unreachable_internally_levels"] == list(
        UNREACHABLE_INTERNALLY_LEVELS)
    assert set(UNREACHABLE_INTERNALLY_LEVELS) == {
        "L4_SOURCE_TRUSTED", "L5_AUTHENTICITY_VERIFIED",
        "L6_ACTUALLY_OCCURRED", "L7_CONSUMABLE"}


def test_no_anchor_state_maps_to_L4_or_above():
    c = evidence_ladder_contract()
    mapping = c["mapping_from_v3h15_anchor_states"]
    assert set(mapping) == set(ANCHOR_STATES)
    assert c["no_state_maps_to_L4_or_above"] is True
    for lvl in mapping.values():
        assert not lvl.startswith(("L4", "L5", "L6", "L7")), lvl
    assert c["digest_or_signature_establishes_at_most"] == \
        "L3_CONTENT_CONSISTENT"


def test_forbidden_ladder_inferences_are_closed():
    assert len(FORBIDDEN_LADDER_INFERENCES) == 12
    for x in ("L1_implies_L7", "L2_implies_L5", "L3_implies_L6",
              "L3_implies_L7", "L5_implies_L6",
              "digest_or_signature_implies_L5_or_L6"):
        assert x in FORBIDDEN_LADDER_INFERENCES, x


def test_consumable_level_is_unreachable_in_audit_output():
    r = audit_consumption_closure(cash_state())
    assert r["consumable_reachable"] is False
    assert r["max_internally_reachable_level"] == "L3_CONTENT_CONSISTENT"
    assert r["claim_boundary"]["grants_verified_status"] is False
    assert r["claim_boundary"]["elevates_evidence_level"] is False


# ── 5. 失败类别策略 ────────────────────────────────────────────────
def test_seven_failure_classes_all_fail_closed():
    f = failure_class_policy()
    assert f["failure_classes"] == list(FAILURE_CLASSES)
    assert f["n_failure_classes"] == 7
    assert set(FAILURE_CLASSES) == {
        "UNKNOWN", "MISSING", "CONFLICT", "UNVERIFIED",
        "INTERNALLY_ISSUED", "DIGEST_ONLY", "UNTRUSTED_SOURCE"}
    assert f["all_fail_closed"] is True
    assert all(v == "fail_closed" for v in f["policy"].values())
    assert f["default_for_unrecognised"] == "fail_closed"
    assert f["grants_anything_on_failure"] is False
    for k in FAILURE_CLASSES:
        assert k in f["notes"], k


# ── 6. 覆盖范围与盲区（诚实边界）──────────────────────────────────
def test_blind_spots_are_declared_and_substantial():
    assert len(STATIC_AUDIT_BLIND_SPOTS) == 8
    joined = " ".join(STATIC_AUDIT_BLIND_SPOTS)
    for must in ("dynamic_import", "getattr", "exec_eval",
                 "outside_this_repository", "runtime_data_flow",
                 "non_python_artifacts", "monkeypatching",
                 "future_source_changes"):
        assert must in joined, must


def test_limitations_state_that_no_violation_is_not_a_guarantee():
    assert len(LIMITATIONS) == 4
    joined = " ".join(LIMITATIONS)
    assert "does NOT prove that misuse can never occur" in joined
    assert "cannot see dynamic imports" in joined
    assert "not a guarantee about future changes" in joined


def test_audit_never_claims_future_misuse_is_impossible():
    r = audit_consumption_closure(cash_state())
    assert r["future_misuse_guaranteed_impossible"] is False
    assert r["claim_boundary"]["is_future_misuse_guarantee"] is False
    assert r["claim_boundary"]["claims_no_violation_ever"] is False
    assert r["blind_spots"] == list(STATIC_AUDIT_BLIND_SPOTS)
    assert r["limitations"] == list(LIMITATIONS)


def test_coverage_statement_lists_covered_and_not_covered():
    c = audit_coverage_statement(cash_state())
    assert c["present"] is True
    assert c["conclusion_strength"] == "no_violation_found_within_audited_scope"
    assert c["conclusion_is_a_guarantee"] is False
    assert c["not_covered"] == list(STATIC_AUDIT_BLIND_SPOTS)
    assert c["limitations"] == list(LIMITATIONS)
    assert c["n_files_scanned"] > 0
    assert c["n_files_parsed"] > 0


def test_scope_audited_declares_analysis_kinds_and_no_runtime_check():
    r = audit_consumption_closure(cash_state())
    sc = r["scope_audited"]
    assert sc["runtime_analysis_performed"] is False
    assert set(sc["analysis_kinds"]) == {
        "static_import_graph", "ast_scan", "string_literal_flow",
        "contract_test"}
    assert sc["n_files_scanned"] >= sc["n_files_parsed"]
    assert sc["unparseable"] == []


def test_uncertainty_records_structural_blind_spots():
    r = audit_consumption_closure(cash_state())
    u = r["uncertainty"]
    assert u["measured"] is False
    assert u["coverage_is_partial"] is True
    assert "blind_spots" in u["note"]


# ── 7. 零结果 / 零执行 / 零统计 ────────────────────────────────────
def test_never_produces_results_or_executes():
    for r in (audit_consumption_closure(cash_state()),
              audit_coverage_statement(cash_state()),
              evidence_ladder_contract(), failure_class_policy()):
        assert r["executed"] is False
        assert r["result"] is None
        assert r.get("results_available", False) is False
        assert r.get("analysis_executed", False) is False


def test_claim_boundary_all_false_22_flags():
    assert len(CLAIM_BOUNDARY) == 22
    assert all(v is False for v in CLAIM_BOUNDARY.values())
    for k in ("is_future_misuse_guarantee", "claims_no_violation_ever",
              "elevates_evidence_level",
              "infers_high_level_from_low_level",
              "treats_structure_as_verification",
              "treats_digest_or_signature_as_occurrence",
              "grants_verified_status",
              "permits_protected_sink_consumption"):
        assert CLAIM_BOUNDARY[k] is False, k
    r = audit_consumption_closure(cash_state())
    assert all(v is False for v in r["claim_boundary"].values())


def test_flat_key_scan_has_no_result_or_statistic_field():
    r = audit_consumption_closure(cash_state())
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


def test_source_is_read_only_with_no_network_or_process_capability():
    src = MODULE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(src)
    reads = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and n.func.attr == "read_text"]
    writes = [n for n in ast.walk(tree)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
              and n.func.attr in ("write_text", "write_bytes", "unlink",
                                  "mkdir", "remove", "rename")]
    assert len(reads) == 1
    assert writes == []
    imported = sorted({a.name for n in ast.walk(tree)
                       if isinstance(n, ast.Import) for a in n.names} |
                      {n.module or "" for n in ast.walk(tree)
                       if isinstance(n, ast.ImportFrom)})
    assert imported == ["__future__", "ast", "pathlib", "typing"], imported
    for banned in ("socket", "requests", "urllib", "subprocess", "wallet",
                   "sqlite", "click", "pyautogui"):
        assert not any(banned in m for m in imported), banned


# ── 8. 定输入复现 / 不变异 / 可序列化 ──────────────────────────────
def test_deterministic_same_input_same_output():
    st = cash_state()
    outs = [audit_consumption_closure(st) for _ in range(3)]
    assert all(o == outs[0] for o in outs)
    outs2 = [audit_coverage_statement(st) for _ in range(3)]
    assert all(o == outs2[0] for o in outs2)
    outs3 = [evidence_ladder_contract() for _ in range(3)]
    assert all(o == outs3[0] for o in outs3)


def test_violation_lists_are_sorted():
    r = audit_consumption_closure(cash_state())
    assert r["violations"] == sorted(r["violations"])
    assert r["unexpected_anchor_state_referencing_modules"] == sorted(
        r["unexpected_anchor_state_referencing_modules"])
    assert r["unknown"] == sorted(r["unknown"])


def test_state_not_mutated():
    st = cash_state()
    before = json.dumps(st, sort_keys=True, ensure_ascii=False)
    audit_consumption_closure(st)
    audit_coverage_statement(st)
    assert json.dumps(st, sort_keys=True, ensure_ascii=False) == before


def test_outputs_json_serialisable():
    st = cash_state()
    for r in (audit_consumption_closure(st), audit_coverage_statement(st),
              evidence_ladder_contract(), failure_class_policy()):
        assert json.loads(json.dumps(r, ensure_ascii=False)) == r


# ── 9. AST · import 隔离 ─────────────────────────────────────────
def test_declares_version_constant():
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    consts = {n.targets[0].id for n in tree.body
              if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)}
    assert any(c.endswith("_VERSION") for c in consts)
    assert CASH_STRATEGY_STATIC_CLOSURE_AUDIT_VERSION.startswith("v3h17-")


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
        "cash_strategy_static_closure_audit",
        "cash_strategy_trust_anchor_contract",
    }
    assert required <= set(V3H_LAYERS), sorted(required - set(V3H_LAYERS))


# ── 11. frozen / sealed drift ────────────────────────────────────
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


# ── 12. 不重复 V3H-14/15/16 ──────────────────────────────────────
def test_does_not_duplicate_prior_layer_surfaces():
    """本层**不**定义或调用 V3H-14/15/16 的任何公开面（按 AST 结构判定）。"""
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
              "RESOLUTION_CLASSES", "READINESS_VERDICTS",
              "EXTERNAL_REQUIREMENTS", "consumption_gate_matrix",
              "evaluate_consumption_gate", "audit_contract_consistency",
              "trust_anchor_contract", "attestation_schema",
              "validate_attestation", "evaluate_anchor_state"}
    assert not (defined & banned), sorted(defined & banned)
    called: set[str] = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name):
            called.add(n.func.id)
        elif isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute):
            called.add(n.func.attr)
    assert not (called & banned), sorted(called & banned)


def test_anchor_states_are_discovered_not_redeclared():
    """本层**不**把六类锚状态定义为自己的常量。

    允许它们在 `evidence_ladder_contract()` 的**文档性映射**中出现，
    但**必须**存在由契约模块**发现**的路径，且不得被声明为模块级常量。
    """
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    top_level_names: set[str] = set()
    for n in tree.body:
        if isinstance(n, ast.Assign):
            for t in n.targets:
                if isinstance(t, ast.Name):
                    top_level_names.add(t.id)
    assert not any("ANCHOR" in x for x in top_level_names), top_level_names
    src = MODULE_PATH.read_text(encoding="utf-8")
    assert "ANCHOR_" in src                       # 仅以前缀判定
    assert "startswith(\"ANCHOR_\")" in src      # 由契约模块发现
    # 发现结果与契约模块一致（运行时验证）
    r = audit_consumption_closure(cash_state())
    assert sorted(r["anchor_states"]) == sorted(ANCHOR_STATES)
