"""V3I-P17 — 架构收口审计与冻结条件判定的验证（测试）。

本文件验证 V3I-P17 只读审计层 `cash_architecture_closure_audit`：
链路清单、CLAIM_BOUNDARY 审计（含「未声明」与「正向宣称」的区分）、
模块冻结状态、最终决策五类语义、最小冻结条件清单与冻结裁定。

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

from poker_decision_ai.cash_architecture_closure_audit import (
    CHAIN_STAGES,
    CLAIM_BOUNDARY,
    DISAGREEMENT_KINDS,
    EVIDENCE_GRADE,
    FINAL_DECISION_CLASSES,
    FREEZE_CHECKLIST,
    MODULE_STATUSES,
    NOT_PROVABLE_WITH_MODEL_INTERNAL,
    PRIMARY_FLOW,
    PROVABLE_WITH_MODEL_INTERNAL,
    UNFINISHED_MARKERS,
    audit_boundary_coverage,
    audit_chain_inventory,
    audit_claim_boundaries,
    build_freeze_checklist,
    classify_final_semantics,
    extract_claim_boundary,
    final_semantics_summary,
    freeze_decision,
    model_internal_provable,
    module_freeze_status,
    open_capability_gaps,
    p17_contract,
    stable_fingerprint,
)

MODULE = pathlib.Path("poker_decision_ai/cash_architecture_closure_audit.py")
EVIDENCE = pathlib.Path(".tmp/v3i_p17_evidence.json")

_BOUNDARY_OK = "CLAIM_BOUNDARY = {'is_gto': False, 'is_ev': False}\n"
_BOUNDARY_ANN = "CLAIM_BOUNDARY: dict = {'is_gto': False, 'is_ev': False}\n"
_BOUNDARY_BAD = "CLAIM_BOUNDARY = {'is_gto': False, 'is_ev': True}\n"


def _rec(**over) -> dict:
    base = {
        "run_id": "r1", "scenario": "flop_norm", "street": "flop",
        "mode": "B2_conditioned_search_feasible_assembly",
        "present": True, "reason": None, "abstained": False,
        "preferred": {"action": "bet", "amount_bb": 0.495,
                      "selected_by_tier": "search"},
        "disagreements": [], "n_scorers_alignable": 2,
        "tree_root_children": [{"action": "check", "amount_bb": 0.0},
                               {"action": "bet", "amount_bb": 0.495}],
        "stage_trace": [{"stage": "action_policy", "present": True}],
    }
    base.update(over)
    return base


# ---------------------------------------------------------------------------
# T1. 只读契约与冻结边界（要求 2、3、9）
# ---------------------------------------------------------------------------
def test_t1_contract_declares_no_implementation_and_no_escalation():
    c = p17_contract()
    assert c["implements_preflop_unfreeze"] is False
    assert c["modifies_search_tree_or_ranking"] is False
    assert c["modifies_search_budget"] is False
    assert c["relaxes_leaf_coverage_threshold"] is False
    assert c["enables_override"] is False
    assert c["uses_voting"] is False
    assert c["forces_ranking_search_unification"] is False
    assert c["connects_coinpoker_or_auto_betting"] is False
    assert c["enters_mtt_pko"] is False
    assert c["adds_strategy_mechanism"] is False
    assert c["reads_only"] is True
    assert tuple(c["primary_flow"]) == PRIMARY_FLOW
    assert tuple(c["freeze_checklist"]) == FREEZE_CHECKLIST
    assert tuple(c["final_decision_classes"]) == FINAL_DECISION_CLASSES


def test_t1_claim_boundary_all_false():
    for k, v in CLAIM_BOUNDARY.items():
        assert v is False, k
    for flag in ("is_gto", "is_equilibrium", "is_ev", "is_profit_evidence",
                 "wired_to_production", "implements_any_unfreeze_item",
                 "modifies_frozen_contract", "enables_override", "uses_voting",
                 "forces_ranking_search_unification"):
        assert flag in CLAIM_BOUNDARY
    assert EVIDENCE_GRADE == "MODEL_INTERNAL"


def test_t1_module_imports_only_stdlib_and_no_banned_calls():
    tree = ast.parse(MODULE.read_text(encoding="utf-8"))
    mods = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            mods.update(a.name for a in n.names)
        elif isinstance(n, ast.ImportFrom):
            mods.add(n.module or "")
    assert mods <= {"__future__", "typing", "json", "hashlib", "ast"}, mods
    assert not any(m.startswith("poker_decision_ai") for m in mods)
    banned = {"setattr", "delattr", "globals", "locals", "eval", "exec",
              "__import__", "compile", "open", "input"}
    called = {n.func.id for n in ast.walk(tree)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert not (called & banned), called & banned
    assert "random" not in {a.name for n in ast.walk(tree)
                            if isinstance(n, ast.Import) for a in n.names}


#: 受保护层 pin 的**历史链**（只追加、不删除；末位 = 当前封存 v4 值）。
#: 来源：docs/CASH_CORE_FREEZE_MANIFEST.json 的 supersede 链
#: （hash_of_hashes v1 87c22ce79351ee1e → v2 632835d6d0be2e2f →
#: v3 575761af940431ed → v4 be0a7ba3d39924d3）。经授权执行
#: SCORE-HYGIENE-1-STALE-PIN-REBASELINE 与 V3I-G1-PHASE1 的 pin 迁移：
#: **历史值保留**，其余 pin 逐位不变。
PROTECTED_LAYER_PIN_HISTORY = {
    "poker_decision_ai/cash_strategy_search_tree.py": (
        "5ea3854c2bf65631",   # 曾被钉住的历史 pin（V3H-7）
        "f4c4b3ec1c520a80",   # v3 封存值（V3I-RC2-G7-1 街忠实增量）
        "5f5b29be063a6f5a",   # **当前**封存 v5（Phase 2 未改 tree；v4 值）
    ),
    "poker_decision_ai/cash_strategy_search_eval.py": (
        "513ba523781c2c07",   # 曾被钉住的历史 pin（V3H-7）
        "73cebc0314e4a268",   # 中间值（V3I-P8；本文件当时未同步，从未钉住）
        "62f2e01565953368",   # v3 封存值（V3I-RC2-SCORE-HYGIENE-1）
        "2db1bd6995461428",   # v4 封存值（G1 Phase 1 动作价值）
        "e97ccea19d2ffd92",   # **当前**封存 v5（G1 Phase 2 统一动作价值）
    ),
    "poker_decision_ai/cash_strategy_action_ranking.py": (
        "2c1830e77157baaa",   # 曾被钉住的历史 pin（V3H-5..Phase 1）
        "53dc7aa44245eb84",   # **当前**封存 v5（G1 Phase 2 统一动作价值）
    ),
}


def test_t1_protected_layers_untouched():
    frozen = {
        # 历史 pin（V3H-7）：search_tree 5ea3854c2bf65631 / search_eval
        # 513ba523781c2c07 —— 历史值保留于 PROTECTED_LAYER_PIN_HISTORY，未删除
        "poker_decision_ai/cash_strategy_search_tree.py": "5f5b29be063a6f5a",
        "poker_decision_ai/cash_strategy_search_eval.py": "e97ccea19d2ffd92",
        "poker_decision_ai/cash_strategy_action_ranking.py": "53dc7aa44245eb84",
        "poker_decision_ai/cash_strategy_decision_assembly.py":
            "3732314ab0182063",
        "poker_decision_ai/cash_feasible_domain_wiring.py": "42a90a9bdff15a2b",
        "poker_decision_ai/cash_decision_loop.py": "7c4b0a11c1e73a7f",
        "poker_decision_ai/domain/poker/strategy_core.py": "378a68671ff09e3f",
        "poker_decision_ai/agents/judge/agent.py": "239436fae7299c47",
        "poker_decision_ai/realtime.py": "3400a20ff857880d",
    }
    # 历史保留：历史链只追加、不得覆盖/重复；当前 pin 必须等于链末位
    for path, hist in PROTECTED_LAYER_PIN_HISTORY.items():
        assert len(set(hist)) == len(hist), (path, "历史链不得重复/覆盖")
        assert frozen[path] == hist[-1], (path, frozen[path], hist[-1])
    for path, want in frozen.items():
        got = hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()[:16]
        assert got == want, (path, got, want)


def test_t1_threshold_and_budget_intact():
    from poker_decision_ai.cash_strategy_decision_assembly import (
        ASSEMBLY_POLICY)
    from poker_decision_ai.cash_strategy_search_tree import DEFAULT_BUDGET
    assert float(ASSEMBLY_POLICY["min_leaf_coverage"]) == 0.8
    assert ASSEMBLY_POLICY["override_default"] is False
    assert ASSEMBLY_POLICY["on_disagreement"] == "abstain_unknown"
    assert DEFAULT_BUDGET == {"max_street_depth": 2, "max_response_depth": 1,
                              "max_candidate_actions": 6, "max_leaves": 200,
                              "max_runout_children": 8}


def test_t1_no_production_wiring_added():
    def imports_of(p):
        tree = ast.parse(pathlib.Path(p).read_text(encoding="utf-8"))
        mods = set()
        for n in ast.walk(tree):
            if isinstance(n, ast.ImportFrom):
                if n.module:
                    mods.add(n.module.split(".")[-1])
                for a in n.names:
                    mods.add(a.name.split(".")[-1])
            elif isinstance(n, ast.Import):
                for a in n.names:
                    mods.add(a.name.split(".")[-1])
        return mods

    found = {}
    for f in ("poker_decision_ai/realtime.py",
              "poker_decision_ai/agents/judge/agent.py"):
        for m in sorted(imports_of(f)):
            if m.startswith("cash_"):
                found.setdefault(m, []).append(pathlib.Path(f).name)
    assert {k: sorted(set(v)) for k, v in found.items()} == {
        "cash_integrity_context": ["agent.py", "realtime.py"],
        "cash_unknown_e2e": ["realtime.py"],
    }, found
    assert "cash_architecture_closure_audit" not in found


# ---------------------------------------------------------------------------
# T2. 链路清单（要求 4）
# ---------------------------------------------------------------------------
def _mods(complete=True):
    return {m: {"sha256_prefix": "x" * 16, "n_lines": 10}
            for _, m in CHAIN_STAGES} if complete else {
        m: {"sha256_prefix": "x" * 16, "n_lines": 10}
        for _, m in CHAIN_STAGES[:3]}


def test_t2_inventory_complete_and_hash_registered():
    inv = audit_chain_inventory(modules=_mods(),
                                module_sources={})
    assert inv["present"] is True
    assert inv["chain_complete"] is True
    assert inv["all_hashes_registered"] is True
    assert inv["no_unfinished_markers"] is True
    assert inv["missing_modules"] == []
    assert inv["n_modules"] == len(CHAIN_STAGES)
    assert inv["n_stages"] == len({s for s, _ in CHAIN_STAGES})
    assert tuple(inv["primary_flow"]) == PRIMARY_FLOW


def test_t2_inventory_detects_missing_module_and_markers():
    m = _mods()
    m.pop("cash_strategy_search_tree")
    inv = audit_chain_inventory(modules=m)
    assert inv["chain_complete"] is False
    assert "cash_strategy_search_tree" in inv["missing_modules"]
    inv2 = audit_chain_inventory(
        modules=_mods(),
        module_sources={"cash_multistreet": "x = 1  # TODO fix"})
    assert inv2["no_unfinished_markers"] is False
    assert "cash_multistreet" in inv2["modules_with_unfinished_markers"]
    assert set(UNFINISHED_MARKERS) >= {"TODO", "FIXME", "NotImplementedError"}


def test_t2_inventory_fail_closed():
    assert audit_chain_inventory(modules=None).get("present") is False
    assert audit_chain_inventory(modules={}).get("present") is False


# ---------------------------------------------------------------------------
# T3. CLAIM_BOUNDARY 审计（要求 5）
# ---------------------------------------------------------------------------
def test_t3_extract_supports_assign_and_annassign():
    a = extract_claim_boundary(source=_BOUNDARY_OK)
    assert a["present"] is True and a["all_false"] is True
    assert a["n_flags"] == 2
    b = extract_claim_boundary(source=_BOUNDARY_ANN)
    assert b["present"] is True and b["all_false"] is True
    assert b["n_flags"] == 2


def test_t3_extract_detects_positive_claim():
    c = extract_claim_boundary(source=_BOUNDARY_BAD)
    assert c["present"] is True
    assert c["all_false"] is False
    assert c["non_false_flags"] == ["is_ev"]


def test_t3_extract_fail_closed():
    assert extract_claim_boundary(source="x = 1\n").get("present") is False
    assert extract_claim_boundary(source="").get("present") is False
    assert extract_claim_boundary(source=None).get("present") is False
    assert extract_claim_boundary(source="CLAIM_BOUNDARY = not_a_literal").get(
        "present") is False


def test_t3_batch_separates_undeclared_from_positive_claim():
    r = audit_claim_boundaries(module_sources={
        "a": _BOUNDARY_OK, "b": _BOUNDARY_BAD, "c": "x = 1\n"})
    assert r["n_modules"] == 3
    assert r["n_declared"] == 2
    assert r["n_undeclared"] == 1
    assert r["undeclared_modules"] == ["c"]
    assert r["modules_with_positive_claim"] == ["b"]
    assert r["all_declared_are_all_false"] is False
    assert r["every_module_declares_boundary"] is False
    assert audit_claim_boundaries(module_sources={}).get("present") is False


def test_t3_boundary_coverage_splits_documentation_gap():
    r = audit_boundary_coverage(
        module_sources={"a": _BOUNDARY_OK, "c": "x = 1\n"},
        decision_surface_modules=["a"])
    assert r["declared_all_false"] is True
    assert r["decision_surface_declared"] is True
    assert r["every_module_declares"] is False
    assert r["undeclared_modules"] == ["c"]
    g = r["documentation_gap"]
    assert g["kind"] == "in_source_boundary_documentation"
    assert g["n"] == 1
    assert "不产出决策" in g["why_not_a_correctness_gap"]
    assert "只登记、不修改" in g["why_it_cannot_be_closed_inside_the_freeze"]
    # 决策面未声明 ⇒ 不算通过
    r2 = audit_boundary_coverage(module_sources={"a": _BOUNDARY_OK, "c": "x=1"},
                                 decision_surface_modules=["c"])
    assert r2["decision_surface_declared"] is False
    assert audit_boundary_coverage(module_sources={}).get("present") is False


# ---------------------------------------------------------------------------
# T4. 模块冻结状态（要求 5）
# ---------------------------------------------------------------------------
def test_t4_module_status_counts_and_blocked():
    inv = audit_chain_inventory(modules=_mods())
    st = module_freeze_status(
        inventory=inv,
        verification_layers={"cash_strategy_search_tree": "V3H-7"},
        externally_blocked=["cash_solver_interface"])
    assert st["per_module"]["cash_strategy_search_tree"] == \
        "ENGINEERING_FROZEN"
    assert st["per_module"]["cash_solver_interface"] == "EXTERNALLY_BLOCKED"
    assert st["per_module"]["cash_multistreet"] == "CAPABILITY_GAP_OPEN"
    assert sum(st["counts"].values()) == len(CHAIN_STAGES)
    assert set(st["counts"]) == set(MODULE_STATUSES)
    assert "工程可用但能力未经外部证据验证" in st["statement"]
    assert module_freeze_status(inventory=None).get("present") is False


# ---------------------------------------------------------------------------
# T5. 最终决策五类语义（要求 7）
# ---------------------------------------------------------------------------
def test_t5_full_vs_ranking_only_accept_are_separated():
    full = classify_final_semantics(record=_rec(n_scorers_alignable=2))
    assert full["outcome_class"] == "FULL_MULTISOURCE_ACCEPT"
    assert full["is_full_multisource"] is True
    assert full["is_ranking_only"] is False
    ro = classify_final_semantics(record=_rec(
        n_scorers_alignable=1,
        preferred={"action": "bet", "amount_bb": 0.495,
                   "selected_by_tier": "ranking"}))
    assert ro["outcome_class"] == "RANKING_ONLY_ACCEPT"
    assert ro["is_ranking_only"] is True
    assert ro["is_full_multisource"] is False
    assert "**不是**" in ro["must_not_be_conflated"]


def test_t5_disagreement_subkinds_need_both_keys_inside():
    same = classify_final_semantics(record=_rec(
        abstained=True, preferred=None, disagreements=[
            {"tier_a": "search", "key_a": ["check", 0.0],
             "tier_b": "ranking", "key_b": ["bet", 0.495]}]))
    assert same["outcome_class"] == "TRUE_DISAGREEMENT_ABSTAIN"
    assert same["disagreement_kind"] == "same_domain_disagreement"
    structural = classify_final_semantics(record=_rec(
        abstained=True, preferred=None, disagreements=[
            {"tier_a": "search", "key_a": ["bet", 0.495],
             "tier_b": "ranking", "key_b": ["bet", 100.0]}]))
    assert structural["disagreement_kind"] == "structural_domain_mismatch"
    assert set(DISAGREEMENT_KINDS) == {"same_domain_disagreement",
                                       "structural_domain_mismatch"}


def test_t5_river_blocked_and_evidence_insufficient():
    river = classify_final_semantics(record=_rec(
        street="river", present=False, reason="stage_failed:action_policy",
        abstained=False, preferred=None,
        stage_trace=[{"stage": "action_policy", "present": False}]))
    assert river["outcome_class"] == "RIVER_BLOCKED"
    nc = classify_final_semantics(record=_rec(
        present=False, reason="not_cash_mode_fail_closed",
        abstained=False, preferred=None, stage_trace=[]))
    assert nc["outcome_class"] == "EVIDENCE_INSUFFICIENT"
    no_pref = classify_final_semantics(record=_rec(
        abstained=False, preferred=None))
    assert no_pref["outcome_class"] == "EVIDENCE_INSUFFICIENT"
    assert classify_final_semantics(record=None).get("present") is False


def test_t5_summary_counts_and_shares():
    recs = [
        _rec(run_id="a", n_scorers_alignable=2),
        _rec(run_id="b", n_scorers_alignable=1),
        _rec(run_id="c", abstained=True, preferred=None, disagreements=[
            {"tier_a": "search", "key_a": ["check", 0.0],
             "tier_b": "ranking", "key_b": ["bet", 0.495]}]),
        _rec(run_id="d", street="river", present=False,
             reason="stage_failed:action_policy", abstained=False,
             preferred=None,
             stage_trace=[{"stage": "action_policy", "present": False}]),
    ]
    s = final_semantics_summary(records=recs)
    assert s["n_runs"] == 4
    assert s["class_counts"]["FULL_MULTISOURCE_ACCEPT"] == 1
    assert s["class_counts"]["RANKING_ONLY_ACCEPT"] == 1
    assert s["class_counts"]["TRUE_DISAGREEMENT_ABSTAIN"] == 1
    assert s["class_counts"]["RIVER_BLOCKED"] == 1
    assert s["accept_total"] == 2
    assert s["full_multisource_share_of_accepts"] == 0.5
    assert s["disagreement_kind_counts"] == {"same_domain_disagreement": 1}
    assert final_semantics_summary(records=[]).get("present") is False


# ---------------------------------------------------------------------------
# T6. 冻结条件清单与裁定（要求 6、11）
# ---------------------------------------------------------------------------
def _ev(**over):
    base = {
        "chain_complete": True, "n_required_stages": 8,
        "n_required_stages_ok": True, "full_chain_run_ok": True,
        "deterministic": True, "n_fingerprints_compared": 2,
        "fail_closed_ok": True, "fail_closed_detail": {"x": 1},
        "claim_boundaries_all_false": True,
        "decision_surface_declared": True,
        "decision_surface_modules": ["cash_decision_loop"],
        "n_claim_boundary_modules": 22, "n_boundary_declared": 15,
        "n_boundary_undeclared": 7, "boundary_undeclared_modules": ["m"],
        "claim_boundary_offenders": [],
        "documented_gaps": [{"kind": "doc"}],
        "evidence_level_internal_only": True,
        "allowed_evidence_levels": ["MODEL_INTERNAL"],
        "rejected_evidence_levels": ["EXTERNAL_VALIDATION"],
        "no_external_validation_claims": True,
        "override_never_used": True, "leaf_coverage_threshold": 0.8,
        "budget_unchanged": True, "budget": {},
        "production_graph_unchanged": True, "production_graph": {},
        "unwired_layers_still_unwired": True, "unwired_layers": [],
        "semantics_classes_reachable": True, "classes_with_instances": [],
        "evidence_insufficient_reachable": True,
        "evidence_insufficient_detail": {}, "capability_gaps": ["G1"],
        "gaps_are_explicit": True, "no_unfinished_markers": True,
        "modules_with_unfinished_markers": {},
    }
    base.update(over)
    return base


def test_t6_checklist_all_pass_gives_candidate():
    chk = build_freeze_checklist(evidence=_ev())
    assert chk["n_conditions"] == len(FREEZE_CHECKLIST)
    assert chk["all_passed"] is True
    assert chk["failed"] == []
    assert chk["n_documented_gaps"] == 1
    assert chk["documented_gaps_are_non_blocking"] is True
    d = freeze_decision(checklist=chk, gaps={"open_gaps": [{"code": "G1"}]})
    assert d["decision"] == "CASH_CORE_FREEZE_CANDIDATE"
    assert d["n_open_capability_gaps"] == 1
    assert d["n_documented_non_blocking_gaps"] == 1
    assert "建议**冻结**" in d["recommendation"]


def test_t6_each_condition_can_fail_independently():
    cases = {
        "F1_chain_completeness": {"chain_complete": False},
        "F2_determinism": {"deterministic": False},
        "F3_fail_closed": {"fail_closed_ok": False},
        "F4_claim_boundary_declared_no_positive_claims": {
            "claim_boundaries_all_false": False},
        "F5_evidence_grade_internal_only": {
            "evidence_level_internal_only": False},
        "F6_no_privilege_escalation": {"override_never_used": False},
        "F7_production_isolation": {"production_graph_unchanged": False},
        "F8_decision_semantics_closed": {
            "evidence_insufficient_reachable": False},
        "F9_capability_gaps_declared": {"capability_gaps": []},
        "F10_no_unfinished_markers": {"no_unfinished_markers": False},
    }
    assert set(cases) == set(FREEZE_CHECKLIST)
    for cid, over in cases.items():
        chk = build_freeze_checklist(evidence=_ev(**over))
        assert chk["all_passed"] is False, cid
        assert cid in chk["failed"], (cid, chk["failed"])
    # 决策面未声明也要让 F4 失败
    chk = build_freeze_checklist(evidence=_ev(decision_surface_declared=False))
    assert "F4_claim_boundary_declared_no_positive_claims" in chk["failed"]


def test_t6_blocker_blocks_freeze():
    chk = build_freeze_checklist(evidence=_ev())
    d = freeze_decision(checklist=chk, blockers=["some real blocker"])
    assert d["decision"] == "NOT_READY"
    assert d["blockers"] == ["some real blocker"]
    assert "不建议" in d["recommendation"]
    assert freeze_decision(checklist=None).get("present") is False


def test_t6_freeze_meaning_is_explicit():
    chk = build_freeze_checklist(evidence=_ev())
    d = freeze_decision(checklist=chk)
    assert "停止新增策略机制" in d["what_freezing_means"]
    assert "**不**表示决策质量已被证明" in d["what_freezing_does_not_mean"]


# ---------------------------------------------------------------------------
# T7. MODEL_INTERNAL 边界与能力缺口（要求 8）
# ---------------------------------------------------------------------------
def test_t7_model_internal_provable_and_not():
    m = model_internal_provable()
    assert m["evidence_grade"] == "MODEL_INTERNAL"
    assert any("覆盖率" in s for s in m["provable"])
    assert any("决定论" in s or "决定" in s for s in m["provable"])
    assert any("更优" in s for s in m["not_provable"])
    assert any("盈利" in s for s in m["not_provable"])
    assert "EV" in m["must_not_be_read_as"]
    assert "GTO" in m["must_not_be_read_as"]
    assert tuple(m["provable"]) == PROVABLE_WITH_MODEL_INTERNAL
    assert tuple(m["not_provable"]) == NOT_PROVABLE_WITH_MODEL_INTERNAL


def test_t7_open_gaps_are_not_blocking():
    g = open_capability_gaps(sources={"G1": "detail", "G2": "detail"})
    assert g["n_open_gaps"] == 2
    assert [x["code"] for x in g["open_gaps"]] == ["G1", "G2"]
    assert "**不阻塞工程冻结**" in g["note"]
    assert open_capability_gaps(sources={})["n_open_gaps"] == 0


# ---------------------------------------------------------------------------
# T8. 实测守护（要求 6、10、11）
# ---------------------------------------------------------------------------
pytestmark_ev = pytest.mark.skipif(
    not EVIDENCE.exists(), reason="需要 .tmp/v3i_p17_evidence.json")


@pytestmark_ev
def test_t8_freeze_checklist_all_passed_measured():
    d = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    chk = d["freeze_checklist"]
    assert chk["n_conditions"] == 10
    assert chk["all_passed"] is True
    assert chk["failed"] == []
    assert [i["id"] for i in chk["items"]] == list(FREEZE_CHECKLIST)
    assert all(i["passed"] for i in chk["items"])
    assert d["freeze_decision"]["decision"] == "CASH_CORE_FREEZE_CANDIDATE"
    assert d["freeze_decision"]["conditions_passed"] == 10
    assert d["freeze_decision"]["blockers"] == []


@pytestmark_ev
def test_t8_chain_inventory_and_markers_measured():
    d = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    inv = d["chain_inventory"]
    assert inv["chain_complete"] is True
    assert inv["missing_modules"] == []
    assert inv["all_hashes_registered"] is True
    assert inv["no_unfinished_markers"] is True
    assert inv["modules_with_unfinished_markers"] == {}
    assert inv["n_modules"] == len(CHAIN_STAGES)
    assert len(d["modules"]) == len(CHAIN_STAGES)


@pytestmark_ev
def test_t8_boundary_documentation_gap_is_recorded_not_hidden():
    d = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    b = d["boundary_coverage"]
    assert b["declared_all_false"] is True
    assert b["modules_with_positive_claim"] == []
    assert b["decision_surface_declared"] is True
    assert b["every_module_declares"] is False
    assert b["n_undeclared"] == 7
    assert set(b["undeclared_modules"]) == {
        "cash_state_integrity", "cash_integrity_context", "cash_multistreet",
        "cash_opponent_model", "cash_range_model", "cash_action_tree",
        "cash_solver_interface"}
    g = b["documentation_gap"]
    assert g["n"] == 7
    assert "只登记、不修改" in g["why_it_cannot_be_closed_inside_the_freeze"]
    # 该缺口不得被算成 blocker
    assert d["freeze_decision"]["n_documented_non_blocking_gaps"] == 1
    assert d["freeze_decision"]["conditions_failed"] == []


@pytestmark_ev
def test_t8_final_semantics_measured_matches_p14_p15():
    d = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    sem = d["final_semantics"]["factorial_grid_only"]
    assert sem["n_runs"] == 108
    c = sem["class_counts"]
    assert c["FULL_MULTISOURCE_ACCEPT"] == 8
    assert c["RANKING_ONLY_ACCEPT"] == 24
    assert c["TRUE_DISAGREEMENT_ABSTAIN"] == 64
    assert c["RIVER_BLOCKED"] == 12
    assert c["FULL_MULTISOURCE_ACCEPT"] + c["RANKING_ONLY_ACCEPT"] == 32
    assert sem["disagreement_kind_counts"] == {
        "structural_domain_mismatch": 36, "same_domain_disagreement": 28}
    assert sem["full_multisource_share_of_accepts"] == 0.25
    b2 = d["final_semantics"]["b2_mode"]["class_counts"]
    assert b2["FULL_MULTISOURCE_ACCEPT"] == 2
    assert b2["RANKING_ONLY_ACCEPT"] == 10
    assert b2["TRUE_DISAGREEMENT_ABSTAIN"] == 28
    assert b2["RIVER_BLOCKED"] == 5


@pytestmark_ev
def test_t8_determinism_and_fail_closed_measured():
    d = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    det = d["determinism"]
    assert det["deterministic"] is True
    assert len(set(det["fingerprints"])) == 1
    probes = d["fail_closed_probes"]
    assert probes["non_cash"]["present"] is False
    assert probes["non_cash"]["reason"] == "not_cash_mode_fail_closed"
    assert probes["forbidden_evidence"]["present"] is False
    assert probes["missing_stage"]["present"] is False
    assert "pipeline_missing_stages" in probes["missing_stage"]["reason"]
    assert probes["reference_full_pipeline"]["present"] is True
    assert d["model_internal"]["must_not_be_read_as"]


@pytestmark_ev
def test_t8_module_status_and_gaps_measured():
    d = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    st = d["module_status"]
    assert st["counts"]["EXTERNALLY_BLOCKED"] == 1
    assert st["per_module"]["cash_solver_interface"] == "EXTERNALLY_BLOCKED"
    assert sum(st["counts"].values()) == len(CHAIN_STAGES)
    assert d["capability_gaps"]["n_open_gaps"] >= 7
    codes = [g["code"] for g in d["capability_gaps"]["open_gaps"]]
    for need in ("G5_decision_quality_not_provable_model_internal",
                 "G6_river_terminal_never_reaches_assembly",
                 "G7_preflop_no_real_equity_in_frozen_expansion",
                 "G8_postflop_no_legitimate_resolution"):
        assert need in codes, (need, codes)
    assert d["p17_contract"]["implements_preflop_unfreeze"] is False
    assert d["p17_contract"]["relaxes_leaf_coverage_threshold"] is False
    assert d["fingerprint"] == stable_fingerprint(
        {"inv": d["chain_inventory"], "cb": d["claim_boundary_audit"],
         "chk": d["freeze_checklist"],
         "sem": d["final_semantics"]["factorial_grid_only"]})
