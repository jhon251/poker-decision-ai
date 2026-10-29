"""V3I-P18 — 冻结后能力验证计划与可执行性审计的验证（测试）。

本文件验证 V3I-P18 只读规划层 `cash_capability_verification_plan`：
逐缺口审计（四问）、闭合路径分类、外部依赖图与「不再重复寻找」登记、
最小能力验收矩阵、以及**禁止用内部证据替代外部能力证据**的守卫。

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

from poker_decision_ai.cash_capability_verification_plan import (
    ACCEPTANCE_DIMENSIONS,
    CLAIM_BOUNDARY,
    CLOSURE_CATEGORIES,
    EVIDENCE_CLASSES,
    EXHAUSTED_PUBLIC_SOURCES,
    EXTERNAL_REQUIRED_DIMENSIONS,
    EXTERNAL_SOURCES,
    FORBIDDEN_DIMENSIONS,
    GAP_IDS,
    INTERNAL_SUFFICIENT_DIMENSIONS,
    acceptance_matrix,
    audit_gap,
    build_gap_verification_plan,
    external_dependency_graph,
    p18_contract,
    prioritise,
    stable_fingerprint,
    validate_claim,
)

MODULE = pathlib.Path("poker_decision_ai/cash_capability_verification_plan.py")
EVIDENCE = pathlib.Path(".tmp/v3i_p18_plan.json")


def _gap(**over) -> dict:
    base = {
        "gap_id": "G5_decision_quality_not_provable_model_internal",
        "title": "t",
        "closure_category": "REQUIRES_EXTERNAL_EVIDENCE",
        "external_evidence_needed": "RC-2 / Solver / 真人之一",
        "current_evidence": "P17",
        "what_is_missing": "三个外部源都不可得",
        "minimal_closure_experiment": "当前不可执行",
        "closable_without_solver": False,
        "closable_without_rc2": True,
        "closable_without_human_validation": True,
        "closable_without_any_external": False,
        "long_term_open": True,
        "is_a_defect": False,
        "requires_unfreeze": False,
        "blocked_by_freeze": False,
        "do_not_search_again": ["RC2_real_opponent_corpus"],
    }
    base.update(over)
    return base


# ---------------------------------------------------------------------------
# T1. 冻结基线与只读契约（要求 1、2、8、9）
# ---------------------------------------------------------------------------
def test_t1_contract_declares_frozen_baseline_and_no_mechanism():
    c = p18_contract()
    assert c["frozen_baseline"] is True
    assert c["adds_strategy_mechanism"] is False
    assert c["modifies_frozen_contract"] is False
    assert c["implements_any_unfreeze_item"] is False
    assert c["modifies_chain_state_belief_population_conditioning_search_"
             "ranking_assembly"] is False
    assert c["modifies_p13_feasible_domain"] is False
    assert c["relaxes_leaf_coverage_threshold"] is False
    assert c["modifies_budget_or_strategy_core_or_cfi_or_river_terminal"] is False
    assert c["substitutes_internal_for_external_evidence"] is False
    assert c["predicts_strength"] is False
    assert c["fabricates_results"] is False
    assert c["connects_coinpoker_or_auto_betting"] is False
    assert c["enters_mtt_pko"] is False
    assert c["reads_only"] is True
    assert tuple(c["gap_ids"]) == GAP_IDS
    assert tuple(c["closure_categories"]) == CLOSURE_CATEGORIES
    assert tuple(c["acceptance_dimensions"]) == ACCEPTANCE_DIMENSIONS


def test_t1_claim_boundary_all_false():
    for k, v in CLAIM_BOUNDARY.items():
        assert v is False, k
    for flag in ("is_gto", "is_equilibrium", "is_ev", "is_profit_evidence",
                 "is_external_validation", "modifies_frozen_contract",
                 "substitutes_internal_for_external_evidence",
                 "predicts_strength", "connects_coinpoker_or_auto_betting",
                 "enters_mtt_pko"):
        assert flag in CLAIM_BOUNDARY


def test_t1_module_imports_only_stdlib_and_no_banned_calls():
    tree = ast.parse(MODULE.read_text(encoding="utf-8"))
    mods = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            mods.update(a.name for a in n.names)
        elif isinstance(n, ast.ImportFrom):
            mods.add(n.module or "")
    assert mods <= {"__future__", "typing", "json", "hashlib"}, mods
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
    "poker_decision_ai/cash_conditioned_search_wiring.py": (
        "bcf564ca3e99fff5",   # 曾被钉住的历史 pin（P7/P8 时代）
        "0f8a0bd1118c009a",   # v4 封存值（G1 Phase 1 测量量物化）
        "740b9edff66a1982",   # **当前**封存 v5（G1 Phase 2 统一路径编排）
    ),
}


def test_t1_frozen_chain_modules_untouched():
    """要求 2：冻结基线逐哈希未变。"""
    frozen = {
        "poker_decision_ai/cash_state_integrity.py": "f8dca76a2fa3fe9c",
        "poker_decision_ai/cash_integrity_context.py": "4f7fe96ea43ceff6",
        "poker_decision_ai/cash_decision_environment.py": "ccbf7f0048eee525",
        "poker_decision_ai/cash_strategy_dynamic_belief.py":
            "01549df2a3385d01",
        "poker_decision_ai/cash_strategy_action_ranking.py":
            "53dc7aa44245eb84",
        # 历史 pin（V3H-7）：search_tree 5ea3854c2bf65631 / search_eval
        # 513ba523781c2c07 —— 历史值保留于 PROTECTED_LAYER_PIN_HISTORY，未删除
        "poker_decision_ai/cash_strategy_search_tree.py": "5f5b29be063a6f5a",
        "poker_decision_ai/cash_strategy_search_eval.py": "e97ccea19d2ffd92",
        "poker_decision_ai/cash_strategy_decision_assembly.py":
            "3732314ab0182063",
        "poker_decision_ai/cash_decision_loop.py": "7c4b0a11c1e73a7f",
        "poker_decision_ai/cash_feasible_domain_wiring.py":
            "42a90a9bdff15a2b",
        "poker_decision_ai/cash_opponent_conditioned_bridge.py":
            "548f14a1c9d4db32",
        "poker_decision_ai/cash_conditioned_search_wiring.py":
            "740b9edff66a1982",
        "poker_decision_ai/cash_opponent_conditioned_loop.py":
            "debc421b6f2af242",
        "poker_decision_ai/cash_multistreet.py": "e896c978b2763108",
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


def test_t1_gate_budget_and_solver_status_intact():
    from poker_decision_ai.cash_strategy_decision_assembly import (
        ASSEMBLY_POLICY)
    from poker_decision_ai.cash_strategy_search_tree import DEFAULT_BUDGET
    from poker_decision_ai.cash_solver_interface import solver_status
    from poker_decision_ai.cash_decision_environment import build_scenario
    assert float(ASSEMBLY_POLICY["min_leaf_coverage"]) == 0.8
    assert DEFAULT_BUDGET == {"max_street_depth": 2, "max_response_depth": 1,
                              "max_candidate_actions": 6, "max_leaves": 200,
                              "max_runout_children": 8}
    sc = build_scenario(seed=11, street="flop", pot_bb=1.5, stack_bb=100.0)
    assert solver_status(sc["state"]).get("solver_status") == "BLOCKED"


def test_t1_new_module_is_not_wired_to_production():
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
    assert "cash_capability_verification_plan" not in found


# ---------------------------------------------------------------------------
# T2. 逐缺口审计（要求 4）
# ---------------------------------------------------------------------------
def test_t2_gap_audit_requires_all_four_questions():
    r = audit_gap(gap=_gap())
    assert r["present"] is True
    assert r["gap_id"] == "G5_decision_quality_not_provable_model_internal"
    for k in ("external_evidence_needed", "current_evidence",
              "what_is_missing", "minimal_closure_experiment"):
        assert r[k], k
    for k in ("external_evidence_needed", "current_evidence",
              "what_is_missing"):
        bad = _gap()
        bad.pop(k)
        assert audit_gap(gap=bad).get("present") is False, k


def test_t2_gap_audit_fail_closed_on_bad_input():
    assert audit_gap(gap=None).get("present") is False
    assert audit_gap(gap=_gap(gap_id="GX")).get("present") is False
    assert audit_gap(gap=_gap(closure_category="WHATEVER")).get(
        "present") is False


def test_t2_gap_audit_records_external_dependency_flags():
    r = audit_gap(gap=_gap())
    assert r["closable_without_solver"] is False
    assert r["closable_without_rc2"] is True
    assert r["closable_without_any_external"] is False
    assert r["long_term_open"] is True
    assert r["do_not_search_again"] == ["RC2_real_opponent_corpus"]


def test_t2_plan_requires_all_seven_gaps():
    plan = build_gap_verification_plan(gaps=[_gap()])
    assert plan["n_gaps"] == 1
    assert plan["all_gaps_audited"] is False
    assert len(plan["gaps_not_audited"]) == len(GAP_IDS) - 1
    full = build_gap_verification_plan(
        gaps=[_gap(gap_id=g) for g in GAP_IDS])
    assert full["all_gaps_audited"] is True
    assert build_gap_verification_plan(gaps=[]).get("present") is False


def test_t2_plan_groups_by_closure_category_and_counts():
    plan = build_gap_verification_plan(gaps=[
        _gap(gap_id=GAP_IDS[0], closure_category="ARCHITECTURE_BOUND_FROZEN",
             requires_unfreeze=True, blocked_by_freeze=True,
             long_term_open=False),
        _gap(gap_id=GAP_IDS[1], closure_category="SETTLED_NOT_A_DEFECT",
             long_term_open=False, requires_unfreeze=False,
             blocked_by_freeze=False),
        _gap(gap_id=GAP_IDS[3], closure_category="REQUIRES_EXTERNAL_EVIDENCE",
             long_term_open=True),
    ])
    assert plan["n_category"]["ARCHITECTURE_BOUND_FROZEN"] == 1
    assert plan["n_category"]["SETTLED_NOT_A_DEFECT"] == 1
    assert plan["n_requires_unfreeze"] == 1
    assert plan["n_long_term_open"] == 1
    assert plan["n_closable_now_within_freeze"] == 0
    assert "冻结期内**可真正能力闭合**的缺口数 = 0" in plan["headline"]


# ---------------------------------------------------------------------------
# T3. 外部依赖关系（要求 5、6）
# ---------------------------------------------------------------------------
def test_t3_dependency_graph_three_sources_all_unavailable():
    g = external_dependency_graph()
    assert set(g["nodes"]) == {"RC2_real_opponent_corpus",
                               "solver_equilibrium_reference",
                               "real_human_validation"}
    assert g["nodes"]["RC2_real_opponent_corpus"]["status"] == "EXHAUSTED"
    assert g["nodes"]["solver_equilibrium_reference"]["status"] == "BLOCKED"
    assert g["nodes"]["real_human_validation"]["status"] == "UNAVAILABLE"
    assert all(not n["re_searchable"] for n in g["nodes"].values())


def test_t3_no_substitution_between_sources():
    g = external_dependency_graph()
    assert len(g["why_not_substitutable"]) >= 3
    txt = " ".join(g["why_not_substitutable"])
    assert "不能" in txt
    assert "MODEL_INTERNAL" in txt
    assert "没有任何可达的闭合路径" in g["single_point_of_failure"]


def test_t3_exhausted_sources_registry_prevents_repeat_search():
    g = external_dependency_graph()
    reg = g["avoid_repeat_search"]["exhausted_public_sources"]
    assert len(reg) == len(EXHAUSTED_PUBLIC_SOURCES) >= 5
    names = " ".join(x["source"] for x in reg)
    for need in ("IRC", "C4", "TwoPlusTwo", "pokerftp", "Kaggle"):
        assert need in names, need
    assert "不再" in g["avoid_repeat_search"]["policy"]


def test_t3_evidence_classes_match_loop_vocabulary():
    from poker_decision_ai.cash_decision_loop import (
        ALLOWED_EVIDENCE_LEVELS, REJECTED_EVIDENCE_LEVELS)
    assert tuple(ALLOWED_EVIDENCE_LEVELS) == ("MODEL_INTERNAL",)
    assert set(ALLOWED_EVIDENCE_LEVELS) | set(REJECTED_EVIDENCE_LEVELS) \
        <= set(EVIDENCE_CLASSES)
    assert "MODEL_INTERNAL" in EVIDENCE_CLASSES


# ---------------------------------------------------------------------------
# T4. 最小能力验收矩阵（要求 7）
# ---------------------------------------------------------------------------
def test_t4_matrix_defaults_to_no_evidence_not_pass():
    m = acceptance_matrix()
    assert m["n_dimensions"] == len(ACCEPTANCE_DIMENSIONS)
    assert m["n_satisfied"] == 0
    assert all(r["status"] == "NO_EVIDENCE" for r in m["rows"])
    assert all(r["satisfied"] is False for r in m["rows"])
    assert "不猜测、不预测强度" in m["note"]


def test_t4_matrix_records_required_evidence_class_per_dimension():
    m = acceptance_matrix(statuses={
        "A5_strategy_quality_vs_equilibrium": {
            "criterion": "x", "required_evidence_class": "SOLVER_EQUILIBRIUM",
            "evidence_class": None, "satisfied": False,
            "status": "NO_EVIDENCE_BLOCKED", "falsifier": "f"},
        "A1_engineering_behavior": {
            "criterion": "y", "required_evidence_class": "MODEL_INTERNAL",
            "evidence_class": "MODEL_INTERNAL", "satisfied": True,
            "status": "SATISFIED", "falsifier": "g"},
    })
    by = {r["dimension"]: r for r in m["rows"]}
    assert by["A5_strategy_quality_vs_equilibrium"][
        "required_evidence_class"] == "SOLVER_EQUILIBRIUM"
    assert by["A5_strategy_quality_vs_equilibrium"]["satisfied"] is False
    assert by["A1_engineering_behavior"]["satisfied"] is True
    assert m["n_satisfied"] == 1
    assert m["n_external_satisfied"] == 0
    assert m["n_external_required"] == len(EXTERNAL_REQUIRED_DIMENSIONS)


def test_t4_matrix_falsifier_and_reference_columns_exist():
    m = acceptance_matrix(statuses={"A2_decision_coverage_and_provenance": {
        "criterion": "c", "required_evidence_class": "MODEL_INTERNAL",
        "evidence_class": "MODEL_INTERNAL", "satisfied": True,
        "status": "SATISFIED", "falsifier": "note 不在展开集内",
        "evidence_ref": "docs/V3I_P14_RESULT.json"}})
    row = [r for r in m["rows"]
           if r["dimension"] == "A2_decision_coverage_and_provenance"][0]
    assert row["falsifier"]
    assert row["evidence_ref"] == "docs/V3I_P14_RESULT.json"
    assert set(INTERNAL_SUFFICIENT_DIMENSIONS) <= set(ACCEPTANCE_DIMENSIONS)


# ---------------------------------------------------------------------------
# T5. 禁止替代守卫（要求 5）
# ---------------------------------------------------------------------------
def test_t5_internal_evidence_cannot_close_external_dimensions():
    r = validate_claim(criterion="A5_strategy_quality_vs_equilibrium",
                       evidence_class="MODEL_INTERNAL")
    assert r["accepted"] is False
    assert r["verdict"] == "REJECTED_INTERNAL_SUBSTITUTION"
    assert r["requires_external_evidence"] is True
    assert "外部能力证据" in r["reason"] or "外部" in r["reason"]
    r2 = validate_claim(criterion="A6_real_opponent_external_validity",
                        evidence_class="MODEL_INTERNAL")
    assert r2["verdict"] == "REJECTED_INTERNAL_SUBSTITUTION"
    # 用外部证据则不被"替代"守卫拒绝
    r3 = validate_claim(criterion="A6_real_opponent_external_validity",
                        evidence_class="REAL_OPPONENT_DATA")
    assert r3["accepted"] is True


def test_t5_profitability_claims_are_forbidden_entirely():
    for cls in ("MODEL_INTERNAL", "EXTERNAL_VALIDATION", "PROFIT_EVIDENCE"):
        r = validate_claim(criterion="A7_profitability", evidence_class=cls)
        assert r["accepted"] is False, cls
        assert r["verdict"] == "REJECTED_FORBIDDEN_CLAIM", cls
        assert r["is_forbidden_dimension"] is True
    assert "A7_profitability" in FORBIDDEN_DIMENSIONS


def test_t5_internal_dimensions_can_be_closed_internally():
    r = validate_claim(criterion="A1_engineering_behavior",
                       evidence_class="MODEL_INTERNAL")
    assert r["accepted"] is True and r["verdict"] == "ACCEPTED"
    assert r["requires_external_evidence"] is False
    assert validate_claim(criterion="NOPE",
                          evidence_class="MODEL_INTERNAL").get(
        "present") is False
    assert validate_claim(criterion="A1_engineering_behavior",
                          evidence_class="NOPE").get("present") is False


# ---------------------------------------------------------------------------
# T6. 优先级与可执行性（要求 5）
# ---------------------------------------------------------------------------
def test_t6_prioritise_splits_into_four_buckets():
    plan = build_gap_verification_plan(gaps=[
        _gap(gap_id=GAP_IDS[0], closure_category="ARCHITECTURE_BOUND_FROZEN",
             requires_unfreeze=True, blocked_by_freeze=True,
             long_term_open=False),
        _gap(gap_id=GAP_IDS[1], closure_category="SETTLED_NOT_A_DEFECT",
             long_term_open=False),
        _gap(gap_id=GAP_IDS[3], closure_category="REQUIRES_EXTERNAL_EVIDENCE",
             long_term_open=True),
    ])
    p = prioritise(plan=plan)
    assert p["blocked_by_freeze_needs_unfreeze"] == [GAP_IDS[0]]
    assert p["settled_already_closed_as_questions"] == [GAP_IDS[1]]
    assert p["long_term_open_pending_external"] == [GAP_IDS[3]]
    assert p["closable_now_without_external"] == []
    assert "**不得**用更多 `MODEL_INTERNAL` 实验" in p["statement"]
    assert prioritise(plan=None).get("present") is False


# ---------------------------------------------------------------------------
# T7. 实测守护（要求 4、5、6、7、9）
# ---------------------------------------------------------------------------
pytestmark_ev = pytest.mark.skipif(
    not EVIDENCE.exists(), reason="需要 .tmp/v3i_p18_plan.json")


@pytestmark_ev
def test_t7_all_seven_gaps_audited_with_four_questions():
    d = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    plan = d["gap_plan"]
    assert plan["n_gaps"] == 7
    assert plan["all_gaps_audited"] is True
    assert plan["gaps_not_audited"] == []
    assert set(plan["per_gap"]) == set(GAP_IDS)
    for gid, r in plan["per_gap"].items():
        for k in ("external_evidence_needed", "current_evidence",
                  "what_is_missing", "minimal_closure_experiment"):
            assert r[k], (gid, k)
        assert r["closure_category"] in CLOSURE_CATEGORIES


@pytestmark_ev
def test_t7_measured_category_split_is_honest():
    d = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    plan = d["gap_plan"]
    assert plan["n_category"]["ARCHITECTURE_BOUND_FROZEN"] == 3
    assert plan["n_category"]["SETTLED_NOT_A_DEFECT"] == 2
    assert plan["n_category"]["REQUIRES_EXTERNAL_EVIDENCE"] == 1
    assert plan["n_category"]["REQUIRES_EXTERNAL_COVERAGE_BENCHMARK"] == 1
    assert plan["n_closable_now_within_freeze"] == 0
    p = d["prioritisation"]
    assert len(p["blocked_by_freeze_needs_unfreeze"]) == 3
    assert len(p["long_term_open_pending_external"]) == 2
    assert len(p["settled_already_closed_as_questions"]) == 2
    assert p["closable_now_without_external"] == []


@pytestmark_ev
def test_t7_matrix_measured_statuses_are_not_inflated():
    d = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    m = d["acceptance_matrix"]
    assert m["n_dimensions"] == 9
    assert m["n_satisfied"] == 3
    assert m["n_external_satisfied"] == 0
    by = {r["dimension"]: r for r in m["rows"]}
    assert by["A5_strategy_quality_vs_equilibrium"]["satisfied"] is False
    assert by["A6_real_opponent_external_validity"]["satisfied"] is False
    assert by["A7_profitability"]["satisfied"] is False
    assert by["A8_river_street_decision_capability"]["satisfied"] is False
    assert by["A9_preflop_real_equity_evidence"]["satisfied"] is False
    assert by["A9_preflop_real_equity_evidence"]["evidence_class_held"] is None


@pytestmark_ev
def test_t7_guards_measured_reject_substitution_and_profit_claims():
    d = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    vs = [(g["criterion"], g["verdict"]) for g in d["substitution_guards"]]
    assert ("A5_strategy_quality_vs_equilibrium",
            "REJECTED_INTERNAL_SUBSTITUTION") in vs
    assert ("A6_real_opponent_external_validity",
            "REJECTED_INTERNAL_SUBSTITUTION") in vs
    assert ("A7_profitability", "REJECTED_FORBIDDEN_CLAIM") in vs
    assert ("A1_engineering_behavior", "ACCEPTED") in vs


@pytestmark_ev
def test_t7_dependency_and_exhausted_registry_measured():
    d = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    dep = d["external_dependency_graph"]
    assert dep["nodes"]["RC2_real_opponent_corpus"]["status"] == "EXHAUSTED"
    assert dep["nodes"]["solver_equilibrium_reference"]["status"] == "BLOCKED"
    assert dep["nodes"]["real_human_validation"]["status"] == "UNAVAILABLE"
    assert len(dep["avoid_repeat_search"]["exhausted_public_sources"]) >= 5
    assert dep["single_point_of_failure"]


@pytestmark_ev
def test_t7_evidence_declares_frozen_baseline_and_no_fabrication():
    d = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    c = d["p18_contract"]
    assert c["frozen_baseline"] is True
    assert c["modifies_frozen_contract"] is False
    assert c["adds_strategy_mechanism"] is False
    assert c["substitutes_internal_for_external_evidence"] is False
    assert c["predicts_strength"] is False
    assert c["fabricates_results"] is False
    assert d["frozen_baseline_confirmed"] is True
    assert d["fingerprint"] == stable_fingerprint(
        {"plan": d["gap_plan"]["per_gap"], "prio": d["prioritisation"],
         "matrix": d["acceptance_matrix"]["rows"],
         "dep": d["external_dependency_graph"]["nodes"]})
