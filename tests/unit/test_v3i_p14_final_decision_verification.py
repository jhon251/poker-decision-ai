"""V3I-P14 — 最终决策覆盖 / 跨街稳定性 / 输出质量边界的**验证**（测试）。

本文件验证的是 V3I-P14 **只读验证层** `cash_final_decision_verification`：
分类分离、稳定性测量、ACCEPT 出处验证、证据响应、注入组合/harness 透明性、
能力验证缺口声明，以及 **实测网格守护**（禁止把预测/单 fixture 当覆盖结论）。

环境约束：本沙箱禁止 `%TEMP%` 写入，`tmp_path`/`ctx`/`runtime` fixture 会抛
`PermissionError [WinError 5]`，故本文件不使用需要临时目录的 fixture；
所有新测试均为**纯函数 + 合成记录**，不写盘、不重算上游。
"""
from __future__ import annotations

import ast
import hashlib
import inspect
import json
import pathlib
import sys

import pytest

sys.path.insert(0, ".")

from poker_decision_ai.cash_final_decision_verification import (
    CAPABILITY_GAP_CODES,
    CLAIM_BOUNDARY,
    EVIDENCE_GRADE,
    OUTCOMES,
    RUN_MODES,
    aggregate_accept_provenance,
    assess_capability_gaps,
    classify_run,
    compose_dual_injection,
    decision_quality_boundary,
    harness_equivalence,
    measure_evidence_response,
    measure_stability,
    memoize_stage,
    p14_contract,
    stable_fingerprint,
    summarize_runs,
    verify_accept_provenance,
)
from poker_decision_ai.cash_strategy_decision_assembly import assemble_decision
from poker_decision_ai.cash_feasible_domain_wiring import (
    feasible_keys_from_search_tree)

MODULE = pathlib.Path("poker_decision_ai/cash_final_decision_verification.py")
RECORDS = pathlib.Path(".tmp/v3i_p14_records.json")
SUMMARY = pathlib.Path(".tmp/v3i_p14_summary.json")

COND_MODE = "B2_conditioned_search_feasible_assembly"
REF_MODE = "A2_frozen_search_feasible_assembly"


# ---------------------------------------------------------------------------
# 合成记录（不依赖任何真实运行）
# ---------------------------------------------------------------------------
def _rec(**over) -> dict:
    base = {
        "run_id": "r1", "scenario": "flop_norm", "street": "flop", "seed": 11,
        "pot_bb": 1.5, "stack_bb": 100.0,
        "population": "DECLARED_BALANCED", "mode": COND_MODE,
        "present": True, "reason": None, "loop_status": "UNKNOWN",
        "abstained": False, "preferred": None, "disagreements": [],
        "selected_by_tier": None, "n_scorers_alignable": 0,
        "scorer_tops": {}, "rejections": [],
        "feasible_applied": True, "feasible_domain_size": 2,
        "unknown": [], "stage_trace": [],
        "action_policy_present": True, "search_tree_present": True,
        "search_eval_present": True,
        "search_routed": True, "search_routing_path": "p",
        "search_candidates": [], "search_preferred": None, "ranking_top": None,
        "tree_root_children": [{"action": "check", "amount_bb": 0.0},
                               {"action": "bet", "amount_bb": 0.495}],
        "conditioned_per_candidate": {
            "check|check_no_investment|0.0": {
                "action": "check", "amount_bb": 0.0,
                "hero_equity_vs_continuing": 0.87},
            "bet|bet_pot_fraction|0.495": {
                "action": "bet", "amount_bb": 0.495,
                "hero_equity_vs_continuing": 0.695}},
    }
    base.update(over)
    return base


def _search_cands():
    return [
        {"action": "check", "amount_bb": 0.0, "future_equity_mean": 0.87,
         "search_score": 0.87, "leaf_coverage": 1.0, "n_leaves_evaluable": 9},
        {"action": "bet", "amount_bb": 0.495, "future_equity_mean": 0.695,
         "search_score": 0.8237, "leaf_coverage": 1.0, "n_leaves_evaluable": 9},
    ]


def _accept_rec(**over) -> dict:
    """一个**完整**的 B2 ACCEPT 记录（出处链四环全通）。"""
    tops = {"search": {"action": "bet", "amount_bb": 0.495, "score": 0.9},
            "ranking": {"action": "bet", "amount_bb": 0.495, "score": 1.1}}
    base = dict(
        preferred={"action": "bet", "amount_bb": 0.495,
                   "selected_by_tier": "search", "candidate_key": ["bet", 0.495]},
        selected_by_tier="search", n_scorers_alignable=2, scorer_tops=tops,
        search_candidates=_search_cands(),
        search_preferred={"action": "bet", "amount_bb": 0.495,
                          "search_score": 0.8237},
        disagreements=[], abstained=False,
    )
    base.update(over)
    return _rec(**base)


# ---------------------------------------------------------------------------
# T1. 只读契约与冻结边界（要求 6、7、8）
# ---------------------------------------------------------------------------
def test_t1_contract_declares_read_only_and_no_mechanism():
    c = p14_contract()
    assert c["adds_strategy_mechanism"] is False
    assert c["modifies_p13_contract"] is False
    assert c["modifies_solver_v3h4_v3h6_cfi_lambda_strategy_core"] is False
    assert c["modifies_search_budget_or_river_terminal"] is False
    assert c["enables_override"] is False
    assert c["relaxes_unknown_or_abstain_standard"] is False
    assert c["imports_project_layers"] is False
    assert c["reads_only"] is True
    assert tuple(c["run_modes"]) == RUN_MODES
    assert tuple(c["outcomes"]) == OUTCOMES


def test_t1_claim_boundary_all_false():
    for k, v in CLAIM_BOUNDARY.items():
        assert v is False, k
    for flag in ("is_gto", "is_equilibrium", "is_solver_output", "is_ev",
                 "is_external_validation", "wired_to_production",
                 "adds_strategy_mechanism", "modifies_frozen_contract",
                 "relaxes_fail_closed", "enables_override",
                 "is_strategy_strength_claim"):
        assert flag in CLAIM_BOUNDARY
    assert EVIDENCE_GRADE == "MODEL_INTERNAL"


def test_t1_module_imports_only_stdlib():
    tree = ast.parse(MODULE.read_text(encoding="utf-8"))
    mods = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            mods.update(a.name for a in n.names)
        elif isinstance(n, ast.ImportFrom):
            mods.add(n.module or "")
    assert mods <= {"__future__", "typing", "json", "hashlib"}, mods
    assert not any(m.startswith("poker_decision_ai") for m in mods)


def test_t1_no_banned_dynamic_calls_or_random():
    tree = ast.parse(MODULE.read_text(encoding="utf-8"))
    banned = {"setattr", "delattr", "globals", "locals", "eval", "exec",
              "__import__", "compile", "open", "input"}
    called = {n.func.id for n in ast.walk(tree)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert not (called & banned), called & banned
    imported = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import)
                for a in n.names}
    assert "random" not in imported
    assert "random" not in {n.module for n in ast.walk(tree)
                            if isinstance(n, ast.ImportFrom)}


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
    "poker_decision_ai/cash_conditioned_search_wiring.py": (
        "bcf564ca3e99fff5",   # 曾被钉住的历史 pin（P7/P8 时代）
        "0f8a0bd1118c009a",   # v4 封存值（G1 Phase 1 测量量物化）
        "740b9edff66a1982",   # **当前**封存 v5（G1 Phase 2 统一路径编排）
    ),
}


def test_t1_p13_contract_and_frozen_layers_untouched():
    """要求 6：本阶段不得修改 P13 契约与任何受保护层。"""
    frozen = {
        "poker_decision_ai/cash_strategy_decision_assembly.py":
            "3732314ab0182063",       # V3I-P13 授权状态
        "poker_decision_ai/cash_feasible_domain_wiring.py": "42a90a9bdff15a2b",
        "poker_decision_ai/cash_decision_loop.py": "7c4b0a11c1e73a7f",
        # 历史 pin（V3H-7）：search_eval 513ba523781c2c07 / search_tree
        # 5ea3854c2bf65631 —— 历史值保留于 PROTECTED_LAYER_PIN_HISTORY，未删除
        "poker_decision_ai/cash_strategy_search_eval.py": "e97ccea19d2ffd92",
        "poker_decision_ai/cash_strategy_search_tree.py": "5f5b29be063a6f5a",
        "poker_decision_ai/cash_opponent_conditioned_loop.py":
            "debc421b6f2af242",
        "poker_decision_ai/cash_conditioned_search_wiring.py":
            "740b9edff66a1982",
        "poker_decision_ai/cash_decision_environment.py": "ccbf7f0048eee525",
        "poker_decision_ai/cash_strategy_dynamic_belief.py": "01549df2a3385d01",
        "poker_decision_ai/cash_strategy_likelihood_model.py":
            "df207954ade71fa6",
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


def test_t1_p13_feasible_contract_still_present_and_default_none():
    sig = inspect.signature(assemble_decision)
    assert "feasible_keys" in sig.parameters
    assert sig.parameters["feasible_keys"].default is None
    r = feasible_keys_from_search_tree(
        search_tree={"present": True, "root": {"children": [
            {"action": "bet", "amount_bb": 0.495}]}})
    assert r["present"] is True and r["feasible_keys"] == [["bet", 0.495]]


def test_t1_search_budget_unchanged():
    from poker_decision_ai.cash_strategy_search_tree import DEFAULT_BUDGET
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
    assert "cash_final_decision_verification" not in found


# ---------------------------------------------------------------------------
# T2. 结果分类**分离**（要求 2）
# ---------------------------------------------------------------------------
def test_t2_accept_classified_with_route():
    c = classify_run(record=_accept_rec())
    assert c["outcome"] == "ACCEPT"
    assert c["accept_route"] == "search_feasible"
    assert c["cross_scorer_verified"] is True


def test_t2_accept_route_distinguishes_feasible_and_unconstrained():
    a = classify_run(record=_accept_rec(feasible_applied=False,
                                        mode="A1_frozen_search_frozen_assembly"))
    assert a["accept_route"] == "search_unconstrained"
    b = classify_run(record=_accept_rec(
        preferred={"action": "bet", "amount_bb": 0.495,
                   "selected_by_tier": "ranking"}))
    assert b["accept_route"] == "ranking_feasible"


def test_t2_structural_vs_true_disagreement_separated():
    """A1（冻结 search）的分歧是**结构性域不匹配**；限域内同域分歧是**真分歧**。"""
    structural = _rec(
        mode="A1_frozen_search_frozen_assembly", feasible_applied=False,
        abstained=True, preferred=None,
        disagreements=[{"tier_a": "search", "tier_b": "ranking",
                        "key_a": ["bet", 0.495], "key_b": ["bet", 100.0]}])
    c1 = classify_run(record=structural)
    assert c1["outcome"] == "ABSTAIN"
    assert c1["abstain_kinds"] == ["structural_domain_mismatch"]
    true_dis = _rec(
        abstained=True, preferred=None,
        disagreements=[{"tier_a": "search", "tier_b": "ranking",
                        "key_a": ["check", 0.0], "key_b": ["bet", 0.495]}])
    c2 = classify_run(record=true_dis)
    assert c2["outcome"] == "ABSTAIN"
    assert c2["abstain_kinds"] == ["true_in_domain_disagreement"]


def test_t2_blocked_kind_pre_search_not_abstain():
    """river 式阻断：冻结终局 → `pre_search_blocked`，且**不**算 ABSTAIN。"""
    river = _rec(street="river", scenario="river_norm", present=False,
                 reason="stage_failed:action_policy", loop_status="BLOCKED",
                 abstained=False, preferred=None, feasible_applied=False,
                 action_policy_present=None, search_tree_present=None,
                 tree_root_children=[], search_routed=False,
                 stage_trace=[{"stage": "action_policy", "present": False}],
                 unknown=["stage_failed:action_policy"])
    c = classify_run(record=river)
    assert c["outcome"] == "BLOCKED"
    assert c["blocked_kind"] == "pre_search_blocked"
    assert c["loop_fail_closed"] is False
    assert c["loop_present_false"] is True


def test_t2_loop_fail_closed_is_separate_kind():
    nc = _rec(present=False, reason="not_cash_mode_fail_closed",
              loop_status="BLOCKED", action_policy_present=None,
              search_tree_present=None, tree_root_children=[],
              unknown=["not_cash_mode_fail_closed"])
    c = classify_run(record=nc)
    assert c["outcome"] == "BLOCKED"
    assert c["blocked_kind"] == "loop_fail_closed"
    assert c["loop_fail_closed"] is True


def test_t2_unknown_no_preferred_is_separate_from_accept():
    u = _rec(loop_status="UNKNOWN", preferred=None, abstained=False,
             unknown=["search_tree_unavailable"])
    c = classify_run(record=u)
    assert c["outcome"] == "UNKNOWN_NO_PREFERRED"
    assert c["accept_route"] is None


def test_t2_loop_unknown_with_preferred_is_flagged_not_a_failure():
    """平台缺 `likelihood_model` 时 loop 报 UNKNOWN，但装配层已给出 preferred：
    必须**同时**算 ACCEPT 并单独标记该平台缺口。"""
    c = classify_run(record=_accept_rec(
        loop_status="UNKNOWN",
        unknown=["likelihood_model_not_supplied",
                 "belief_conditioned_not_supplied"]))
    assert c["outcome"] == "ACCEPT"
    assert c["loop_unknown_due_to_missing_likelihood"] is True


def test_t2_summarize_separates_categories():
    recs = [
        _accept_rec(run_id="a1"),
        _rec(run_id="b1", abstained=True,
             disagreements=[{"tier_a": "search", "tier_b": "ranking",
                             "key_a": ["check", 0.0], "key_b": ["bet", 0.495]}]),
        _rec(run_id="c1", street="river", present=False,
             reason="stage_failed:action_policy", loop_status="BLOCKED",
             action_policy_present=None, search_tree_present=None,
             tree_root_children=[], search_routed=False,
             stage_trace=[{"stage": "action_policy", "present": False}]),
        _rec(run_id="d1", loop_status="UNKNOWN", preferred=None,
             unknown=["search_tree_unavailable"]),
    ]
    s = summarize_runs(records=recs)
    o = s["overall"]
    assert o["n_runs"] == 4
    assert o["outcome_counts"] == {"ACCEPT": 1, "ABSTAIN": 1, "BLOCKED": 1,
                                   "UNKNOWN_NO_PREFERRED": 1}
    assert o["final_decision_coverage"] == 0.25
    assert o["accept_routes"] == {"search_feasible": 1}
    assert o["accept_cross_scorer_verified"] == 1
    assert o["abstain_kinds"] == {"true_in_domain_disagreement": 1}
    assert o["blocked_kinds"] == {"pre_search_blocked": 1}


def test_t2_summarize_fail_closed_on_bad_input():
    assert summarize_runs(records=[]).get("present") is False
    assert summarize_runs(records="nope").get("present") is False


# ---------------------------------------------------------------------------
# T3. 跨维度稳定性（要求 3）
# ---------------------------------------------------------------------------
def test_t3_stability_detects_unstable_group_and_accept_amounts():
    recs = [
        _accept_rec(run_id="x1", street="flop", scenario="flop_norm",
                    population="P1", scorer_tops={
                        "search": {"action": "bet", "amount_bb": 0.495},
                        "ranking": {"action": "bet", "amount_bb": 0.495}},
                    preferred={"action": "bet", "amount_bb": 0.495,
                               "selected_by_tier": "search"}),
        _rec(run_id="x2", street="flop", scenario="flop_norm", population="P2",
             abstained=True,
             disagreements=[{"tier_a": "search", "tier_b": "ranking",
                             "key_a": ["check", 0.0], "key_b": ["bet", 0.495]}]),
    ]
    st = measure_stability(records=recs, dimension="street")
    assert st["present"] is True
    assert st["all_groups_stable"] is False
    assert st["unstable_groups"] == ["flop"]
    assert st["per_group"]["flop"]["n_distinct_outcomes"] == 2
    assert st["per_group"]["flop"]["accept_amounts"] == [0.495]


def test_t3_stability_restricts_to_mode_and_reports_stable():
    recs = [
        _accept_rec(run_id="y1", street="flop"),
        _accept_rec(run_id="y2", street="flop"),
        _rec(run_id="y3", street="flop", mode="A1_frozen_search_frozen_assembly",
             abstained=True, preferred=None,
             disagreements=[{"tier_a": "search", "tier_b": "ranking",
                             "key_a": ["bet", 0.495], "key_b": ["bet", 100.0]}]),
    ]
    st = measure_stability(records=recs, dimension="street", mode=COND_MODE)
    assert st["per_group"]["flop"]["n_runs"] == 2
    assert st["all_groups_stable"] is True
    assert st["distinct_outcomes_overall"] == ["ACCEPT"]


def test_t3_stability_fail_closed():
    assert measure_stability(records=[]).get("present") is False
    assert measure_stability(records=[_rec()], dimension="").get(
        "present") is False
    assert measure_stability(
        records=[_rec()], dimension="street",
        mode="NOPE_MODE").get("present") is False


# ---------------------------------------------------------------------------
# T4. ACCEPT 出处验证（要求 4）
# ---------------------------------------------------------------------------
def test_t4_complete_chain_accept_passes_all_core_checks():
    p = verify_accept_provenance(record=_accept_rec())
    assert p["applicable"] is True
    assert p["provenance_verified"] is True
    assert p["accept_derivation"] == "search_candidate_local_conditioned"
    assert p["outside_search_expansion"] is False
    assert all(p["checks"][k] for k in p["core_checks"])
    assert p["checks"]["selected_search_candidate_used_own_conditioned_equity"] \
        is True
    assert p["amount_over_pot"] == pytest.approx(0.33, abs=1e-6)
    assert p["tree_root_candidate_amounts"] == [0.0, 0.495]


def test_t4_non_search_tier_accept_inside_feasible_domain_is_second_tier():
    """选中层非 search，但候选落在可行域（= 实际展开集）内且与该层 top 一致。"""
    p = verify_accept_provenance(record=_accept_rec(
        preferred={"action": "bet", "amount_bb": 0.495,
                   "selected_by_tier": "ranking"},
        scorer_tops={"ranking": {"action": "bet", "amount_bb": 0.495}}))
    assert p["provenance_verified"] is True
    assert p["accept_derivation"] == "feasible_domain_non_search_tier"


def test_t4_accept_outside_search_expansion_is_flagged_not_passed():
    """未限域时 ranking 层可选中 search **从未展开**的候选（如 allin 100bb）：
    必须如实标为 `outside_search_expansion`，且**不**算通过出处验证。"""
    p = verify_accept_provenance(record=_accept_rec(
        feasible_applied=False, mode="A1_frozen_search_frozen_assembly",
        search_candidates=[{"action": "check", "amount_bb": 0.0,
                            "future_equity_mean": 0.87},
                           {"action": "bet", "amount_bb": 0.495,
                            "future_equity_mean": 0.695}],
        preferred={"action": "bet", "amount_bb": 100.0,
                   "selected_by_tier": "ranking"},
        scorer_tops={"ranking": {"action": "bet", "amount_bb": 100.0}}))
    assert p["outside_search_expansion"] is True
    assert p["accept_derivation"] == "outside_search_expansion"
    assert p["provenance_verified"] is False
    # 同一记录：限域后即被排除（域外候选不得成为决策）
    p2 = verify_accept_provenance(record=_accept_rec(
        preferred={"action": "bet", "amount_bb": 100.0,
                   "selected_by_tier": "ranking"},
        scorer_tops={"ranking": {"action": "bet", "amount_bb": 100.0}}))
    assert p2["outside_search_expansion"] is True


def test_t4_aggregate_provenance_reports_derivations_honestly():
    a = verify_accept_provenance(record=_accept_rec(run_id="a"))
    b = verify_accept_provenance(record=_accept_rec(
        run_id="b", feasible_applied=False,
        mode="A1_frozen_search_frozen_assembly",
        preferred={"action": "bet", "amount_bb": 100.0,
                   "selected_by_tier": "ranking"},
        scorer_tops={"ranking": {"action": "bet", "amount_bb": 100.0}}))
    c = verify_accept_provenance(record=_rec(run_id="c", abstained=True,
                                             preferred=None))
    agg = aggregate_accept_provenance(provenances=[a, b, c])
    assert agg["n_accepts"] == 2
    assert agg["n_verified"] == 1
    assert agg["n_unverified"] == 1
    assert agg["all_verified"] is False
    assert agg["n_search_candidate_local_conditioned"] == 1
    assert agg["n_outside_search_expansion"] == 1
    assert agg["all_feasible_domain_accepts_verified"] is True
    assert agg["unverified_examples"][0]["run_id"] == "b"
    assert aggregate_accept_provenance(provenances=[]).get("present") is False


def test_t4_amount_outside_actual_candidates_fails():
    p = verify_accept_provenance(record=_accept_rec(
        preferred={"action": "bet", "amount_bb": 1.5,
                   "selected_by_tier": "search"},
        scorer_tops={"search": {"action": "bet", "amount_bb": 1.5}}))
    assert p["checks"]["amount_is_actual_root_candidate"] is False
    assert p["provenance_verified"] is False


def test_t4_amount_must_match_selected_tier_candidate():
    p = verify_accept_provenance(record=_accept_rec(
        scorer_tops={"search": {"action": "check", "amount_bb": 0.0}}))
    assert p["checks"]["amount_matches_selected_tier_candidate"] is False
    assert p["provenance_verified"] is False


def test_t4_pooled_means_fail_conditioned_per_candidate_check():
    """P7 池化缺陷的判据：提供了逐候选条件化且已路由时，各候选的
    `future_equity_mean` 必须**逐候选**吻合输入；全部被池成同一个值时即失败。"""
    pooled = [{"action": "check", "amount_bb": 0.0, "future_equity_mean": 0.8},
              {"action": "bet", "amount_bb": 0.495, "future_equity_mean": 0.8}]
    p = verify_accept_provenance(record=_accept_rec(search_candidates=pooled))
    assert p["checks"][
        "conditioned_consumption_matches_per_candidate_input"] is False
    assert p["n_conditioned_candidates_checked"] == 2
    assert p["n_conditioned_candidates_matching"] == 0
    assert p["search_candidate_means_are_distinct"] is False
    assert p["provenance_verified"] is False


def test_t4_identical_but_matching_conditioned_equities_do_not_false_fail():
    """两个候选自身的条件化权益**本来就可能相同**（如 preflop check/bet）：
    只要逐候选吻合，就**不得**判为池化（避免假失败）。"""
    same = [{"action": "check", "amount_bb": 0.0, "future_equity_mean": 0.5125},
            {"action": "bet", "amount_bb": 0.495, "future_equity_mean": 0.5125}]
    p = verify_accept_provenance(record=_accept_rec(
        search_candidates=same,
        conditioned_per_candidate={
            "check|check_no_investment|0.0": {
                "action": "check", "amount_bb": 0.0,
                "hero_equity_vs_continuing": 0.5125},
            "bet|bet_pot_fraction|0.495": {
                "action": "bet", "amount_bb": 0.495,
                "hero_equity_vs_continuing": 0.5125}},
        search_preferred={"action": "bet", "amount_bb": 0.495}))
    assert p["search_candidate_means_are_distinct"] is False
    assert p["n_conditioned_candidates_matching"] == 2
    assert p["checks"][
        "conditioned_consumption_matches_per_candidate_input"] is True
    assert p["provenance_verified"] is True
    # 未提供条件化输入时该项 vacuously 通过，但 derivation 会如实区分
    q = verify_accept_provenance(record=_accept_rec(
        conditioned_per_candidate=None, search_routed=False))
    assert q["conditioned_input_present"] is False
    assert q["conditioned_consumption_verified"] is False
    assert q["accept_derivation"] == "search_tier_within_feasible_domain"


def test_t4_zero_amount_non_check_is_a_sentinel():
    p = verify_accept_provenance(record=_accept_rec(
        tree_root_children=[{"action": "bet", "amount_bb": 0.0}],
        conditioned_per_candidate={},
        preferred={"action": "bet", "amount_bb": 0.0,
                   "selected_by_tier": "ranking"},
        scorer_tops={"ranking": {"action": "bet", "amount_bb": 0.0}}))
    assert p["checks"]["amount_not_a_default_sentinel"] is False
    assert p["provenance_verified"] is False


def test_t4_search_tier_accept_must_consume_own_conditioned_equity():
    """选中层=search 且条件化已路由时，`future_equity_mean` 必须等于
    **该候选自身**的条件化权益（而不是别的候选/全局值）。"""
    p = verify_accept_provenance(record=_accept_rec(
        conditioned_per_candidate={
            "check|check_no_investment|0.0": {
                "action": "check", "amount_bb": 0.0,
                "hero_equity_vs_continuing": 0.87},
            "bet|bet_pot_fraction|0.495": {
                "action": "bet", "amount_bb": 0.495,
                "hero_equity_vs_continuing": 0.10}}))
    assert p["checks"][
        "selected_search_candidate_used_own_conditioned_equity"] is False
    assert p["provenance_verified"] is False


def test_t4_non_accept_is_not_applicable():
    p = verify_accept_provenance(record=_rec(abstained=True, preferred=None))
    assert p["applicable"] is False
    assert p["provenance_verified"] is False
    assert verify_accept_provenance(record=None).get("present") is False


# ---------------------------------------------------------------------------
# T5. 证据响应与 fail-closed（要求 5）
# ---------------------------------------------------------------------------
def _pair(scenario, pop, *, cond_top, ref_top, cond_out, ref_out,
          cond_sig, ref_sig):
    def _mk(rid, mode, top, out, sig):
        accept = out == "ACCEPT"
        return _rec(
            run_id=rid, scenario=scenario, mode=mode, population=pop,
            search_preferred={"action": top[0], "amount_bb": top[1]},
            abstained=(out == "ABSTAIN"), preferred=(
                {"action": top[0], "amount_bb": top[1],
                 "selected_by_tier": "search"} if accept else None),
            disagreements=([{"tier_a": "search", "tier_b": "ranking",
                             "key_a": ["check", 0.0], "key_b": ["bet", 0.495]}]
                           if out == "ABSTAIN" else []),
            scorer_tops=({"search": {"action": top[0], "amount_bb": top[1]}}
                         if accept else {}),
            search_candidates=_search_cands(),
            conditioned_per_candidate={
                "check|check_no_investment|0.0": {
                    "action": "check", "amount_bb": 0.0,
                    "hero_equity_vs_continuing": sig[0]},
                "bet|bet_pot_fraction|0.495": {
                    "action": "bet", "amount_bb": 0.495,
                    "hero_equity_vs_continuing": sig[1]}},
        )
    return [_mk(f"{scenario}|{pop}|cond", COND_MODE, cond_top, cond_out, cond_sig),
            _mk(f"{scenario}|{pop}|ref", REF_MODE, ref_top, ref_out, ref_sig)]


def test_t5_response_detects_conditioning_changing_search_top_and_outcome():
    recs = []
    for pop, sig in (("P1", (0.87, 0.695)), ("P2", (0.87, 0.90))):
        recs += _pair("flop_norm", pop, cond_top=("check", 0.0),
                      ref_top=("bet", 0.495), cond_out="ABSTAIN",
                      ref_out="ACCEPT", cond_sig=sig, ref_sig=(0.87, 0.695))
    r = measure_evidence_response(records=recs)
    assert r["present"] is True
    assert r["n_pairs_total"] == 2
    assert r["n_pairs_search_top_changed"] == 2
    assert r["n_pairs_outcome_changed"] == 2
    assert r["scenarios_where_outcome_changed_by_conditioning"] == ["flop_norm"]
    # 条件化签名随 population 变化（P1 与 P2 不同）
    per = r["per_scenario"]["flop_norm"]
    assert per["n_distinct_conditioned_signatures"] == 2
    # outcome 在 population 之间不变（都为 ABSTAIN）→ 稳定性可分辨
    assert per["outcome_changes_with_population"] is False


def test_t5_response_reports_population_sensitivity_when_present():
    recs = []
    for pop, out, sig in (("P1", "ABSTAIN", (0.87, 0.695)),
                          ("P2", "ACCEPT", (0.60, 0.90))):
        recs += _pair("turn_norm", pop, cond_top=("bet", 2.64),
                      ref_top=("bet", 2.64), cond_out=out, ref_out="ACCEPT",
                      cond_sig=sig, ref_sig=(0.9, 0.5))
    r = measure_evidence_response(records=recs)
    per = r["per_scenario"]["turn_norm"]
    assert per["outcome_changes_with_population"] is True
    assert per["n_distinct_outcomes"] == 2
    assert r["scenarios_where_outcome_varies_with_population"] == ["turn_norm"]


def test_t5_fail_closed_runs_are_still_classified():
    """未路由（条件化不可用）的运行必须**仍**给出可分类结果，不得变成未定义态。"""
    recs = [_rec(run_id="f1", search_routed=False,
                 search_routing_path="passthrough_no_conditioning",
                 abstained=True, preferred=None,
                 disagreements=[{"tier_a": "search", "tier_b": "ranking",
                                 "key_a": ["check", 0.0],
                                 "key_b": ["bet", 0.495]}]),
            _rec(run_id="f2", search_routed=False, loop_status="BLOCKED",
                 present=False, reason="stage_failed:action_policy",
                 action_policy_present=None, search_tree_present=None,
                 tree_root_children=[],
                 stage_trace=[{"stage": "action_policy", "present": False}])]
    r = measure_evidence_response(records=recs)
    assert r["not_routed_runs"] == 2
    assert r["not_routed_runs_all_have_result"] is True
    assert r["all_runs_classified"] is True
    assert r["fail_closed_preserved"] is True


def test_t5_response_fail_closed_on_bad_input():
    assert measure_evidence_response(records=[]).get("present") is False
    assert measure_evidence_response(records="nope").get("present") is False
    # 单条记录（无可配对 reference）仍必须给出结构完整的输出
    one = measure_evidence_response(records=[_rec()])
    assert one["present"] is True
    assert one["n_pairs_total"] == 0
    assert one["all_runs_classified"] is True


# ---------------------------------------------------------------------------
# T6. 注入组合与 harness 透明性（测量工具，不改任何层）
# ---------------------------------------------------------------------------
def _pipe():
    def f(state, **kw):
        return {"present": True}
    return {k: f for k in ("build_action_policy", "build_dynamic_belief",
                           "update_belief", "rank_actions",
                           "evaluate_belief_conditioned", "build_search_tree",
                           "evaluate_search", "assemble_decision")}


def test_t6_compose_injects_both_keys_without_mutating_input():
    base = _pipe()
    es, ad = (lambda state, **kw: {}), (lambda state, **kw: {})
    r = compose_dual_injection(pipeline=base, evaluate_search_closure=es,
                               assemble_decision_closure=ad)
    assert r["present"] is True
    assert r["injected_keys"] == ["evaluate_search", "assemble_decision"]
    assert r["pipeline"]["evaluate_search"] is es
    assert r["pipeline"]["assemble_decision"] is ad
    assert base["evaluate_search"] is not es
    assert base["assemble_decision"] is not ad
    assert r["mutates_input_pipeline"] is False


def test_t6_compose_records_skipped_and_fails_closed_on_missing_stage():
    r = compose_dual_injection(pipeline=_pipe(), evaluate_search_closure=None,
                               assemble_decision_closure=None)
    assert r["present"] is True
    assert r["injected_keys"] == []
    assert r["skipped_keys"] == ["evaluate_search", "assemble_decision"]
    bad = compose_dual_injection(pipeline={"build_action_policy": lambda s: {}})
    assert bad["present"] is False
    assert "pipeline_missing_stages" in bad["reason"]
    assert compose_dual_injection(pipeline=None).get("present") is False


def test_t6_memoize_returns_same_object_and_counts_hits():
    calls = {"n": 0}

    def fn(state, **kw):
        calls["n"] += 1
        return {"present": True, "n": calls["n"]}

    m = memoize_stage(fn=fn)
    assert m["present"] is True
    a = m["callable"]({"s": 1}, x=1)
    b = m["callable"]({"s": 1}, x=1)
    c = m["callable"]({"s": 1}, x=2)
    assert a is b
    assert a is not c
    assert calls["n"] == 2
    assert m["stats"] == {"calls": 3, "hits": 1, "misses": 2,
                          "distinct_inputs": 2}
    assert m["uses_monkeypatch"] is False
    assert m["uses_global_state"] is False


def test_t6_memoize_fail_closed_on_non_callable():
    assert memoize_stage(fn=None).get("present") is False
    assert memoize_stage(fn=42).get("present") is False


def test_t6_harness_equivalence_is_measured_not_assumed():
    base = {"present": True, "loop_status": "UNKNOWN",
            "decision_record": {"decision": {"a": 1}, "preferred": {"x": 1},
                                "abstained": False, "disagreements": []},
            "stage_outputs": {"search_eval": {"candidates": [
                {"action": "bet", "amount_bb": 0.5, "future_equity_mean": 0.7,
                 "search_score": 0.8, "leaf_coverage": 1.0,
                 "n_leaves_evaluable": 3}]}},
            "stage_trace": [{"stage": "action_policy", "present": True}],
            "unknown": []}
    assert harness_equivalence(memoized_out=base,
                               raw_out=json.loads(json.dumps(base)))[
        "identical"] is True
    other = json.loads(json.dumps(base))
    other["stage_outputs"]["search_eval"]["candidates"][0][
        "future_equity_mean"] = 0.71
    eq = harness_equivalence(memoized_out=base, raw_out=other)
    assert eq["identical"] is False
    assert eq["memoization_is_transparent"] is False
    assert harness_equivalence(memoized_out=None, raw_out=base).get(
        "present") is False


# ---------------------------------------------------------------------------
# T7. 能力验证缺口（要求 9）
# ---------------------------------------------------------------------------
def test_t7_gaps_declare_quality_not_provable():
    summary = summarize_runs(records=[_accept_rec(), _rec(run_id="b",
                                                          abstained=True,
                                                          preferred=None,
                                                          disagreements=[])])
    g = assess_capability_gaps(summary=summary)
    assert "G5_decision_quality_not_provable_model_internal" in g["gap_codes"]
    assert "G1_accept_not_cross_scorer_verified" not in g["gap_codes"]
    assert g["decision_quality_provable_with_model_internal_evidence"] is False
    assert set(g["declared_codes"]) == set(CAPABILITY_GAP_CODES)


def test_t7_single_tier_accept_is_flagged_as_quality_boundary():
    summary = summarize_runs(records=[_accept_rec(n_scorers_alignable=1)])
    g = assess_capability_gaps(summary=summary)
    assert "G1_accept_not_cross_scorer_verified" in g["gap_codes"]
    assert "G1_accept_not_cross_scorer_verified" in g["quality_boundaries"]


def test_t7_conditioning_coverage_drop_is_flagged_as_capability_gap():
    a2 = _accept_rec(run_id="a", mode=REF_MODE)
    b2 = _rec(run_id="b", abstained=True, preferred=None,
              disagreements=[{"tier_a": "search", "tier_b": "ranking",
                              "key_a": ["check", 0.0],
                              "key_b": ["bet", 0.495]}])
    summary = summarize_runs(records=[a2, b2])
    g = assess_capability_gaps(summary=summary)
    assert "G2_conditioning_routing_collapses_accept" in g["capability_gaps"]


def test_t7_quality_boundary_lists_provable_and_not_provable():
    summary = summarize_runs(records=[_accept_rec()])
    g = assess_capability_gaps(summary=summary)
    b = decision_quality_boundary(summary=summary, gaps=g)
    assert b["measured_final_coverage"] == 1.0
    assert "EV" in b["must_not_be_read_as"]
    assert any("覆盖率" in s for s in b["provable_with_model_internal_evidence"])
    assert any("更优" in s for s in
               b["not_provable_with_model_internal_evidence"])
    assert assess_capability_gaps(summary=None).get("present") is False


# ---------------------------------------------------------------------------
# T8. 实测网格守护（要求 1、3、4、5：禁止单一 fixture / 禁止预测当实测）
# ---------------------------------------------------------------------------
pytestmark_records = pytest.mark.skipif(
    not RECORDS.exists(), reason="需要 .tmp/v3i_p14_records.json（扫描脚本产出）")


@pytestmark_records
def test_t8_all_four_streets_measured_not_single_fixture():
    recs = json.loads(RECORDS.read_text(encoding="utf-8"))
    streets = {r["street"] for r in recs}
    assert streets == {"preflop", "flop", "turn", "river"}, streets
    seeds = {r["seed"] for r in recs}
    assert len(seeds) >= 3, seeds
    pots = {r["pot_bb"] for r in recs}
    stacks = {r["stack_bb"] for r in recs}
    assert len(pots) >= 4 and len(stacks) >= 4, (pots, stacks)
    pops = {r["population"] for r in recs}
    assert len(pops) >= 5, pops
    modes = {r["mode"] for r in recs}
    assert set(RUN_MODES) <= modes, modes


@pytestmark_records
def test_t8_outcome_categories_are_separated_by_measurement():
    recs = json.loads(RECORDS.read_text(encoding="utf-8"))
    cls = [classify_run(record=r) for r in recs]
    meas = [c for c in cls if c["mode"] in RUN_MODES]
    assert all(c["present"] for c in meas)
    seen = {c["outcome"] for c in meas}
    assert "ACCEPT" in seen and "ABSTAIN" in seen and "BLOCKED" in seen
    # 结构性域不匹配与真分歧**都**被实测到，且分开计数
    kinds = {k for c in meas for k in c["abstain_kinds"]}
    assert kinds == {"structural_domain_mismatch", "true_in_domain_disagreement"}
    # river 全部为 pre_search_blocked
    river = [c for c in meas if c["street"] == "river"]
    assert river and all(c["outcome"] == "BLOCKED"
                         and c["blocked_kind"] == "pre_search_blocked"
                         for c in river)


@pytestmark_records
def test_t8_every_accept_has_verified_provenance():
    recs = json.loads(RECORDS.read_text(encoding="utf-8"))
    grid = [r for r in recs if r["mode"] in RUN_MODES]
    prov = [verify_accept_provenance(record=r) for r in grid]
    ok = [p for p in prov if p.get("applicable")]
    assert ok, "网格中必须至少有一个 ACCEPT（否则本阶段结论需改写）"
    # 1. **限域**（可行域 = search 实际展开集）的 ACCEPT 必须**全部**通过出处验证
    constrained = [p for p in ok if p["feasible_domain_applied"]]
    assert constrained, "限域 ACCEPT 缺失"
    bad = [p for p in constrained if not p.get("provenance_verified")]
    assert bad == [], bad
    # 2. 未通过的只能是"候选在 search 展开集之外"的未限域 ACCEPT —— 如实标记，
    #    不得被当作通过
    unverified = [p for p in ok if not p.get("provenance_verified")]
    assert all(p["accept_derivation"] == "outside_search_expansion"
               for p in unverified), unverified
    assert all(not p["feasible_domain_applied"] for p in unverified), unverified
    # 3. 至少存在一档"最强出处"（search 层消费自身条件化权益）或限域非 search 档
    derivations = {p["accept_derivation"] for p in ok}
    assert derivations & {"search_candidate_local_conditioned",
                          "search_tier_within_feasible_domain",
                          "feasible_domain_non_search_tier"}, derivations
    # 4. 金额随 pot 变化（不是硬编码常量）：至少 3 个不同注额，且都落在实际候选上
    amounts = {p["amount_bb"] for p in constrained}
    assert len(amounts) >= 3, amounts
    for p in constrained:
        assert p["amount_bb"] in p["tree_root_candidate_amounts"], p
        if p["action"] != "check":
            assert p["amount_bb"] > 0, p


@pytestmark_records
def test_t8_accept_provenance_aggregate_measured():
    summ = json.loads(SUMMARY.read_text(encoding="utf-8"))
    agg = summ["accept_provenance_aggregate"]
    assert agg["present"] is True
    assert agg["n_accepts"] == len(summ["accept_provenance"])
    assert agg["all_feasible_domain_accepts_verified"] is True
    # 未限域路径**确实**会选出 search 从未展开的候选 → 该边界被实测到
    assert agg["n_outside_search_expansion"] >= 1, agg
    assert agg["n_verified"] == agg["n_accepts"] - \
        agg["accept_derivations"]["outside_search_expansion"]

@pytestmark_records
def test_t8_perturbation_proves_per_candidate_dependence_not_hardcoding():
    """决定性检验：**保集合**的逐候选权益置换必须改变结果；
    恒等扰动必须复现结果（确定性）。"""
    recs = json.loads(RECORDS.read_text(encoding="utf-8"))
    pert = {}
    for r in recs:
        pert.setdefault(r["mode"], []).append(r)
    null_recs = pert.get("P1_null_conditioned_feasible") or []
    swap_recs = pert.get("P2_swap_root_candidate_equities") or []
    base_recs = [r for r in recs if r["mode"] == COND_MODE
                 and r["population"] == "DECLARED_BALANCED"]
    assert null_recs and swap_recs and base_recs

    def _sig(items):
        return {r["scenario"]: r["decision_fingerprint"] for r in items}

    # 1. 恒等扰动 → 决策指纹与基线在**共同 scenario** 上逐一相同（确定性）
    a, b = _sig(null_recs), _sig(base_recs)
    common = sorted(set(a) & set(b))
    assert len(common) >= 6, common          # river 无条件化输入 → 无扰动
    assert {k: a[k] for k in common} == {k: b[k] for k in common}
    # 2. 置换扰动 → 至少一个 scenario 的决策指纹改变（per-candidate 生效）
    sa, sb = _sig(swap_recs), _sig(base_recs)
    changed = [k for k in sa if k in sb and sa[k] != sb[k]]
    assert changed, (sa, sb)

    # 3. 置换把**每个候选自身的**条件化权益互换 ⇒ search 层逐候选均值必须随之互换
    #    （池化型评分对"保集合置换"不变，故这是决定性判据）。
    #    若某 scenario 的两个 root 候选自身权益本来就相同，则置换是恒等 → 不构成检验。
    def _root_means(rec):
        return {(c["action"], c["amount_bb"]): c["future_equity_mean"]
                for c in rec["search_candidates"]}

    n_perm = 0
    for s in swap_recs:
        b = [x for x in base_recs if x["scenario"] == s["scenario"]]
        if not b:
            continue
        bm, sm = _root_means(b[0]), _root_means(s)
        keys = sorted(bm)
        if len(keys) < 2 or set(keys) != set(sm):
            continue
        vals = [bm[k] for k in keys]
        if len({v for v in vals if v is not None}) < 2:
            continue
        n_perm += 1
        assert [sm[k] for k in keys] == [bm[k] for k in reversed(keys)], (
            s["scenario"], bm, sm)
    assert n_perm >= 3, n_perm


@pytestmark_records
def test_t8_shift_perturbation_moves_each_candidate_mean_by_delta():
    """整体下移 0.30 → 每个候选的 future_equity_mean 必须各自下移 0.30
    （证明 search 消费的是**该候选自身**的条件化权益）。"""
    recs = json.loads(RECORDS.read_text(encoding="utf-8"))
    base = [r for r in recs if r["mode"] == COND_MODE
            and r["population"] == "DECLARED_BALANCED"]
    shift = [r for r in recs if r["mode"] == "P3_shift_all_equities_down"]
    assert shift
    checked = 0
    for s in shift:
        b = [x for x in base if x["scenario"] == s["scenario"]]
        if not b:
            continue
        bm = {(c["action"], c["amount_bb"]): c["future_equity_mean"]
              for c in b[0]["search_candidates"]}
        sm = {(c["action"], c["amount_bb"]): c["future_equity_mean"]
              for c in s["search_candidates"]}
        for k in set(bm) & set(sm):
            if bm[k] is None or sm[k] is None:
                continue
            assert sm[k] == pytest.approx(max(0.001, bm[k] - 0.30), abs=1e-6), k
            checked += 1
    assert checked > 0


@pytestmark_records
def test_t8_fail_closed_probes_measured():
    summ = json.loads(SUMMARY.read_text(encoding="utf-8"))
    probes = summ["extra_evidence"]["fail_closed_probes"]
    assert probes["non_cash"]["present"] is False
    assert probes["non_cash"]["reason"] == "not_cash_mode_fail_closed"
    assert probes["forbidden_evidence_level"]["present"] is False
    f = probes["flop_norm"]
    assert f["F1_no_prepared"]["will_route"] is False
    assert f["F1_no_prepared"]["routed"] is False
    assert f["F1_no_prepared"]["routing_path"] == "passthrough_no_conditioning"
    # conditioned_input 不可物化 → 闭包**仍会尝试**路由，但实际回退（不伪造适应）
    assert f["F2_materialize_unavailable"]["will_route"] is True
    assert f["F2_materialize_unavailable"]["routed"] is False
    assert f["F2_materialize_unavailable"]["routing_path"] == \
        "passthrough_materialize_failed"
    assert f["F3_conditioned_ok"]["will_route"] is True
    assert f["F3_conditioned_ok"]["routed"] is True
    assert f["F4_empty_feasible_keys"]["feasible_applied"] is False


@pytestmark_records
def test_t8_harness_transparency_measured_identical():
    summ = json.loads(SUMMARY.read_text(encoding="utf-8"))
    eq = summ["extra_evidence"]["harness_equivalence"]
    assert eq["identical"] is True, eq
    assert eq["memoized_fingerprint"] == eq["raw_fingerprint"]


@pytestmark_records
def test_t8_summary_uses_only_factorial_grid():
    summ = json.loads(SUMMARY.read_text(encoding="utf-8"))
    n = summ["n_records_in_factorial_grid"]
    assert summ["n_records"] == n + summ["n_records_perturbation"]
    by_mode = summ["summary_by_mode"]["by_mode"]
    assert sum(v["n_runs"] for v in by_mode.values()) == n
    # 每个 scenario 上 A1/A2 各跑一次、B1/B2 各跑一次（population 维度相同）
    assert by_mode["A1_frozen_search_frozen_assembly"]["n_runs"] == \
        by_mode["A2_frozen_search_feasible_assembly"]["n_runs"]
    assert by_mode["B1_conditioned_search_frozen_assembly"]["n_runs"] == \
        by_mode["B2_conditioned_search_feasible_assembly"]["n_runs"]
    # 扰动记录不得混入覆盖口径
    assert all(v["n_runs"] > 0 for v in by_mode.values())


@pytestmark_records
def test_t8_stability_measured_across_streets_and_modes():
    summ = json.loads(SUMMARY.read_text(encoding="utf-8"))
    st = summ["stability_street_B2"]
    assert st["present"] is True
    assert "preflop" in st["per_group"] and "flop" in st["per_group"]
    assert st["per_group"]["river"]["outcome_counts"].get("BLOCKED", 0) > 0
    # B2（完整链路）跨 population：每 scenario 内结果是否一致
    scn = summ["stability_scenario_B2"]
    assert scn["n_groups"] == 9
    assert len(scn["stable_groups"]) >= 7, scn["unstable_groups"]
    # A2（P13 配置）在每个 scenario 上都是稳定 ACCEPT（river 除外）
    a2 = summ["stability_scenario_A2"]
    assert a2["unstable_groups"] == []
    assert len(a2["distinct_accept_amounts_overall"]) >= 4


@pytestmark_records
def test_t8_evidence_response_measured_and_paired_explicitly():
    """要求 5：必须真的**配对**比较过（禁止空集伪装成"无变化"）。"""
    summ = json.loads(SUMMARY.read_text(encoding="utf-8"))
    r = summ["evidence_response"]
    assert r["present"] is True
    assert r["n_pairs_total"] >= 9, r["n_pairs_total"]
    assert r["pairing_strategies"], r
    assert "same_scenario_population_independent_reference" in \
        r["pairing_strategies"]
    # reference（A2）未注入条件化闭包 ⇒ 不得出现"reference 已路由"
    assert r["n_pairs_reference_routed"] == 0
    assert r["n_pairs_conditioned_routed"] > 0
    # 条件化确实改变了 search 层 top（并在部分场景改变了最终结果）
    assert r["n_pairs_search_top_changed"] > 0
    assert r["all_runs_classified"] is True
    assert r["fail_closed_preserved"] is True


@pytestmark_records
def test_t8_feasible_domain_effect_isolated_within_conditioned_branch():
    """B1→B2（同 population 配对）隔离 feasible-domain 的作用：
    它把**结构性**域不匹配转成**真**同域分歧，而不是把证据变强。"""
    summ = json.loads(SUMMARY.read_text(encoding="utf-8"))
    r = summ["evidence_response_feasible_domain_effect"]
    assert r["pairing_strategies"] == ["same_population"]
    assert r["n_pairs_total"] == 45
    by_mode = summ["summary_by_mode"]["by_mode"]
    b1 = by_mode["B1_conditioned_search_frozen_assembly"]
    b2 = by_mode["B2_conditioned_search_feasible_assembly"]
    # B1 的弃权全部是结构性域不匹配；B2 的弃权全部是真同域分歧
    assert set(b1["abstain_kinds"]) == {"structural_domain_mismatch"}
    assert set(b2["abstain_kinds"]) == {"true_in_domain_disagreement"}
    assert b1["abstain_count"] >= b2["abstain_count"]


@pytestmark_records
def test_t8_conditioned_branch_coverage_is_lower_than_p13_configuration():
    """**本阶段最重要的诚实结论**：把 P9 条件化 search 真正路由后，
    最终决策覆盖显著低于 P13 报告的冻结-search 配置。不得混为一谈。"""
    summ = json.loads(SUMMARY.read_text(encoding="utf-8"))
    by_mode = summ["summary_by_mode"]["by_mode"]
    a2 = by_mode["A2_frozen_search_feasible_assembly"]["final_decision_coverage"]
    b2 = by_mode["B2_conditioned_search_feasible_assembly"]["final_decision_coverage"]
    b1 = by_mode["B1_conditioned_search_frozen_assembly"]["final_decision_coverage"]
    assert a2 > b2, (a2, b2)
    assert b2 >= b1, (b1, b2)
    # 完整链路（B2）的 ACCEPT 无一来自"search 消费自身条件化权益"以外的路径时，
    # 必须能从缺口里读到该事实
    assert "G2_conditioning_routing_collapses_accept" in \
        summ["capability_gaps"]["gap_codes"]


@pytestmark_records
def test_t8_capability_gaps_measured_and_quality_not_provable():
    summ = json.loads(SUMMARY.read_text(encoding="utf-8"))
    g = summ["capability_gaps"]
    for code in ("G2_conditioning_routing_collapses_accept",
                 "G5_decision_quality_not_provable_model_internal",
                 "G6_river_terminal_never_reaches_assembly"):
        assert code in g["gap_codes"], g["gap_codes"]
    assert g["decision_quality_provable_with_model_internal_evidence"] is False
    assert g["structural_boundaries"] == ["G6_river_terminal_never_reaches_assembly"]
    assert summ["capability_gaps_quality"]["must_not_be_read_as"]


@pytestmark_records
def test_t8_population_dimension_evidence_volume_is_measured():
    """要求 3/9：population 维度能提供的**独立**证据量必须如实计量。
    实测结论：条件化生效的 scenario 上 5 个种群给出 **5 个不同**签名
    （维度**未**退化）—— 这是实测反证，不代表方向正确。"""
    summ = json.loads(SUMMARY.read_text(encoding="utf-8"))
    n_pop = summ["n_declared_populations"]
    assert n_pop == 5
    per_scn = summ["evidence_response"]["per_scenario"]
    prepared = [s for s in per_scn
                if any(r["scenario"] == s and r["mode"] == COND_MODE
                       and r.get("prepared_present")
                       for r in json.loads(RECORDS.read_text(encoding="utf-8")))]
    assert len(prepared) >= 8, prepared
    assert all(per_scn[s]["n_distinct_conditioned_signatures"] == n_pop
               for s in prepared), {
        s: per_scn[s]["n_distinct_conditioned_signatures"] for s in prepared}
    assert "G4_population_dimension_not_degenerate" in \
        summ["capability_gaps"]["gap_codes"]
    assert summ["capability_gaps"]["evidence_boundaries"] == \
        ["G4_population_dimension_not_degenerate"]
