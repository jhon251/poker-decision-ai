"""V3I-P15 — B2 覆盖率根因审计与最小反事实的验证（测试）。

本文件验证 V3I-P15 只读审计层 `cash_b2_root_cause_audit`：
非 ACCEPT 分解、`leaf_coverage` 根因、四个最小反事实、工程/质量分离、冻结边界与
实测守护（禁止把反事实当成配置变更）。

环境约束：本沙箱禁止 `%TEMP%` 写入，`tmp_path`/`ctx`/`runtime` fixture 会抛
`PermissionError [WinError 5]`，故本文件不使用需要临时目录的 fixture。
"""
from __future__ import annotations

import ast
import hashlib
import json
import pathlib
import re
import sys

import pytest

sys.path.insert(0, ".")

from poker_decision_ai.cash_b2_root_cause_audit import (
    B2_MODE,
    CLAIM_BOUNDARY,
    EXPECTED_BOARD,
    LEAF_COVERAGE_CAUSES,
    NON_ACCEPT_REASON_CODES,
    VALID_EQUITY_BOARD_SIZES,
    analyze_conditioning_top_flip,
    audit_leaf_anatomy,
    break_even_conditioned_equity,
    classify_leaf_cause,
    counterfactual_admit_search_top,
    counterfactual_restrict_to_expansion,
    decompose_non_accept,
    decompose_run,
    p15_contract,
    separate_engineering_vs_quality,
    stable_fingerprint,
    summarize_admission_counterfactual,
)

MODULE = pathlib.Path("poker_decision_ai/cash_b2_root_cause_audit.py")
EVIDENCE = pathlib.Path(".tmp/v3i_p15_evidence.json")


# ---------------------------------------------------------------------------
# 合成记录
# ---------------------------------------------------------------------------
def _rec(**over) -> dict:
    base = {
        "run_id": "r1", "scenario": "flop_norm", "street": "flop", "seed": 11,
        "pot_bb": 1.5, "stack_bb": 100.0, "population": "DECLARED_BALANCED",
        "mode": B2_MODE, "present": True, "reason": None,
        "loop_status": "UNKNOWN", "abstained": False, "preferred": None,
        "disagreements": [], "n_scorers_alignable": 0, "scorer_tops": {},
        "rejections": [], "feasible_applied": True, "feasible_keys": [
            ["check", 0.0], ["bet", 0.495]],
        "tree_root_children": [{"action": "check", "amount_bb": 0.0},
                               {"action": "bet", "amount_bb": 0.495}],
        "search_candidates": [
            {"action": "check", "amount_bb": 0.0, "future_equity_mean": 0.83,
             "search_score": 0.83, "n_leaves_evaluable": 73,
             "n_leaves_total": 73, "leaf_coverage": 1.0},
            {"action": "bet", "amount_bb": 0.495, "future_equity_mean": 0.665,
             "search_score": 0.7739, "n_leaves_evaluable": 73,
             "n_leaves_total": 73, "leaf_coverage": 1.0}],
        "search_preferred": {"action": "check", "amount_bb": 0.0,
                             "search_score": 0.83},
        "unknown": ["likelihood_model_not_supplied"],
        "stage_trace": [],
    }
    base.update(over)
    return base


def _leaf(**over) -> dict:
    base = {"leaf_id": "l1", "board": ["2s"], "future_board_state": "enumerated",
            "terminal_reason": "no_further_expansion_after_deal",
            "equity_present": False,
            "equity_source": None}
    base.update(over)
    return base


# ---------------------------------------------------------------------------
# T1. 只读契约与冻结边界（要求 5、6、9）
# ---------------------------------------------------------------------------
def test_t1_contract_declares_no_mechanism_and_no_threshold_change():
    c = p15_contract()
    assert c["adds_strategy_mechanism"] is False
    assert c["modifies_frozen_contract"] is False
    assert c["modifies_budget_depth_or_cap"] is False
    assert c["relaxes_leaf_coverage_threshold"] is False
    assert c["enables_override"] is False
    assert c["changes_any_layer_output"] is False
    assert c["counterfactuals_are_read_only_recomputations"] is True
    assert c["imports_project_layers"] is False
    assert c["b2_mode"] == B2_MODE
    assert tuple(c["non_accept_reason_codes"]) == NON_ACCEPT_REASON_CODES
    assert tuple(c["leaf_coverage_causes"]) == LEAF_COVERAGE_CAUSES


def test_t1_claim_boundary_all_false():
    for k, v in CLAIM_BOUNDARY.items():
        assert v is False, k
    for flag in ("is_gto", "is_equilibrium", "is_ev", "is_profit_evidence",
                 "is_external_validation", "modifies_frozen_contract",
                 "modifies_budget_depth_or_cap",
                 "relaxes_leaf_coverage_threshold", "enables_override",
                 "changes_any_layer_output"):
        assert flag in CLAIM_BOUNDARY


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


def test_t1_no_banned_calls_or_random():
    tree = ast.parse(MODULE.read_text(encoding="utf-8"))
    banned = {"setattr", "delattr", "globals", "locals", "eval", "exec",
              "__import__", "compile", "open", "input"}
    called = {n.func.id for n in ast.walk(tree)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert not (called & banned), called & banned
    imported = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import)
                for a in n.names}
    assert "random" not in imported


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
}


def test_t1_protected_layers_untouched():
    """要求 5：不得修改 Solver / V3H-4 / V3H-6 / CFI·λ / strategy_core /
    P13 契约 / P1 loop / river terminal。"""
    frozen = {
        "poker_decision_ai/cash_strategy_decision_assembly.py":
            "3732314ab0182063",
        "poker_decision_ai/cash_feasible_domain_wiring.py": "42a90a9bdff15a2b",
        "poker_decision_ai/cash_decision_loop.py": "7c4b0a11c1e73a7f",
        # 历史 pin（V3H-7）：search_tree 5ea3854c2bf65631 / search_eval
        # 513ba523781c2c07 —— 历史值保留于 PROTECTED_LAYER_PIN_HISTORY，未删除
        "poker_decision_ai/cash_strategy_search_tree.py": "5f5b29be063a6f5a",
        "poker_decision_ai/cash_strategy_search_eval.py": "e97ccea19d2ffd92",
        "poker_decision_ai/domain/poker/strategy_core.py": "378a68671ff09e3f",
        "poker_decision_ai/cash_strategy_dynamic_belief.py": "01549df2a3385d01",
        "poker_decision_ai/cash_strategy_likelihood_model.py":
            "df207954ade71fa6",
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


def test_t1_search_budget_and_coverage_threshold_unchanged():
    from poker_decision_ai.cash_strategy_search_tree import DEFAULT_BUDGET
    from poker_decision_ai.cash_strategy_decision_assembly import (
        ASSEMBLY_POLICY)
    assert DEFAULT_BUDGET == {"max_street_depth": 2, "max_response_depth": 1,
                              "max_candidate_actions": 6, "max_leaves": 200,
                              "max_runout_children": 8}
    # 要求 6：**不得**降低 0.8 门槛
    assert float(ASSEMBLY_POLICY["min_leaf_coverage"]) == 0.8
    assert ASSEMBLY_POLICY["override_default"] is False
    p = p15_contract()
    assert p["relaxes_leaf_coverage_threshold"] is False


def test_t1_valid_equity_board_sizes_match_frozen_guard():
    """本层声明的白名单必须与冻结 `_leaf_equity` 的源码判据**一致**（实测核对）。"""
    src = pathlib.Path(
        "poker_decision_ai/cash_strategy_search_tree.py").read_text(
        encoding="utf-8")
    m = re.search(r"len\(board\)\s+not\s+in\s+\(([^)]*)\)", src)
    assert m, "未找到冻结的 board 张数判据"
    frozen = tuple(int(x.strip()) for x in m.group(1).split(","))
    assert frozen == VALID_EQUITY_BOARD_SIZES, (frozen,
                                                VALID_EQUITY_BOARD_SIZES)
    assert EXPECTED_BOARD == {"preflop": 0, "flop": 3, "turn": 4, "river": 5}
    exp = re.search(r"_EXPECTED_BOARD[^=]*=\s*\{([^}]*)\}", src)
    assert exp, "未找到冻结的 _EXPECTED_BOARD"
    assert "preflop" in exp.group(1) and "river" in exp.group(1)


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
    assert "cash_b2_root_cause_audit" not in found


# ---------------------------------------------------------------------------
# T2. B2 非 ACCEPT 分解（要求 1）
# ---------------------------------------------------------------------------
def test_t2_river_terminal_blocked_reason():
    r = _rec(street="river", scenario="river_norm", present=False,
             reason="stage_failed:action_policy", loop_status="BLOCKED",
             abstained=False, preferred=None,
             unknown=["stage_failed:action_policy",
                      "terminal_or_no_legal_actions"])
    d = decompose_run(record=r)
    assert d["outcome"] == "BLOCKED"
    assert d["reason_codes"] == ["river_terminal_pre_search_blocked"]
    assert d["detail"]["blocking_stage"] == "action_policy"


def test_t2_true_disagreement_reason_and_pair():
    r = _rec(abstained=True, preferred=None,
             disagreements=[{"tier_a": "search", "tier_b": "ranking",
                             "key_a": ["check", 0.0],
                             "key_b": ["bet", 0.495]}])
    d = decompose_run(record=r)
    assert d["outcome"] == "ABSTAIN"
    assert "cross_scorer_true_disagreement" in d["reason_codes"]
    assert d["detail"]["tier_pairs"] == ["search|ranking"]
    assert d["detail"]["key_patterns"] == ["['check', 0.0] vs ['bet', 0.495]"]


def test_t2_single_tier_accept_is_labelled_not_dressed_up():
    """要求 6：单层 ranking ACCEPT 必须**明确标注**，不得当成条件化决策。"""
    r = _rec(preferred={"action": "bet", "amount_bb": 0.495,
                        "selected_by_tier": "ranking"},
             n_scorers_alignable=1,
             scorer_tops={"ranking": {"action": "bet", "amount_bb": 0.495}})
    d = decompose_run(record=r)
    assert d["outcome"] == "ACCEPT"
    assert d["is_single_tier_accept"] is True
    assert d["is_cross_scorer_accept"] is False
    assert "single_tier_decision" in d["reason_codes"]


def test_t2_coverage_gate_reason_when_search_tier_unavailable():
    r = _rec(preferred={"action": "bet", "amount_bb": 0.495,
                        "selected_by_tier": "ranking"},
             n_scorers_alignable=1,
             scorer_tops={"ranking": {"action": "bet", "amount_bb": 0.495}},
             rejections=[{"tier": "search", "action": "bet",
                          "leaf_coverage": 0.111111,
                          "min_leaf_coverage": 0.8,
                          "reason": "leaf_coverage_below_minimum"},
                         {"tier": "search", "action": "check",
                          "leaf_coverage": 0.111111,
                          "min_leaf_coverage": 0.8,
                          "reason": "leaf_coverage_below_minimum"}])
    d = decompose_run(record=r)
    assert "search_tier_unavailable_by_coverage_gate" in d["reason_codes"]
    assert d["detail"]["n_coverage_rejections"] == 2
    assert d["detail"]["rejected_leaf_coverages"] == [0.111111]


def test_t2_cross_scorer_accept_gets_no_gap_reason():
    r = _rec(preferred={"action": "bet", "amount_bb": 0.99,
                        "selected_by_tier": "search"},
             n_scorers_alignable=2,
             scorer_tops={"search": {"action": "bet", "amount_bb": 0.99},
                          "ranking": {"action": "bet", "amount_bb": 0.99}})
    d = decompose_run(record=r)
    assert d["outcome"] == "ACCEPT"
    assert d["is_cross_scorer_accept"] is True
    assert d["reason_codes"] == []


def test_t2_decompose_non_accept_aggregates():
    recs = [
        _rec(run_id="a", preferred={"action": "bet", "amount_bb": 0.495,
                                    "selected_by_tier": "ranking"},
             n_scorers_alignable=1),
        _rec(run_id="b", abstained=True, preferred=None,
             disagreements=[{"tier_a": "search", "tier_b": "ranking",
                             "key_a": ["check", 0.0],
                             "key_b": ["bet", 0.495]}]),
        _rec(run_id="c", street="river", present=False,
             reason="stage_failed:action_policy", loop_status="BLOCKED",
             unknown=["terminal_or_no_legal_actions"]),
    ]
    d = decompose_non_accept(records=recs)
    assert d["n_runs"] == 3 and d["n_accept"] == 1 and d["n_non_accept"] == 2
    assert d["outcome_counts"] == {"ACCEPT": 1, "ABSTAIN": 1, "BLOCKED": 1,
                                   "UNKNOWN_NO_PREFERRED": 0}
    assert d["non_accept_reason_counts"] == {
        "cross_scorer_true_disagreement": 1,
        "river_terminal_pre_search_blocked": 1}
    assert d["accept_single_tier"] == 1
    assert d["disagreement_tier_pairs"] == {"search|ranking": 1}
    assert d["by_street"]["river"]["outcome_counts"]["BLOCKED"] == 1


def test_t2_decompose_fail_closed_on_bad_input():
    assert decompose_non_accept(records=[]).get("present") is False
    assert decompose_non_accept(records="nope").get("present") is False
    assert decompose_non_accept(records=[_rec()],
                                mode="NOPE").get("present") is False
    assert decompose_run(record=None).get("present") is False


# ---------------------------------------------------------------------------
# T3. leaf_coverage 根因（要求 2）
# ---------------------------------------------------------------------------
def test_t3_classify_leaf_cause_one_card_board():
    r = classify_leaf_cause(leaf=_leaf(), street="preflop")
    assert r["board_size"] == 1
    assert r["board_size_in_validity_whitelist"] is False
    assert r["expected_board_size_for_street"] == 0
    assert set(r["causes"]) == {
        "board_size_outside_validity_whitelist",
        "single_card_street_enumeration",
        "street_board_size_mismatch_blocks_further_expansion"}


def test_t3_classify_leaf_cause_budget_and_terminal_variants():
    a = classify_leaf_cause(leaf=_leaf(board=[], board_size=0,
                                       terminal_reason="max_leaves_reached",
                                       future_board_state="unavailable"),
                            street="preflop")
    assert a["causes"] == ["max_leaves_budget_reached"]
    b = classify_leaf_cause(leaf=_leaf(board=["a", "b", "c", "d", "e"],
                                       terminal_reason="river_no_future_street"),
                            street="turn")
    assert b["equity_present"] is False
    # 已可评估的叶子不附任何原因（不制造伪根因）
    c = classify_leaf_cause(leaf=_leaf(equity_present=True), street="turn")
    assert c["causes"] == []


def test_t3_audit_leaf_anatomy_separates_preflop_from_postflop():
    pre = [_leaf(leaf_id=f"p{i}") for i in range(16)]
    pre += [_leaf(leaf_id="p16", board=[], equity_present=True,
                  equity_source="arithmetic:pot_share_on_opponent_fold"),
            _leaf(leaf_id="p17", board=[], equity_present=True,
                  equity_source="arithmetic:pot_share_on_opponent_fold")]
    flop = [_leaf(leaf_id=f"f{i}", board=["a", "b", "c", "d"],
                  equity_present=True)
            for i in range(8)]
    rows = [classify_leaf_cause(leaf=l, street="preflop") for l in pre] + \
           [classify_leaf_cause(leaf=l, street="flop") for l in flop]
    a = audit_leaf_anatomy(anatomy=rows)
    assert a["streets_below_coverage_gate"] == ["preflop"]
    assert a["streets_above_coverage_gate"] == ["flop"]
    assert a["by_street"]["preflop"]["leaf_coverage_from_anatomy"] == 0.111111
    assert a["by_street"]["preflop"]["dominant_cause"] == \
        "board_size_outside_validity_whitelist"
    assert a["by_street"]["preflop"]["evaluable_equity_sources"] == [
        "arithmetic:pot_share_on_opponent_fold"]
    assert a["by_street"]["flop"]["leaf_coverage_from_anatomy"] == 1.0
    assert "只补**1 张**" in a["root_cause_statement"]
    assert any("max_leaves" in s for s in a["not_caused_by"])
    assert audit_leaf_anatomy(anatomy=[]).get("present") is False


def test_t3_root_cause_is_not_budget_or_cap():
    """要求 2：必须**排除** budget/cap/pruning 作为根因（用实测预算用量核对）。"""
    ev = json.loads(EVIDENCE.read_text(encoding="utf-8")) \
        if EVIDENCE.exists() else None
    if ev is None:
        pytest.skip("需要 .tmp/v3i_p15_evidence.json")
    for street in ("preflop", "flop", "turn"):
        bu = ev["tree_level_coverage"][street]["budget_used"]
        assert bu["leaves"] < bu["max_leaves"], street
        assert bu["root_candidates"] == bu["candidate_cap"] == 2, street
        assert ev["tree_level_coverage"][street]["budget_hits"] == \
            ["max_runout_children"], street


# ---------------------------------------------------------------------------
# T4. 反事实接纳 search top（要求 4-2）
# ---------------------------------------------------------------------------
def _gated_rec(**over):
    base = dict(
        preferred={"action": "bet", "amount_bb": 0.495,
                   "selected_by_tier": "ranking"},
        n_scorers_alignable=1,
        scorer_tops={"ranking": {"action": "bet", "amount_bb": 0.495}},
        rejections=[{"tier": "search", "action": "check",
                     "leaf_coverage": 0.111111, "min_leaf_coverage": 0.8,
                     "reason": "leaf_coverage_below_minimum"},
                    {"tier": "search", "action": "bet",
                     "leaf_coverage": 0.111111, "min_leaf_coverage": 0.8,
                     "reason": "leaf_coverage_below_minimum"}],
        search_preferred={"action": "check", "amount_bb": 0.0,
                          "search_score": 0.5125},
        search_candidates=[
            {"action": "check", "amount_bb": 0.0, "future_equity_mean": 0.5125,
             "search_score": 0.5125, "n_leaves_evaluable": 1,
             "n_leaves_total": 9, "leaf_coverage": 0.111111},
            {"action": "bet", "amount_bb": 0.495,
             "future_equity_mean": 0.4975, "search_score": 0.49585,
             "n_leaves_evaluable": 1, "n_leaves_total": 9,
             "leaf_coverage": 0.111111}],
        street="preflop", scenario="preflop_norm",
    )
    base.update(over)
    return _rec(**base)


def test_t4_admission_counterfactual_detects_gate_and_disagreement():
    c = counterfactual_admit_search_top(record=_gated_rec())
    assert c["gate_blocked_search_tier"] is True
    assert c["actual_outcome"] == "ACCEPT"
    assert c["counterfactual_outcome_if_admitted"] == "ABSTAIN"
    assert c["counterfactual_would_agree"] is False
    assert c["outcome_would_change"] is True
    assert c["counterfactual_only"] is True
    assert c["threshold_unchanged"] == 0.8
    assert c["modifies_any_layer"] is False


def test_t4_admission_flags_single_leaf_evidence():
    c = counterfactual_admit_search_top(record=_gated_rec())
    assert c["search_top_n_leaves_evaluable"] == 1
    assert c["search_top_rests_on_single_leaf"] is True
    assert c["search_leaf_coverages"] == [0.111111]


def test_t4_admission_summary_counts_and_direction():
    rows = [counterfactual_admit_search_top(record=_gated_rec(run_id=f"g{i}"))
            for i in range(6)]
    rows += [counterfactual_admit_search_top(record=_gated_rec(
        run_id="ok", search_preferred={"action": "bet", "amount_bb": 0.495,
                                       "search_score": 0.49585},
        scorer_tops={"ranking": {"action": "bet", "amount_bb": 0.495}}))]
    s = summarize_admission_counterfactual(results=rows)
    assert s["n_runs"] == 7
    assert s["n_gate_blocked"] == 7
    assert s["n_accept_to_abstain"] == 6
    assert s["n_abstain_to_accept"] == 0
    assert s["n_unchanged_accept_via_coincidental_agreement"] == 1
    assert s["n_unchanged_accept_resting_on_single_leaf"] == 1
    assert s["counterfactual_accept"] == 1
    assert s["coverage_would_rise_or_fall"] == "fall"
    assert s["threshold_unchanged"] == 0.8
    assert summarize_admission_counterfactual(results=[]).get(
        "present") is False


def test_t4_non_gated_record_has_no_counterfactual_change():
    r = _rec(preferred={"action": "bet", "amount_bb": 0.495,
                        "selected_by_tier": "search"},
             n_scorers_alignable=2,
             scorer_tops={"search": {"action": "bet", "amount_bb": 0.495},
                          "ranking": {"action": "bet", "amount_bb": 0.495}},
             search_preferred={"action": "bet", "amount_bb": 0.495})
    c = counterfactual_admit_search_top(record=r)
    assert c["gate_blocked_search_tier"] is False
    assert c["counterfactual_would_agree"] is True


# ---------------------------------------------------------------------------
# T5. 反事实限域（要求 4-3）
# ---------------------------------------------------------------------------
def test_t5_restriction_not_applicable_without_selection():
    r = _rec(abstained=True, preferred=None,
             disagreements=[{"tier_a": "search", "tier_b": "ranking",
                             "key_a": ["check", 0.0],
                             "key_b": ["bet", 0.495]}])
    c = counterfactual_restrict_to_expansion(record=r)
    assert c["applicable"] is False
    assert c["reason"] == "no_preferred_selection"
    assert c["outside_search_expansion"] is None
    assert c["counterfactual_changes_selection"] is False


def test_t5_restriction_recomputes_in_domain_top_from_ranking_rows():
    r = _rec(mode="B1_conditioned_search_frozen_assembly",
             feasible_applied=False,
             preferred={"action": "bet", "amount_bb": 100.0,
                        "selected_by_tier": "ranking"},
             n_scorers_alignable=1,
             scorer_tops={"ranking": {"action": "bet", "amount_bb": 100.0}})
    rows = [{"action": "bet", "amount_bb": 100.0, "score": 53.5},
            {"action": "bet", "amount_bb": 0.495, "score": 1.155},
            {"action": "check", "amount_bb": 0.0, "score": 0.895}]
    c = counterfactual_restrict_to_expansion(record=r, ranking_rows=rows)
    assert c["applicable"] is True
    assert c["outside_search_expansion"] is True
    assert c["ranking_rows_available"] is True
    assert c["ranking_top_within_expansion"] == ["bet", 0.495]
    assert c["counterfactual_selection"] == ["bet", 0.495]
    assert c["counterfactual_changes_selection"] is True
    assert c["actual_selected_key"] == ["bet", 100.0]


def test_t5_restriction_inside_expansion_is_a_no_op():
    r = _rec(preferred={"action": "bet", "amount_bb": 0.495,
                        "selected_by_tier": "ranking"},
             n_scorers_alignable=2,
             scorer_tops={"ranking": {"action": "bet", "amount_bb": 0.495}})
    c = counterfactual_restrict_to_expansion(record=r)
    assert c["outside_search_expansion"] is False
    assert c["counterfactual_changes_selection"] is False
    assert c["counterfactual_selection"] == ["bet", 0.495]
    assert counterfactual_restrict_to_expansion(record=None).get(
        "present") is False


# ---------------------------------------------------------------------------
# T6. 条件化 top 翻转与打平点（要求 3、4-4）
# ---------------------------------------------------------------------------
def test_t6_break_even_formula_matches_frozen_scoring():
    be = break_even_conditioned_equity(pot_bb=1.5, amount_bb=0.495,
                                       check_equity=0.8275)
    assert be["present"] is True
    assert be["bet_score_scale"] == pytest.approx(1.66, abs=1e-9)
    assert be["bet_invest_penalty"] == pytest.approx(0.33, abs=1e-9)
    assert be["break_even_bet_equity"] == pytest.approx(0.697289, abs=1e-6)
    # 解析核对：用打平权益代入冻结公式，bet 与 check 分数相等
    e = be["break_even_bet_equity"]
    score_bet = e * (1.5 + 2 * 0.495) / 1.5 - 0.495 / 1.5
    # 打平点按 6 位小数落盘；代入冻结公式后与 check 分数一致（含舍入误差）
    assert score_bet == pytest.approx(0.8275, abs=1e-6)
    assert break_even_conditioned_equity(pot_bb=0, amount_bb=1,
                                         check_equity=0.5).get(
        "present") is False
    assert break_even_conditioned_equity(pot_bb=1.5, amount_bb=None,
                                         check_equity=0.5).get(
        "present") is False


def test_t6_top_flip_explained_by_break_even():
    r = _rec(pot_bb=1.5)
    a = analyze_conditioning_top_flip(record=r)
    assert a["present"] is True
    assert a["conditioning_flipped_search_top"] is True
    assert a["bet_below_break_even"] is True
    assert a["gap_to_break_even"] == pytest.approx(0.033795, abs=1e-6)
    assert a["both_candidates_in_feasible_domain"] is True
    assert "P13" in a["p13_contract_is_not_the_cause"] or \
        "可行域" in a["p13_contract_is_not_the_cause"]


def test_t6_top_flip_absent_when_bet_above_break_even():
    r = _rec(pot_bb=1.5, search_candidates=[
        {"action": "check", "amount_bb": 0.0, "future_equity_mean": 0.70,
         "search_score": 0.70},
        {"action": "bet", "amount_bb": 0.495, "future_equity_mean": 0.80,
         "search_score": 0.998}],
        search_preferred={"action": "bet", "amount_bb": 0.495})
    a = analyze_conditioning_top_flip(record=r)
    assert a["conditioning_flipped_search_top"] is False
    assert a["bet_below_break_even"] is False
    assert a["gap_to_break_even"] < 0


def test_t6_top_flip_reports_unconditioned_reference():
    r = _rec(pot_bb=1.5)
    ref = _rec(search_candidates=[
        {"action": "check", "amount_bb": 0.0, "future_equity_mean": 0.796495},
        {"action": "bet", "amount_bb": 0.495,
         "future_equity_mean": 0.796495}])
    a = analyze_conditioning_top_flip(record=r, reference_record=ref)
    assert a["bet_unconditioned_equity_mean_reference"] == 0.796495
    assert a["bet_equity_mean"] == 0.665
    assert analyze_conditioning_top_flip(record=_rec(
        search_candidates=[{"action": "check", "amount_bb": 0.0,
                            "future_equity_mean": 0.5}])).get(
        "present") is False


# ---------------------------------------------------------------------------
# T7. 工程覆盖率问题 vs 决策质量问题（要求 8）
# ---------------------------------------------------------------------------
def test_t7_separation_reports_both_dimensions_distinctly():
    dec = decompose_non_accept(records=[
        _gated_rec(run_id="a"),
        _rec(run_id="b", abstained=True, preferred=None,
             disagreements=[{"tier_a": "search", "tier_b": "ranking",
                             "key_a": ["check", 0.0],
                             "key_b": ["bet", 0.495]}]),
        _rec(run_id="c", street="river", present=False,
             reason="stage_failed:action_policy", loop_status="BLOCKED"),
    ])
    sep = separate_engineering_vs_quality(decomposition=dec)
    e = sep["engineering_coverage_problem"]
    q = sep["decision_quality_problem"]
    assert e["measured"]["n_runs"] == 3
    assert e["measured"]["n_accept"] == 1
    assert e["is_engineering_fix_possible_within_frozen_boundary"] is False
    assert e["attribution"]["cross_scorer_true_disagreement"] == 1
    assert e["attribution"]["river_terminal"] == 1
    assert e["attribution"]["single_tier_decision_among_accepts"] == 1
    assert q["provable_with_model_internal_evidence"] is False
    assert q["measured_proxies"]["single_tier_share_of_accepts"] == 1.0
    assert "不等于" in sep["must_not_conflate"]
    assert separate_engineering_vs_quality(decomposition=None).get(
        "present") is False


# ---------------------------------------------------------------------------
# T8. 实测守护（要求 1、2、3、4、8、10）
# ---------------------------------------------------------------------------
pytestmark_ev = pytest.mark.skipif(
    not EVIDENCE.exists(), reason="需要 .tmp/v3i_p15_evidence.json")


@pytestmark_ev
def test_t8_b2_decomposition_measured():
    ev = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    d = ev["b2_decomposition"]
    assert d["n_runs"] == 45
    assert d["outcome_counts"] == {"ACCEPT": 12, "ABSTAIN": 28, "BLOCKED": 5,
                                   "UNKNOWN_NO_PREFERRED": 0}
    assert d["n_non_accept"] == 33
    assert d["non_accept_reason_counts"] == {
        "cross_scorer_true_disagreement": 28,
        "river_terminal_pre_search_blocked": 5}
    assert d["accept_cross_scorer"] == 2
    assert d["accept_single_tier"] == 10
    assert d["disagreement_tier_pairs"] == {"search|ranking": 28}


@pytestmark_ev
def test_t8_leaf_coverage_cause_is_preflop_only_and_not_budget():
    ev = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    la = ev["leaf_coverage_audit"]
    assert la["streets_below_coverage_gate"] == ["preflop"]
    assert set(la["streets_above_coverage_gate"]) == {"flop", "turn"}
    pre = la["by_street"]["preflop"]
    assert pre["n_leaves"] == 18 and pre["n_evaluable"] == 2
    assert pre["leaf_coverage_from_anatomy"] == 0.111111
    assert pre["board_sizes"] == [0, 1]
    assert pre["dominant_cause"] == "board_size_outside_validity_whitelist"
    # 唯一可评估来源是"对手弃牌"的算术终局叶，**没有**真实权益计算
    assert pre["evaluable_equity_sources"] == [
        "arithmetic:pot_share_on_opponent_fold"]
    # 明确排除 budget / cap / pruning
    assert any("max_leaves" in s for s in la["not_caused_by"])
    for street in ("preflop", "flop", "turn"):
        bu = ev["tree_level_coverage"][street]["budget_used"]
        assert bu["leaves"] < bu["max_leaves"]


@pytestmark_ev
def test_t8_admission_counterfactual_measured_direction():
    """要求 4-2：放宽覆盖率门槛**不会**提高 B2 覆盖（实测方向）。"""
    ev = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    a = ev["admission_summary"]
    assert a["n_gate_blocked"] == 10
    assert a["n_accept_to_abstain"] == 6
    assert a["n_abstain_to_accept"] == 0
    assert a["n_unchanged_accept_resting_on_single_leaf"] == 4
    assert a["actual_accept"] == 12 and a["counterfactual_accept"] == 6
    assert a["coverage_would_rise_or_fall"] == "fall"
    assert a["threshold_unchanged"] == 0.8
    assert a["scenarios_gate_blocked"] == ["preflop_norm", "preflop_short"]


@pytestmark_ev
def test_t8_top_flip_is_exactly_explained_by_break_even():
    """要求 3：条件化翻转 search top 的每一例都由解析打平点解释。"""
    ev = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    t = ev["conditioning_top_flip_summary"]
    assert t["n_runs"] == 30
    assert t["n_flipped_to_check"] == 28
    assert t["n_bet_below_break_even"] == 28
    assert t["flip_is_exactly_explained_by_break_even"] is True
    assert t["all_candidates_in_feasible_domain"] is True
    assert t["n_non_flipped"] == 2
    for r in t["non_flipped_runs"]:
        assert r["scenario"] == "flop_board23"
        assert r["gap"] < 0


@pytestmark_ev
def test_t8_true_disagreement_both_keys_in_feasible_domain():
    """要求 4-4：28 条真分歧的两侧 key **都**在可行域内 ⇒ 不是 P13 契约造成的。"""
    ev = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    s = ev["true_disagreement_summary"]
    assert s["n_disagreements"] == 28
    assert s["n_both_keys_in_domain"] == 28
    assert s["all_both_in_domain"] is True
    assert all("check" in p for p in s["distinct_patterns"])


@pytestmark_ev
def test_t8_restriction_counterfactual_measured():
    ev = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    s = ev["restriction_summary"]
    assert s["n_applicable_with_selection"] == 12
    assert s["n_outside_expansion"] == 12
    assert s["n_inside_expansion"] == 0
    assert s["n_would_change_selection"] == 12
    for p in s["pairs"]:
        assert p["actual"][1] not in [k[1] for k in p["expanded"]]
        assert p["counterfactual_selection"][1] in [k[1] for k in p["expanded"]]


@pytestmark_ev
def test_t8_engineering_vs_quality_measured_and_not_conflated():
    ev = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    sep = ev["engineering_vs_quality"]
    e = sep["engineering_coverage_problem"]
    assert e["measured"]["coverage"] == 0.266667
    assert e["is_engineering_fix_possible_within_frozen_boundary"] is False
    assert "冻结" in e["why_not"]
    q = sep["decision_quality_problem"]
    assert q["provable_with_model_internal_evidence"] is False
    assert q["measured_proxies"]["n_cross_scorer_verified_accept"] == 2
    assert q["measured_proxies"]["n_single_tier_accept"] == 10
    assert sep["admission_counterfactual_summary"]["n_accept_to_abstain"] == 6


@pytestmark_ev
def test_t8_determinism_and_evidence_fingerprints():
    ev = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    d = ev["determinism"]
    assert d["deterministic"] is True
    assert d["decomposition_fingerprint_1"] == \
        d["decomposition_fingerprint_2"]
    assert d["admission_fingerprint_1"] == d["admission_fingerprint_2"]
    assert d["run_decomposition_fingerprint_1"] == \
        d["run_decomposition_fingerprint_2"]
    # 同输入重算 → 同指纹（本层纯函数）
    assert stable_fingerprint(ev["b2_decomposition"]) == \
        stable_fingerprint(json.loads(json.dumps(ev["b2_decomposition"])))


@pytestmark_ev
def test_t8_evidence_declares_no_layer_modification():
    ev = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    c = ev["p15_contract"]
    assert c["adds_strategy_mechanism"] is False
    assert c["modifies_frozen_contract"] is False
    assert c["modifies_budget_depth_or_cap"] is False
    assert c["relaxes_leaf_coverage_threshold"] is False
    assert c["changes_any_layer_output"] is False
    for r in ev["admission_counterfactual"]:
        assert r["counterfactual_only"] is True
        assert r["modifies_any_layer"] is False
        assert r["threshold_unchanged"] == 0.8
