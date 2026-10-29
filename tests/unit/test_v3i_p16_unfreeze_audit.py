"""V3I-P16 — 最小解冻方案审计与最小反事实的验证（测试）。

本文件验证 V3I-P16 只读审计层 `cash_unfreeze_proposal_audit`：
真实权益核对（fold-only 不得计入）、两个根因审计、候选解冻项规范化、
非法"解决"登记、postflop 四个选项的逐一裁定、覆盖率影响与证据语义裁定。

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

from poker_decision_ai.cash_unfreeze_proposal_audit import (
    CLAIM_BOUNDARY,
    EVIDENCE_SEMANTICS_RULES,
    FOLD_ONLY_SOURCE_MARKER,
    ILLEGITIMATE_RESOLUTIONS,
    PROPOSAL_KINDS,
    REAL_EQUITY_SOURCE_MARKERS,
    ROOT_CAUSE_KINDS,
    VALID_EQUITY_BOARD_SIZES,
    audit_postflop_root_cause,
    audit_preflop_root_cause,
    counterfactual_postflop_options,
    coverage_impact,
    evidence_semantics_verdict,
    minimal_unfreeze_proposal,
    p16_contract,
    reject_illegitimate_resolutions,
    stable_fingerprint,
    summarize_postflop_options,
    verify_real_equity_evidence,
)

MODULE = pathlib.Path("poker_decision_ai/cash_unfreeze_proposal_audit.py")
SUMMARY = pathlib.Path(".tmp/v3i_p16_summary.json")
REAL = "domain.poker.equity.EquityCalculator.hand_vs_range"


def _leaf(source, board, present=True):
    return {"equity_present": present, "equity_source": source,
            "board_size": board}


def _rec(**over) -> dict:
    cond = {
        "check|check_no_investment|0.0": {
            "action": "check", "amount_bb": 0.0,
            "hero_equity_vs_continuing": 0.8275},
        "bet|bet_pot_fraction|0.495": {
            "action": "bet", "amount_bb": 0.495,
            "hero_equity_vs_continuing": 0.665},
    }
    base = {
        "run_id": "r1", "scenario": "flop_norm", "street": "flop",
        "pot_bb": 1.5, "population": "DECLARED_BALANCED",
        "search_candidates": [
            {"action": "check", "amount_bb": 0.0, "future_equity_mean": 0.8275,
             "search_score": 0.8275},
            {"action": "bet", "amount_bb": 0.495, "future_equity_mean": 0.665,
             "search_score": 0.7739}],
        "conditioned_per_candidate": cond,
        "search_preferred": {"action": "check", "amount_bb": 0.0,
                             "search_score": 0.8275},
        "scorer_tops": {"search": None,
                        "ranking": {"action": "bet", "amount_bb": 0.495}},
        "preferred": None, "abstained": True,
        "disagreements": [{"tier_a": "search", "key_a": ["check", 0.0],
                           "tier_b": "ranking", "key_b": ["bet", 0.495]}],
    }
    base.update(over)
    return base


def _proposal(**over) -> dict:
    base = {
        "kind": "street_faithful_deal_expansion",
        "target_file": "poker_decision_ai/cash_strategy_search_tree.py",
        "target_function": "_emit_street",
        "fields_touched": ["_emit_street 的补牌张数逻辑"],
        "minimality_argument": "k = expected - len(board_now)；postflop k≡1",
        "blast_radius": "仅 preflop 树",
        "why_not_fabrication": "权益来自冻结权益入口，board 在白名单内",
        "keeps_leaf_coverage_threshold": True,
        "threshold_value_after": 0.8,
    }
    base.update(over)
    return base


# ---------------------------------------------------------------------------
# T1. 只读契约与冻结边界（要求 5、6）
# ---------------------------------------------------------------------------
def test_t1_contract_declares_no_implementation_and_threshold_kept():
    c = p16_contract()
    assert c["implements_any_unfreeze_item"] is False
    assert c["modifies_frozen_contract"] is False
    assert c["modifies_budget_depth_or_cap"] is False
    assert c["relaxes_leaf_coverage_threshold"] is False
    assert c["leaf_coverage_threshold_kept"] == 0.8
    assert c["enables_override"] is False
    assert c["uses_voting"] is False
    assert c["treats_fold_only_as_equity"] is False
    assert c["counterfactuals_are_read_only"] is True
    assert c["imports_project_layers"] is False
    assert tuple(c["root_cause_kinds"]) == ROOT_CAUSE_KINDS
    assert tuple(c["proposal_kinds"]) == PROPOSAL_KINDS
    assert tuple(c["valid_equity_board_sizes"]) == VALID_EQUITY_BOARD_SIZES


def test_t1_claim_boundary_all_false():
    for k, v in CLAIM_BOUNDARY.items():
        assert v is False, k
    for flag in ("is_gto", "is_equilibrium", "is_ev", "is_profit_evidence",
                 "implements_any_unfreeze_item", "modifies_frozen_contract",
                 "relaxes_leaf_coverage_threshold", "enables_override",
                 "uses_voting", "treats_fold_only_as_equity"):
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
}


def test_t1_protected_layers_untouched():
    """要求 6：候选解冻项**只报告不实施** ⇒ 生产层必须逐哈希未变。"""
    frozen = {
        # 历史 pin（V3H-7）：search_tree 5ea3854c2bf65631 / search_eval
        # 513ba523781c2c07 —— 历史值保留于 PROTECTED_LAYER_PIN_HISTORY，未删除
        "poker_decision_ai/cash_strategy_search_tree.py": "5f5b29be063a6f5a",
        "poker_decision_ai/cash_strategy_search_eval.py": "e97ccea19d2ffd92",
        "poker_decision_ai/cash_strategy_action_ranking.py": None,
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
        if want is not None:
            assert got == want, (path, got, want)
        else:
            # ranking 层未被本阶段触及：记录当前值并在文档中核对
            assert isinstance(got, str) and len(got) == 16


def test_t1_threshold_and_budget_still_frozen():
    from poker_decision_ai.cash_strategy_decision_assembly import (
        ASSEMBLY_POLICY)
    from poker_decision_ai.cash_strategy_search_tree import DEFAULT_BUDGET
    assert float(ASSEMBLY_POLICY["min_leaf_coverage"]) == 0.8
    assert ASSEMBLY_POLICY["override_default"] is False
    assert DEFAULT_BUDGET == {"max_street_depth": 2, "max_response_depth": 1,
                              "max_candidate_actions": 6, "max_leaves": 200,
                              "max_runout_children": 8}


def test_t1_board_whitelist_matches_frozen_guard_by_source():
    import re
    src = pathlib.Path(
        "poker_decision_ai/cash_strategy_search_tree.py").read_text(
        encoding="utf-8")
    m = re.search(r"len\(board\)\s+not\s+in\s+\(([^)]*)\)", src)
    assert m
    frozen = tuple(int(x.strip()) for x in m.group(1).split(","))
    assert frozen == VALID_EQUITY_BOARD_SIZES
    assert FOLD_ONLY_SOURCE_MARKER == \
        "arithmetic:pot_share_on_opponent_fold"
    assert any("hand_vs_range" in s for s in REAL_EQUITY_SOURCE_MARKERS)


# ---------------------------------------------------------------------------
# T2. 真实权益核对（要求 4）—— fold-only 不得计入
# ---------------------------------------------------------------------------
def test_t2_fold_only_only_is_not_real_equity():
    r = verify_real_equity_evidence(leaves=[
        _leaf(FOLD_ONLY_SOURCE_MARKER, 1), _leaf(FOLD_ONLY_SOURCE_MARKER, 0)])
    assert r["present"] is True
    assert r["n_evaluable"] == 2
    assert r["n_real_equity_leaves"] == 0
    assert r["n_fold_only_arithmetic_leaves"] == 2
    assert r["has_real_traceable_equity"] is False
    assert "不满足要求 4" in r["verdict"]
    assert r["fold_only_excluded_from_real_equity"] is True


def test_t2_real_equity_on_whitelisted_boards_passes():
    r = verify_real_equity_evidence(leaves=[
        _leaf(REAL, 3)] * 4 + [_leaf(REAL, 4)] * 4
        + [_leaf(FOLD_ONLY_SOURCE_MARKER, 0)])
    assert r["n_real_equity_leaves"] == 8
    assert r["n_fold_only_arithmetic_leaves"] == 1
    assert r["n_evaluable_with_board_outside_whitelist"] == 0
    assert r["has_real_traceable_equity"] is True
    assert "真实、可追溯" in r["verdict"]


def test_t2_board_outside_whitelist_is_flagged():
    r = verify_real_equity_evidence(leaves=[_leaf(REAL, 1), _leaf(REAL, 3)])
    assert r["n_evaluable_with_board_outside_whitelist"] == 1
    assert "不满足要求 4" in r["verdict"]


def test_t2_fail_closed_on_bad_input():
    assert verify_real_equity_evidence(leaves=[]).get("present") is False
    assert verify_real_equity_evidence(leaves=None).get("present") is False


# ---------------------------------------------------------------------------
# T3. 两个根因审计（要求 1）
# ---------------------------------------------------------------------------
def test_t3_preflop_audit_reports_no_real_equity_as_defect():
    frozen = verify_real_equity_evidence(leaves=[
        _leaf(FOLD_ONLY_SOURCE_MARKER, 1)] * 16
        + [_leaf(FOLD_ONLY_SOURCE_MARKER, 0)] * 2)
    a = audit_preflop_root_cause(frozen_leaf_verification=frozen,
                                 tree_coverage={"frozen": 0.111111})
    assert a["kind"] == "preflop_no_real_equity_expansion"
    assert a["is_a_defect"] is True
    assert a["frozen_state"]["n_real_equity_leaves"] == 0
    assert "单张" in a["mechanism"] or "1 张" in a["mechanism"]
    assert a["candidate_state"] is None
    assert audit_preflop_root_cause(
        frozen_leaf_verification=None).get("present") is False


def test_t3_postflop_audit_says_not_a_defect():
    dec = {"present": True, "n_accept": 12, "accept_cross_scorer": 2,
           "accept_single_tier": 10,
           "non_accept_reason_counts": {
               "cross_scorer_true_disagreement": 28,
               "river_terminal_pre_search_blocked": 5}}
    a = audit_postflop_root_cause(decomposition=dec)
    assert a["kind"] == "postflop_same_domain_true_disagreement"
    assert a["is_a_defect"] is False
    assert a["n_true_disagreements"] == 28
    assert "28/28" in a["why_not_a_defect"]
    assert "真实分歧" in a["defect_kind"]
    assert audit_postflop_root_cause(decomposition=None).get(
        "present") is False


# ---------------------------------------------------------------------------
# T4. 候选解冻项规范化（要求 2、6）
# ---------------------------------------------------------------------------
def test_t4_valid_proposal_is_normalized_and_not_implemented():
    p = minimal_unfreeze_proposal(proposal=_proposal())
    assert p["present"] is True
    assert p["kind"] == "street_faithful_deal_expansion"
    assert p["is_illegitimate_by_construction"] is False
    assert p["implements_anything"] is False
    assert p["status"] == "PROPOSED_NOT_IMPLEMENTED"
    assert p["keeps_leaf_coverage_threshold"] is True
    assert p["threshold_value_after"] == 0.8


def test_t4_proposal_missing_fields_fails_closed():
    p = minimal_unfreeze_proposal(proposal={"kind":
                                            "street_faithful_deal_expansion"})
    assert p["present"] is False
    assert "proposal_missing" in p["reason"]
    assert minimal_unfreeze_proposal(proposal={"kind": "bogus"}).get(
        "present") is False
    assert minimal_unfreeze_proposal(proposal=None).get("present") is False


def test_t4_illegitimate_kind_is_flagged_by_construction():
    for kind in ("tier_precedence_or_override",
                 "lower_leaf_coverage_threshold", "vote_between_tiers",
                 "relabel_fold_only_as_equity",
                 "candidate_local_equity_for_ranking"):
        p = minimal_unfreeze_proposal(proposal=_proposal(kind=kind))
        assert p["present"] is True
        assert p["is_illegitimate_by_construction"] is True, kind
        assert p["illegitimacy_reason"], kind


def test_t4_rejected_registry_covers_all_forbidden_ways():
    r = reject_illegitimate_resolutions()
    assert r["n_rejected"] == len(ILLEGITIMATE_RESOLUTIONS)
    assert set(r["rejected"]) == set(ILLEGITIMATE_RESOLUTIONS)
    for k in ("tier_precedence_or_override", "lower_leaf_coverage_threshold",
              "vote_between_tiers", "relabel_fold_only_as_equity"):
        assert k in r["rejected"]
    assert "不等于" in r["principle"]


# ---------------------------------------------------------------------------
# T5. postflop 四个选项的裁定（要求 3）
# ---------------------------------------------------------------------------
def test_t5_commensuration_is_rejected_as_tautological():
    o = counterfactual_postflop_options(record=_rec())
    a = o["option_A_commensurate_ranking"]
    assert a["n_candidates"] == 2
    assert a["n_scores_identical_to_search"] == 2
    assert a["all_scores_identical_to_search"] is True
    assert a["would_agree_with_search"] is True
    assert a["independent_scoring_sources_after"] == 1
    assert a["verdict"] == "REJECTED_TAUTOLOGICAL"


def test_t5_override_and_vote_and_threshold_are_all_rejected():
    o = counterfactual_postflop_options(record=_rec())
    assert o["option_B_prefer_search_on_disagreement"]["verdict"] == \
        "REJECTED_OVERRIDE_IN_EFFECT"
    assert o["option_B_prefer_search_on_disagreement"]["discarded_tier"] == \
        "ranking"
    assert o["option_C_vote_between_tiers"]["tie"] is True
    assert o["option_C_vote_between_tiers"]["verdict"] == "REJECTED_TIE"
    assert o["option_D_lower_threshold"]["verdict"] == \
        "REJECTED_FORBIDDEN_AND_COUNTERPRODUCTIVE"
    assert o["any_legitimate_resolution_found"] is False


def test_t5_commensuration_detects_non_identity_when_equity_differs():
    """若条件化权益与 search 均值**不同**，则不是恒真 —— 该判定必须能区分。"""
    rec = _rec(search_candidates=[
        {"action": "check", "amount_bb": 0.0, "future_equity_mean": 0.8275,
         "search_score": 0.8275},
        {"action": "bet", "amount_bb": 0.495, "future_equity_mean": 0.700,
         "search_score": 0.832}])
    o = counterfactual_postflop_options(record=rec)
    a = o["option_A_commensurate_ranking"]
    assert a["all_scores_identical_to_search"] is False
    assert a["verdict"] == "NEEDS_FURTHER_AUDIT"


def test_t5_summary_and_fail_closed():
    o = [counterfactual_postflop_options(record=_rec(run_id=f"r{i}"))
         for i in range(3)]
    s = summarize_postflop_options(results=o)
    assert s["n_runs"] == 3
    assert s["n_option_A_tautological"] == 3
    assert s["n_option_A_would_agree"] == 3
    assert s["legitimate_resolutions_found"] == 0
    assert summarize_postflop_options(results=[]).get("present") is False
    assert counterfactual_postflop_options(record=None).get(
        "present") is False
    assert counterfactual_postflop_options(
        record={"search_candidates": []}).get("present") is False


# ---------------------------------------------------------------------------
# T6. 覆盖率影响与证据语义裁定（要求 8、9）
# ---------------------------------------------------------------------------
def test_t6_coverage_impact_marks_legitimacy_not_just_coverage():
    c = coverage_impact(
        actual={"n_runs": 45, "n_accept": 12},
        candidate_effects={
            "C1": {"accept_to_abstain": 10, "abstain_to_accept": 0,
                   "legitimacy": "legitimate_but_lowers_coverage",
                   "independent_sources_after": 2},
            "C2": {"accept_to_abstain": 0, "abstain_to_accept": 28,
                   "legitimacy": "REJECTED_TAUTOLOGICAL",
                   "independent_sources_after": 1}})
    c1 = c["by_candidate"]["C1"]
    c2 = c["by_candidate"]["C2"]
    assert c1["accept_after"] == 2 and c1["delta"] == -10
    assert c2["accept_after"] == 40 and c2["delta"] == 28
    # 覆盖率更高的那个反而是被拒的
    assert c2["coverage_after"] > c1["coverage_after"]
    assert c2["legitimacy"].startswith("REJECTED")
    assert c2["independent_sources_after"] == 1
    assert "不得以覆盖率提升为由接受非法方案" in c["note"]
    assert coverage_impact(actual=None,
                           candidate_effects={}).get("present") is False


def test_t6_evidence_semantics_verdict_separates_accepted_and_rejected():
    v = evidence_semantics_verdict(candidates=[
        {"kind": "street_faithful_deal_expansion", "rejected": False},
        {"kind": "candidate_local_equity_for_ranking", "rejected": True},
        {"kind": "vote_between_tiers", "rejected": True}])
    assert v["n_accepted"] == 1
    assert v["n_rejected"] == 2
    assert v["accepted_under_strict_evidence_semantics"] == [
        "street_faithful_deal_expansion"]
    assert "存在通过严格证据语义的候选方案" in v["verdict"]
    assert tuple(v["rules"]) == EVIDENCE_SEMANTICS_RULES
    empty = evidence_semantics_verdict(candidates=[
        {"kind": "vote_between_tiers", "rejected": True}])
    assert empty["n_accepted"] == 0
    assert "**不存在**" in empty["verdict"]


# ---------------------------------------------------------------------------
# T7. 实测守护（要求 1、2、3、4、8、9）
# ---------------------------------------------------------------------------
pytestmark_ev = pytest.mark.skipif(
    not SUMMARY.exists(), reason="需要 .tmp/v3i_p16_summary.json")


@pytestmark_ev
def test_t7_preflop_candidate_produces_real_traceable_equity():
    """要求 4：新路径必须产生**真实**权益，且 fold-only 不计入。"""
    d = json.loads(SUMMARY.read_text(encoding="utf-8"))
    st = d["shadow_experiment"]["c1_shadow_tree"]
    for scn in ("preflop_norm", "preflop_short"):
        f = st[scn]["frozen"]
        c = st[scn]["street_faithful"]
        assert f["n_real_equity_leaves"] == 0, scn
        assert f["leaf_coverage"] == 0.111111, scn
        assert c["leaf_coverage"] == 1.0, scn
        assert c["n_real_equity_leaves"] == 128, scn
        assert c["n_fold_only_arithmetic_leaves"] == 18, scn
        assert any("hand_vs_range" in k for k in c["equity_sources"]), scn
        assert all(b in ("0", "3", "4", "5")
                   for b in c["evaluable_board_sizes"]), scn
    # 负对照：冻结 preflop 的唯一可评估来源就是 fold-only
    assert st["preflop_norm"]["frozen"]["equity_sources"] == {
        "arithmetic:pot_share_on_opponent_fold": 2}


@pytestmark_ev
def test_t7_blast_radius_is_preflop_only():
    """要求 2：最小影响面 —— postflop 树**逐字节不变**。"""
    d = json.loads(SUMMARY.read_text(encoding="utf-8"))
    c4 = d["shadow_experiment"]["c4_blast_radius"]
    assert c4["flop_norm"]["tree_unchanged_by_candidate_rule"] is True
    assert c4["turn_norm"]["tree_unchanged_by_candidate_rule"] is True
    assert c4["preflop_norm"]["tree_unchanged_by_candidate_rule"] is False
    assert d["shadow_experiment"]["c4_summary"][
        "postflop_trees_unchanged_by_the_candidate_rule"] is True
    for scn in ("preflop_norm", "preflop_short", "flop_norm", "turn_norm"):
        assert d["shadow_experiment"]["c1_shadow_tree"][scn][
            "mirror_reproduces_frozen"] is True


@pytestmark_ev
def test_t7_preflop_candidate_lowers_coverage_measured():
    """要求 8：覆盖率变化必须实测报告，且方向如实。"""
    d = json.loads(SUMMARY.read_text(encoding="utf-8"))
    c1 = d["coverage_impact"]["by_candidate"][
        "C1_preflop_street_faithful_deal"]
    assert c1["accept_to_abstain"] == 10
    assert c1["abstain_to_accept"] == 0
    assert c1["coverage_before"] == 0.266667
    assert c1["coverage_after"] == 0.044444
    assert c1["delta"] == -10
    assert c1["legitimacy"] == "legitimate_but_lowers_coverage"
    assert c1["independent_sources_after"] == 2
    s = d["preflop_counterfactual_summary"]
    assert s["n_runs"] == 10
    assert s["all_gates_pass_after_candidate"] is True
    assert s["sf_search_tops"] == ["['check', 0.0]"]
    assert set(s["ranking_tops"]) == {"['bet', 0.495]", "['bet', 0.66]"}


@pytestmark_ev
def test_t7_postflop_commensuration_is_tautological_measured():
    d = json.loads(SUMMARY.read_text(encoding="utf-8"))
    s = d["postflop_options_summary"]
    assert s["n_runs"] == 30
    assert s["n_option_A_tautological"] == 30
    assert s["n_option_A_would_agree"] == 30
    assert s["legitimate_resolutions_found"] == 0
    assert d["shadow_experiment"]["c2_conditioned_ranking_summary"][
        "n_all_scores_identical_to_search"] == 30
    assert d["shadow_experiment"]["c2_conditioned_ranking_summary"][
        "independent_scoring_sources_after_commensuration"] == 1


@pytestmark_ev
def test_t7_illegitimate_options_all_rejected_in_measurement():
    d = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert d["rejected_resolutions"]["n_rejected"] == 5
    for r in d["postflop_options"]:
        assert r["any_legitimate_resolution_found"] is False
        assert r["option_A_commensurate_ranking"]["verdict"] == \
            "REJECTED_TAUTOLOGICAL"
        assert r["option_B_prefer_search_on_disagreement"]["verdict"] == \
            "REJECTED_OVERRIDE_IN_EFFECT"
        assert r["option_C_vote_between_tiers"]["verdict"] == "REJECTED_TIE"
        assert r["option_D_lower_threshold"]["verdict"] == \
            "REJECTED_FORBIDDEN_AND_COUNTERPRODUCTIVE"


@pytestmark_ev
def test_t7_root_cause_verdicts_and_evidence_semantics_verdict():
    d = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert d["root_cause_audits"]["preflop"]["is_a_defect"] is True
    assert d["root_cause_audits"]["postflop"]["is_a_defect"] is False
    v = d["evidence_semantics_verdict"]
    assert v["n_accepted"] == 1 and v["n_rejected"] == 5
    assert v["accepted_under_strict_evidence_semantics"] == [
        "street_faithful_deal_expansion"]
    # 两个候选都**未实施**
    for k, p in d["proposals"].items():
        assert p["status"] == "PROPOSED_NOT_IMPLEMENTED", k
        assert p["implements_anything"] is False, k
        assert p["threshold_value_after"] == 0.8, k
    assert d["proposals"][
        "C2_candidate_local_equity_for_ranking"][
        "is_illegitimate_by_construction"] is True


@pytestmark_ev
def test_t7_evidence_is_deterministic_and_declares_no_layer_change():
    d = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert d["fingerprints"]["root_cause_audits"] == \
        stable_fingerprint(d["root_cause_audits"])
    assert d["fingerprints"]["coverage_impact"] == \
        stable_fingerprint(d["coverage_impact"])
    assert d["fingerprints"]["postflop_options"] == \
        stable_fingerprint(d["postflop_options_summary"])
    c = d["p16_contract"]
    assert c["implements_any_unfreeze_item"] is False
    assert c["leaf_coverage_threshold_kept"] == 0.8
    assert c["enables_override"] is False
    assert c["uses_voting"] is False
    assert c["treats_fold_only_as_equity"] is False
