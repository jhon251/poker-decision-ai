# -*- coding: utf-8 -*-
"""V3I-G1 **Phase 2** — Search / Ranking 动作价值统一（统一路径测试）。

覆盖授权要求的 6 项必需测试：
1. 同候选集一致性；2. 同记账一致性；3. action-value 反事实；
4. Search / Ranking **独立重算**；5. candidate mismatch 不再计入 disagreement；
6. 缺失证据 fail-closed。

纪律（沿用 Phase 1）
-------------------
* 本文件**不调用**被测 value aggregation 实现（名称见下方 `_FORBIDDEN`，
  以拼接构造，避免本文件自身包含该字面量）；统一价值一律经**公开入口**
  `unified_action_value_path(...)` / `evaluate_search(...)` /
  `rank_actions(...)` 取得；
* 反事实只通过**构造输入**实现；
* 公式的**独立重算**由本文件自行实现（见 §4），不复用被测实现。

环境约束：本沙箱禁止 `%TEMP%` 写入，`tmp_path`/`ctx`/`runtime` fixture 会抛
`PermissionError [WinError 5]`，故本文件不使用需要临时目录的 fixture。
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from poker_decision_ai.cash_action_value_contract import (  # noqa: E402
    CONTRACT as AV_CONTRACT, UNIFIED_MODE_POLICY, candidate_key,
    shared_action_domain, unified_action_value_input,
    unified_mode_contract)
from poker_decision_ai.cash_conditioned_search_wiring import (  # noqa: E402
    _candidate_key_str, unified_action_value_path, unified_path_contract)
from poker_decision_ai.cash_strategy_action_ranking import rank_actions  # noqa: E402
from poker_decision_ai.cash_strategy_decision_assembly import (  # noqa: E402
    ASSEMBLY_POLICY, TIERS, assemble_decision)
from poker_decision_ai.cash_strategy_search_eval import (  # noqa: E402
    SEARCH_SCORE_MODEL)

_FORBIDDEN = ("aggregate" + "_action_values",
              "candidate_action_value" + "(")

WIRING = ROOT / "poker_decision_ai/cash_conditioned_search_wiring.py"

_M: dict = {}


def _p1():
    """复用 Phase 1 测试模块的装配与缓存（同一 pytest 进程内零重复构建）。"""
    if "p1" not in _M:
        spec = importlib.util.spec_from_file_location(
            "_g1p1_mod",
            ROOT / "tests" / "unit" / "test_v3i_g1_phase1_action_value.py")
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        sys.modules["_g1p1_mod"] = mod
        spec.loader.exec_module(mod)
        _M["p1"] = mod
    return _M["p1"]


def _base(street: str) -> dict:
    return _p1()._base(street)


def _tree(street: str, cap: int) -> dict:
    return _p1()._tree(street, cap)


def _bridge(street: str, population: str):
    b = _base(street)
    return _p1()._p8()._bridge_for(b["st"], b["rm"], b["ap"], b["bel"], population)


def _unified(street: str = "flop", cap: int = 3,
             population: str = "DECLARED_BALANCED",
             unified_in: dict | None = None, tree: dict | None = None,
             with_legacy: bool = True) -> dict:
    key = f"u::{street}::{cap}::{population}::{with_legacy}::{tree is not None}"
    if unified_in is None and tree is None and key in _M:
        return _M[key]
    b = _base(street)
    if unified_in is None:
        # 默认：由 bridge 测量量构造统一输入（公开路径）
        res = unified_action_value_path(
            state=b["st"], action_policy=b["ap"], belief=b["bel"],
            bridge_out=_bridge(street, population),
            search_tree=tree if tree is not None else _tree(street, cap),
            unified_input_fn=unified_action_value_input,
            search_eval_fn=b["d"]["evaluate_search"],
            rank_actions_fn=rank_actions,
            assemble_decision_fn=assemble_decision,
            include_legacy_comparison=with_legacy)
    else:
        # 反事实：直接注入构造好的统一输入
        res = unified_action_value_path(
            state=b["st"], action_policy=b["ap"], belief=b["bel"],
            bridge_out=_bridge(street, population),
            search_tree=tree if tree is not None else _tree(street, cap),
            unified_input_fn=lambda **kw: unified_in,
            search_eval_fn=b["d"]["evaluate_search"],
            rank_actions_fn=rank_actions,
            assemble_decision_fn=assemble_decision,
            include_legacy_comparison=with_legacy)
    if unified_in is None and tree is None:
        _M[key] = res
    return res


def _rows_by_key(block: dict) -> dict:
    return {f"{r['action']}|{r['amount_bb']}": r for r in block["ranked"]}


def _keys(block: dict) -> list:
    return [f"{r['action']}|{r['amount_bb']}" for r in block["ranked"]]


def _records(unified_in: dict) -> list:
    return list(unified_in.get("records") or [])


# ══ 1. 同候选集一致性 ═════════════════════════════════════════════════
def test_phase2_same_candidate_domain_truncated_search():
    res = _unified("flop", 3)
    dom = res["domain"]
    search_keys = sorted(f"{c['action']}|{c['amount_bb']}"
                         for c in res["search_eval"]["candidates"])
    ranking_keys = sorted(f"{c['action']}|{c['amount_bb']}"
                          for c in res["ranking"]["ranked_legacy"])
    assert dom["candidate_set_mismatch"] is True
    assert dom["per_tier_keys"]["search"] == search_keys
    assert dom["per_tier_keys"]["ranking"] == ranking_keys
    assert dom["common_keys"] == search_keys                  # 共同域 = search 实际展开集
    assert dom["only_in_tier"]["ranking"] == ["bet|0.99", "bet|1.5", "bet|100.0"]
    assert dom["counts_as_evidence_disagreement"] is False
    # 两层都只在共同域内排序
    assert sorted(_keys(res["search_eval"])) == dom["common_keys"]
    assert sorted(_keys(res["ranking"])) == dom["common_keys"]
    # 域外候选**被报告**（不隐藏），且带 legacy 分数
    ood = res["ranking"]["unified_out_of_domain"]
    assert sorted(x["key"] for x in ood) == dom["only_in_tier"]["ranking"]
    assert all(x["counts_as_evidence_disagreement"] is False for x in ood)
    assert all(isinstance(x["score_legacy_v3"], (int, float)) for x in ood)


def test_phase2_same_candidate_domain_full_search():
    res = _unified("flop", 6)
    dom = res["domain"]
    assert dom["candidate_set_mismatch"] is False
    assert dom["n_common"] == 6
    assert sorted(_keys(res["search_eval"])) == sorted(_keys(res["ranking"]))
    assert res["search_eval"]["unified_out_of_domain"] == []
    assert res["ranking"]["unified_out_of_domain"] == []


def test_phase2_key_format_agrees_with_contract():
    b = _base("flop")
    for c in b["ap"]["candidates"]:
        assert _candidate_key_str(c["action"], c["amount_bb"]) == \
            candidate_key(c["action"], c["amount_bb"])
    dom = shared_action_domain(
        search_keys=[candidate_key("bet", 0.75)],
        ranking_keys=[candidate_key("bet", 0.75), candidate_key("bet", 1.5)])
    assert dom["common_keys"] == ["bet|0.75"]
    assert dom["only_in_tier"]["ranking"] == ["bet|1.5"]
    assert dom["candidate_set_mismatch"] is True


# ══ 2. 同记账一致性（pot/stack + fold/continue + terminal value）═══════
def test_phase2_same_accounting_and_inputs_for_same_candidate():
    res = _unified("flop", 6)
    shared = res["unified_input"]["unified_values"]
    se_rows = _rows_by_key(res["search_eval"])
    rk_rows = _rows_by_key(res["ranking"])
    assert res["search_eval"]["value_mode"] == "unified"
    assert res["ranking"]["value_mode"] == "unified"
    assert sorted(se_rows) == sorted(rk_rows) == sorted(shared)
    for k in shared:
        v = shared[k]["value"]
        assert se_rows[k]["action_value_pot_share"] == v
        assert rk_rows[k]["action_value_pot_share"] == v
        # 同一记账：terms（fold/showdown）与 measured（权重/权益）逐位一致
        assert se_rows[k]["action_value_terms"] == rk_rows[k]["action_value_terms"]
        assert se_rows[k]["action_value_measured"] == \
            rk_rows[k]["action_value_measured"]
        # 同一 pot/stack 后果（来自 search tree 记账）
        assert se_rows[k]["unified_value_consequences"] == \
            rk_rows[k]["unified_value_consequences"]
        assert rk_rows[k]["action_value_provenance"]["free_parameters"] == 0
    # 单一记账定义（两层引用同一 contract）
    assert res["unified_input"]["contract"]["name"] == AV_CONTRACT["name"]
    assert AV_CONTRACT["free_parameters"] == 0


def test_phase2_legacy_calibres_preserved():
    res = _unified("flop", 6)
    se = res["search_eval"]
    rk = res["ranking"]
    # search：统一模式下 search_score 语义为统一价值，旧 v3 值并列保留
    legacy_pairs = {f"{r['action']}|{r['amount_bb']}":
                    r["search_score_legacy_v3"] for r in se["ranked"]}
    assert legacy_pairs["bet|0.75"] == 1.035754
    assert legacy_pairs["bet|100.0"] == 36.48481
    assert se["legacy_search_score_preserved"] is True
    assert se["dual_report"]["legacy_model_unchanged"] is True
    assert se["dual_report"]["value_mode"] == "unified"
    assert se["ranked_legacy"] and se["preferred_legacy"] is not None
    # ranking：旧 pot-share 分数并列保留
    rk_legacy = {f"{r['action']}|{r['amount_bb']}": r["score_legacy_v3"]
                 for r in rk["ranked"]}
    assert rk_legacy["bet|100.0"] == 53.518008
    assert rk["legacy_ranking_preserved"] is True
    assert rk["dual_report"]["legacy_model"]["name"] == "pot_share_arithmetic_v1"
    assert rk["ranked_legacy"] and rk["preferred_legacy"] is not None
    # 旧口径确实与统一口径**不同**（不是同源复述的假一致）
    assert se["ranked_legacy"][0]["action"] == "bet"
    assert se["ranked_legacy"][0]["amount_bb"] == 100.0


def test_phase2_terminal_value_and_consequences_from_one_source():
    res = _unified("flop", 6)
    tree = _tree("flop", 6)
    cons = {_candidate_key_str(n["action"], n["amount_bb"]):
            (n["hero_investment_bb"], n["pot_after_hero_bb"],
             n["stack_after_hero_bb"]) for n in tree["root"]["children"]}
    for key, row in res["unified_input"]["unified_values"].items():
        c = row["consequences"]
        assert (c["hero_investment_bb"], c["pot_after_hero_bb"],
                c["stack_after_hero_bb"]) == cons[key]
        assert c["source"] == "search_tree.accounting"
        assert row["terms"]["terminal_values_distinct"] >= 1
    assert res["unified_input"]["policy"]["accounting"].startswith("两层消费")


# ══ 3. action-value 反事实（两层同步）═════════════════════════════════
def test_phase2_counterfactual_c0_degenerates_to_legacy_in_both_tiers():
    res = _unified("turn", 3)
    pot = res["unified_input"]["pot_bb"]
    recs = []
    for r in res["search_eval"]["ranked_legacy"]:
        recs.append({
            "action": r["action"], "amount_bb": r["amount_bb"],
            "equity_vs_continuing": r["future_equity_mean_real_equity_only"],
            "p_continue": 1.0, "p_fold": 0.0,
            "threshold": 0.5, "n_continuation_classes": 1,
            "population": "C0", "equity_basis": "equity_vs_continuing_measured"})
    from poker_decision_ai.cash_action_value_contract import basis_fingerprint
    for rec in recs:
        rec["basis_fingerprint"] = basis_fingerprint(rec)
    legacy_by_key = {f"{r['action']}|{r['amount_bb']}": r["search_score"]
                     for r in res["search_eval"]["ranked_legacy"]}
    variant = unified_action_value_input(
        pot_bb=pot,
        candidates=[{"action": r["action"], "amount_bb": r["amount_bb"]}
                    for r in res["search_eval"]["ranked_legacy"]],
        records=recs, search_keys=list(legacy_by_key),
        ranking_keys=list(legacy_by_key), consequences=None,
        population="C0", source="test_c0")
    out = _unified("turn", 3, unified_in=variant, with_legacy=False)
    for tier in ("search_eval", "ranking"):
        rows = _rows_by_key(out[tier])
        for k, row in rows.items():
            assert row["action_value_pot_share"] == legacy_by_key[k], (tier, k)
    assert _keys(out["search_eval"]) == _keys(out["ranking"])


def test_phase2_counterfactual_weight_permutation_moves_both_tiers():
    res = _unified("flop", 6)
    rows = {r["key"]: r for r in res["search_eval"]["ranked_by_action_value"]}
    a, bkey = "bet|0.495", "bet|100.0"
    recs = []
    for r in rows.values():
        p_c, p_f = r["p_continue"], r["p_fold"]
        if r["key"] == a:
            p_c, p_f = rows[bkey]["p_continue"], rows[bkey]["p_fold"]
        elif r["key"] == bkey:
            p_c, p_f = rows[a]["p_continue"], rows[a]["p_fold"]
        rec = {"action": r["action"], "amount_bb": r["amount_bb"],
               "equity_vs_continuing": r["equity_vs_continuing"],
               "p_continue": p_c, "p_fold": p_f, "threshold": 0.5,
               "n_continuation_classes": 1, "population": "SWAP",
               "equity_basis": "equity_vs_continuing_measured"}
        from poker_decision_ai.cash_action_value_contract import basis_fingerprint
        rec["basis_fingerprint"] = basis_fingerprint(rec)
        recs.append(rec)
    variant = unified_action_value_input(
        pot_bb=res["unified_input"]["pot_bb"],
        candidates=[{"action": r["action"], "amount_bb": r["amount_bb"]}
                    for r in rows.values()],
        records=recs, search_keys=list(rows), ranking_keys=list(rows),
        population="SWAP", source="test_swap")
    out = _unified("flop", 6, unified_in=variant, with_legacy=False)
    assert _keys(out["search_eval"]) == _keys(out["ranking"])
    assert _keys(out["search_eval"])[0] == a
    assert out["tier_tops_equal"] is True


@pytest.mark.parametrize("population", ["DECLARED_TIGHT_PASSIVE",
                                       "DECLARED_BALANCED",
                                       "DECLARED_AGGRESSIVE"])
def test_phase2_counterfactual_population_switch_keeps_tiers_unified(population):
    res = _unified("turn", 6, population)
    assert _keys(res["search_eval"]) == _keys(res["ranking"])
    assert res["tier_tops_equal"] is True
    assert res["disagreement_provenance"]["unified_disagreements"] == []


# ══ 4. Search / Ranking 独立重算 ══════════════════════════════════════
def test_phase2_independent_recomputation_matches_both_tiers():
    """本测试**自行实现**契约公式，从原始测量记录重算，不调用被测聚合实现。"""
    res = _unified("flop", 6)
    unified_in = res["unified_input"]
    pot = float(unified_in["pot_bb"])
    recomputed: dict[str, float] = {}
    for rec in _records(unified_in):
        a = float(rec["amount_bb"] or 0.0)
        eq = float(rec["equity_vs_continuing"])
        p_c = float(rec["p_continue"])
        p_f = float(rec["p_fold"])
        if a <= 0.0:
            value = eq                       # 零投入：无 fold 分支（定义）
        else:
            value = p_f * 1.0 + p_c * (eq * (pot + 2.0 * a) / pot - a / pot)
        recomputed[f"{rec['action']}|{rec['amount_bb']}"] = round(value, 6)
    assert recomputed, "records required"
    for tier in ("search_eval", "ranking"):
        rows = _rows_by_key(res[tier])
        for k, row in rows.items():
            assert row["action_value_pot_share"] == recomputed[k], (tier, k)
    # 生产侧一致性自检（search 层独立重算共享价值表）
    uc = res["search_eval"]["unified_consistency"]
    assert uc["checked"] == len(recomputed)
    assert uc["mismatched_keys"] == []
    assert res["ranking"]["unified_provenance_issues"] == []


# ══ 5. candidate mismatch 不再计入 disagreement ═══════════════════════
def test_phase2_candidate_mismatch_not_counted_as_disagreement():
    res = _unified("flop", 3)
    prov = res["disagreement_provenance"]
    assert prov["candidate_set_mismatch"] is True
    assert prov["candidate_set_mismatch_counts_as_disagreement"] is False
    assert prov["unified_disagreements"] == []
    # **不隐藏**：legacy 对照仍能看到那条 disagreement（截断假象）
    assert prov["legacy_disagreements"], prov
    assert prov["pseudo_disagreement_keys_removed"], prov
    assert res["decision"]["abstained"] is False
    assert res["legacy_decision"]["abstained"] is True
    assert res["domain"]["counts_as_evidence_disagreement"] is False


# ══ 6. 缺失证据 fail-closed ═══════════════════════════════════════════
def test_phase2_missing_bridge_evidence_fails_closed_both_tiers():
    b = _base("turn")
    res = unified_action_value_path(
        state=b["st"], action_policy=b["ap"], belief=b["bel"],
        bridge_out={"present": False}, search_tree=_tree("turn", 3),
        unified_input_fn=unified_action_value_input,
        search_eval_fn=b["d"]["evaluate_search"],
        rank_actions_fn=rank_actions, assemble_decision_fn=assemble_decision)
    se, rk, dec = res["search_eval"], res["ranking"], res["decision"]
    assert se["value_mode"] == "unified" and rk["value_mode"] == "unified"
    assert se["ranked"] == [] and rk["ranked"] == []
    assert se["preferred"] is None and rk["preferred"] is None
    assert se["n_candidates_scored"] == 0
    unk = dec["unknown"] or []
    assert "search_top_candidate_unavailable" in unk
    assert "ranking_top_candidate_unavailable" in unk
    assert res["unified_input"]["n_valued"] == 0
    assert res["unified_input"]["unavailable_reason"]


def test_phase2_budget_violation_fails_closed_and_legacy_still_reported():
    b = _base("flop")
    tree_default = b["tree_default"]                      # 219 > 200 ⇒ 超限
    assert tree_default["max_leaves_exceeded"] is True
    res = unified_action_value_path(
        state=b["st"], action_policy=b["ap"], belief=b["bel"],
        bridge_out=_bridge("flop", "DECLARED_BALANCED"),
        search_tree=tree_default,
        unified_input_fn=unified_action_value_input,
        search_eval_fn=b["d"]["evaluate_search"],
        rank_actions_fn=rank_actions, assemble_decision_fn=assemble_decision)
    assert res["unified_input"]["unavailable_reason"] == "leaf_budget_violated"
    assert res["search_eval"]["ranked"] == []
    assert res["ranking"]["ranked"] == []
    # legacy 口径仍并列报告（不删除历史口径）：`ranked_legacy` 行的 `search_score`
    # 即 v3 旧值（统一模式下只有 `ranked` 行才带 `search_score_legacy_v3`）
    assert res["search_eval"]["ranked_legacy"]
    assert all(isinstance(r["search_score"], (int, float))
               for r in res["search_eval"]["ranked_legacy"])
    assert res["search_eval"]["dual_report"]["legacy_model_unchanged"] is True
    assert res["decision"]["abstained"] is False
    assert "search_top_candidate_unavailable" in (res["decision"]["unknown"] or [])


def test_phase2_missing_single_candidate_evidence_excludes_only_that_candidate():
    res = _unified("turn", 3)
    recs = []
    for rec in _records(res["unified_input"]):
        rec = dict(rec)
        if str(rec["action"]) == "bet" and float(rec["amount_bb"]) == 0.495:
            rec.pop("equity_vs_continuing", None)
        from poker_decision_ai.cash_action_value_contract import basis_fingerprint
        rec["basis_fingerprint"] = basis_fingerprint(rec)
        recs.append(rec)
    variant = unified_action_value_input(
        pot_bb=res["unified_input"]["pot_bb"],
        candidates=[{"action": r["action"], "amount_bb": r["amount_bb"]}
                    for r in res["search_eval"]["candidates"]],
        records=recs,
        search_keys=[_candidate_key_str(r["action"], r["amount_bb"])
                     for r in res["search_eval"]["candidates"]],
        ranking_keys=[_candidate_key_str(r["action"], r["amount_bb"])
                      for r in res["search_eval"]["candidates"]],
        population="DROP_ONE", source="test_drop")
    out = _unified("turn", 3, unified_in=variant, with_legacy=False)
    for tier in ("search_eval", "ranking"):
        keys = _keys(out[tier])
        assert "bet|0.495" not in keys
        assert len(keys) == 2
    assert _keys(out["search_eval"]) == _keys(out["ranking"])
    assert out["search_eval"]["unified_unavailable_reason"] is None


# ══ 7. 其它不变量 ═════════════════════════════════════════════════════
def test_phase2_gate_tiers_override_and_voting_unchanged():
    assert float(ASSEMBLY_POLICY["min_leaf_coverage"]) == 0.8
    assert ASSEMBLY_POLICY["override_default"] is False
    assert ASSEMBLY_POLICY["on_disagreement"] == "abstain_unknown"
    assert TIERS == ("search", "belief_conditioned", "ranking")
    assert UNIFIED_MODE_POLICY["no_voting_no_override"] is True
    assert UNIFIED_MODE_POLICY["gate_changed"] is False
    assert UNIFIED_MODE_POLICY["external_data_used"] is False
    assert UNIFIED_MODE_POLICY["free_parameters"] == 0
    c = unified_path_contract()
    assert c["no_voting_no_override"] is True and c["gate_changed"] is False
    assert c["imports_project_modules"] is False
    assert unified_mode_contract()["tier_consumers"] == [
        "cash_strategy_search_eval.evaluate_search",
        "cash_strategy_action_ranking.rank_actions"]
    assert SEARCH_SCORE_MODEL["name"] == "search_runout_pot_share_v1"


def test_phase2_untouched_modules_unchanged():
    for rel, want in (
            ("poker_decision_ai/cash_strategy_decision_assembly.py",
             "3732314ab0182063"),
            ("poker_decision_ai/cash_decision_loop.py", "7c4b0a11c1e73a7f"),
            ("poker_decision_ai/cash_strategy_likelihood_model.py",
             "df207954ade71fa6"),
            ("poker_decision_ai/cash_solver_interface.py", "a3d55431669e25c6"),
            ("poker_decision_ai/cash_opponent_conditioned_bridge.py",
             "548f14a1c9d4db32")):
        got = hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()[:16]
        assert got == want, (rel, got, want)


def test_phase2_blackbox_discipline_and_wiring_stdlib_only():
    src = pathlib.Path(__file__).read_text(encoding="utf-8")
    for bad in _FORBIDDEN:
        assert bad not in src, f"测试不得调用被测聚合实现：{bad}"
    assert "unified_action_value_path" in src
    mods = set()
    for n in ast.walk(ast.parse(WIRING.read_text(encoding="utf-8"))):
        if isinstance(n, ast.Import):
            mods.update(a.name for a in n.names)
        elif isinstance(n, ast.ImportFrom):
            mods.add(n.module or "")
    assert mods <= {"__future__", "typing", "json", "hashlib"}, mods
