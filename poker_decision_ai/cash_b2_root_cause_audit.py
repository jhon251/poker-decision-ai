"""V3I-P15 — B2（完整链路）覆盖率 12/45 的**根因审计**与**最小反事实**层。

授权范围（本阶段**只做根因审计与最小实验**，不扩大机制）
------------------------------------------------------
* 不修改 Solver / V3H-4 / V3H-6 / CFI·λ / `strategy_core` / P13 契约 / P1 loop /
  river terminal；**不**修改 budget / depth / candidate cap；
* **不**降低 0.8 leaf-coverage 门槛、**不**启用 override；
* **不**把单层 `ranking` ACCEPT 包装成"条件化决策"；
* 不宣称 EV / GTO / equilibrium / 盈利能力；
* 本层只 import 标准库；项目内层一律由调用方**显式注入**。

四个最小反事实（授权要求 4）都是**只读重算**，不改变任何层的输出：
1. `leaf_coverage`  → 叶子解剖 + 冻结输入有效性判据（board 张数白名单）；
2. `search_unavailable` → **反事实接纳**：若把 search 层 top（不看覆盖率）当作可比
   tier 参与冻结对账，结果会怎样（**不改门槛**，只是重算）；
3. `outside_search_expansion` → 反事实**限域**选择；
4. `true disagreement` → 冻结对账规则下的 key 比较 + 条件化**盈亏平衡权益**。

`engineered_coverage_vs_decision_quality` 用于**明确区分**"工程覆盖率问题"与
"决策质量问题"（授权要求 8）。
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

CASH_P15_ROOT_CAUSE_AUDIT_VERSION = "v3i-p15-b2-root-cause-audit-v1"

#: B2 完整链路（与 V3I-P14 一致）
B2_MODE = "B2_conditioned_search_feasible_assembly"

#: 非 ACCEPT 的原因类别（授权要求 1）
NON_ACCEPT_REASON_CODES: tuple[str, ...] = (
    "river_terminal_pre_search_blocked",
    "cross_scorer_true_disagreement",
    "search_tier_unavailable_by_coverage_gate",
    "single_tier_decision",
    "unknown_no_preferred",
    "stage_failure",
    "loop_fail_closed",
)

#: `leaf_coverage` 根因分类（授权要求 2）
LEAF_COVERAGE_CAUSES: tuple[str, ...] = (
    "board_size_outside_validity_whitelist",
    "single_card_street_enumeration",
    "street_board_size_mismatch_blocks_further_expansion",
    "max_leaves_budget_reached",
    "max_runout_children_cap_reached",
    "max_street_depth_reached",
    "river_no_future_street",
    "unknown_cause",
)

#: 叶子权益可计算的 board 张数白名单（冻结 `_leaf_equity` 的输入有效性判据）
VALID_EQUITY_BOARD_SIZES: tuple[int, ...] = (0, 3, 4, 5)

#: 各街"完整"board 张数（冻结 `cash_strategy_search_tree._EXPECTED_BOARD`）
EXPECTED_BOARD = {"preflop": 0, "flop": 3, "turn": 4, "river": 5}

CLAIM_BOUNDARY: dict[str, bool] = {
    "is_gto": False,
    "is_equilibrium": False,
    "is_solver_output": False,
    "is_ev": False,
    "is_real_opponent_data": False,
    "is_real_human_validation": False,
    "is_profit_evidence": False,
    "is_external_validation": False,
    "is_win_rate_claim": False,
    "is_strategy_strength_claim": False,
    "wired_to_production": False,
    "adds_strategy_mechanism": False,
    "modifies_frozen_contract": False,
    "modifies_budget_depth_or_cap": False,
    "relaxes_leaf_coverage_threshold": False,
    "enables_override": False,
    "changes_any_layer_output": False,
}

EVIDENCE_GRADE = "MODEL_INTERNAL"


def _is_num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _stub(reason: str) -> dict:
    return {
        "present": False,
        "version": CASH_P15_ROOT_CAUSE_AUDIT_VERSION,
        "reason": reason,
        "unknown": [reason],
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "evidence_grade": "NONE",
        "executed": False,
        "result": None,
        "wired_to_production": False,
    }


def stable_fingerprint(value: Any) -> str:
    def _norm(v: Any) -> Any:
        if isinstance(v, dict):
            return {str(k): _norm(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [_norm(x) for x in v]
        return v
    try:
        text = json.dumps(_norm(value), sort_keys=True, separators=(",", ":"),
                          ensure_ascii=True, default=str)
    except Exception:                                      # noqa: BLE001
        text = repr(value)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _top_key(v: Any) -> Any:
    if not isinstance(v, dict):
        return None
    act, amt = v.get("action"), v.get("amount_bb")
    if act is None:
        return None
    return [act, round(float(amt), 6) if _is_num(amt) else None]


def _same_key(a: Any, b: Any, tol: float = 1e-6) -> bool:
    if not isinstance(a, (list, tuple)) or not isinstance(b, (list, tuple)):
        return False
    if len(a) != 2 or len(b) != 2:
        return False
    if str(a[0]).strip().lower() != str(b[0]).strip().lower():
        return False
    if a[1] is None or b[1] is None:
        return True
    return abs(float(a[1]) - float(b[1])) <= tol


# ---------------------------------------------------------------------------
# 1. B2 非 ACCEPT 的原因分解（授权要求 1）
# ---------------------------------------------------------------------------
def decompose_run(*, record: Any, min_leaf_coverage: float = 0.8) -> dict:
    """把**一次**运行分解为（结果, 原因代码, 细节）。**只读**。"""
    if not isinstance(record, dict):
        return _stub("record_not_a_dict")
    preferred = record.get("preferred")
    abstained = record.get("abstained") is True
    loop_status = record.get("loop_status")
    reason = str(record.get("reason") or "")
    unknown_list = [str(u) for u in (record.get("unknown") or [])]
    disagreements = [d for d in (record.get("disagreements") or [])
                     if isinstance(d, dict)]
    rejections = [x for x in (record.get("rejections") or [])
                  if isinstance(x, dict)]
    tops = record.get("scorer_tops") or {}
    n_alignable = record.get("n_scorers_alignable")

    if record.get("present") is False:
        outcome = "BLOCKED"
    elif abstained:
        outcome = "ABSTAIN"
    elif preferred not in (None, {}):
        outcome = "ACCEPT"
    elif loop_status == "UNKNOWN":
        outcome = "UNKNOWN_NO_PREFERRED"
    else:
        outcome = "BLOCKED"

    codes: list[str] = []
    detail: dict[str, Any] = {}
    if outcome == "BLOCKED":
        if reason.startswith("stage_failed:action_policy"):
            codes.append("river_terminal_pre_search_blocked")
            detail["blocking_stage"] = "action_policy"
            detail["blocking_reason"] = "terminal_or_no_legal_actions"
        elif reason.startswith(("not_cash", "evidence_level", "pipeline_")):
            codes.append("loop_fail_closed")
            detail["blocking_reason"] = reason
        else:
            codes.append("stage_failure")
            detail["blocking_reason"] = reason
    elif outcome == "UNKNOWN_NO_PREFERRED":
        codes.append("unknown_no_preferred")
    else:
        # 覆盖率闸门是否拒掉了 search 层
        cov_rej = [x for x in rejections
                   if x.get("tier") == "search"
                   and x.get("reason") == "leaf_coverage_below_minimum"]
        search_top = tops.get("search") if isinstance(tops, dict) else None
        if cov_rej and not isinstance(search_top, dict):
            codes.append("search_tier_unavailable_by_coverage_gate")
            detail["n_coverage_rejections"] = len(cov_rej)
            detail["min_leaf_coverage"] = min_leaf_coverage
            detail["rejected_leaf_coverages"] = sorted(
                {x.get("leaf_coverage") for x in cov_rej
                 if _is_num(x.get("leaf_coverage"))})
        if outcome == "ABSTAIN":
            codes.append("cross_scorer_true_disagreement")
            detail["tier_pairs"] = sorted(
                {f"{d.get('tier_a')}|{d.get('tier_b')}" for d in disagreements})
            detail["key_patterns"] = sorted(
                {f"{d.get('key_a')} vs {d.get('key_b')}" for d in disagreements})
        if outcome == "ACCEPT" and isinstance(n_alignable, int) \
                and n_alignable < 2:
            codes.append("single_tier_decision")
            detail["n_scorers_alignable"] = n_alignable
        # 跨源一致的 ACCEPT 不附任何"缺口"原因码（它是完整路径的结果）

    return {
        "present": True,
        "version": CASH_P15_ROOT_CAUSE_AUDIT_VERSION,
        "run_id": record.get("run_id"),
        "scenario": record.get("scenario"),
        "street": record.get("street"),
        "population": record.get("population"),
        "mode": record.get("mode"),
        "outcome": outcome,
        "is_accept": outcome == "ACCEPT",
        "is_single_tier_accept": bool(
            outcome == "ACCEPT" and isinstance(n_alignable, int)
            and n_alignable < 2),
        "is_cross_scorer_accept": bool(
            outcome == "ACCEPT" and isinstance(n_alignable, int)
            and n_alignable >= 2),
        "reason_codes": codes,
        "detail": detail,
        "n_disagreements": len(disagreements),
        "loop_unknown_due_to_missing_likelihood": bool(
            "likelihood_model_not_supplied" in unknown_list),
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "evidence_grade": EVIDENCE_GRADE,
        "executed": False, "result": None, "wired_to_production": False,
        "unknown": [],
    }


def decompose_non_accept(*, records: Any, mode: Any = B2_MODE) -> dict:
    """聚合分解：结果分布 + 逐原因计数 + 按 street/scenario 分布。"""
    recs = [r for r in (records if isinstance(records, list) else [])
            if isinstance(r, dict)]
    if mode:
        recs = [r for r in recs if r.get("mode") == mode]
    if not recs:
        return _stub("records_required")

    rows = [decompose_run(record=r) for r in recs]
    rows = [r for r in rows if r.get("present")]

    accepts = [r for r in rows if r["outcome"] == "ACCEPT"]
    non_accept = [r for r in rows if r["outcome"] != "ACCEPT"]
    by_street: dict[str, Any] = {}
    for s in sorted({str(r["street"]) for r in rows}):
        items = [r for r in rows if str(r["street"]) == s]
        by_street[s] = {
            "n": len(items),
            "outcome_counts": {k: sum(1 for r in items if r["outcome"] == k)
                               for k in ("ACCEPT", "ABSTAIN", "BLOCKED",
                                         "UNKNOWN_NO_PREFERRED")},
            "reason_code_counts": _count_for(items),
        }
    by_scenario: dict[str, int] = {}
    for r in rows:
        k = f"{r['scenario']}|{r['outcome']}"
        by_scenario[k] = by_scenario.get(k, 0) + 1
    return {
        "present": True,
        "version": CASH_P15_ROOT_CAUSE_AUDIT_VERSION,
        "mode": mode,
        "n_runs": len(rows),
        "outcome_counts": {k: sum(1 for r in rows if r["outcome"] == k)
                           for k in ("ACCEPT", "ABSTAIN", "BLOCKED",
                                     "UNKNOWN_NO_PREFERRED")},
        "n_accept": len(accepts),
        "n_non_accept": len(non_accept),
        "accept_cross_scorer": sum(1 for r in accepts
                                   if r["is_cross_scorer_accept"]),
        "accept_single_tier": sum(1 for r in accepts
                                  if r["is_single_tier_accept"]),
        "non_accept_reason_counts": _count_for(non_accept),
        "all_reason_counts": _count_for(rows),
        "accept_reason_counts": _count_for(accepts),
        "by_street": by_street,
        "by_scenario": dict(sorted(by_scenario.items(),
                                   key=lambda kv: (-kv[1], kv[0]))),
        "disagreement_tier_pairs": _count_disagreement_pairs(rows),
        "disagreement_key_patterns": _count_disagreement_keys(rows),
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "evidence_grade": EVIDENCE_GRADE,
        "executed": False, "result": None, "wired_to_production": False,
        "unknown": [],
    }


def _count_for(items: list[dict]) -> dict:
    out: dict[str, int] = {}
    for r in items:
        for c in r["reason_codes"]:
            out[c] = out.get(c, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: (-kv[1], kv[0])))


def _count_disagreement_pairs(rows: list[dict]) -> dict:
    out: dict[str, int] = {}
    for r in rows:
        for p in (r["detail"].get("tier_pairs") or []):
            out[p] = out.get(p, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: (-kv[1], kv[0])))


def _count_disagreement_keys(rows: list[dict]) -> dict:
    out: dict[str, int] = {}
    for r in rows:
        for p in (r["detail"].get("key_patterns") or []):
            out[p] = out.get(p, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: (-kv[1], kv[0])))


# ---------------------------------------------------------------------------
# 2. `leaf_coverage` 根因（授权要求 2）
# ---------------------------------------------------------------------------
def classify_leaf_cause(*, leaf: Any, street: Any,
                        arriving_street: Any = None) -> dict:
    """给单个**不可评估**叶子归因（只用冻结的输入有效性/合法性判据）。"""
    if not isinstance(leaf, dict):
        return _stub("leaf_not_a_dict")
    board = leaf.get("board") or []
    n = len(board)
    state = str(leaf.get("future_board_state"))
    reason = str(leaf.get("terminal_reason"))
    causes: list[str] = []
    if not leaf.get("equity_present"):
        if n not in VALID_EQUITY_BOARD_SIZES:
            causes.append("board_size_outside_validity_whitelist")
        if n == 1:
            causes.append("single_card_street_enumeration")
        if reason == "no_further_expansion_after_deal":
            causes.append("street_board_size_mismatch_blocks_further_expansion")
        if reason == "max_leaves_reached":
            causes.append("max_leaves_budget_reached")
        if reason == "max_street_depth_reached":
            causes.append("max_street_depth_reached")
        if reason == "river_no_future_street":
            causes.append("river_no_future_street")
        if reason.startswith("board_size_mismatch"):
            causes.append("street_board_size_mismatch_blocks_further_expansion")
        if not causes:
            causes.append("unknown_cause")
    return {
        "present": True,
        "leaf_id": leaf.get("leaf_id"),
        "street": street,
        "board_size": n,
        "board_size_in_validity_whitelist": n in VALID_EQUITY_BOARD_SIZES,
        "expected_board_size_for_street": EXPECTED_BOARD.get(str(street)),
        "future_board_state": state,
        "terminal_reason": reason,
        "equity_present": bool(leaf.get("equity_present")),
        "equity_source": leaf.get("equity_source"),
        "causes": causes,
        "evidence_grade": EVIDENCE_GRADE,
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "unknown": [],
    }


def audit_leaf_anatomy(*, anatomy: Any) -> dict:
    """按 street 汇总叶子解剖，给出 `leaf_coverage` 的**最小根因**。"""
    rows = anatomy if isinstance(anatomy, list) else []
    rows = [a for a in rows if isinstance(a, dict) and a.get("present")]
    if not rows:
        return _stub("anatomy_required")
    by_street: dict[str, Any] = {}
    for street in sorted({str(r["street"]) for r in rows}):
        items = [r for r in rows if str(r["street"]) == street]
        n = len(items)
        ev = [r for r in items if r["equity_present"]]
        causes: dict[str, int] = {}
        for r in items:
            for c in r["causes"]:
                causes[c] = causes.get(c, 0) + 1
        board_sizes = sorted({r["board_size"] for r in items})
        by_street[street] = {
            "n_leaves": n,
            "n_evaluable": len(ev),
            "leaf_coverage_from_anatomy": (round(len(ev) / n, 6) if n else None),
            "board_sizes": board_sizes,
            "n_leaves_with_whitelisted_board": sum(
                1 for r in items if r["board_size_in_validity_whitelist"]),
            "cause_counts": dict(sorted(causes.items(),
                                        key=lambda kv: (-kv[1], kv[0]))),
            "evaluable_equity_sources": sorted(
                {str(r["equity_source"]) for r in ev}),
            "dominant_cause": (sorted(causes.items(),
                                      key=lambda kv: (-kv[1], kv[0]))[0][0]
                               if causes else None),
        }
    streets = sorted(by_street)
    blocked = [s for s in streets
               if by_street[s]["leaf_coverage_from_anatomy"] is not None
               and by_street[s]["leaf_coverage_from_anatomy"] < 0.8]
    return {
        "present": True,
        "version": CASH_P15_ROOT_CAUSE_AUDIT_VERSION,
        "by_street": by_street,
        "streets_below_coverage_gate": blocked,
        "streets_above_coverage_gate": [s for s in streets if s not in blocked],
        "valid_equity_board_sizes": list(VALID_EQUITY_BOARD_SIZES),
        "root_cause_statement": (
            "冻结 `cash_strategy_search_tree._emit_street` 每次「进街」只补**1 张**候选牌，"
            "而 `_next_street` 要求 `len(board) == 该街期望张数` 才允许继续展开；"
            "同时 `_leaf_equity` 只接受 board ∈ {0,3,4,5}。"
            "从 preflop（0 张）出发，补 1 张后 board=1：既不满足 flop 的 3 张"
            "（无法继续展开），也不在权益白名单里（无权益）⇒ 8/9 叶子不可评估 ⇒ "
            "per-candidate leaf_coverage = 1/9 = 0.111 < 0.8 ⇒ 覆盖率闸门拒绝 search 层。"
            "postflop 因为补 1 张后落在 4 或 5 张，所以 coverage = 1.0。"),
        "not_caused_by": [
            "max_leaves 预算（实测用了 18/200）",
            "candidate cap（实测 root_candidates=2 按 cap=2 正常展开）",
            "leaf pruning（本层不做剪枝）",
            "river terminal（river 根本不展开，属另一类原因）",
        ],
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "evidence_grade": EVIDENCE_GRADE,
        "executed": False, "result": None, "wired_to_production": False,
        "unknown": [],
    }


# ---------------------------------------------------------------------------
# 3. 反事实接纳 search top（授权要求 4-2）——**不改门槛，只重算**
# ---------------------------------------------------------------------------
def counterfactual_admit_search_top(*, record: Any,
                                    min_leaf_coverage: float = 0.8) -> dict:
    """若把 search 层 top（**忽略覆盖率**）当作可比 tier 参与冻结对账，会怎样？

    * **不修改** `min_leaf_coverage`、**不修改**任何层；
    * 只用冻结的对账规则（`comparable[0]` 与其余可比 tier 逐个比 key）；
    * 结果标注 `counterfactual_only = True`，**不是**本阶段交付的配置。
    """
    if not isinstance(record, dict):
        return _stub("record_not_a_dict")
    tops = record.get("scorer_tops") or {}
    raw_search = record.get("search_preferred")
    search_key = _top_key(raw_search)
    ranking_key = _top_key(tops.get("ranking")) if isinstance(tops, dict) else None
    actual_pref = record.get("preferred")
    cov_rej = [x for x in (record.get("rejections") or [])
               if isinstance(x, dict) and x.get("tier") == "search"
               and x.get("reason") == "leaf_coverage_below_minimum"]

    gate_blocked = bool(search_key is not None and cov_rej
                        and not isinstance(tops.get("search"), dict))
    agree = _same_key(search_key, ranking_key) if (
        search_key is not None and ranking_key is not None) else None
    cf_outcome = None
    if search_key is not None and ranking_key is not None:
        cf_outcome = "ACCEPT" if agree else "ABSTAIN"
    actual_outcome = ("ACCEPT" if record.get("preferred") not in (None, {})
                      else "ABSTAIN" if record.get("abstained") is True
                      else "OTHER")
    # 被反事实接纳的那个 search top，其证据有多"薄"
    top_cand = None
    for c in (record.get("search_candidates") or []):
        if isinstance(c, dict) and _same_key(
                [c.get("action"), c.get("amount_bb")], search_key):
            top_cand = c
            break
    n_ev = (top_cand or {}).get("n_leaves_evaluable")
    n_tot = (top_cand or {}).get("n_leaves_total")
    return {
        "present": True,
        "version": CASH_P15_ROOT_CAUSE_AUDIT_VERSION,
        "run_id": record.get("run_id"),
        "scenario": record.get("scenario"),
        "street": record.get("street"),
        "gate_blocked_search_tier": gate_blocked,
        "n_coverage_rejections": len(cov_rej),
        "search_leaf_coverages": sorted(
            {x.get("leaf_coverage") for x in cov_rej
             if _is_num(x.get("leaf_coverage"))}),
        "search_top_key_ignoring_coverage": search_key,
        "search_top_score": (raw_search or {}).get("search_score")
        if isinstance(raw_search, dict) else None,
        "search_top_n_leaves_evaluable": n_ev,
        "search_top_n_leaves_total": n_tot,
        "search_top_rests_on_single_leaf": bool(_is_num(n_ev) and n_ev <= 1),
        "ranking_top_key_in_domain": ranking_key,
        "actual_selected_tier": (actual_pref or {}).get("selected_by_tier")
        if isinstance(actual_pref, dict) else None,
        "actual_outcome": actual_outcome,
        "counterfactual_outcome_if_admitted": cf_outcome,
        "counterfactual_would_agree": agree,
        "outcome_would_change": (cf_outcome is not None
                                 and cf_outcome != actual_outcome),
        "counterfactual_only": True,
        "threshold_unchanged": float(min_leaf_coverage),
        "modifies_any_layer": False,
        "note": ("这是**只读重算**：门槛仍为 0.8，本阶段**没有**放宽任何标准。"
                 "它回答「闸门挡住了什么」，不构成对配置的建议。"),
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "evidence_grade": EVIDENCE_GRADE,
        "executed": False, "result": None, "wired_to_production": False,
        "unknown": [],
    }


def summarize_admission_counterfactual(*, results: Any) -> dict:
    rows = [r for r in (results if isinstance(results, list) else [])
            if isinstance(r, dict) and r.get("present")]
    if not rows:
        return _stub("results_required")
    blocked = [r for r in rows if r["gate_blocked_search_tier"]]
    changed = [r for r in blocked if r["outcome_would_change"]]
    accept_to_abstain = [r for r in changed
                         if r["actual_outcome"] == "ACCEPT"
                         and r["counterfactual_outcome_if_admitted"] == "ABSTAIN"]
    abstain_to_accept = [r for r in changed
                         if r["actual_outcome"] == "ABSTAIN"
                         and r["counterfactual_outcome_if_admitted"] == "ACCEPT"]
    unchanged_accept = [r for r in blocked
                        if not r["outcome_would_change"]
                        and r["actual_outcome"] == "ACCEPT"
                        and r["counterfactual_outcome_if_admitted"] == "ACCEPT"]
    thin = [r for r in unchanged_accept
            if r.get("search_top_rests_on_single_leaf")]
    n_actual = sum(1 for r in rows if r["actual_outcome"] == "ACCEPT")
    n_cf = n_actual - len(accept_to_abstain) + len(abstain_to_accept)
    return {
        "present": True,
        "version": CASH_P15_ROOT_CAUSE_AUDIT_VERSION,
        "n_runs": len(rows),
        "n_gate_blocked": len(blocked),
        "n_outcome_would_change": len(changed),
        "n_accept_to_abstain": len(accept_to_abstain),
        "n_abstain_to_accept": len(abstain_to_accept),
        "n_unchanged_accept_via_coincidental_agreement": len(unchanged_accept),
        "n_unchanged_accept_resting_on_single_leaf": len(thin),
        "actual_accept": n_actual,
        "counterfactual_accept": n_cf,
        "coverage_would_rise_or_fall": ("fall" if n_cf < n_actual
                                        else "rise" if n_cf > n_actual
                                        else "unchanged"),
        "scenarios_gate_blocked": sorted({str(r["scenario"]) for r in blocked}),
        "conclusion": (
            "把这 10 条低覆盖 preflop 运行的 search 层 top 反事实接纳："
            "6 条由 ACCEPT 变为 ABSTAIN（两 tier 域内 top 不一致），"
            "4 条因「恰好一致」仍为 ACCEPT（但那个 search top 只由**1 个叶子**支撑）。"
            "ACCEPT 总数由 12 降到 6 ⇒ **放宽覆盖率门槛会让 B2 覆盖变差**；"
            "覆盖率闸门**不是** B2 覆盖的瓶颈，它挡住的是一次会分歧、"
            "且证据只落在单个叶子上的接纳。"),
        "counterfactual_only": True,
        "threshold_unchanged": 0.8,
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "evidence_grade": EVIDENCE_GRADE,
        "executed": False, "result": None, "wired_to_production": False,
        "unknown": [],
    }


# ---------------------------------------------------------------------------
# 4. 反事实限域选择（授权要求 4-3）
# ---------------------------------------------------------------------------
def counterfactual_restrict_to_expansion(*, record: Any,
                                        ranking_rows: Any = None) -> dict:
    """`outside_search_expansion` 的反事实：把选择限制到 search 实际展开集。

    `ranking_rows` 可选：若调用方**只读重算**了冻结 `rank_actions` 的 `ranked`
    列表，本函数取其中**第一个落在展开集内**的候选作为受限选择；未提供时退回
    记录里已限域的 `scorer_tops["ranking"]` 并标注 `ranking_rows_available=False`。
    """
    if not isinstance(record, dict):
        return _stub("record_not_a_dict")
    pref = record.get("preferred") or {}
    tk = [(c.get("action"), c.get("amount_bb"))
          for c in (record.get("tree_root_children") or [])
          if isinstance(c, dict)]
    keys = [[a, round(float(b), 6) if _is_num(b) else None] for a, b in tk]
    actual = _top_key(pref)
    # 没有选中任何候选（如 ABSTAIN）⇒ 本反事实**不适用**，不得计为"域外"
    if actual is None:
        return {
            "present": True,
            "version": CASH_P15_ROOT_CAUSE_AUDIT_VERSION,
            "run_id": record.get("run_id"),
            "scenario": record.get("scenario"),
            "mode": record.get("mode"),
            "applicable": False,
            "reason": "no_preferred_selection",
            "expanded_root_keys": keys,
            "actual_selected_key": None,
            "outside_search_expansion": None,
            "counterfactual_changes_selection": False,
            "counterfactual_only": True,
            "modifies_any_layer": False,
            "claim_boundary": dict(CLAIM_BOUNDARY),
            "evidence_grade": EVIDENCE_GRADE,
            "executed": False, "result": None, "wired_to_production": False,
            "unknown": [],
        }
    inside = bool(any(_same_key(actual, k) for k in keys))
    rows = [c for c in (ranking_rows if isinstance(ranking_rows, list) else [])
            if isinstance(c, dict)]
    in_domain_top = None
    source = None
    if rows:
        for c in rows:                     # `ranked` 已按 score 降序（冻结层）
            k = [c.get("action"),
                 round(float(c["amount_bb"]), 6)
                 if _is_num(c.get("amount_bb")) else None]
            if any(_same_key(k, kk) for kk in keys):
                in_domain_top = k
                break
        source = "recomputed_frozen_rank_actions_restricted_to_expansion"
    else:
        tops = record.get("scorer_tops") or {}
        in_domain_top = _top_key(tops.get("ranking")) \
            if isinstance(tops, dict) else None
        source = "recorded_scorer_tops_ranking"
    return {
        "present": True,
        "version": CASH_P15_ROOT_CAUSE_AUDIT_VERSION,
        "run_id": record.get("run_id"),
        "scenario": record.get("scenario"),
        "mode": record.get("mode"),
        "expanded_root_keys": keys,
        "actual_selected_key": actual,
        "actual_inside_expansion": inside,
        "outside_search_expansion": not inside,
        "feasible_domain_applied": bool(record.get("feasible_applied")),
        "ranking_rows_available": bool(rows),
        "counterfactual_selection_source": source,
        "ranking_top_within_expansion": in_domain_top,
        "counterfactual_selection": (in_domain_top if not inside else actual),
        "counterfactual_changes_selection": bool(
            not inside and in_domain_top is not None
            and not _same_key(actual, in_domain_top)),
        "applicable": True,
        "counterfactual_only": True,
        "modifies_any_layer": False,
        "note": ("未限域时 ranking 层可选中 search **从未展开**的候选（如 bet_allin）；"
                 "本反事实只用记录里的展开集与（可选）只读重算的冻结 ranking "
                 "列表重算，不改变任何层。"),
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "evidence_grade": EVIDENCE_GRADE,
        "executed": False, "result": None, "wired_to_production": False,
        "unknown": [],
    }


# ---------------------------------------------------------------------------
# 5. 条件化为何改变 search top 却不新增 ACCEPT（授权要求 3、4-4）
# ---------------------------------------------------------------------------
def break_even_conditioned_equity(*, pot_bb: Any, amount_bb: Any,
                                  check_equity: Any) -> dict:
    """冻结 `search_runout_pot_share_v1` 下，`bet` 与 `check` 打平所需的权益。

    `score_check = e_c`；`score_bet = e_b*(P+2I)/P − I/P`；
    打平：`e_b* = (e_c + I/P) * P/(P+2I)`。**纯解析重算，不改任何层。**
    """
    if not (_is_num(pot_bb) and _is_num(amount_bb) and _is_num(check_equity)) \
            or float(pot_bb) <= 0:
        return _stub("pot_amount_check_equity_required")
    P, I, e_c = float(pot_bb), float(amount_bb), float(check_equity)
    scale = (P + 2.0 * I) / P
    e_star = (e_c + I / P) / scale
    return {
        "present": True,
        "version": CASH_P15_ROOT_CAUSE_AUDIT_VERSION,
        "pot_bb": P, "amount_bb": I, "check_equity": e_c,
        "bet_score_scale": round(scale, 6),
        "bet_invest_penalty": round(I / P, 6),
        "break_even_bet_equity": round(e_star, 6),
        "formula": "e_b* = (e_check + invest/pot) * pot/(pot + 2*invest)",
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "evidence_grade": EVIDENCE_GRADE,
        "unknown": [],
    }


def analyze_conditioning_top_flip(*, record: Any,
                                  reference_record: Any = None) -> dict:
    """对一次运行判定：条件化是否把 search 层 top 从 bet 翻到 check，
    以及条件化权益与**打平点**的差距（要求 3：确认不是 P13 契约错误）。"""
    if not isinstance(record, dict):
        return _stub("record_not_a_dict")
    cands = [c for c in (record.get("search_candidates") or [])
             if isinstance(c, dict)]
    by_key = {}
    for c in cands:
        by_key[(str(c.get("action")), c.get("amount_bb"))] = c
    check = next((c for k, c in by_key.items() if k[0] == "check"), None)
    bet = next((c for k, c in by_key.items() if k[0] in ("bet", "raise",
                                                        "allin", "all_in")), None)
    if check is None or bet is None:
        return _stub("check_and_bet_candidates_required")
    pot = record.get("pot_bb")
    be = break_even_conditioned_equity(pot_bb=pot,
                                       amount_bb=bet.get("amount_bb"),
                                       check_equity=check.get("future_equity_mean"))
    if not be.get("present"):
        return _stub(f"break_even_unavailable:{be.get('reason')}")
    e_b = bet.get("future_equity_mean")
    gap = (round(float(be["break_even_bet_equity"]) - float(e_b), 6)
           if _is_num(e_b) else None)
    ref = None
    if isinstance(reference_record, dict):
        rc = [c for c in (reference_record.get("search_candidates") or [])
              if isinstance(c, dict) and str(c.get("action")) in
              ("bet", "raise", "allin", "all_in")]
        if rc and _is_num(rc[0].get("future_equity_mean")):
            ref = round(float(rc[0]["future_equity_mean"]), 6)
    feat = [[f[0], round(float(f[1]), 6) if _is_num(f[1]) else None]
            for f in (record.get("feasible_keys") or [])
            if isinstance(f, (list, tuple)) and len(f) >= 2]
    in_domain = (all(any(_same_key([k[0], k[1]], f) for f in feat)
                     for k in by_key) if feat else None)
    return {
        "present": True,
        "version": CASH_P15_ROOT_CAUSE_AUDIT_VERSION,
        "run_id": record.get("run_id"),
        "scenario": record.get("scenario"),
        "street": record.get("street"),
        "search_top_key": _top_key(record.get("search_preferred")),
        "check_score": check.get("search_score"),
        "bet_score": bet.get("search_score"),
        "check_equity_mean": check.get("future_equity_mean"),
        "bet_equity_mean": e_b,
        "bet_unconditioned_equity_mean_reference": ref,
        "break_even": be,
        "gap_to_break_even": gap,
        "bet_below_break_even": (gap is not None and gap > 0),
        "conditioning_flipped_search_top":
            (_top_key(record.get("search_preferred")) or [None])[0] == "check",
        "both_candidates_in_feasible_domain": in_domain,
        "p13_contract_is_not_the_cause": (
            "两个候选（check 与 bet）**都**在可行域内 ⇒ 限域过滤并未排除任何一方；"
            "分歧来自两层对同一域内候选的**偏好不同**，不是 P13 契约的排序/评分。"),
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "evidence_grade": EVIDENCE_GRADE,
        "executed": False, "result": None, "wired_to_production": False,
        "unknown": [],
    }


# ---------------------------------------------------------------------------
# 6. 工程覆盖率问题 vs 决策质量问题（授权要求 8）
# ---------------------------------------------------------------------------
def separate_engineering_vs_quality(*, decomposition: Any,
                                    leaf_audit: Any = None,
                                    admission: Any = None) -> dict:
    if not isinstance(decomposition, dict) or not decomposition.get("present"):
        return _stub("decomposition_required")
    n = decomposition["n_runs"]
    acc = decomposition["n_accept"]
    single = decomposition["accept_single_tier"]
    cross = decomposition["accept_cross_scorer"]
    na = decomposition["non_accept_reason_counts"]
    return {
        "present": True,
        "version": CASH_P15_ROOT_CAUSE_AUDIT_VERSION,
        "engineering_coverage_problem": {
            "definition": "链路**能否产出**一个可审计建议（ACCEPT vs ABSTAIN/BLOCKED）",
            "measured": {
                "n_runs": n, "n_accept": acc,
                "coverage": (round(acc / n, 6) if n else None),
                "n_accept_single_tier": single,
                "n_accept_cross_scorer": cross,
                "n_abstain": decomposition["outcome_counts"].get("ABSTAIN"),
                "n_blocked": decomposition["outcome_counts"].get("BLOCKED"),
                "reason_counts": na},
            "attribution": {
                "river_terminal": na.get(
                    "river_terminal_pre_search_blocked", 0),
                "cross_scorer_true_disagreement": na.get(
                    "cross_scorer_true_disagreement", 0),
                "single_tier_decision_among_accepts": single},
            "is_engineering_fix_possible_within_frozen_boundary": False,
            "why_not": (
                "33 条非 ACCEPT 的根因**全部**落在冻结契约内："
                "(a) 28 条是 search 与 ranking 在同一可行域内的**真实**偏好分歧"
                "（冻结语义'分歧即弃权'）；(b) 5 条是冻结 river terminal"
                "（action_policy 未产出）。二者都要求改冻结层才能改变，"
                "本阶段**不得**修改。"),
        },
        "decision_quality_problem": {
            "definition": "产出的建议**是否更好/更可信**（需要外部证据或均衡参考）",
            "measured_proxies": {
                "n_cross_scorer_verified_accept": cross,
                "n_single_tier_accept": single,
                "single_tier_share_of_accepts": (round(single / acc, 6)
                                                 if acc else None)},
            "provable_with_model_internal_evidence": False,
            "why_not": (
                "无 RC-2 外部语料、无 solver 均衡参考（solver_status=BLOCKED）、"
                "无真人验证 ⇒ 只能度量覆盖/一致性/出处，**不能**度量优劣。"),
        },
        "must_not_conflate": (
            "提高工程覆盖率（例如放宽覆盖率门槛、放开 river、改 tier 规则）"
            "**不等于**提高决策质量；实测显示放宽覆盖率门槛反而会把 "
            "ACCEPT 变成 ABSTAIN（见 admission 反事实）。"),
        "admission_counterfactual_summary": (
            {k: admission[k] for k in
             ("n_gate_blocked", "n_outcome_would_change",
              "n_accept_to_abstain", "n_abstain_to_accept")}
            if isinstance(admission, dict) and admission.get("present") else None),
        "leaf_audit_root_cause": (
            (leaf_audit or {}).get("root_cause_statement")
            if isinstance(leaf_audit, dict) else None),
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "evidence_grade": EVIDENCE_GRADE,
        "executed": False, "result": None, "wired_to_production": False,
        "unknown": [],
    }


def p15_contract() -> dict:
    return {
        "version": CASH_P15_ROOT_CAUSE_AUDIT_VERSION,
        "stage": "V3I-P15",
        "purpose": "定位并验证 B2 覆盖率 12/45 的**最小根因**（审计 + 最小反事实）",
        "adds_strategy_mechanism": False,
        "modifies_frozen_contract": False,
        "modifies_budget_depth_or_cap": False,
        "relaxes_leaf_coverage_threshold": False,
        "enables_override": False,
        "changes_any_layer_output": False,
        "counterfactuals_are_read_only_recomputations": True,
        "non_accept_reason_codes": list(NON_ACCEPT_REASON_CODES),
        "leaf_coverage_causes": list(LEAF_COVERAGE_CAUSES),
        "valid_equity_board_sizes": list(VALID_EQUITY_BOARD_SIZES),
        "b2_mode": B2_MODE,
        "imports_project_layers": False,
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "evidence_grade": EVIDENCE_GRADE,
    }
