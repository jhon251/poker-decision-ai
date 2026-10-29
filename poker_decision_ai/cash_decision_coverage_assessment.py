"""V3I-P11 — CASH Decision Loop 的**可决策覆盖率**与**真实可用决策边界**评估层。

**本层不新增任何扑克策略机制、不修改任何冻结契约**：只做统计、分类与边界刻画。

审计结论（授权要求 1）
----------------------
P1→P10 当前完整链路（全部为 `MODEL_INTERNAL`，无外部证据）：

```
V3I-P1 run_decision_loop（依赖注入，8 个必需阶段）
  1 action_policy     cash_strategy_action_policy      （冻结）
  2 dynamic_belief    cash_strategy_dynamic_belief     （V3H-4，冻结）
  3 belief_update     cash_strategy_belief_update      （V3H-6，仅当提供 likelihood）
  4 action_ranking    cash_strategy_action_ranking     （冻结）
  5 belief_conditioned cash_strategy_belief_conditioned_eval（仅在 belief_update 可用时）
  6 search_tree       cash_strategy_search_tree        （冻结，root 展开 ap.candidates[:cap]）
  7 search_eval       cash_strategy_search_eval        （冻结；P8 后可被 P9 闭包条件化）
  8 decision_assembly cash_strategy_decision_assembly  （V3H-8，冻结）
```

V3H-8 的**唯一**接受路径（源码核对）
------------------------------------
* `TIERS = ("search", "belief_conditioned", "ranking")`，`on_disagreement="abstain_unknown"`；
* `agreed_key` 取自 `comparable[0]`（= search），随后与其他每个 tier 的 top key 比较；
  任何不一致 ⇒ 记一条 `disagreements`，`note="top candidates differ; no averaging, no guessing"`；
* `abstained = bool(disagreements) and not override_used`；
* `override_used` 需**调用方显式**传 `allow_higher_tier_override=True`
  **且** `override_justification` 为非空字符串；
* 因此 **ACCEPT 的必要条件**：所有可比 tier 的 top key 一致（或被合法 override）。

本层只 import 标准库。
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

CASH_P11_COVERAGE_VERSION = "v3i-p11-decision-coverage-assessment-v1"

#: 覆盖率分层（授权要求 3）
COVERAGE_LAYERS: tuple[str, ...] = (
    "search_capability",
    "conditioning_capability",
    "ranking_capability",
    "final_decision_capability",
)

#: 最终决策结果分类
DECISION_CLASSES: tuple[str, ...] = (
    "ACCEPT",      # assembly 产出了 preferred（未弃权）
    "ABSTAIN",     # assembly 按冻结规则弃权
    "BLOCKED",     # loop 在 assembly 之前即停止
    "UNKNOWN",     # loop 明确报 UNKNOWN
)

#: 结构化原因分类（授权要求 4）——不得把 ABSTAIN/BLOCKED 当测试失败
REASON_TAXONOMY: dict[str, tuple[str, ...]] = {
    "pre_search_blocked": (
        "action_policy_unavailable",
        "dynamic_belief_unavailable",
        "action_ranking_unavailable",
        "environment_action_policy_blocked",
    ),
    "search_layer_abstain": (
        "no_root_candidates_expanded",
        "search_tree_unavailable",
        "no_evaluable_leaves",
        "leaf_coverage_below_minimum",
        "search_score_unavailable",
    ),
    "scorer_unavailable": (
        "search_top_candidate_unavailable",
        "belief_conditioned_top_candidate_unavailable",
        "ranking_top_candidate_unavailable",
        "belief_conditioned_not_supplied",
        "likelihood_model_not_supplied",
    ),
    "cross_scorer_disagreement": (
        "cross_scorer_disagreement",
        "search_vs_belief_conditioned_top_key_mismatch",
        "search_vs_ranking_top_key_mismatch",
    ),
    "assembly_blocked": (
        "stage_failed:decision_assembly",
        "decision_assembly_unavailable",
    ),
    "conditioning_fail_closed": (
        "prepared_missing",
        "bridge_unavailable",
        "prior_only_has_no_opponent_evidence",
        "conditioned_input_not_per_candidate",
        "passthrough_materialize_failed",
        "population_tolerance_unavailable",
    ),
}

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
    "wired_to_production": False,
    "adds_strategy_capability": False,
    "modifies_frozen_contract": False,
    "relaxes_fail_closed_for_coverage": False,
}

EVIDENCE_GRADE = "MODEL_INTERNAL"


def _is_num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _stub(reason: str) -> dict:
    return {
        "present": False,
        "version": CASH_P11_COVERAGE_VERSION,
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


# ---------------------------------------------------------------------------
# 1. 单条记录 → 覆盖分层与原因分类
# ---------------------------------------------------------------------------
def classify_record(*, record: Any) -> dict:
    """把一次运行的记录解析为 (分层覆盖, 决策分类, 结构化原因)。"""
    if not isinstance(record, dict):
        return _stub("record_not_a_dict")

    trace = record.get("stage_trace") or []
    reached = {t.get("stage"): t.get("present") for t in trace
               if isinstance(t, dict)}
    n_cand = record.get("n_candidates")
    n_cand = int(n_cand) if isinstance(n_cand, int) else 0
    abstained = record.get("abstained")
    loop_status = record.get("loop_status")
    preferred = record.get("assembly_preferred")
    disagreements = record.get("disagreements") or []
    unknown_list = [str(u) for u in (record.get("unknown") or [])]

    # ── 分层覆盖 ──
    search_ok = n_cand > 0
    means = [v for v in (record.get("candidate_means") or {}).values()
             if _is_num(v)]
    non_pooled = bool(len(set(means)) > 1)
    # 「条件化**已应用**」以 routed 标记为准（稳健）；non_pooled 是"候选级生效"
    # 的附加证据。二者不可互相替代：条件化可能已应用而各候选恰好同值。
    routed = record.get("routed")
    conditioning_applied = bool(routed is True) if routed is not None \
        else bool(str(record.get("conditioning_status") or "") == "conditioned")
    conditioning_ok = bool(search_ok and conditioning_applied and non_pooled)
    ranking_ok = bool(reached.get("action_ranking") is True)
    if abstained is True:
        cls = "ABSTAIN"
    elif preferred not in (None, {}):
        cls = "ACCEPT"
    elif loop_status == "UNKNOWN":
        cls = "UNKNOWN"
    else:
        cls = "BLOCKED"
    final_ok = cls == "ACCEPT"

    # ── 结构化原因 ──
    reasons: list[dict] = []
    if cls == "ABSTAIN":
        for d in disagreements:
            if not isinstance(d, dict):
                continue
            a = d.get("tier_a")
            b = d.get("tier_b")
            a_s = str(a) if a is not None else "unavailable_tier"
            b_s = str(b) if b is not None else "unavailable_tier"
            reasons.append({
                "category": "cross_scorer_disagreement",
                "code": f"{a_s}_vs_{b_s}_top_key_mismatch",
                "detail": str(d.get("note")),
                "tier_a": a, "key_a": d.get("key_a"),
                "tier_b": b, "key_b": d.get("key_b"),
                "both_tiers_comparable": a is not None and b is not None,
            })
        if not reasons:
            reasons.append({"category": "cross_scorer_disagreement",
                            "code": "cross_scorer_disagreement",
                            "detail": record.get("abstention_reason")})
    if cls in ("BLOCKED", "UNKNOWN"):
        absent = [k for k, v in reached.items() if v is False]
        if not search_ok:
            if reached.get("action_policy") is False:
                reasons.append({"category": "pre_search_blocked",
                                "code": "environment_action_policy_blocked",
                                "detail": ("action_policy 阶段未产出（既有冻结语义，"
                                           "如 river terminal）")})
            elif reached.get("search_tree") is False:
                reasons.append({"category": "search_layer_abstain",
                                "code": "search_tree_unavailable",
                                "detail": "search_tree 阶段未产出"})
            else:
                reasons.append({"category": "pre_search_blocked",
                                "code": "stage_absent_before_search",
                                "detail": f"absent_stages={absent}"})
        else:
            reasons.append({"category": "assembly_blocked",
                            "code": "stage_failed:decision_assembly",
                            "detail": f"absent_stages={absent}"})
    for u in unknown_list:
        for cat, codes in REASON_TAXONOMY.items():
            for c in codes:
                if c in u:
                    reasons.append({"category": cat, "code": c, "detail": u})
    # conditioning 层：只有"**期望**条件化但未生效"才算 fail-closed。
    # 基线透传（population 为基线）本来就是预期行为，不计为失败。
    pop_name = str(record.get("population") or "")
    is_baseline = "BASELINE" in pop_name.upper()
    cs = str(record.get("conditioning_status") or "")
    if not conditioning_applied and not is_baseline:
        reasons.append({"category": "conditioning_fail_closed",
                        "code": cs or "conditioning_not_applied",
                        "detail": "条件化未生效（回退原树）"})
    elif is_baseline and cs == "passthrough":
        reasons.append({"category": "expected_passthrough",
                        "code": "baseline_passthrough_by_design",
                        "detail": "基线运行本来就不带条件化输入"})
    if conditioning_applied and not non_pooled and search_ok:
        reasons.append({"category": "conditioning_fail_closed",
                        "code": "conditioned_but_pooled",
                        "detail": "条件化已应用但各候选评估未区分（池化）"})

    return {
        "present": True,
        "version": CASH_P11_COVERAGE_VERSION,
        "scenario": record.get("scenario"),
        "street": record.get("street"),
        "population": record.get("population"),
        "n_candidates": n_cand,
        "coverage": {
            "search_capability": search_ok,
            "conditioning_capability": conditioning_ok,
            "ranking_capability": ranking_ok,
            "final_decision_capability": final_ok,
        },
        "decision_class": cls,
        "loop_status": loop_status,
        "abstained": abstained,
        "assembly_preferred": preferred,
        "selected_by_tier": record.get("selected_by_tier"),
        "n_disagreements": len(disagreements),
        "reasons": reasons,
        "reason_categories": sorted({r["category"] for r in reasons}),
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "evidence_grade": EVIDENCE_GRADE,
        "executed": False, "result": None, "wired_to_production": False,
        "unknown": [],
    }


# ---------------------------------------------------------------------------
# 2. 汇总覆盖率与原因分布
# ---------------------------------------------------------------------------
def aggregate_coverage(*, records: Any, group_by: Any = None) -> dict:
    """汇总覆盖率；可按 `group_by`（如 "street" / "population"）分组。"""
    recs = records if isinstance(records, list) else []
    if not recs:
        return _stub("records_required")

    classified = [classify_record(record=r) for r in recs]
    ok = [c for c in classified if c.get("present")]
    if not ok:
        return _stub("no_classifiable_records")

    def summ(items: list[dict]) -> dict:
        n = len(items)
        counts = {k: 0 for k in DECISION_CLASSES}
        for c in items:
            counts[c["decision_class"]] = counts.get(c["decision_class"], 0) + 1
        cov = {k: 0 for k in COVERAGE_LAYERS}
        for c in items:
            for k in COVERAGE_LAYERS:
                if c["coverage"].get(k):
                    cov[k] += 1
        reason_counts: dict[str, int] = {}
        for c in items:
            for r in c["reasons"]:
                reason_counts[r["code"]] = reason_counts.get(r["code"], 0) + 1
        cat_counts: dict[str, int] = {}
        for c in items:
            for cat in c["reason_categories"]:
                cat_counts[cat] = cat_counts.get(cat, 0) + 1
        return {
            "n_runs": n,
            "decision_class_counts": counts,
            "decision_class_fraction": {
                k: (round(v / n, 6) if n else None) for k, v in counts.items()},
            "coverage_counts": cov,
            "coverage_fraction": {k: (round(v / n, 6) if n else None)
                                  for k, v in cov.items()},
            "reason_code_counts": dict(sorted(reason_counts.items(),
                                              key=lambda kv: (-kv[1], kv[0]))),
            "reason_category_counts": dict(sorted(cat_counts.items(),
                                                  key=lambda kv: (-kv[1], kv[0]))),
        }

    groups: dict[str, Any] = {}
    if isinstance(group_by, str) and group_by:
        keys = sorted({str(c.get(group_by)) for c in ok})
        for k in keys:
            groups[k] = summ([c for c in ok if str(c.get(group_by)) == k])

    overall = summ(ok)
    # 冻结 tier 分歧对最终覆盖率的影响
    dis = [c for c in ok if c["n_disagreements"] > 0]
    no_dis = [c for c in ok if c["n_disagreements"] == 0]
    return {
        "present": True,
        "version": CASH_P11_COVERAGE_VERSION,
        "overall": overall,
        "groups": groups,
        "group_by": group_by,
        "search_capability_runs": overall["coverage_counts"]["search_capability"],
        "ranking_capability_runs": overall["coverage_counts"]["ranking_capability"],
        "final_decision_runs": overall["coverage_counts"]["final_decision_capability"],
        "gap_search_to_final": (
            overall["coverage_counts"]["search_capability"]
            - overall["coverage_counts"]["final_decision_capability"]),
        "abstain_attributable_to_tier_disagreement": len(dis),
        "abstain_without_disagreement": len(no_dis) - len(
            [c for c in no_dis if c["decision_class"] == "ACCEPT"]),
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "evidence_grade": EVIDENCE_GRADE,
        "executed": False, "result": None, "wired_to_production": False,
        "unknown": [],
    }


# ---------------------------------------------------------------------------
# 3. V3H-8 tier 分歧的专项分析（要求 5）
# ---------------------------------------------------------------------------
def analyze_tier_disagreement(*, classified_records: Any) -> dict:
    """量化 V3H-8 tier 分歧对最终决策覆盖率的影响。"""
    recs = classified_records if isinstance(classified_records, list) else []
    ok = [c for c in recs if isinstance(c, dict) and c.get("present")]
    if not ok:
        return _stub("classified_records_required")

    total = len(ok)
    with_dis = [c for c in ok if c["n_disagreements"] > 0]
    accept = [c for c in ok if c["decision_class"] == "ACCEPT"]
    abstain = [c for c in ok if c["decision_class"] == "ABSTAIN"]
    blocked = [c for c in ok if c["decision_class"] == "BLOCKED"]

    pair_counts: dict[str, int] = {}
    key_patterns: dict[str, int] = {}
    for c in with_dis:
        for r in c["reasons"]:
            if r["category"] != "cross_scorer_disagreement":
                continue
            a = r.get("tier_a")
            b = r.get("tier_b")
            a_lab = str(a) if isinstance(a, str) and a else "(unavailable_tier)"
            b_lab = str(b) if isinstance(b, str) and b else "(unavailable_tier)"
            pair = f"{a_lab}|{b_lab}"
            pair_counts[pair] = pair_counts.get(pair, 0) + 1
            ka = r.get("key_a")
            kb = r.get("key_b")
            pat = f"{ka} vs {kb}"
            key_patterns[pat] = key_patterns.get(pat, 0) + 1

    # 反事实：若分歧被合法 override，覆盖率上限是多少
    upper_bound = len(accept) + len(abstain)

    return {
        "present": True,
        "version": CASH_P11_COVERAGE_VERSION,
        "total_runs": total,
        "runs_with_disagreement": len(with_dis),
        "accept_runs": len(accept),
        "abstain_runs": len(abstain),
        "blocked_runs": len(blocked),
        "final_coverage": round(len(accept) / total, 6) if total else None,
        "abstain_rate": round(len(abstain) / total, 6) if total else None,
        "disagreement_pair_counts": dict(sorted(
            pair_counts.items(), key=lambda kv: (-kv[1], kv[0]))),
        "disagreement_key_patterns": dict(sorted(
            key_patterns.items(), key=lambda kv: (-kv[1], kv[0]))),
        "upper_bound_coverage_if_override_allowed": (
            round(upper_bound / total, 6) if total else None),
        "frozen_accept_condition": (
            "V3H-8: abstained = bool(disagreements) and not override_used；"
            "override_used 需调用方显式 allow_higher_tier_override=True 且 "
            "override_justification 非空"),
        "override_used_in_runs": any(c.get("selected_by_tier") is not None
                                     and c["decision_class"] == "ACCEPT"
                                     for c in ok),
        "impact_statement": (
            "在本网格中，凡三层 top key 不一致的运行，最终决策能力即被冻结的"
            "'分歧即弃权'规则阻断；这是 V3H-8 的**既定语义**，不是缺陷。"),
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "evidence_grade": EVIDENCE_GRADE,
        "executed": False, "result": None, "wired_to_production": False,
        "unknown": [],
    }


def coverage_boundary_statement(*, coverage: Any, disagreement: Any) -> dict:
    """给出"真实可用决策边界"的分层结论。"""
    if not isinstance(coverage, dict) or not coverage.get("present"):
        return _stub("coverage_required")
    if not isinstance(disagreement, dict) or not disagreement.get("present"):
        return _stub("disagreement_analysis_required")
    o = coverage["overall"]
    return {
        "present": True,
        "version": CASH_P11_COVERAGE_VERSION,
        "search_capability_fraction": o["coverage_fraction"]["search_capability"],
        "conditioning_capability_fraction":
            o["coverage_fraction"]["conditioning_capability"],
        "ranking_capability_fraction": o["coverage_fraction"]["ranking_capability"],
        "final_decision_capability_fraction":
            o["coverage_fraction"]["final_decision_capability"],
        "dominant_blocking_layer": (
            "final_decision_capability" if
            o["coverage_fraction"]["search_capability"] >
            o["coverage_fraction"]["final_decision_capability"]
            else "none"),
        "boundary": (
            "可用决策边界：search 与 ranking 能力可达，但**最终决策**受冻结 V3H-8 "
            "'三层 top key 不一致即弃权'约束；在本网格中该约束是主要阻断来源。"),
        "to_raise_final_coverage_requires": (
            "要么改变 tier 规则/容忍度（冻结契约），要么由调用方提供显式且可辩护的 "
            "override_justification（本阶段**未**启用任何 override）。"),
        "requires_frozen_contract_change": True,
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "evidence_grade": EVIDENCE_GRADE,
        "executed": False, "result": None, "wired_to_production": False,
        "unknown": [],
    }


def p11_contract() -> dict:
    return {
        "version": CASH_P11_COVERAGE_VERSION,
        "adds_strategy_mechanism": False,
        "modifies_frozen_contract": False,
        "relaxes_fail_closed": False,
        "coverage_layers": list(COVERAGE_LAYERS),
        "decision_classes": list(DECISION_CLASSES),
        "reason_taxonomy": {k: list(v) for k, v in REASON_TAXONOMY.items()},
        "abstain_and_blocked_are_not_test_failures": True,
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "evidence_grade": EVIDENCE_GRADE,
    }
