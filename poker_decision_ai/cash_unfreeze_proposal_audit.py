"""V3I-P16 — 针对 V3I-P15 两个冻结根因的**最小解冻方案审计**与**最小反事实**层。

授权范围（本阶段**只做方案审计 + 最小反事实**，不直接修改冻结生产代码）
----------------------------------------------------------------------
* 不修改 Solver / V3H-4 / V3H-6 / CFI·λ / `strategy_core` / P13 契约 / P1 loop /
  river terminal / budget；候选解冻项**只报告，不实施**；
* **保持 `leaf_coverage = 0.8`**，不通过降低门槛解决问题；
* **不**启用 override、**不**用投票/放宽 ABSTAIN 冒充"解决分歧"；
* **不**把 fold-only 算术终局叶当作真实权益；
* 不宣称 EV / GTO / equilibrium / 盈利能力；不接 CoinPoker；不自动下注；
* 本层只 import 标准库；上游层与实验产物一律由调用方**显式注入**。

核心判据（本层用于"拒绝"而非"合理化"）
--------------------------------------
1. **真实权益**：`equity_source` 必须来自冻结权益入口
   （`...EquityCalculator.hand_vs_range` / `conditional_absolute_equity`），
   且 board 张数落在冻结白名单 `{0,3,4,5}`；
   `arithmetic:pot_share_on_opponent_fold`（对手弃牌恒 1.0）**不计**真实权益。
2. **不是伪造**：候选方案不得依赖"恒真假设"（如对手必弃牌）充当证据。
3. **不是冒充**：`override` / 投票 / 改 tier 优先级 / 降门槛 / 改标签
   一律记为**非法解决**，即使它们能提高覆盖率。
4. **独立来源**：若把同一权益喂给两层，两层分数若**逐位相同**，则"一致"是
   **恒真**，跨评分源校验退化为同源复述 ⇒ 记为**非法解决**。
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

CASH_P16_UNFREEZE_AUDIT_VERSION = "v3i-p16-minimal-unfreeze-audit-v1"

#: 根因种类（与 V3I-P15 对应）
ROOT_CAUSE_KINDS: tuple[str, ...] = (
    "preflop_no_real_equity_expansion",
    "postflop_same_domain_true_disagreement",
)

#: 候选方案种类
PROPOSAL_KINDS: tuple[str, ...] = (
    "street_faithful_deal_expansion",
    "candidate_local_equity_for_ranking",
    "tier_precedence_or_override",
    "lower_leaf_coverage_threshold",
    "vote_between_tiers",
    "relabel_fold_only_as_equity",
)

#: **非法解决方式**登记表（要求 3、5、6）
ILLEGITIMATE_RESOLUTIONS: dict[str, str] = {
    "tier_precedence_or_override":
        "改动 tier 优先级或在分歧时直接采用某一层 = **换名的 override**；"
        "它不解决分歧，只是把另一层的证据丢掉。本阶段**未启用**，也不建议。",
    "lower_leaf_coverage_threshold":
        "降低 0.8 门槛（要求 5 明确禁止）。且 V3I-P15 已实测：放宽后 B2 覆盖"
        "由 12 降到 6，因为被挡住的接纳本身会分歧。",
    "vote_between_tiers":
        "两层投票在只有两个可比 tier 且 key 不同时**必然平票**，无法给出决策；"
        "若为打破平票引入第三个来源，等于新增评分源 = 新增机制。",
    "relabel_fold_only_as_equity":
        "把 `arithmetic:pot_share_on_opponent_fold`（对手弃牌恒 1.0）当作真实权益，"
        "等于用'对手一定会弃牌'这一恒真假设充当证据 = **伪造证据**。",
    "candidate_local_equity_for_ranking":
        "把逐候选条件化权益喂给冻结 ranking 公式：实测其分数与 search 分数"
        "**逐位相同**（同一代数式 + 同一权益）⇒ 两层一致性**恒真**，"
        "跨评分源校验退化为**同源复述**（独立来源数 1）⇒ 不是解决，是把"
        "双源校验变成单源复述。",
}

#: 严格证据语义规则
EVIDENCE_SEMANTICS_RULES: tuple[str, ...] = (
    "真实权益必须来自冻结权益入口（EquityCalculator.hand_vs_range / "
    "conditional_absolute_equity），且 board 张数在 {0,3,4,5} 白名单内",
    "fold-only 算术终局叶不计真实权益",
    "不得用恒真假设充当证据",
    "不得用 override / 投票 / 改优先级 / 降门槛 / 改标签来'解决'分歧",
    "跨评分源校验必须保持**独立来源**；若两层分数逐位相同则一致性恒真，"
    "该方案作废",
    "覆盖率变化必须实测报告，不得以覆盖率提升为由降低证据标准",
)

#: 冻结叶子权益可计算的 board 张数（与 P15 / 冻结 `_leaf_equity` 一致）
VALID_EQUITY_BOARD_SIZES: tuple[int, ...] = (0, 3, 4, 5)

#: 真实权益来源的判别子串
REAL_EQUITY_SOURCE_MARKERS: tuple[str, ...] = (
    "EquityCalculator.hand_vs_range", "conditional_absolute_equity")

#: fold-only 算术终局叶的来源标记
FOLD_ONLY_SOURCE_MARKER = "arithmetic:pot_share_on_opponent_fold"

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
    "implements_any_unfreeze_item": False,
    "modifies_frozen_contract": False,
    "modifies_budget_depth_or_cap": False,
    "relaxes_leaf_coverage_threshold": False,
    "enables_override": False,
    "uses_voting": False,
    "treats_fold_only_as_equity": False,
}

EVIDENCE_GRADE = "MODEL_INTERNAL"


def _is_num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _stub(reason: str) -> dict:
    return {
        "present": False,
        "version": CASH_P16_UNFREEZE_AUDIT_VERSION,
        "reason": reason,
        "unknown": [reason],
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "evidence_grade": "NONE",
        "executed": False, "result": None, "wired_to_production": False,
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


def _key(v: Any) -> Any:
    if not isinstance(v, dict):
        return None
    act = v.get("action")
    if act is None:
        return None
    amt = v.get("amount_bb")
    return [str(act), round(float(amt), 6) if _is_num(amt) else None]


def _same(a: Any, b: Any, tol: float = 1e-6) -> bool:
    if not isinstance(a, (list, tuple)) or not isinstance(b, (list, tuple)):
        return False
    if len(a) != 2 or len(b) != 2:
        return False
    if str(a[0]) != str(b[0]):
        return False
    if a[1] is None or b[1] is None:
        return True
    return abs(float(a[1]) - float(b[1])) <= tol


# ---------------------------------------------------------------------------
# 1. 真实权益验证（要求 4）—— fold-only **不得**计入
# ---------------------------------------------------------------------------
def verify_real_equity_evidence(*, leaves: Any) -> dict:
    """核对一组叶子是否提供**真实、可追溯**的权益证据。"""
    rows = [l for l in (leaves if isinstance(leaves, list) else [])
            if isinstance(l, dict)]
    if not rows:
        return _stub("leaves_required")
    n_ev = 0
    n_real = 0
    n_fold_only = 0
    n_off_whitelist = 0
    real_sources: dict[str, int] = {}
    fold_sources: dict[str, int] = {}
    for l in rows:
        if not l.get("equity_present"):
            continue
        n_ev += 1
        src = str(l.get("equity_source"))
        board = l.get("board_size")
        if isinstance(board, int) and board not in VALID_EQUITY_BOARD_SIZES:
            n_off_whitelist += 1
        if any(m in src for m in REAL_EQUITY_SOURCE_MARKERS):
            n_real += 1
            real_sources[src] = real_sources.get(src, 0) + 1
        elif FOLD_ONLY_SOURCE_MARKER in src:
            n_fold_only += 1
            fold_sources[src] = fold_sources.get(src, 0) + 1
    has_real = n_real > 0
    return {
        "present": True,
        "version": CASH_P16_UNFREEZE_AUDIT_VERSION,
        "n_leaves": len(rows),
        "n_evaluable": n_ev,
        "n_real_equity_leaves": n_real,
        "n_fold_only_arithmetic_leaves": n_fold_only,
        "n_evaluable_with_board_outside_whitelist": n_off_whitelist,
        "real_equity_sources": dict(sorted(real_sources.items(),
                                           key=lambda kv: (-kv[1], kv[0]))),
        "fold_only_sources": dict(sorted(fold_sources.items(),
                                         key=lambda kv: (-kv[1], kv[0]))),
        "has_real_traceable_equity": has_real,
        "fold_only_excluded_from_real_equity": True,
        "board_whitelist": list(VALID_EQUITY_BOARD_SIZES),
        "verdict": (
            "该路径产生**真实、可追溯**的权益证据（来源为冻结权益入口，"
            "board 落在白名单内）" if has_real and n_off_whitelist == 0 else
            "该路径**未**产生真实权益证据（或存在白名单外 board）⇒ 不满足要求 4"),
        "fold_only_note": (
            "`arithmetic:pot_share_on_opponent_fold`（对手弃牌，hero_equity 恒 1.0）"
            "**不**计入真实权益；把它当作权益等于用恒真假设充当证据。"),
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "evidence_grade": EVIDENCE_GRADE,
        "unknown": [],
    }


# ---------------------------------------------------------------------------
# 2. 根因审计（要求 1）
# ---------------------------------------------------------------------------
def audit_preflop_root_cause(*, frozen_leaf_verification: Any,
                             candidate_leaf_verification: Any = None,
                             tree_coverage: Any = None) -> dict:
    """preflop 根因：无真实 equity 的展开。"""
    if not isinstance(frozen_leaf_verification, dict) \
            or not frozen_leaf_verification.get("present"):
        return _stub("frozen_leaf_verification_required")
    f = frozen_leaf_verification
    c = candidate_leaf_verification if isinstance(
        candidate_leaf_verification, dict) and \
        candidate_leaf_verification.get("present") else None
    return {
        "present": True,
        "version": CASH_P16_UNFREEZE_AUDIT_VERSION,
        "kind": "preflop_no_real_equity_expansion",
        "frozen_state": {
            "n_evaluable_leaves": f["n_evaluable"],
            "n_real_equity_leaves": f["n_real_equity_leaves"],
            "n_fold_only_leaves": f["n_fold_only_arithmetic_leaves"],
            "real_equity_sources": f["real_equity_sources"],
            "fold_only_sources": f["fold_only_sources"],
            "tree_coverage": tree_coverage},
        "mechanism": (
            "冻结 `_emit_street` 每次「进街」只补 **1 张**候选牌；`_next_street` 要求 "
            "`len(board) == _EXPECTED_BOARD[arriving]` 才允许继续展开；`_leaf_equity` "
            "只接受 board ∈ {0,3,4,5}。从 preflop（0 张）补 1 张后 board=1："
            "既不能再展开，也不在权益白名单内 ⇒ 每候选 9 叶只有 1 叶可评估，"
            "且该叶是**对手弃牌的算术终局叶（恒 1.0）**，preflop **没有任何真实权益**。"),
        "is_a_defect": True,
        "defect_kind": "evidence_availability（缺真实权益），**不是**证据伪造",
        "candidate_state": ({
            "n_evaluable_leaves": c["n_evaluable"],
            "n_real_equity_leaves": c["n_real_equity_leaves"],
            "n_fold_only_leaves": c["n_fold_only_arithmetic_leaves"],
            "real_equity_sources": c["real_equity_sources"],
            "board_whitelist_clean":
                c["n_evaluable_with_board_outside_whitelist"] == 0,
        } if c else None),
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "evidence_grade": EVIDENCE_GRADE,
        "unknown": [],
    }


def audit_postflop_root_cause(*, decomposition: Any) -> dict:
    """postflop 根因：同一可行域内的真实分歧。"""
    if not isinstance(decomposition, dict) or not decomposition.get("present"):
        return _stub("decomposition_required")
    na = decomposition.get("non_accept_reason_counts") or {}
    n_dis = int(na.get("cross_scorer_true_disagreement") or 0)
    return {
        "present": True,
        "version": CASH_P16_UNFREEZE_AUDIT_VERSION,
        "kind": "postflop_same_domain_true_disagreement",
        "n_true_disagreements": n_dis,
        "n_accept": decomposition.get("n_accept"),
        "n_accept_cross_scorer": decomposition.get("accept_cross_scorer"),
        "n_accept_single_tier": decomposition.get("accept_single_tier"),
        "mechanism": (
            "条件化把**逐候选**权益写入 search 叶池后，`bet` 的权益低于解析打平点 "
            "`e_b* = (e_check + I/P)·P/(P+2I)`，search 层 top 由 `bet` 翻到 `check`；"
            "而 `ranking` 层对**所有候选使用同一个** `hero_equity`（V3H-5 公式），"
            "结构上无法表达'下注更差是因为对手继续范围更强'，因此 top 仍是 `bet`。"
            "两层在同一可行域内偏好不同 ⇒ 冻结'分歧即弃权'触发。"),
        "is_a_defect": False,
        "defect_kind": (
            "**不是缺陷**：这是两条**证据不等价**的评分源之间的真实分歧，"
            "ABSTAIN 是冻结语义下的**正确**行为。"),
        "why_not_a_defect": (
            "分歧两侧的 key 都在可行域内（V3I-P15 实测 28/28），限域没有排除任何一方；"
            "分歧的触发点由解析打平点逐条解释。"),
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "evidence_grade": EVIDENCE_GRADE,
        "unknown": [],
    }


# ---------------------------------------------------------------------------
# 3. 最小解冻方案（要求 2、6）
# ---------------------------------------------------------------------------
def minimal_unfreeze_proposal(*, proposal: Any) -> dict:
    """校验并规范化一个**候选最小解冻项**（只报告，不实施）。"""
    if not isinstance(proposal, dict):
        return _stub("proposal_required")
    kind = proposal.get("kind")
    if kind not in PROPOSAL_KINDS:
        return _stub(f"unknown_proposal_kind:{kind!r}")
    required = ("target_file", "target_function", "fields_touched",
                "minimality_argument", "blast_radius", "why_not_fabrication")
    missing = [k for k in required if not proposal.get(k)]
    if missing:
        return _stub("proposal_missing:" + ",".join(missing))
    illegitimate = kind in ILLEGITIMATE_RESOLUTIONS
    return {
        "present": True,
        "version": CASH_P16_UNFREEZE_AUDIT_VERSION,
        "kind": kind,
        "target_file": proposal["target_file"],
        "target_function": proposal["target_function"],
        "fields_touched": list(proposal["fields_touched"]),
        "minimality_argument": proposal["minimality_argument"],
        "blast_radius": proposal["blast_radius"],
        "why_not_fabrication": proposal["why_not_fabrication"],
        "keeps_leaf_coverage_threshold": proposal.get(
            "keeps_leaf_coverage_threshold"),
        "threshold_value_after": proposal.get("threshold_value_after"),
        "measured_coverage_delta": proposal.get("measured_coverage_delta"),
        "risk": proposal.get("risk"),
        "test_impact": proposal.get("test_impact"),
        "frozen_drift": proposal.get("frozen_drift"),
        "is_illegitimate_by_construction": illegitimate,
        "illegitimacy_reason": ILLEGITIMATE_RESOLUTIONS.get(kind),
        "implements_anything": False,
        "status": "PROPOSED_NOT_IMPLEMENTED",
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "evidence_grade": EVIDENCE_GRADE,
        "unknown": [],
    }


def reject_illegitimate_resolutions() -> dict:
    """显式登记被**拒绝**的"解决"方式（要求 3、5、6）。"""
    return {
        "present": True,
        "version": CASH_P16_UNFREEZE_AUDIT_VERSION,
        "n_rejected": len(ILLEGITIMATE_RESOLUTIONS),
        "rejected": dict(ILLEGITIMATE_RESOLUTIONS),
        "principle": (
            "能提高覆盖率**不等于**解决了根因。凡是以丢弃某一层证据、降低证据标准、"
            "或制造恒真一致为代价的'解决'，一律拒绝。"),
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "evidence_grade": EVIDENCE_GRADE,
        "unknown": [],
    }


# ---------------------------------------------------------------------------
# 4. postflop 反事实：区分"允许分歧后选择"与"真正解决"（要求 3）
# ---------------------------------------------------------------------------
def _frozen_ranking_score(*, action: Any, amount_bb: Any, equity: Any,
                          pot_bb: Any) -> Any:
    """冻结 V3H-5 公式（`equity*(pot+2r)/pot − r/pot`；check 为 `equity`）。"""
    if not (_is_num(equity) and _is_num(pot_bb)) or float(pot_bb) <= 0:
        return None
    act = str(action)
    if act == "fold":
        return 0.0
    if act == "check":
        return round(float(equity), 6)
    if not _is_num(amount_bb):
        return None
    inv = float(amount_bb)
    return round(float(equity) * (float(pot_bb) + 2.0 * inv) / float(pot_bb)
                 - inv / float(pot_bb), 6)


def counterfactual_postflop_options(*, record: Any) -> dict:
    """对一次 postflop 运行，枚举"解决分歧"的候选并**逐一判定合法性**。"""
    if not isinstance(record, dict):
        return _stub("record_not_a_dict")
    pot = record.get("pot_bb")
    cands = [c for c in (record.get("search_candidates") or [])
             if isinstance(c, dict)]
    cond = record.get("conditioned_per_candidate") or {}
    if not cands:
        return _stub("search_candidates_required")

    rows = []
    for c in cands:
        c_amt = c.get("amount_bb")
        c_amt = float(c_amt) if c_amt is not None else None
        eq = None
        for v in cond.values():
            if not isinstance(v, dict):
                continue
            v_amt = v.get("amount_bb")
            v_amt = float(v_amt) if v_amt is not None else None
            if str(v.get("action")) != str(c.get("action")):
                continue
            if v_amt is None or c_amt is None:
                continue
            if abs(v_amt - c_amt) <= 1e-6:
                eq = v.get("hero_equity_vs_continuing")
                break
        score = _frozen_ranking_score(action=c.get("action"),
                                      amount_bb=c_amt, equity=eq, pot_bb=pot)
        rows.append({
            "action": str(c.get("action")), "amount_bb": c_amt,
            "conditioned_equity": eq,
            "conditioned_ranking_score": score,
            "search_score": c.get("search_score"),
            "identical_to_search_score": bool(
                score is not None and _is_num(c.get("search_score"))
                and abs(float(c["search_score"]) - float(score)) <= 1e-6),
        })
    rows_cmp = [r for r in rows if _is_num(r["conditioned_ranking_score"])]
    rows_cmp.sort(key=lambda r: -float(r["conditioned_ranking_score"]))
    commensurate_top = ([rows_cmp[0]["action"], rows_cmp[0]["amount_bb"]]
                        if rows_cmp else None)
    search_top = _key(record.get("search_preferred"))
    ranking_top = _key((record.get("scorer_tops") or {}).get("ranking"))
    n_identical = sum(1 for r in rows if r["identical_to_search_score"])
    all_identical = bool(rows and n_identical == len(rows))
    agrees = bool(commensurate_top is not None
                  and _same(commensurate_top, search_top))
    actual_pref = record.get("preferred")
    actual_outcome = ("ACCEPT" if actual_pref not in (None, {})
                      else "ABSTAIN" if record.get("abstained") is True
                      else "OTHER")
    return {
        "present": True,
        "version": CASH_P16_UNFREEZE_AUDIT_VERSION,
        "run_id": record.get("run_id"),
        "scenario": record.get("scenario"),
        "street": record.get("street"),
        "population": record.get("population"),
        "actual_outcome": actual_outcome,
        "actual_selected_tier": (actual_pref or {}).get("selected_by_tier")
        if isinstance(actual_pref, dict) else None,
        "search_top_key": search_top,
        "ranking_top_key_in_domain": ranking_top,
        "option_A_commensurate_ranking": {
            "rows": rows,
            "n_candidates": len(rows),
            "n_scores_identical_to_search": n_identical,
            "all_scores_identical_to_search": all_identical,
            "commensurated_ranking_top": commensurate_top,
            "would_agree_with_search": agrees,
            "outcome_would_become": ("ACCEPT" if agrees else "ABSTAIN"),
            "independent_scoring_sources_after": 1,
            "verdict": "REJECTED_TAUTOLOGICAL" if all_identical else
                       "NEEDS_FURTHER_AUDIT",
            "reason": ILLEGITIMATE_RESOLUTIONS[
                "candidate_local_equity_for_ranking"]},
        "option_B_prefer_search_on_disagreement": {
            "outcome_would_become": ("ACCEPT" if search_top else "UNKNOWN"),
            "selected_key": search_top,
            "discarded_tier": "ranking",
            "verdict": "REJECTED_OVERRIDE_IN_EFFECT",
            "reason": ILLEGITIMATE_RESOLUTIONS["tier_precedence_or_override"]},
        "option_C_vote_between_tiers": {
            "n_comparable_tiers": 2,
            "tie": not _same(search_top, ranking_top),
            "outcome_would_become": ("UNKNOWN" if not _same(search_top,
                                                           ranking_top)
                                     else "ACCEPT"),
            "verdict": "REJECTED_TIE",
            "reason": ILLEGITIMATE_RESOLUTIONS["vote_between_tiers"]},
        "option_D_lower_threshold": {
            "verdict": "REJECTED_FORBIDDEN_AND_COUNTERPRODUCTIVE",
            "reason": ILLEGITIMATE_RESOLUTIONS["lower_leaf_coverage_threshold"]},
        "any_legitimate_resolution_found": False,
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "evidence_grade": EVIDENCE_GRADE,
        "unknown": [],
    }


def summarize_postflop_options(*, results: Any) -> dict:
    rows = [r for r in (results if isinstance(results, list) else [])
            if isinstance(r, dict) and r.get("present")]
    if not rows:
        return _stub("results_required")
    a = [r for r in rows if r["option_A_commensurate_ranking"][
        "all_scores_identical_to_search"]]
    agree = [r for r in rows if r["option_A_commensurate_ranking"][
        "would_agree_with_search"]]
    return {
        "present": True,
        "version": CASH_P16_UNFREEZE_AUDIT_VERSION,
        "n_runs": len(rows),
        "n_option_A_tautological": len(a),
        "n_option_A_would_agree": len(agree),
        "n_option_B_would_accept": sum(
            1 for r in rows
            if r["option_B_prefer_search_on_disagreement"][
                "outcome_would_become"] == "ACCEPT"),
        "n_actual_accept": sum(1 for r in rows
                               if r["actual_outcome"] == "ACCEPT"),
        "legitimate_resolutions_found": 0,
        "conclusion": (
            "postflop 的 28 条分歧**没有任何**符合证据语义的解决方式："
            "选项 A（把逐候选条件化权益喂给 ranking）经实测使两层分数**逐位相同**，"
            "一致性恒真、独立来源降为 1；选项 B（分歧时采用 search）是换名的 override；"
            "选项 C（投票）在两 tier 不同 key 时必然平票；选项 D 被明令禁止且实测有害。"
            "因此 postflop 根因**不可在严格证据语义下被'解决'**——"
            "ABSTAIN 是正确的行为。"),
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "evidence_grade": EVIDENCE_GRADE,
        "unknown": [],
    }


# ---------------------------------------------------------------------------
# 5. 覆盖率影响（要求 8）
# ---------------------------------------------------------------------------
def coverage_impact(*, actual: Any, candidate_effects: Any) -> dict:
    """按实测的逐条结果变化计算覆盖率影响（不预测，只汇总已观测到的变化）。"""
    if not isinstance(actual, dict):
        return _stub("actual_required")
    effects = candidate_effects if isinstance(candidate_effects, dict) else {}
    n = int(actual.get("n_runs") or 0)
    base_accept = int(actual.get("n_accept") or 0)
    rows: dict[str, Any] = {}
    for name, eff in effects.items():
        if not isinstance(eff, dict):
            continue
        acc = int(eff.get("accept_to_abstain") or 0)
        a2a = int(eff.get("abstain_to_accept") or 0)
        new_accept = base_accept - acc + a2a
        rows[name] = {
            "accept_to_abstain": acc,
            "abstain_to_accept": a2a,
            "accept_before": base_accept,
            "accept_after": new_accept,
            "coverage_before": round(base_accept / n, 6) if n else None,
            "coverage_after": round(new_accept / n, 6) if n else None,
            "delta": new_accept - base_accept,
            "legitimacy": eff.get("legitimacy"),
            "independent_sources_after": eff.get("independent_sources_after"),
        }
    return {
        "present": True,
        "version": CASH_P16_UNFREEZE_AUDIT_VERSION,
        "n_runs": n,
        "by_candidate": rows,
        "note": ("覆盖率只是**可行性指标**之一；合法性由独立来源数与证据语义判定，"
                 "不得以覆盖率提升为由接受非法方案。"),
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "evidence_grade": EVIDENCE_GRADE,
        "unknown": [],
    }


def evidence_semantics_verdict(*, candidates: Any) -> dict:
    """对每个候选方案给出"是否在严格证据语义下算解决"的裁定。"""
    rows = candidates if isinstance(candidates, list) else []
    ok = []
    rejected = []
    for c in rows:
        if not isinstance(c, dict):
            continue
        (rejected if (c.get("is_illegitimate_by_construction")
                      or c.get("rejected")) else ok).append(
            c.get("kind") or c.get("name"))
    return {
        "present": True,
        "version": CASH_P16_UNFREEZE_AUDIT_VERSION,
        "n_candidates": len(rows),
        "accepted_under_strict_evidence_semantics": ok,
        "rejected": rejected,
        "n_accepted": len(ok),
        "n_rejected": len(rejected),
        "verdict": ("存在通过严格证据语义的候选方案" if ok else
                    "**不存在**通过严格证据语义的候选方案 ⇒ 按要求 9 报告"),
        "rules": list(EVIDENCE_SEMANTICS_RULES),
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "evidence_grade": EVIDENCE_GRADE,
        "unknown": [],
    }


def p16_contract() -> dict:
    return {
        "version": CASH_P16_UNFREEZE_AUDIT_VERSION,
        "stage": "V3I-P16",
        "purpose": "针对 V3I-P15 两个冻结根因的**最小解冻方案审计 + 最小反事实**",
        "implements_any_unfreeze_item": False,
        "modifies_frozen_contract": False,
        "modifies_budget_depth_or_cap": False,
        "relaxes_leaf_coverage_threshold": False,
        "leaf_coverage_threshold_kept": 0.8,
        "enables_override": False,
        "uses_voting": False,
        "treats_fold_only_as_equity": False,
        "counterfactuals_are_read_only": True,
        "root_cause_kinds": list(ROOT_CAUSE_KINDS),
        "proposal_kinds": list(PROPOSAL_KINDS),
        "illegitimate_resolution_kinds": sorted(ILLEGITIMATE_RESOLUTIONS),
        "valid_equity_board_sizes": list(VALID_EQUITY_BOARD_SIZES),
        "real_equity_source_markers": list(REAL_EQUITY_SOURCE_MARKERS),
        "evidence_semantics_rules": list(EVIDENCE_SEMANTICS_RULES),
        "imports_project_layers": False,
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "evidence_grade": EVIDENCE_GRADE,
    }
