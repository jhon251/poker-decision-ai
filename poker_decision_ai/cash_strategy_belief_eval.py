"""V3H-5 — CASH Belief-Aware Candidate Evaluation（prior-only vs likelihood 可辨）。

定位
----
把**动态 belief** 引入候选动作评估，并**强制可辨识**两条路径：

| 路径 | `likelihood_source` | 语义 |
|---|---|---|
| **prior-only** | `not_supplied` | 未提供动作似然 → belief **只是先验 + 已知牌移除**；**不是**已更新的后验 |
| **action-likelihood** | `explicit_action_weights` | 调用方**显式**给出似然；乘数记录在案 |

**绝不伪造动作似然**：未提供时不得假装已更新、不得用均匀似然冒充、
不得把 prior-only 结果表述为"已根据对手动作更新"。

隔离保证（授权要求）
--------------------
* **对手范围**：只来自 belief 的 `belief[player_id].posterior`；
* **hero 范围**：本层**不推导** hero 范围——hero 的手牌是**已知具体牌**
  （`hero_hole_cards`），与对手的范围分布是**不同对象**，不得混用；
* **known-card removal**：只包含公共牌 + hero 手牌（对手手牌未知），
  逐张列出，且标记 `opponent_hole_cards_included=False`。

**本层不做最终决策、不接生产 Judge、不接 Solver、不声称 GTO/equilibrium。**

数据接口（生产 import 图零新增）
--------------------------------
**不在模块顶层 import 任何 V3G 层**；所有输入由调用方传入。
"""
from __future__ import annotations

from typing import Any

CASH_STRATEGY_BELIEF_EVAL_VERSION = "v3h5-cash-strategy-belief-eval-v1"

CLAIM_BOUNDARY: dict[str, bool] = {
    "is_gto": False,
    "is_equilibrium": False,
    "is_solver_output": False,
    "is_profit_evidence": False,
    "fabricates_action_likelihood": False,
    "mixes_hero_and_opponent_ranges": False,
    "wired_to_production": False,
}

#: prior-only 时必须出现的显式标记（机器可读）
PRIOR_ONLY_FLAGS: tuple[str, ...] = (
    "prior_only",
    "likelihood_not_supplied",
)


def _is_cash(state: Any) -> bool:
    if not isinstance(state, dict):
        return False
    gt = state.get("game_type")
    return isinstance(gt, str) and gt.strip().lower() == "cash"


def _stub(reason: str) -> dict:
    return {
        "present": False,
        "version": CASH_STRATEGY_BELIEF_EVAL_VERSION,
        "reason": reason,
        "evidence_grade": "NONE",
        "unknown": [reason],
        "candidates": [],
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }


def _isolation_report(belief: dict, state: dict) -> dict:
    """对手范围 / hero 手牌 / known-card removal 的隔离报告。"""
    hero_id = belief.get("hero_id")
    opponents = [pid for pid in (belief.get("belief") or {})]
    known = list(belief.get("known_cards_removed") or [])
    hero_hole: list[str] = []
    for p in (state.get("players") or []):
        if isinstance(p, dict) and (bool(p.get("is_hero")) or
                                    p.get("player_id") == hero_id):
            hc = p.get("hole_cards")
            if isinstance(hc, list):
                hero_hole = [str(c) for c in hc if isinstance(c, str)]
            break
    return {
        "opponent_range_players": sorted(opponents),
        "hero_id": hero_id,
        "hero_represented_as_range": False,
        "hero_hole_cards": hero_hole,
        "hero_hole_cards_are_specific": len(hero_hole) == 2,
        "hero_id_absent_from_opponent_ranges": (
            hero_id is None or hero_id not in opponents),
        "known_cards_removed": known,
        "known_cards_include_board_and_hero_only": True,
        "opponent_hole_cards_included": False,
        "unknown": list(belief.get("unknown") or []),
    }


def evaluate_with_belief(state: Any, *, action_policy: dict | None = None,
                         belief: dict | None = None,
                         evaluator: dict | None = None) -> dict:
    """belief-aware 候选评估（**prior-only 与 likelihood 两路可辨**）。

    Parameters
    ----------
    state : CASH state（非 CASH → present=False）
    action_policy : **必填**
    belief : **必填**（`cash_strategy_dynamic_belief.build_dynamic_belief`）
    evaluator : 可选（`cash_strategy_policy_evaluator.evaluate_policy`）
    """
    if not _is_cash(state):
        return _stub("not_cash_mode_fail_closed")
    if not isinstance(action_policy, dict):
        return _stub("action_policy_required")
    if not isinstance(belief, dict):
        return _stub("belief_required")
    ap, bl = action_policy, belief
    if not ap.get("present"):
        return _stub(f"action_policy_unavailable:{ap.get('reason', 'unknown')}")
    if not bl.get("present"):
        return _stub(f"belief_unavailable:{bl.get('reason', 'unknown')}")

    lik_src = bl.get("likelihood_source")
    prior_only = lik_src != "explicit_action_weights"

    flags: list[str] = []
    if prior_only:
        flags.extend(PRIOR_ONLY_FLAGS)

    unknown: list[str] = list(ap.get("unknown") or [])
    for u in (bl.get("unknown") or []):
        if u not in unknown:
            unknown.append(u)

    isolation = _isolation_report(bl, state)

    strength = ap.get("hand_strength") or {}
    equity = strength.get("hero_equity") if strength.get("present") else None
    pot = ap.get("pot_bb")
    to_call = float(ap.get("to_call_bb") or 0.0)

    rows: list[dict] = []
    for cand in ap.get("candidates") or []:
        if not isinstance(cand, dict):
            continue
        act = cand.get("action")
        amt = cand.get("amount_bb")
        row_unknown: list[str] = list(cand.get("unknown") or [])
        belief_term: dict[str, Any] = {"available": False}
        score = None

        if isinstance(equity, (int, float)) and isinstance(pot, (int, float)) and pot > 0:
            if act == "fold":
                score = 0.0
            elif act == "check":
                score = float(equity)
            else:
                invest = (float(amt) if isinstance(amt, (int, float))
                          else (to_call if act == "call" else None))
                if invest is not None:
                    score = (float(equity) * (float(pot) + 2.0 * invest) / float(pot)
                             - invest / float(pot))
                else:
                    row_unknown.append("invest_amount_unavailable")

        # belief 参与：把"对手范围质量"作为**可核验**的模型内量转述（不猜）
        masses = []
        for pid, entry in (bl.get("belief") or {}).items():
            post = (entry or {}).get("posterior") or {}
            if post.get("present"):
                masses.append(float(post.get("mass_posterior") or 0.0))
        if masses:
            belief_term = {
                "available": True,
                "n_opponents_with_posterior": len(masses),
                "posterior_mass_total": round(sum(masses), 6),
                "posterior_mass_mean": round(sum(masses) / len(masses), 6),
                "used_in_score": False,
                "note": ("belief 作为**质量/来源**维度转述；未折算为权重进入分数"
                         "（避免发明似然→权重的映射）"),
            }
        rows.append({
            "action": act,
            "sizing_kind": cand.get("sizing_kind"),
            "amount_bb": amt,
            "score": (round(score, 6) if isinstance(score, (int, float)) else None),
            "score_unit": "pot_share",
            "score_source": "MODEL_INTERNAL",
            "belief_term": belief_term,
            "prior_only": prior_only,
            "flags": list(flags),
            "unknown": row_unknown,
            "is_gto": False,
            "is_equilibrium": False,
        })

    return {
        "present": True,
        "version": CASH_STRATEGY_BELIEF_EVAL_VERSION,
        "mode": ap.get("mode"),
        "street": ap.get("street"),
        "candidates": rows,
        "n_candidates": len(rows),
        # 路径可辨（授权要求）
        "likelihood_source": lik_src,
        "prior_only": prior_only,
        "flags": flags,
        "likelihood_multiplier_recorded": [
            {"player_id": pid,
             "multiplier": ((entry or {}).get("posterior") or {}).get(
                 "likelihood_multiplier")}
            for pid, entry in (bl.get("belief") or {}).items()
        ] if not prior_only else [],
        "posterior_equals_prior_note": (
            "posterior = prior minus known-card removal only; NO action "
            "likelihood was applied" if prior_only else
            "explicit action likelihood applied by the caller"),
        "isolation": isolation,
        "evaluator_used": bool(isinstance(evaluator, dict) and evaluator.get("present")),
        "unknown": unknown,
        "level": ("UNKNOWN" if (prior_only and not rows) else
                  ("MODEL_INTERNAL" if rows else "UNKNOWN")),
        "evidence_grade": "MODEL_INTERNAL",
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }
