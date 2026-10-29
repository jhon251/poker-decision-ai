"""V3H-6 — CASH Belief-Conditioned Policy Evaluation（四态显式区分）。

定位
----
把 V3H-6 的 **likelihood 条件化 belief** 接入 V3H-5 的候选排序链路，并在输出中
**显式区分四个状态**（授权要求）：

| 状态 | 含义 |
|---|---|
| `prior_only` | 未提供调用方声明的 likelihood 模型 → belief 只是先验 + 已知牌移除 |
| `likelihood_conditioned` | 已按声明式 likelihood 条件化（乘数逐项记录） |
| `MODEL_INTERNAL` | 分数/比较量的证据等级（**非** GTO、**非**真实盈利） |
| `UNAVAILABLE` | 无法计算的量（例如 exploitability：无均衡参照 → 无分母） |

**Solver 仍必须 `BLOCKED`**：本层**不读取也不翻转** solver 状态；未传入时记为
`UNKNOWN` 且 `blocked=True`。

**不输出** GTO / equilibrium / 真实 exploitability。

复用而非重实现（授权要求）
--------------------------
* 候选与几何量：V3H-4 `cash_strategy_action_policy`（由调用方传入）；
* 排序与 tie-break：V3H-5 `cash_strategy_action_ranking`（由调用方传入，**不重写**）；
* 条件化 belief：V3H-6 `cash_strategy_belief_update`（由调用方传入）；
* 本层只做**汇总与状态标注**，不新增评分口径。

数据接口（生产 import 图零新增）
--------------------------------
**不在模块顶层 import 任何 V3G 层**；所有输入由调用方传入。
"""
from __future__ import annotations

from typing import Any

CASH_STRATEGY_BELIEF_CONDITIONED_EVAL_VERSION = (
    "v3h6-cash-strategy-belief-conditioned-eval-v1")

#: 四个状态（枚举封闭）
STATES: tuple[str, ...] = (
    "prior_only", "likelihood_conditioned", "MODEL_INTERNAL", "UNAVAILABLE",
)

#: 无均衡参照 → 不可计算的量（不计算）
UNAVAILABLE_METRICS: tuple[str, ...] = (
    "exploitability", "equilibrium_ev_gap", "frequency_mistake_rate",
    "real_opponent_range",
)

CLAIM_BOUNDARY: dict[str, bool] = {
    "is_gto": False,
    "is_equilibrium": False,
    "is_solver_output": False,
    "is_profit_evidence": False,
    "computes_exploitability": False,
    "claims_real_opponent_range": False,
    "wired_to_production": False,
}


def _is_cash(state: Any) -> bool:
    if not isinstance(state, dict):
        return False
    gt = state.get("game_type")
    return isinstance(gt, str) and gt.strip().lower() == "cash"


def _stub(reason: str) -> dict:
    return {
        "present": False,
        "version": CASH_STRATEGY_BELIEF_CONDITIONED_EVAL_VERSION,
        "reason": reason,
        "evidence_grade": "NONE",
        "unknown": [reason],
        "ranked": [],
        "state": None,
        "unavailable_metrics": list(UNAVAILABLE_METRICS),
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }


def evaluate_belief_conditioned(state: Any, *, action_policy: dict | None = None,
                                ranking: dict | None = None,
                                belief_update: dict | None = None,
                                ranking_evaluator: dict | None = None,
                                solver: dict | None = None) -> dict:
    """把条件化 belief 接入排序结果，并标注四态。

    Parameters
    ----------
    state : CASH state（非 CASH → present=False）
    action_policy : 可选，V3H-4 输出（仅用于转述几何量）
    ranking : **必填**，V3H-5 `cash_strategy_action_ranking.rank_actions` 输出
    belief_update : 可选，V3H-6 `cash_strategy_belief_update` 输出
    ranking_evaluator : 可选，V3H-4 `cash_strategy_policy_evaluator` 输出
    solver : 可选，**只读转述**的 solver 状态
    """
    if not _is_cash(state):
        return _stub("not_cash_mode_fail_closed")
    if not isinstance(ranking, dict):
        return _stub("ranking_required")
    if not ranking.get("present"):
        return _stub(f"ranking_unavailable:{ranking.get('reason', 'unknown')}")

    bu = belief_update if isinstance(belief_update, dict) and \
        belief_update.get("present") else None
    if bu is None:
        conditioning = "prior_only"
        conditioning_source = "belief_update_not_supplied"
    else:
        conditioning = bu.get("path")
        conditioning_source = bu.get("version")

    unknown: list[str] = list(ranking.get("unknown") or [])
    if bu:
        for u in (bu.get("unknown") or []):
            if u not in unknown:
                unknown.append(u)
    if bu is None:
        unknown.append("belief_update_not_supplied")

    # Solver：只读转述；未传入 → UNKNOWN + blocked
    if isinstance(solver, dict):
        solver_label = solver.get("solver_status", "UNKNOWN")
        solver_blocked = solver_label != "VALID_RESULT"
        passthrough = False
    else:
        solver_label = "UNKNOWN"
        solver_blocked = True
        passthrough = True

    # 条件化质量摘要（只转述，不重算）
    opp_summary: list[dict] = []
    if bu:
        for pid, e in (bu.get("belief") or {}).items():
            opp_summary.append({
                "player_id": pid,
                "position": (e or {}).get("position"),
                "n_observed_actions": len((e or {}).get("observed_actions") or []),
                "likelihood_conditioned": bool((e or {}).get("likelihood_conditioned")),
                "likelihood_total_multiplier": (e or {}).get(
                    "likelihood_total_multiplier"),
                "mass_before": (e or {}).get("mass_before_likelihood"),
                "mass_after": (e or {}).get("mass_after_likelihood"),
                "no_known_card_reentered": ((e or {}).get("invariants") or {}).get(
                    "no_known_card_reentered"),
            })

    ranked = []
    for row in ranking.get("ranked") or []:
        if not isinstance(row, dict):
            continue
        ranked.append({
            "rank": row.get("rank"),
            "action": row.get("action"),
            "sizing_kind": row.get("sizing_kind"),
            "amount_bb": row.get("amount_bb"),
            "score": row.get("score"),
            "score_source": row.get("score_source"),
            # 证据等级：分数本身是模型内量；条件化状态另标
            "evidence_grade": "MODEL_INTERNAL",
            "conditioning": conditioning,
            "is_gto": False,
            "is_equilibrium": False,
        })

    state_labels = ["MODEL_INTERNAL"]
    state_labels.append(conditioning if conditioning in STATES[0:2]
                        else "prior_only")
    if not ranked:
        state_labels.append("UNAVAILABLE")

    return {
        "present": True,
        "version": CASH_STRATEGY_BELIEF_CONDITIONED_EVAL_VERSION,
        "mode": "cash",
        "street": ranking.get("street"),
        "states": sorted(set(state_labels)),
        "state_definitions": {
            "prior_only": "no caller-declared likelihood model applied",
            "likelihood_conditioned": "conditioned on declared likelihood "
                                      "parameters (multipliers recorded)",
            "MODEL_INTERNAL": "model-internal quantity; not GTO, not real profit",
            "UNAVAILABLE": "cannot be computed (no equilibrium reference)",
        },
        "conditioning": conditioning,
        "conditioning_source": conditioning_source,
        "belief_update_path": (bu or {}).get("path"),
        "opponent_conditioning": opp_summary,
        "n_opponents_conditioned": sum(
            1 for o in opp_summary if o["likelihood_conditioned"]),
        "hero_actions_in_opponent_likelihood": (
            (bu or {}).get("hero_actions_in_opponent_likelihood")),
        "ranked": ranked,
        "n_ranked": len(ranked),
        "preferred": (ranking.get("preferred") if conditioning != "UNAVAILABLE"
                      else None),
        "preferred_reason": ranking.get("preferred_reason"),
        "ranking_evaluator_used": bool(
            isinstance(ranking_evaluator, dict) and ranking_evaluator.get("present")),
        "solver": {"solver_status": solver_label, "blocked": bool(solver_blocked),
                   "read_only_passthrough": passthrough},
        "unavailable_metrics": list(UNAVAILABLE_METRICS),
        "unknown": unknown,
        "level": ("UNKNOWN" if (unknown and not ranked) else
                  ("MODEL_INTERNAL" if ranked else "UNKNOWN")),
        "evidence_grade": "MODEL_INTERNAL",
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "final_decision_made": False,
        "note": ("belief 是**模型内**量；本层不主张其为真实对手范围，"
                 "也不输出任何 GTO/equilibrium/真实 exploitability 量。"),
    }
