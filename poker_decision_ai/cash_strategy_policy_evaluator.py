"""V3H-4 — CASH Policy Evaluator（模型内、机制级候选比较）。

定位
----
把动作策略层的候选动作与 belief 层的对手 belief 组合，产出**模型内、机制级**
的比较量，并显式区分三个来源等级：

  * `HEURISTIC`      —— 明确标注为启发式（本层**不使用**，仅保留枚举值）
  * `MODEL_INTERNAL` —— 由本项目冻结/离线原语推导的量
  * `UNAVAILABLE`    —— 无法计算（缺输入 / 缺参照）

**本层不做最终决策、不接生产 Judge、不接 Solver。**

数据接口（审计要求：生产 import 图零新增）
------------------------------------------
本层**不在模块顶层 import 任何 V3G 层**；`action_policy` / `belief` /
`solver` 全部由调用方传入，缺失即 `present=False`，**不隐式导入、不猜**。

关键诚实声明（`claim_boundary`，机器可读）
------------------------------------------
* **不声称 GTO / equilibrium**；本层**不读取也不翻转** solver 状态：
  无 solver 传入时 `solver` 记为 `UNKNOWN` 且 `blocked=True`；
* **不把 equity / pot odds / 启发式分数冒充 GTO**：一律 `MODEL_INTERNAL`；
* **无均衡参照 → exploitability / 频率错误 / 均衡 EV 差一律 UNAVAILABLE**，
  本层**不计算**这些指标。

硬约束
------
1. **只读 + 确定性**：不修改输入；不 import `random`；无 IO；同输入 → 同输出。
2. **UNKNOWN-first**：缺输入 → `UNAVAILABLE`，**不猜**、不用 0 填充。
3. **CASH-only，fail closed**：非 CASH → `present=False`。
4. **不做筛选**：不排序、不推荐、不产出 `best`；只给可比量与来源标签。
"""
from __future__ import annotations

from typing import Any


CASH_STRATEGY_POLICY_EVALUATOR_VERSION = "v3h4-cash-strategy-policy-evaluator-v1"

SOURCE_GRADES: tuple[str, ...] = ("HEURISTIC", "MODEL_INTERNAL", "UNAVAILABLE")

CLAIM_BOUNDARY: dict[str, bool] = {
    "is_gto": False,
    "is_equilibrium": False,
    "is_solver_output": False,
    "is_profit_evidence": False,
    "selects_action": False,
    "computes_exploitability": False,
    "wired_to_production": False,
}

UNAVAILABLE_METRICS: tuple[str, ...] = (
    "exploitability", "frequency_mistake_rate", "equilibrium_ev_gap",
)


def _is_cash(state: Any) -> bool:
    if not isinstance(state, dict):
        return False
    gt = state.get("game_type")
    return isinstance(gt, str) and gt.strip().lower() == "cash"


def _stub(reason: str) -> dict:
    return {
        "present": False,
        "version": CASH_STRATEGY_POLICY_EVALUATOR_VERSION,
        "reason": reason,
        "evidence_grade": "NONE",
        "unknown": [reason],
        "comparisons": [],
        "unavailable_metrics": list(UNAVAILABLE_METRICS),
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }


def _compare(action_policy: dict, belief: dict) -> list[dict]:
    """逐候选产出模型内比较量（不排序、不筛选）。"""
    out: list[dict] = []
    strength = action_policy.get("hand_strength") or {}
    odds = action_policy.get("pot_odds") or {}
    belief_quality = {
        "n_opponents": belief.get("n_opponents"),
        "known_cards_count": belief.get("known_cards_count"),
        "level": belief.get("level"),
        "unknown_count": len(belief.get("unknown") or []),
    }
    for cand in action_policy.get("candidates") or []:
        if not isinstance(cand, dict):
            continue
        unknown: list[str] = list(cand.get("unknown") or [])
        comparables: dict[str, Any] = {}
        sources: dict[str, str] = {}

        eq = strength.get("hero_equity") if strength.get("present") else None
        if isinstance(eq, (int, float)):
            comparables["hero_equity_uniform"] = round(float(eq), 6)
            sources["hero_equity_uniform"] = "MODEL_INTERNAL"
        else:
            sources["hero_equity_uniform"] = "UNAVAILABLE"

        if odds.get("present") and cand.get("action") == "call":
            req = odds.get("required_equity")
            if isinstance(req, (int, float)):
                comparables["required_equity"] = round(float(req), 6)
                sources["required_equity"] = "MODEL_INTERNAL"
                if isinstance(eq, (int, float)):
                    comparables["equity_minus_required"] = round(
                        float(eq) - float(req), 6)
                    sources["equity_minus_required"] = "MODEL_INTERNAL"
            else:
                sources["required_equity"] = "UNAVAILABLE"
        elif cand.get("action") == "call":
            sources["required_equity"] = "UNAVAILABLE"
            unknown.append("pot_odds_unavailable")

        amt = cand.get("amount_bb")
        if isinstance(amt, (int, float)):
            comparables["risk_in_bb"] = round(float(amt), 6)
            sources["risk_in_bb"] = "MODEL_INTERNAL"

        out.append({
            "action": cand.get("action"),
            "sizing_kind": cand.get("sizing_kind"),
            "amount_bb": amt,
            "comparables": comparables,
            "comparables_sources": sources,
            "belief_quality": dict(belief_quality),
            "evidence_grade": "MODEL_INTERNAL",
            "heuristic_used": False,
            "unknown": unknown,
            "is_gto": False,
            "is_equilibrium": False,
        })
    return out


def evaluate_policy(state: Any, *, action_policy: dict | None = None,
                    belief: dict | None = None,
                    solver: dict | None = None,
                    integrity_context: dict | None = None) -> dict:
    """模型内、机制级候选评估（**不选动作、不接生产、不接 Solver**）。

    Parameters
    ----------
    state : CASH state（非 CASH → present=False）
    action_policy : **必填**，由 `cash_strategy_action_policy.build_action_policy` 产出
    belief : **必填**，由 `cash_strategy_dynamic_belief.build_dynamic_belief` 产出
    solver : 可选，**只读转述**的 solver 状态；未传入时记为 UNKNOWN + blocked
    """
    if not _is_cash(state):
        return _stub("not_cash_mode_fail_closed")
    if not isinstance(action_policy, dict):
        return _stub("action_policy_required")
    if not isinstance(belief, dict):
        return _stub("belief_required")

    ap = action_policy
    if not ap.get("present"):
        return _stub(f"action_policy_unavailable:{ap.get('reason', 'unknown')}")
    bl = belief
    if not bl.get("present"):
        return _stub(f"belief_unavailable:{bl.get('reason', 'unknown')}")

    ctx = integrity_context
    mode = (ctx.get("mode") if isinstance(ctx, dict)
            else (state.get("game_type") if isinstance(state, dict) else None))

    # Solver 状态：**只读转述**；未传入 → UNKNOWN 且 blocked=True（不猜、不翻转）
    if isinstance(solver, dict):
        sv = solver
        solver_label = sv.get("solver_status", "UNKNOWN")
        solver_blocked = solver_label != "VALID_RESULT"
        solver_passthrough = False
    else:
        solver_label = "UNKNOWN"
        solver_blocked = True
        solver_passthrough = True

    comparisons = _compare(ap, bl)

    unknown = list(ap.get("unknown") or [])
    for u in (bl.get("unknown") or []):
        if u not in unknown:
            unknown.append(u)
    if solver_passthrough:
        unknown.append("solver_status_not_supplied")
    for c in comparisons:
        for u in c.get("unknown") or []:
            if u not in unknown:
                unknown.append(u)

    return {
        "present": True,
        "version": CASH_STRATEGY_POLICY_EVALUATOR_VERSION,
        "mode": mode,
        "street": ap.get("street"),
        "comparisons": comparisons,
        "n_candidates": len(comparisons),
        "source_grades_used": sorted({
            g for c in comparisons
            for g in (c.get("comparables_sources") or {}).values()
        }),
        "heuristic_used": False,
        "unavailable_metrics": list(UNAVAILABLE_METRICS),
        "solver": {"solver_status": solver_label,
                   "blocked": bool(solver_blocked),
                   "read_only_passthrough": solver_passthrough},
        "belief_level": bl.get("level"),
        "unknown": unknown,
        "evidence_grade": "MODEL_INTERNAL",
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "ranking_provided": False,
        "selection_implemented": False,
    }
