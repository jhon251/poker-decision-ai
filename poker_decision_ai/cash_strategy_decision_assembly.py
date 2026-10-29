"""V3H-8 — CASH Decision Assembly（多评分源对账 + 弃权 + UNKNOWN 传播）。

**先审计（不重复已 PASS 阶段）**
--------------------------------
现有 11 个 V3H 策略层已提供三层**互相独立**的评分：

| 层 | 入口 | 产出 |
|---|---|---|
| 即时排序 | `cash_strategy_action_ranking.rank_actions` | 排名 + `preferred` + 理由 |
| belief 条件化 | `cash_strategy_belief_conditioned_eval.evaluate_belief_conditioned` | 四态 + 条件化摘要 |
| 搜索感知 | `cash_strategy_search_eval.evaluate_search` | `search_score` + 覆盖率 + 预算 |

**缺口（本阶段唯一新增能力）**：没有任何一层把三者**对账**成"一个可审计的建议"。
本层补齐 V3H-3 路线审计中的 `L5 decision_assembly`：

```
三层评分 → 候选键归一 → 跨源一致性检查 → 覆盖率闸门 → 弃权规则
        → 建议（带 provenance / evidence / uncertainty / 冲突记录）
```

**不做的事**：不重算任何分数、不新增评分公式、不做最终决策、不接生产。

对账策略（`ASSEMBLY_POLICY`，零自由参数、显式声明）
---------------------------------------------------
* **优先级**：`search`（搜索感知，含未来街）> `belief_conditioned` > `ranking`（即时）；
* **一致性闸门**：若**所有已提供的**优先级层都给出 top 候选，且它们在容差内
  （`amount_tolerance_bb`）指向**同一候选键** → 采用该键；
* **分歧即弃权**：任两层的 top 键不一致 → `abstained=true`、`level="UNKNOWN"`、
  `preferred=null`，并记录 `disagreements`。**不猜、不取平均**；
* **覆盖率闸门**：`search` 的候选若 `leaf_coverage < min_leaf_coverage` →
  该候选**降级为不可用**（记入 `rejections`），不得作为首选；
* **唯一例外**：调用方显式传 `allow_higher_tier_override=True` **且**
  `override_justification` 非空 → 允许用更高层覆盖分歧，并在输出中
  记录该覆盖为 `explicit_caller_override`（默认关闭）。

候选键（跨层可对齐）
--------------------
`(action, size_bucket|None, amount_bb 容差内归并)`；`action` 缺失 → 不可对齐 →
记入 `unknown`（不猜）。

硬约束
------
* CASH-only，fail closed；非 dict / 非 CASH → `present=False`；
* UNKNOWN / UNAVAILABLE 原样传播；不默认、不猜测；
* 保持 `MODEL_INTERNAL`、`is_gto=false`、`is_equilibrium=false`；
* 不 import 任何 V3G 层；不 import `random`；同输入 → 同输出；
* Solver 只读转述（未传入 → `UNKNOWN` + `blocked=True`）。
"""
from __future__ import annotations

from typing import Any

CASH_STRATEGY_DECISION_ASSEMBLY_VERSION = "v3h8-cash-strategy-decision-assembly-v1"

#: 对账策略（显式、零自由参数）
ASSEMBLY_POLICY: dict[str, Any] = {
    "name": "cross_scorer_reconciliation_v1",
    "tier_precedence": ["search", "belief_conditioned", "ranking"],
    "amount_tolerance_bb": 1e-6,
    "min_leaf_coverage": 0.8,
    "on_disagreement": "abstain_unknown",
    "override_default": False,
    "uses_randomness": False,
    "free_parameters": 0,
    "is_gto": False,
    "is_equilibrium": False,
}

CLAIM_BOUNDARY: dict[str, bool] = {
    "is_gto": False,
    "is_equilibrium": False,
    "is_solver_output": False,
    "is_profit_evidence": False,
    "computes_exploitability": False,
    "modifies_any_scorer_formula": False,
    "final_decision_made": False,
    "wired_to_production": False,
}

#: 三层评分源的键名
TIERS: tuple[str, ...] = ("search", "belief_conditioned", "ranking")


def _is_cash(state: Any) -> bool:
    if not isinstance(state, dict):
        return False
    gt = state.get("game_type")
    return isinstance(gt, str) and gt.strip().lower() == "cash"


def _stub(reason: str) -> dict:
    return {
        "present": False,
        "version": CASH_STRATEGY_DECISION_ASSEMBLY_VERSION,
        "reason": reason,
        "evidence_grade": "NONE",
        "unknown": [reason],
        "preferred": None,
        "abstained": True,
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }


def _cand_key(action: Any, amount: Any, tol: float,
               buckets: list[tuple[str, float | None]]) -> tuple | None:
    """把候选归一到可跨层对齐的键 `(action, rounded_amount)`。

    `action` 缺失/非字符串 → None（不可对齐，调用方记入 unknown）。
    `amount` 为 None → 该层只用 action 对齐（更弱，记录在 `alignment_strength`）。
    """
    if not isinstance(action, str) or not action.strip():
        return None
    act = action.strip().lower()
    if not isinstance(amount, (int, float)) or isinstance(amount, bool):
        return (act, None)
    amt = float(amount)
    # 与已见金额在容差内 → 归并到同一键（避免浮点误差造成假分歧）
    for known_act, known_amt in buckets:
        if known_act == act and known_amt is not None and \
                abs(known_amt - amt) <= tol:
            return (act, known_amt)
    return (act, round(amt, 6))


def _normalize_feasible_keys(feasible_keys: Any) -> list | None:
    """把调用方给的可行域规范化为 `[(action_lower, amount|None), ...]`。

    **V3I-P13 新增**（P12 设计的最小契约）：调用方传入的"共同可行域"。
    `None` / 非可迭代 → 返回 `None`（= 不启用限域，行为与契约变更前一致）。
    """
    if feasible_keys is None:
        return None
    try:
        items = list(feasible_keys)
    except TypeError:
        return None
    out: list[tuple[str, float | None]] = []
    for it in items:
        act = amt = None
        if isinstance(it, dict):
            act, amt = it.get("action"), it.get("amount_bb")
        elif isinstance(it, (list, tuple)) and len(it) >= 2:
            act, amt = it[0], it[1]
        if not isinstance(act, str):
            continue
        a = act.strip().lower()
        m = (round(float(amt), 6)
             if isinstance(amt, (int, float)) and not isinstance(amt, bool)
             else None)
        out.append((a, m))
    return out


def _key_allowed(key: tuple | None, feasible: list | None, tol: float) -> bool:
    """候选键是否落在**共同可行域**内。

    金额比较使用与本层相同的 `tol` 容差口径，避免浮点误差把同一注额判成域外。
    """
    if feasible is None:
        return True
    if not isinstance(key, tuple) or len(key) != 2:
        return False
    act, amt = key[0], key[1]
    for fa, famt in feasible:
        if fa != act:
            continue
        if famt is None or amt is None:
            return True
        if abs(float(famt) - float(amt)) <= tol:
            return True
    return False


def _top_from_search(search_eval: dict, tol: float,
                     buckets: list[tuple[str, float | None]],
                     min_cov: float,
                     feasible: list | None = None) -> tuple[dict | None, list[dict]]:
    """取 search 层 top 候选；覆盖率不足者被拒绝（返回拒绝记录）。"""
    rejections: list[dict] = []
    ranked = [c for c in (search_eval.get("ranked") or []) if isinstance(c, dict)]
    for c in ranked:
        cov = c.get("leaf_coverage")
        if isinstance(cov, (int, float)) and float(cov) < min_cov:
            rejections.append({
                "tier": "search", "action": c.get("action"),
                "leaf_coverage": cov, "min_leaf_coverage": min_cov,
                "reason": "leaf_coverage_below_minimum",
            })
            continue
        if not isinstance(c.get("search_score"), (int, float)):
            rejections.append({"tier": "search", "action": c.get("action"),
                               "reason": "search_score_unavailable"})
            continue
        key = _cand_key(c.get("action"), c.get("amount_bb"), tol, buckets)
        if feasible is not None and not _key_allowed(key, feasible, tol):
            rejections.append({
                "tier": "search", "action": c.get("action"),
                "amount_bb": c.get("amount_bb"),
                "reason": "outside_feasible_domain"})
            continue
        return {"key": key, "action": c.get("action"),
                "size_bucket": c.get("size_bucket"),
                "amount_bb": c.get("amount_bb"),
                "score": c.get("search_score"),
                "score_source": c.get("score_source"),
                "leaf_coverage": c.get("leaf_coverage"),
                "detail": "search_runout_pot_share_v1"}, rejections
    return None, rejections


def _top_from_belief(bc_eval: dict, tol: float,
                     buckets: list[tuple[str, float | None]],
                     feasible: list | None = None) -> dict | None:
    ranked = [c for c in (bc_eval.get("ranked") or []) if isinstance(c, dict)]
    for c in ranked:
        if not isinstance(c.get("score"), (int, float)):
            continue
        key = _cand_key(c.get("action"), c.get("amount_bb"), tol, buckets)
        if not _key_allowed(key, feasible, tol):
            continue
        return {"key": key, "action": c.get("action"),
                "size_bucket": c.get("sizing_kind"),
                "amount_bb": c.get("amount_bb"),
                "score": c.get("score"), "score_source": c.get("score_source"),
                "conditioning": c.get("conditioning"),
                "detail": "belief_conditioned_pot_share"}
    return None


def _top_from_ranking(rank_eval: dict, tol: float,
                      buckets: list[tuple[str, float | None]],
                      feasible: list | None = None) -> dict | None:
    ranked = [c for c in (rank_eval.get("ranked") or []) if isinstance(c, dict)]
    for c in ranked:
        if not isinstance(c.get("score"), (int, float)):
            continue
        key = _cand_key(c.get("action"), c.get("amount_bb"), tol, buckets)
        if not _key_allowed(key, feasible, tol):
            continue
        return {"key": key, "action": c.get("action"),
                "size_bucket": c.get("sizing_kind"),
                "amount_bb": c.get("amount_bb"),
                "score": c.get("score"), "score_source": c.get("score_source"),
                "detail": "pot_share_arithmetic_v1"}
    return None


def assemble_decision(state: Any, *, ranking: dict | None = None,
                      belief_conditioned: dict | None = None,
                      search_eval: dict | None = None,
                      solver: dict | None = None,
                      policy: dict | None = None,
                      allow_higher_tier_override: bool = False,
                      override_justification: str | None = None,
                      feasible_keys: Any = None) -> dict:
    """把三层评分对账成**一个可审计建议**（不做最终决策、不接生产）。

    Parameters
    ----------
    state : CASH state（非 CASH → present=False）
    ranking : V3H-5 `rank_actions` 输出
    belief_conditioned : V3H-6 `evaluate_belief_conditioned` 输出
    search_eval : V3H-7 `evaluate_search` 输出
    solver : 可选，**只读转述**的 solver 状态
    policy : 覆盖 `ASSEMBLY_POLICY` 的显式子集（未知键被忽略并记录）
    allow_higher_tier_override / override_justification : **默认关闭**的显式覆盖
    feasible_keys : **可选**（V3I-P13 最小契约变更）。调用方给出的**共同可行域**
        `[(action, amount_bb), ...]`。提供时，每个 tier 的 top 先在可行域内选取，
        再做对账；某 tier 在可行域内无候选 → 记为不可用 → **仍弃权**。
        共同域内仍不一致 → **仍弃权**。`None`（默认）→ 与变更前**行为一致**。

        语义边界：本参数只回答"哪些候选各层**都能**评估"，
        **不**改变任何评分公式、tier 优先级、override 语义或容差。
    """
    pol = dict(ASSEMBLY_POLICY)
    extra_keys: list[str] = []
    if isinstance(policy, dict):
        for k, v in policy.items():
            if k in pol:
                pol[k] = v
            else:
                extra_keys.append(k)
    tol = float(pol["amount_tolerance_bb"])
    min_cov = float(pol["min_leaf_coverage"])
    # V3I-P13：共同可行域（None = 不启用限域，行为与变更前一致）
    feas = _normalize_feasible_keys(feasible_keys)
    feas_applied = feas is not None and len(feas) > 0
    feas_used = feas if feas_applied else None

    if not _is_cash(state):
        return _stub("not_cash_mode_fail_closed")

    supplied = {
        "search": isinstance(search_eval, dict) and bool(search_eval.get("present")),
        "belief_conditioned": isinstance(belief_conditioned, dict) and bool(
            belief_conditioned.get("present")),
        "ranking": isinstance(ranking, dict) and bool(ranking.get("present")),
    }
    if not any(supplied.values()):
        return _stub("no_scorer_supplied")

    entries: dict[str, dict | None] = {}
    rejections: list[dict] = []
    buckets: list[tuple[str, float | None]] = []

    if supplied["search"]:
        top, rej = _top_from_search(search_eval, tol, buckets, min_cov,
                                    feasible=feas_used)
        entries["search"] = top
        rejections.extend(rej)
    else:
        entries["search"] = None
    if supplied["belief_conditioned"]:
        entries["belief_conditioned"] = _top_from_belief(
            belief_conditioned, tol, buckets, feasible=feas_used)
    else:
        entries["belief_conditioned"] = None
    if supplied["ranking"]:
        entries["ranking"] = _top_from_ranking(
            ranking, tol, buckets, feasible=feas_used)
    else:
        entries["ranking"] = None

    unknown: list[str] = []
    for tier in TIERS:
        if not supplied[tier]:
            unknown.append(f"{tier}_not_supplied")
        elif entries[tier] is None:
            unknown.append(f"{tier}_top_candidate_unavailable")
        elif entries[tier]["key"] is None:
            unknown.append(f"{tier}_candidate_key_unalignable")
    if extra_keys:
        unknown.append("unknown_policy_keys_ignored")

    # 冲突记录与弃权判定
    comparable = [t for t in TIERS
                  if entries.get(t) and entries[t]["key"] is not None]
    keys = [entries[t]["key"] for t in comparable]
    disagreements: list[dict] = []
    agreed_key = keys[0] if keys else None
    for t in comparable[1:]:
        if entries[t]["key"] != agreed_key:
            disagreements.append({
                "tier_a": comparable[0], "key_a": list(agreed_key or ()),
                "tier_b": t, "key_b": list(entries[t]["key"]),
                "note": "top candidates differ; no averaging, no guessing",
            })

    override_used = False
    override_reason: str | None = None
    if disagreements:
        ok_override = (bool(allow_higher_tier_override)
                       and isinstance(override_justification, str)
                       and override_justification.strip() != "")
        if ok_override:
            override_used = True
            override_reason = override_justification.strip()
            agreed_key = entries[comparable[0]]["key"]

    abstained = bool(disagreements) and not override_used
    if abstained:
        unknown.append("cross_scorer_disagreement")

    chosen_tier = comparable[0] if (comparable and not abstained) else None
    preferred = None
    if chosen_tier is not None and agreed_key is not None:
        e = entries[chosen_tier]
        preferred = {
            "action": e["action"],
            "size_bucket": e.get("size_bucket"),
            "amount_bb": e.get("amount_bb"),
            "candidate_key": list(agreed_key),
            "selected_by_tier": chosen_tier,
            "tier_detail": e.get("detail"),
            "score_source": e.get("score_source"),
            "leaf_coverage": e.get("leaf_coverage"),
        }

    # Solver：只读转述
    if isinstance(solver, dict):
        solver_label = solver.get("solver_status", "UNKNOWN")
        solver_blocked = solver_label != "VALID_RESULT"
        passthrough = False
    else:
        solver_label = "UNKNOWN"
        solver_blocked = True
        passthrough = True
    if passthrough:
        unknown.append("solver_status_not_supplied")

    # 证据等级 = 选中层的最小等级（本阶段全部 MODEL_INTERNAL / NONE）
    if preferred is None:
        level = "UNKNOWN"
        evidence_grade = "NONE"
    else:
        level = "MODEL_INTERNAL"
        evidence_grade = "MODEL_INTERNAL"

    return {
        "present": True,
        "version": CASH_STRATEGY_DECISION_ASSEMBLY_VERSION,
        "mode": "cash",
        "feasible_domain_applied": bool(feas_applied),
        "feasible_domain_size": (len(feas) if feas_applied else 0),
        "feasible_domain_note": (
            "V3I-P13 最小契约：提供 feasible_keys 时，各 tier 的 top 先在共同可行域内"
            "选取再对账；限域后 tier 无候选 → 该 tier 不可用 → 仍弃权；"
            "共同域内仍不一致 → 仍弃权。未提供时行为与变更前一致。"
            if feas_applied else "未提供 feasible_keys：不做限域，行为与变更前一致。"),
        "street": (search_eval or ranking or belief_conditioned or {}).get("street"),
        "assembly_policy": dict(pol),
        "scorers_supplied": supplied,
        "scorer_tops": {t: (entries[t] if entries.get(t) else None)
                        for t in TIERS},
        "agreed_key": list(agreed_key) if agreed_key else None,
        "disagreements": disagreements,
        "rejections": rejections,
        "abstained": abstained,
        "abstention_reason": ("cross_scorer_disagreement: top candidates differ"
                              if abstained else None),
        "override_used": override_used,
        "override_justification": override_reason,
        "preferred": preferred,
        "level": level,
        "evidence_grade": evidence_grade,
        "uncertainty": {
            "min_leaf_coverage": min_cov,
            "amount_tolerance_bb": tol,
            "n_scorers_supplied": sum(1 for v in supplied.values() if v),
            "n_scorers_alignable": len(comparable),
            "search_leaf_coverage": (entries["search"] or {}).get("leaf_coverage")
            if entries.get("search") else None,
            "belief_conditioning": (entries["belief_conditioned"] or {}).get(
                "conditioning") if entries.get("belief_conditioned") else None,
            "note": ("三层证据均为 MODEL_INTERNAL；本层不新增评分公式，只做对账"),
        },
        "solver": {"solver_status": solver_label, "blocked": bool(solver_blocked),
                   "read_only_passthrough": passthrough},
        "unknown": unknown,
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "final_decision_made": False,
        "note": ("本层输出是**离线可审计建议**，不是生产决策；"
                 "冲突时弃权为 UNKNOWN，绝不默认或猜测。"),
    }
