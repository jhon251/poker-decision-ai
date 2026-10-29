"""V3H-5 — CASH Action Ranking / Selection（显式评分 + 确定性 tie-break）。

定位
----
补齐 V3H-3 路线审计中的 `L2 → L5` 缺口：**在候选动作之间做显式、可审计的
评分与排序**，并给出**选择理由**。这是 V3H-4 已知限制 **L4**（此前未实现
选择/排序）的解决层。

**本层不做最终决策、不接生产 Judge、不接 Solver、不声称 GTO/equilibrium。**

评分口径（`RANKING_MODEL`，全部显式、可复算）
---------------------------------------------
每个候选的分数 = **pot 份额期望**（pot-share units），由已记录的分量相加：

| 动作 | 分数（pot 份额） | 说明 |
|---|---|---|
| `fold` | `0.0` | 定义：弃牌收益为 0（已投入为沉没成本）——**不是**启发式猜测 |
| `check` | `equity` | 摊牌份额（无人下注时） |
| `call` | `equity*(pot + 2*to_call)/pot − to_call/pot` | 跟注后期望份额净值 |
| `bet`/`raise`/`allin` | `equity*(1 + 2r)/pot − r/pot`，`r` = 投入 bb | 被跟注情形；**不含**对手弃牌收益（无对手模型） |

* 上式全部是**算术恒等式**（无自由参数、无魔法系数），因此
  `score_source = "MODEL_INTERNAL"`；
* 缺少 hero 手牌 / 公共牌 → **不产出排序**，`preferred=None` 且
  `level="UNKNOWN"`（信息不足必须 UNKNOWN，不得默认）；
* 存在 `unknown`（如 `min_raise_to` 缺失）→ 仍产出排序，但
  `preferred` **携带** `uncertainty` 与 `unknown`，且 `confidence` 降低。

确定性 tie-break（**不得依赖 dict 顺序或随机**）
------------------------------------------------
同分时依次比较：
1. 动作优先级 `fold < check < call < bet < raise < allin`（保守优先）；
2. `amount_bb` 较小者优先；
3. `sizing_kind` 字典序；
4. `action` 字典序。
→ 完全同分的候选之间**顺序稳定且可复现**。

随机性
------
本层**不 import `random`**。上游 equity 的 Monte Carlo 参数（seed/iterations/
样本量）**原样转述**，并据此给出 `uncertainty`（样本量越大越窄）。

数据接口（生产 import 图零新增）
--------------------------------
**不在模块顶层 import 任何 V3G 层**；`action_policy` / `belief` /
`evaluator` 由调用方传入，缺失即 fail closed。
"""
from __future__ import annotations

from typing import Any

CASH_STRATEGY_ACTION_RANKING_VERSION = "v3h5-cash-strategy-action-ranking-v1"

#: 评分模型身份（显式声明，便于审计与复算）
RANKING_MODEL: dict[str, Any] = {
    "name": "pot_share_arithmetic_v1",
    "unit": "pot_share",
    "score_source": "MODEL_INTERNAL",
    "free_parameters": 0,
    "uses_opponent_fold_equity": False,
    "uses_randomness": False,
    "is_gto": False,
    "is_equilibrium": False,
}

CLAIM_BOUNDARY: dict[str, bool] = {
    "is_gto": False,
    "is_equilibrium": False,
    "is_solver_output": False,
    "is_profit_evidence": False,
    "wired_to_production": False,
    "is_final_decision": False,
}

#: 动作优先级（tie-break 第 1 键；保守优先）
ACTION_ORDER: tuple[str, ...] = ("fold", "check", "call", "bet", "raise", "allin")

_AGGRESSIVE = {"bet", "raise", "allin"}


def _is_cash(state: Any) -> bool:
    if not isinstance(state, dict):
        return False
    gt = state.get("game_type")
    return isinstance(gt, str) and gt.strip().lower() == "cash"


def _stub(reason: str) -> dict:
    return {
        "present": False,
        "version": CASH_STRATEGY_ACTION_RANKING_VERSION,
        "reason": reason,
        "evidence_grade": "NONE",
        "unknown": [reason],
        "ranked": [],
        "preferred": None,
        "level": "UNKNOWN",
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }


def _score_candidate(cand: dict, equity: float | None, pot: float,
                     to_call: float) -> tuple[float | None, dict, list[str]]:
    """返回 (score_pot_share, components, unknown)。分数缺输入时为 None。"""
    act = cand.get("action")
    amt = cand.get("amount_bb")
    unknown: list[str] = []
    comp: dict[str, Any] = {}

    if act == "fold":
        comp["fold_zero_by_definition"] = 0.0
        return 0.0, comp, unknown

    if equity is None:
        unknown.append("equity_unavailable")
        return None, comp, unknown

    if pot <= 0:
        unknown.append("pot_nonpositive")
        return None, comp, unknown

    if act == "check":
        comp["equity_share"] = round(equity, 6)
        return equity, comp, unknown

    if act == "call":
        if to_call <= 0:
            # to_call=0 的 "call" 等价于 check（不新增投入）
            comp["equity_share"] = round(equity, 6)
            comp["note"] = "to_call_le_0_treated_as_no_investment"
            return equity, comp, unknown
        invest = min(to_call, float(cand.get("amount_bb") or to_call))
        share = equity * (pot + 2.0 * invest) / pot - invest / pot
        comp["equity_share_gross"] = round(
            equity * (pot + 2.0 * invest) / pot, 6)
        comp["investment_penalty"] = round(invest / pot, 6)
        return share, comp, unknown

    if act in _AGGRESSIVE:
        if not isinstance(amt, (int, float)):
            unknown.append("amount_bb_missing")
            return None, comp, unknown
        invest = float(amt)
        share = equity * (pot + 2.0 * invest) / pot - invest / pot
        comp["equity_share_gross"] = round(
            equity * (pot + 2.0 * invest) / pot, 6)
        comp["investment_penalty"] = round(invest / pot, 6)
        comp["opponent_fold_equity_excluded"] = True
        return share, comp, unknown

    unknown.append(f"unknown_action:{act!r}")
    return None, comp, unknown


def _sort_key(row: dict) -> tuple:
    """确定性 tie-break（同分时稳定；不依赖 dict 顺序）。"""
    act = str(row.get("action") or "")
    rank = ACTION_ORDER.index(act) if act in ACTION_ORDER else len(ACTION_ORDER)
    amt = row.get("amount_bb")
    amt_key = float(amt) if isinstance(amt, (int, float)) else float("inf")
    score = row.get("score")
    score_key = -float(score) if isinstance(score, (int, float)) else float("inf")
    return (score_key, rank, amt_key, str(row.get("sizing_kind") or ""), act)


def _uncertainty(strength: dict) -> dict:
    """把上游 Monte Carlo 参数转述为不确定性（不新增随机）。"""
    samples = strength.get("samples")
    if not isinstance(samples, int) or samples <= 0:
        return {"available": False, "reason": "samples_unavailable"}
    return {
        "available": True,
        "method": strength.get("method"),
        "samples": samples,
        "seed": strength.get("seed"),
        "iterations": strength.get("iterations"),
        "note": ("Monte Carlo 抽样误差未用解析式估计；样本量越大越窄。"
                 "本层原样转述上游参数，不新增随机。"),
    }


def rank_actions(state: Any, *, action_policy: dict | None = None,
                 belief: dict | None = None,
                 evaluator: dict | None = None,
                 action_value_input: dict | None = None) -> dict:
    """对候选动作排序并给出选择理由（**不接生产、不做最终决策**）。

    Parameters
    ----------
    state : CASH state（非 CASH → present=False）
    action_policy : **必填**（`cash_strategy_action_policy.build_action_policy`）
    belief : 可选（`cash_strategy_dynamic_belief.build_dynamic_belief`）
    evaluator : 可选（`cash_strategy_policy_evaluator.evaluate_policy`）
    action_value_input : 可选（V3I-G1 **Phase 2**）。**统一动作价值输入**：
        由 `cash_action_value_contract.unified_action_value_input` 构造，
        内含**唯一**价值表（`unified_values`）、共同候选域（`domain`）与
        定义性 provenance。

        * `value_mode == "unified"` 时：本层 `ranked`/`preferred` 由**统一动作价值**
          决定，且只在**共同候选域**内；旧 `pot_share_arithmetic_v1` 排序逐位并列
          保留于 `ranked_legacy`/`preferred_legacy`（**不删除历史口径**）；
        * 值缺失/来源不符/域外 ⇒ 不参与排序（UNKNOWN），**禁止**回落 legacy 当作决策；
        * 本层**不 import 契约模块**（保持"只 import 标准库"），统一价值表由调用方注入，
          本层只校验其 provenance 字段。
    """
    if not _is_cash(state):
        return _stub("not_cash_mode_fail_closed")
    if not isinstance(action_policy, dict):
        return _stub("action_policy_required")
    ap = action_policy
    if not ap.get("present"):
        return _stub(f"action_policy_unavailable:{ap.get('reason', 'unknown')}")

    strength = ap.get("hand_strength") or {}
    odds = ap.get("pot_odds") or {}
    pot = float(ap.get("pot_bb") or 0.0)
    to_call = float(ap.get("to_call_bb") or 0.0)
    equity = strength.get("hero_equity") if strength.get("present") else None

    unknown: list[str] = list(ap.get("unknown") or [])
    rows: list[dict] = []
    for cand in ap.get("candidates") or []:
        if not isinstance(cand, dict):
            continue
        score, comp, cand_unknown = _score_candidate(cand, equity, pot, to_call)
        for u in cand_unknown:
            if u not in unknown:
                unknown.append(u)
        rows.append({
            "action": cand.get("action"),
            "sizing_kind": cand.get("sizing_kind"),
            "amount_bb": cand.get("amount_bb"),
            "score": (round(score, 6) if isinstance(score, (int, float))
                      else None),
            "score_unit": RANKING_MODEL["unit"],
            "score_source": RANKING_MODEL["score_source"],
            "components": comp,
            "unknown": cand_unknown,
            "is_gto": False,
            "is_equilibrium": False,
        })

    scored = [r for r in rows if isinstance(r["score"], (int, float))]
    ranked = sorted(scored, key=_sort_key)
    for i, r in enumerate(ranked, 1):
        r["rank"] = i
    unscored = [r for r in rows if not isinstance(r["score"], (int, float))]
    for r in unscored:
        r["rank"] = None

    level = "UNKNOWN" if (equity is None or not ranked) else "MODEL_INTERNAL"
    preferred = ranked[0] if ranked and level != "UNKNOWN" else None

    # 选择理由（必须显式；不得"无理由"地返回 null）
    if preferred is None:
        reason = {
            "level": "UNKNOWN",
            "why": ("insufficient information: hero hole cards / board / equity "
                    "unavailable, so no ranking is produced"
                    if equity is None else "no scorable candidate"),
            "unknown": unknown,
        }
    else:
        runner = ranked[1] if len(ranked) > 1 else None
        reason = {
            "level": level,
            "why": ("highest pot-share score under the explicit arithmetic "
                    "model (opponent fold equity excluded)"),
            "margin_vs_runner_up": (
                round(preferred["score"] - runner["score"], 6) if runner else None),
            "tie_break_applied": bool(
                runner and abs(preferred["score"] - runner["score"]) < 1e-12),
            "components": preferred["components"],
            "unknown": unknown,
            "belief_used": isinstance(belief, dict) and bool(belief.get("present")),
            "evaluator_used": isinstance(evaluator, dict) and bool(
                evaluator.get("present")),
        }

    # ── V3I-G1 Phase 2：**统一动作价值模式**（同域 + 同记账 + 同一价值表）──
    # 触发：`action_value_input["value_mode"] == "unified"`。本层**不 import**
    # 契约模块（保持只 import 标准库），价值表由调用方注入；本层只做
    # ① domain 过滤 ② provenance 校验 ③ 以统一价值重排（legacy 并列保留）。
    unified = bool(isinstance(action_value_input, dict)
                   and str(action_value_input.get("value_mode")) == "unified")
    ranked_legacy = ranked
    preferred_legacy = preferred
    unscored_legacy = unscored
    n_ranked_legacy = len(ranked)
    unified_out_of_domain: list[dict] = []
    unified_provenance_issues: list[dict] = []
    unified_unavailable_reason: str | None = None
    unified_consistency: dict[str, Any] | None = None
    if unified:
        av = action_value_input or {}
        shared = av.get("unified_values") or {}
        dom = av.get("domain") or {}
        common = set(dom.get("common_keys") or list(shared))
        want_version = av.get("version")
        rows_unified: list[dict] = []
        for r in rows:
            act = r.get("action")
            amt = r.get("amount_bb")
            key = f"{act}|{round(float(amt), 6) if isinstance(amt, (int, float)) and not isinstance(amt, bool) else 0.0}"
            in_domain = key in common
            row = shared.get(key)
            reasons: list[str] = []
            if not in_domain:
                reasons.append("outside_common_candidate_domain")
            if row is None:
                reasons.append("shared_value_table_missing")
            else:
                prov = row.get("provenance") or {}
                if row.get("status") != "OK" or row.get("value") is None:
                    reasons.append(str(row.get("status")
                                       or "unified_value_unavailable"))
                if want_version is not None and prov.get("contract_version") \
                        != want_version:
                    reasons.append("unified_value_provenance_version_mismatch")
                    unified_provenance_issues.append(
                        {"key": key, "want": want_version,
                         "got": prov.get("contract_version")})
                if prov.get("free_parameters") != 0:
                    reasons.append("unified_value_free_parameters_not_zero")
                    unified_provenance_issues.append(
                        {"key": key, "free_parameters": prov.get("free_parameters")})
            r["unified_value_status"] = "UNKNOWN" if reasons else "OK"
            r["unified_value"] = None if reasons else row.get("value")
            r["unified_value_terms"] = ({} if reasons
                                        else dict(row.get("terms") or {}))
            r["unified_value_consequences"] = (
                {} if reasons else dict(row.get("consequences") or {}))
            r["unified_value_unknown"] = reasons
            if reasons:
                if not in_domain:
                    unified_out_of_domain.append({
                        "key": key, "action": act, "amount_bb": amt,
                        "reason": "outside_common_candidate_domain",
                        "score_legacy_v3": r.get("score"),
                        "counts_as_evidence_disagreement": False})
                continue
            rr = dict(r)
            rr["score_legacy_v3"] = r.get("score")
            rr["score"] = row.get("value")
            rr["score_source"] = "MODEL_INTERNAL"
            rr["value_mode"] = "unified"
            rr["action_value_pot_share"] = row.get("value")
            rr["action_value_gain_vs_check"] = row.get("gain_vs_check")
            rr["action_value_terms"] = dict(row.get("terms") or {})
            rr["action_value_measured"] = dict(row.get("measured") or {})
            rr["action_value_provenance"] = dict(row.get("provenance") or {})
            rows_unified.append(rr)
        rows_unified.sort(key=lambda r: (-float(r["score"]), str(r.get("action")),
                                         float(r.get("amount_bb") or 0.0)))
        for i, r in enumerate(rows_unified, 1):
            r["rank"] = i
        ranked = rows_unified
        unscored = [r for r in rows if not isinstance(r.get("score"), (int, float))
                    or r.get("unified_value_status") != "OK"]
        preferred = ranked[0] if ranked else None
        if preferred is None:
            unified_unavailable_reason = (
                av.get("unavailable_reason")
                or ("no_candidate_in_common_domain" if not common
                    else "no_candidate_with_unified_value"))
        level = "UNKNOWN" if (not ranked) else "MODEL_INTERNAL"
        reason = ({
            "level": level,
            "value_mode": "unified",
            "why": ("highest **unified action value** on the common candidate "
                    "domain (same accounting as the search tier); the legacy "
                    "pot-share ranking is dual-reported"),
            "margin_vs_runner_up": (
                round(float(preferred["score"]) - float(ranked[1]["score"]), 6)
                if len(ranked) > 1 else None),
            "components": preferred.get("action_value_terms"),
            "unknown": unknown + [unified_unavailable_reason]
            if unified_unavailable_reason else list(unknown),
        } if preferred is not None else {
            "level": "UNKNOWN",
            "value_mode": "unified",
            "why": ("unified action value unavailable for every candidate in the "
                    "common domain ⇒ no ranking (fail closed; the legacy ranking "
                    "is dual-reported but is NOT used as the decision basis)"),
            "unknown": unknown + ([unified_unavailable_reason]
                                  if unified_unavailable_reason else []),
        })
        unified_consistency = {
            "checked": len(shared),
            "provenance_issues": unified_provenance_issues,
            "verified_by": ("contract provenance fields (version + "
                            "free_parameters==0); the value arithmetic itself is "
                            "independently recomputed by the search tier"),
            "common_domain_size": len(common),
            "out_of_domain": len(unified_out_of_domain),
        }

    return {
        "present": True,
        "version": CASH_STRATEGY_ACTION_RANKING_VERSION,
        "mode": ap.get("mode"),
        "street": ap.get("street"),
        "ranking_model": dict(RANKING_MODEL),
        "ranked": ranked,
        "unscored": unscored,
        "n_ranked": len(ranked),
        "preferred": ({"action": preferred["action"],
                       "sizing_kind": preferred["sizing_kind"],
                       "amount_bb": preferred["amount_bb"],
                       "score": preferred.get("score"),
                       "score_source": preferred["score_source"],
                       "rank": 1,
                       "value_mode": ("unified" if unified else "legacy")}
                      if preferred else None),
        "preferred_reason": reason,
        "level": level,
        "uncertainty": _uncertainty(strength),
        "pot_odds": odds,
        "unknown": unknown,
        "evidence_grade": "MODEL_INTERNAL",
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "final_decision_made": False,
        # ── V3I-G1 Phase 2（只新增；legacy 模式行为逐位不变）──────────────
        "value_mode": ("unified" if unified else "legacy"),
        "unified_consistency": unified_consistency,
        "unified_domain": ((action_value_input or {}).get("domain")
                           if unified else None),
        "unified_out_of_domain": unified_out_of_domain,
        "unified_unavailable_reason": unified_unavailable_reason,
        "unified_provenance_issues": unified_provenance_issues,
        "ranked_legacy": list(ranked_legacy) if unified else None,
        "preferred_legacy": preferred_legacy if unified else None,
        "unscored_legacy": list(unscored_legacy) if unified else None,
        "n_ranked_legacy": n_ranked_legacy if unified else None,
        "legacy_ranking_preserved": True,
        "dual_report": {
            "legacy_field": ("score_legacy_v3" if unified else "score"),
            "legacy_model": dict(RANKING_MODEL),
            "legacy_model_unchanged": True,
            "action_value_field": "action_value_pot_share",
            "value_mode": ("unified" if unified else "legacy"),
            "legacy_ranking_field": "ranked_legacy" if unified else "ranked",
            "note": ("旧 `pot_share_arithmetic_v1` 逐位并列保留；统一模式下 "
                     "`score` 语义为统一动作价值，旧值并列于 `score_legacy_v3`；"
                     "不构成质量 / EV / GTO / 盈利证据"),
        },
    }
