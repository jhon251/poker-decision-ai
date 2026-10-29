"""P55 — REAL Decision Divergence Diagnosis（只读诊断；无策略修改）。

目标：解释 P54 的 "REAL fold calibration 改善（Brier 0.1291→0.1259）但
decision agreement 下降（34.1%→31.4%）"。对 action changes 做分类归因：
  `range → fold → equity → EV → selected_action`。

诊断问题（授权清单）：
  1. 变化类型分布（fold/call、check/bet、bet/raise、size change、其他）；
  2. 每类的 equity/fold/lambda/证据样本量差值；
  3. posterior 是否系统性偏松（λ 符号分布）；
  4. conditional fold 是否系统性偏高/偏低；
  5. 小样本过度调整（样本量 × 变化率关联）；
  6. 单街 EV 与职业动作的结构性差异（按 observed action 分层 agreement）。

语义声明：regret_proxy 只表示"按当前模型评价的动作差"，不是真实盈利
或真实 regret。诊断不修改任何公式/权重/阈值。
"""
from __future__ import annotations

import statistics
from typing import Any

from .pluribus_decision_adapter import iter_pluribus_rows, parse_hand_row
from .pluribus_validation import validate

CHANGE_CATEGORIES = ("fold_to_call", "call_to_fold", "check_to_bet",
                     "bet_to_check", "bet_to_raise", "raise_to_bet",
                     "call_to_raise", "raise_to_call", "fold_to_raise",
                     "size_change", "other")


def classify_change(b: str | None, r: str | None,
                    b_amt: float | None, r_amt: float | None) -> str:
    """动作变化分类（确定性）。"""
    if b is None or r is None:
        return "other"
    if b == r:
        if b in ("bet", "raise") and b_amt is not None and r_amt is not None \
                and abs(b_amt - r_amt) > 1e-6:
            return "size_change"
        return "other"
    pair = f"{b}_to_{r}"
    return pair if pair in CHANGE_CATEGORIES else "other"


def _mean(vals):
    vals = [v for v in vals if v is not None]
    return round(statistics.mean(vals), 6) if vals else None


def _median(vals):
    vals = [v for v in vals if v is not None]
    return round(statistics.median(vals), 6) if vals else None


def _classify_rows(rows: list[dict]) -> dict:
    """enriched 行 → 分类 + 每类差值统计（纯聚合；确定性）。"""
    changed = [r for r in rows if r.get("action_change")]
    unchanged = [r for r in rows if r.get("action_change") is False]
    cats: dict[str, list[dict]] = {}
    for r in changed:
        cat = classify_change(r.get("baseline_best"), r.get("real_best"),
                              r.get("baseline_amount_bb"),
                              r.get("real_amount_bb"))
        cats.setdefault(cat, []).append(r)

    def _cat_stats(rs):
        return {
            "n": len(rs),
            "equity_diff_mean": _mean([r.get("equity_real", 0)
                                       - r.get("equity_base", 0)
                                       for r in rs]),
            "p_fold_diff_mean": _mean([r["p_fold_real"] - r["p_fold_baseline"]
                                       for r in rs
                                       if r.get("p_fold_real") is not None
                                       and r.get("p_fold_baseline")
                                       is not None]),
            "lambda_mean": _mean([r.get("real_lambda") for r in rs]),
            "lambda_median": _median([r.get("real_lambda") for r in rs]),
            "villain_cfi_events_median": _median(
                [r.get("villain_cfi_events") for r in rs]),
            "observed_agreement_in_class": (
                round(sum(1 for r in rs
                          if r.get("real_best")
                          == r.get("observed_human_action")) / len(rs), 6)
                if rs else None),
        }

    lam_all = [r.get("real_lambda") for r in rows
               if r.get("real_lambda") is not None]
    lam_changed = [r.get("real_lambda") for r in changed
                   if r.get("real_lambda") is not None]
    # 小样本 × 变化率（证据桶）
    buckets: dict[str, list[int]] = {"0-99": [], "100-299": [],
                                     "300-999": [], "1000+": []}
    for r in rows:
        n_ev = r.get("villain_cfi_events") or 0
        k = ("0-99" if n_ev < 100 else "100-299" if n_ev < 300
             else "300-999" if n_ev < 1000 else "1000+")
        buckets[k].append(1 if r.get("action_change") else 0)
    change_rate_by_evidence = {
        k: round(sum(v) / len(v), 6) if v else None
        for k, v in buckets.items()}
    # 按 observed action 分层 agreement（单街 EV vs 职业动作结构差异）
    by_obs: dict[str, dict] = {}
    for tag, arm in (("baseline", "baseline_best"), ("real", "real_best")):
        agg: dict[str, list[int]] = {}
        for r in rows:
            if r.get(arm) is None:
                continue
            agg.setdefault(r["observed_human_action"], []).append(
                1 if r.get(arm) == r["observed_human_action"] else 0)
        by_obs[tag] = {k: round(sum(v) / len(v), 6) if v else None
                       for k, v in sorted(agg.items())}
    return {
        "n_rows": len(rows),
        "n_changed": len(changed),
        "action_change_rate": (round(len(changed) / len(rows), 6)
                               if rows else None),
        "change_categories": {cat: _cat_stats(rs)
                              for cat, rs in sorted(cats.items())},
        "unchanged": _cat_stats(unchanged),
        "lambda_distribution": {
            "all_mean": _mean(lam_all), "all_median": _median(lam_all),
            "all_lt0_share": (round(sum(1 for x in lam_all if x < 0)
                                    / len(lam_all), 6) if lam_all else None),
            "changed_mean": _mean(lam_changed),
            "changed_median": _median(lam_changed)},
        "fold_prob_diff": {
            "mean": _mean([r["p_fold_real"] - r["p_fold_baseline"]
                           for r in rows
                           if r.get("p_fold_real") is not None
                           and r.get("p_fold_baseline") is not None]),
            "median": _median([r["p_fold_real"] - r["p_fold_baseline"]
                               for r in rows
                               if r.get("p_fold_real") is not None
                               and r.get("p_fold_baseline") is not None]),
            "real_lower_share": (
                round(sum(1 for r in rows
                          if r.get("p_fold_real") is not None
                          and r.get("p_fold_baseline") is not None
                          and r["p_fold_real"] < r["p_fold_baseline"])
                      / max(1, sum(1 for r in rows
                                   if r.get("p_fold_real") is not None
                                   and r.get("p_fold_baseline")
                                   is not None)), 6))},
        "equity_diff": {
            "mean": _mean([r.get("equity_real", 0) - r.get("equity_base", 0)
                           for r in rows if r.get("equity_base") is not None])},
        "change_rate_by_evidence_bucket": change_rate_by_evidence,
        "agreement_by_observed_action": by_obs,
    }


def diagnose_full(hands: list[dict], *, eval_hand_limit: int = 0,
                  decision_hand_limit: int = 400,
                  equity_iterations: int = 1500) -> dict:
    """诊断入口：validate(diagnose) 全行收集 → 分类归因（确定性）。"""
    rep = validate(hands, eval_hand_limit=eval_hand_limit,
                   decision_hand_limit=decision_hand_limit,
                   equity_iterations=equity_iterations, diagnose=True)
    diag = _classify_rows(rep["decision_rows_all"])
    # 结构性检验：单街模型 vs 职业动作的一致率上限代理（fold/call 类
    # 决策的 agreement 通常显著高于 bet 类——混合策略/多街规划主导）
    return {"diagnosis_version": "p55-v1",
            "base_metrics": {k: rep[k] for k in
                             ("split", "labels", "unknown", "legality",
                              "fold_calibration")},
            "decision_ev_headline": {k: v for k, v in
                                     rep["decision_ev"].items()
                                     if k != "rows_all"},
            "classification": diag}
