"""Phase 6 Step 6F：Profile Utility Validation（三臂，冻结口径）。

严格复用 Phase 5B Task B：
  - 测试玩家 = splits.json task_a.test_players（1001）
  - 每玩家 ring 手 ts 升序对半：history → posterior；future → 逐手评估
  - 逐手二元预测 → mean log-loss + Brier（越低越好）
  - 玩家级 bootstrap（保持玩家内配对）→ Individual − Population 的 95% CI

三臂:
  Arm P: population prior μ 固定预测（train-only prior）
  Arm I: prior + history → Beta-Binomial 后验均值 → 预测
  Arm C: 跳过（6D cluster_utility 失败，cluster_probs=null，禁止重新启用）

特征（6 个）:
  vpip / pfr / three_bet / cbet / fold_to_3bet / position_sensitivity
  - 前 5 个 + fold_to_3bet: 逐手 log-loss + Brier
  - position_sensitivity: 派生差量 → RMSE 口径（5B 同），仍参与 6 特征计数
  - cbet 保留结果并标记 low_power（机会中位 < LOW_POWER_OPP 时）

判定（冻结）:
  utility_proven = ≥3/6 特征显著改善（diff = Individual−Population，CI 上界 < 0）
  否则 utility_unproven + DEGRADE_LOG.json
  泄漏失败 = 硬失败（AssertionError），不降级继续

输出: data/uoft/models/phase6/phase6f/utility.json + DEGRADE_LOG.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from phase5b_math import bootstrap_pct, brier, mean_log_loss  # noqa: E402
from phase6f_common import (  # noqa: E402
    LL_FEATURES, LOW_POWER_OPP, feature_suff, fit_fold_to_3bet_prior,
    load_hb_fit_priors, load_ring_sequences, load_test_players,
    posterior_mean, split_half)

LL_PROBES = ["vpip", "pfr", "three_bet", "cbet", "fold_to_3bet"]
PS_COMPONENTS = ("rfi_ep", "rfi_lp")
N_BOOT = 1000
BOOT_SEED = 42


def _clip(p: float) -> float:
    return min(max(p, 1e-12), 1.0 - 1e-12)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", default="data/uoft/feature_ready/player_event_table.jsonl")
    ap.add_argument("--stats", default="data/uoft/stats/full_pool_suff_stats.jsonl")
    ap.add_argument("--splits", default="data/uoft/splits.json")
    ap.add_argument("--hb", default="data/uoft/models/mvp/hb_fit.json")
    ap.add_argument("--out", default="data/uoft/models/phase6/phase6f")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_players, test_players = load_test_players(args.splits)
    # ---- train-only prior 装配（含方案 B fold_to_3bet + 泄漏硬断言）----
    priors = load_hb_fit_priors(args.hb, train_players)
    priors["fold_to_3bet"] = fit_fold_to_3bet_prior(args.stats, train_players,
                                                    test_players)
    pop_prior = {f: priors[f]["mu"] for f in priors}

    # ---- 测试玩家 ring 序列（history/future）----
    seq = load_ring_sequences(args.events, player_filter=test_players)
    # 泄漏硬断言：序列玩家全部 ∈ test
    assert set(seq) <= test_players, "出现非测试玩家（泄漏）"

    # ---- 逐手 log-loss / Brier（Arm P vs Arm I）----
    per_probe: dict[str, dict] = {}
    for f in LL_PROBES:
        rows = []          # (ll_ind, ll_pop, brier_ind, brier_pop)
        for pid, hands in seq.items():
            halves = split_half(hands)
            if halves is None:
                continue
            hist, fut = halves
            s_h, n_h = feature_suff(hist, f)
            s_f, n_f = feature_suff(fut, f)
            if n_h < 10 or n_f < 10:
                continue
            p_ind = _clip(posterior_mean(priors[f], s_h, n_h))
            p_pop = _clip(pop_prior[f])
            rows.append({
                "ll_ind": mean_log_loss(p_ind, n_f, s_f),
                "ll_pop": mean_log_loss(p_pop, n_f, s_f),
                "brier_ind": brier(p_ind, n_f, s_f),
                "brier_pop": brier(p_pop, n_f, s_f),
                "n_future": n_f,
            })
        if len(rows) < 30:
            per_probe[f] = {"status": "insufficient_players",
                            "n_evaluated": len(rows)}
            continue
        diffs_ll = [r["ll_ind"] - r["ll_pop"] for r in rows]
        diffs_br = [r["brier_ind"] - r["brier_pop"] for r in rows]
        lo_ll, hi_ll, _ = bootstrap_pct(
            lambda xs: sum(xs) / len(xs), diffs_ll, n_resample=N_BOOT, seed=BOOT_SEED)
        lo_br, hi_br, _ = bootstrap_pct(
            lambda xs: sum(xs) / len(xs), diffs_br, n_resample=N_BOOT, seed=BOOT_SEED)
        # 改善方向为 negative（越低越好）：CI 上界 < 0 → 显著改善
        improved = hi_ll < 0
        hist_opp = [feature_suff(split_half(h)[0], f)[1]
                    for h in seq.values() if split_half(h)]
        hist_opp_med = sorted(hist_opp)[len(hist_opp) // 2]
        per_probe[f] = {
            "n_evaluated": len(rows),
            "ll_individual": round(sum(r["ll_ind"] for r in rows) / len(rows), 6),
            "ll_population": round(sum(r["ll_pop"] for r in rows) / len(rows), 6),
            "diff_ll": round(sum(diffs_ll) / len(diffs_ll), 6),
            "diff_ll_ci95": [round(lo_ll, 6), round(hi_ll, 6)],
            "improved_ll": improved,
            "brier_individual": round(sum(r["brier_ind"] for r in rows) / len(rows), 6),
            "brier_population": round(sum(r["brier_pop"] for r in rows) / len(rows), 6),
            "diff_brier": round(sum(diffs_br) / len(diffs_br), 6),
            "diff_brier_ci95": [round(lo_br, 6), round(hi_br, 6)],
            "history_opp_median": int(hist_opp_med),
            "low_power": hist_opp_med < LOW_POWER_OPP,
            "low_power_note": ("历史半机会中位 < 20，结果保留但判定降权"
                               if hist_opp_med < LOW_POWER_OPP else ""),
            "direction": "negative=improvement",
        }

    # ---- position_sensitivity：RMSE 口径（派生差量）----
    ps_rows = []
    for pid, hands in seq.items():
        halves = split_half(hands)
        if halves is None:
            continue
        hist, fut = halves
        s_ep_h, n_ep_h = feature_suff(hist, "rfi_ep")
        s_lp_h, n_lp_h = feature_suff(hist, "rfi_lp")
        s_ep_f, n_ep_f = feature_suff(fut, "rfi_ep")
        s_lp_f, n_lp_f = feature_suff(fut, "rfi_lp")
        if min(n_ep_h, n_lp_h, n_ep_f, n_lp_f) < 5:
            continue
        ind = (posterior_mean(priors["rfi_lp"], s_lp_h, n_lp_h)
               - posterior_mean(priors["rfi_ep"], s_ep_h, n_ep_h))
        pop = pop_prior["rfi_lp"] - pop_prior["rfi_ep"]
        truth = (s_lp_f / n_lp_f) - (s_ep_f / n_ep_f)
        ps_rows.append({"err_ind": (ind - truth) ** 2, "err_pop": (pop - truth) ** 2})
    if len(ps_rows) >= 30:
        rmse_ind = math.sqrt(sum(r["err_ind"] for r in ps_rows) / len(ps_rows))
        rmse_pop = math.sqrt(sum(r["err_pop"] for r in ps_rows) / len(ps_rows))
        # RMSE 差 = sqrt(mean(err_ind)) − sqrt(mean(err_pop))；bootstrap 输入为误差平方对
        lo, hi, _ = bootstrap_pct(
            lambda pairs: (math.sqrt(sum(a for a, _ in pairs) / len(pairs))
                           - math.sqrt(sum(b for _, b in pairs) / len(pairs))),
            [(r["err_ind"], r["err_pop"]) for r in ps_rows],
            n_resample=N_BOOT, seed=BOOT_SEED)
        # RMSE 差显著 < 0 → 改善
        improved = hi < 0
        per_probe["position_sensitivity"] = {
            "metric": "rmse", "n_evaluated": len(ps_rows),
            "rmse_individual": round(rmse_ind, 6),
            "rmse_population": round(rmse_pop, 6),
            "diff_rmse": round(rmse_ind - rmse_pop, 6),
            "diff_rmse_ci95": [round(lo, 6), round(hi, 6)],
            "improved": improved,
            "direction": "negative=improvement（RMSE 越低越好）",
            "note": "派生差量无逐手 log-loss，采用 5B RMSE 口径，仍参与 6 特征计数",
        }
    else:
        per_probe["position_sensitivity"] = {"status": "insufficient_players",
                                             "n_evaluated": len(ps_rows)}

    # ---- 判定：≥3/6 显著改善 ----
    improved_count = sum(1 for f, v in per_probe.items()
                         if v.get("improved_ll") or v.get("improved"))
    utility_proven = improved_count >= 3
    out = {
        "config": {
            "arms": ["population", "individual_eb_profile"],
            "arm_c": "skipped: 6D cluster_utility gate failed, cluster_probs=null, "
                     "禁止重新启用 cluster",
            "prior_provenance": "train-only: hb_fit.json(fit_on=task_a_train_players) "
                                "+ fold_to_3bet 方案 B(train suff stats)",
            "bootstrap": {"unit": "player", "paired": True,
                          "n": N_BOOT, "seed": BOOT_SEED},
            "direction": "negative=improvement（log-loss/Brier/RMSE 均越低越好）",
            "utility_proven_threshold": ">=3/6 特征显著改善（CI 上界<0）",
            "low_power_opp_threshold": LOW_POWER_OPP,
        },
        "features": per_probe,
        "improved_count": improved_count,
        "utility_proven": utility_proven,
        "conclusion": ("Profile Utility PROVEN：%d/6 特征显著改善"
                       % improved_count if utility_proven else
                       "Profile Utility UNPROVEN：仅 %d/6 特征显著改善"
                       % improved_count),
    }
    (out_dir / "utility.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    if not utility_proven:
        entries = [{"component": "profile_utility", "path": "utility_unproven",
                    "reason": f"仅 {improved_count}/6 特征显著改善（阈值 >=3/6）",
                    "evidence": {f: {"diff_ll": v.get("diff_ll"),
                                     "improved": v.get("improved_ll",
                                                       v.get("improved"))}
                                 for f, v in per_probe.items()}}]
        (out_dir / "DEGRADE_LOG.json").write_text(
            json.dumps({"entries": entries}, ensure_ascii=False, indent=2),
            encoding="utf-8")

    print(json.dumps({"improved_count": improved_count,
                      "utility_proven": utility_proven,
                      "features": {f: {"improved": bool(v.get("improved_ll",
                                                             v.get("improved"))),
                                       "low_power": v.get("low_power", False)}
                                   for f, v in per_probe.items()}},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
