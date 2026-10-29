"""Phase 6.5：Prediction Baseline & Integration Benchmark（学习曲线）。

冻结原则：
  - 禁止使用 6G static_full_history Profile 作为 prediction input
  - 每 horizon 从 history-only 数据临时构造与 Profile 同 shape 的画像
  - train/test 玩家互斥；prior 必须 train-only（hb_fit + fold_to_3bet 方案 B）
  - future 手不得进入 history；cluster 永远 disabled
  - time_sensitive 仅状态信息，不改变 value/reliability/consumable

设计（Step 0 Audit §8 已确认）：
  - horizon N ∈ {50,100,200}；每玩家需 ≥ N+50 ring 手
  - 特征 6: vpip/pfr/three_bet/cbet/position_sensitivity/fold_to_3bet
  - 四臂: B0 population prior μ / B1 naive / B2 wilson center / P individual EB
  - 主判定: N=200 时 P vs B0 在 ≥3/6 特征显著改善（玩家级 bootstrap CI 上界<0）
  - cbet 全部 horizon low_power（机会中位 2/3/7），保留参与 6 特征计数，
    最小机会门槛显式放宽至 2 并记录（audit 批准，非事后调阈值）
  - ps: N=50 insufficient（分量机会<5），N=100/200 用 RMSE 参与
  - 指标: mean log-loss + Brier（越低越好）；ps 用 RMSE
  - bootstrap: 玩家级配对，n=1000，seed=42

输出: data/uoft/models/phase6/phase65/benchmark.json（+ DEGRADE_LOG.json 若失败）
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from phase5b_math import bootstrap_pct, brier, mean_log_loss, wilson_interval  # noqa: E402
from phase6f_common import (  # noqa: E402
    feature_suff, fit_fold_to_3bet_prior, load_hb_fit_priors,
    load_ring_sequences, load_test_players, posterior_mean)

HORIZONS = (50, 100, 200)
LL_PROBES = ("vpip", "pfr", "three_bet", "cbet", "fold_to_3bet")
PS_COMPONENTS = ("rfi_ep", "rfi_lp")
N_BOOT = 1000
BOOT_SEED = 42
MIN_FUTURE = 50
MIN_OPP_DEFAULT = 10            # history/future 机会门槛（默认特征）
CBET_MIN_OPP = 2                # cbet 低机会显式放宽（audit 批准，config 记录）
PS_MIN_OPP = 5                  # ps 分量机会门槛
LOW_POWER_MED_OPP = 20          # 历史机会中位 < 此值 → low_power


def _clip(p: float) -> float:
    return min(max(p, 1e-12), 1.0 - 1e-12)


def _eval_ll_probe(seq: dict, prior: dict, pop_mu: float, N: int, probe: str
                   ) -> dict:
    """单特征单 horizon：四臂逐手 log-loss/Brier + 玩家级配对 diff。"""
    min_opp = CBET_MIN_OPP if probe == "cbet" else MIN_OPP_DEFAULT
    rows = []
    for pid, hands in seq.items():
        if len(hands) < N + MIN_FUTURE:
            continue
        hands_sorted = sorted(hands, key=lambda x: x["ts"] or "")
        hist, fut = hands_sorted[:N], hands_sorted[N:]
        # 泄漏断言：history 全部早于 future
        assert hist[-1]["ts"] <= fut[0]["ts"], f"{pid} history/future 时间重叠"
        s_h, n_h = feature_suff(hist, probe)
        s_f, n_f = feature_suff(fut, probe)
        # 冻结 5B 派生中 cbet 存在 S>n（cbet_opportunity 排除"加注者后续弃牌"的手，
        # 但该加注者可能已下注 → cbet=1 而 opportunity=0）。比率与 Wilson 要求 S<=n：
        # 以 n_eff = max(S, n) 修正（S 事件必然存在机会；保守、不改成功计数）。
        if s_h > n_h:
            n_h = s_h
        if s_f > n_f:
            n_f = s_f
        if n_h < min_opp or n_f < min_opp:
            continue
        p_ind = _clip(posterior_mean(prior, s_h, n_h))
        p_pop = _clip(pop_mu)
        naive = s_h / n_h
        wlo, whi = wilson_interval(s_h, n_h)
        p_wil = _clip((wlo + whi) / 2.0)
        rows.append({
            "ll_ind": mean_log_loss(p_ind, n_f, s_f),
            "ll_pop": mean_log_loss(p_pop, n_f, s_f),
            "ll_naive": mean_log_loss(_clip(naive), n_f, s_f),
            "ll_wilson": mean_log_loss(p_wil, n_f, s_f),
            "brier_ind": brier(p_ind, n_f, s_f),
            "brier_pop": brier(p_pop, n_f, s_f),
            "n_history_opp": n_h, "n_future_opp": n_f,
        })
    if not rows:
        return {"status": "insufficient_players", "n_evaluated": 0}
    n = len(rows)
    hist_opp_med = sorted(r["n_history_opp"] for r in rows)[n // 2]
    low_power = hist_opp_med < LOW_POWER_MED_OPP

    def _ci(diffs):
        lo, hi, _ = bootstrap_pct(lambda xs: sum(xs) / len(xs), diffs,
                                  n_resample=N_BOOT, seed=BOOT_SEED)
        return round(lo, 6), round(hi, 6)

    diff_ll = [r["ll_ind"] - r["ll_pop"] for r in rows]
    diff_ll_naive = [r["ll_ind"] - r["ll_naive"] for r in rows]
    diff_ll_wilson = [r["ll_ind"] - r["ll_wilson"] for r in rows]
    diff_br = [r["brier_ind"] - r["brier_pop"] for r in rows]
    lo, hi = _ci(diff_ll)
    return {
        "n_evaluated": n,
        "ll_individual": round(sum(r["ll_ind"] for r in rows) / n, 6),
        "ll_population": round(sum(r["ll_pop"] for r in rows) / n, 6),
        "ll_naive": round(sum(r["ll_naive"] for r in rows) / n, 6),
        "ll_wilson": round(sum(r["ll_wilson"] for r in rows) / n, 6),
        "diff_ll_vs_pop": round(sum(diff_ll) / n, 6),
        "diff_ll_vs_pop_ci95": [lo, hi],
        "diff_ll_vs_naive": round(sum(diff_ll_naive) / n, 6),
        "diff_ll_vs_naive_ci95": list(_ci(diff_ll_naive)),
        "diff_ll_vs_wilson": round(sum(diff_ll_wilson) / n, 6),
        "diff_ll_vs_wilson_ci95": list(_ci(diff_ll_wilson)),
        "brier_individual": round(sum(r["brier_ind"] for r in rows) / n, 6),
        "brier_population": round(sum(r["brier_pop"] for r in rows) / n, 6),
        "diff_brier_vs_pop": round(sum(diff_br) / n, 6),
        "diff_brier_vs_pop_ci95": list(_ci(diff_br)),
        "improved_vs_pop": hi < 0,          # 主判定方向: negative=improvement
        "history_opp_median": int(hist_opp_med),
        "low_power": low_power,
        "low_power_note": ("历史机会中位 < %d，结果保留但非高置信结论"
                           % LOW_POWER_MED_OPP) if low_power else "",
        "direction": "negative=improvement（log-loss/Brier 均越低越好）",
    }


def _eval_ps(seq: dict, priors: dict, pop_mu: dict, N: int) -> dict:
    """position_sensitivity：RMSE 口径（派生差量）。"""
    rows = []
    for pid, hands in seq.items():
        if len(hands) < N + MIN_FUTURE:
            continue
        hands_sorted = sorted(hands, key=lambda x: x["ts"] or "")
        hist, fut = hands_sorted[:N], hands_sorted[N:]
        assert hist[-1]["ts"] <= fut[0]["ts"]
        s_ep_h, n_ep_h = feature_suff(hist, "rfi_ep")
        s_lp_h, n_lp_h = feature_suff(hist, "rfi_lp")
        s_ep_f, n_ep_f = feature_suff(fut, "rfi_ep")
        s_lp_f, n_lp_f = feature_suff(fut, "rfi_lp")
        if min(n_ep_h, n_lp_h) < PS_MIN_OPP or min(n_ep_f, n_lp_f) < PS_MIN_OPP:
            continue
        ind = (posterior_mean(priors["rfi_lp"], s_lp_h, n_lp_h)
               - posterior_mean(priors["rfi_ep"], s_ep_h, n_ep_h))
        pop = pop_mu["rfi_lp"] - pop_mu["rfi_ep"]
        naive = s_lp_h / n_lp_h - s_ep_h / n_ep_h
        truth = s_lp_f / n_lp_f - s_ep_f / n_ep_f
        rows.append({"err_ind": (ind - truth) ** 2, "err_pop": (pop - truth) ** 2,
                     "err_naive": (naive - truth) ** 2})
    if len(rows) < 30:
        return {"status": "insufficient_players", "n_evaluated": len(rows)}
    rmse_ind = math.sqrt(sum(r["err_ind"] for r in rows) / len(rows))
    rmse_pop = math.sqrt(sum(r["err_pop"] for r in rows) / len(rows))
    rmse_naive = math.sqrt(sum(r["err_naive"] for r in rows) / len(rows))
    lo, hi, _ = bootstrap_pct(
        lambda pairs: (math.sqrt(sum(a for a, _ in pairs) / len(pairs))
                       - math.sqrt(sum(b for _, b in pairs) / len(pairs))),
        [(r["err_ind"], r["err_pop"]) for r in rows],
        n_resample=N_BOOT, seed=BOOT_SEED)
    return {
        "metric": "rmse", "n_evaluated": len(rows),
        "rmse_individual": round(rmse_ind, 6),
        "rmse_population": round(rmse_pop, 6),
        "rmse_naive": round(rmse_naive, 6),
        "diff_rmse_vs_pop": round(rmse_ind - rmse_pop, 6),
        "diff_rmse_vs_pop_ci95": [round(lo, 6), round(hi, 6)],
        "improved_vs_pop": hi < 0,
        "direction": "negative=improvement（RMSE 越低越好）",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", default="data/uoft/feature_ready/player_event_table.jsonl")
    ap.add_argument("--stats", default="data/uoft/stats/full_pool_suff_stats.jsonl")
    ap.add_argument("--splits", default="data/uoft/splits.json")
    ap.add_argument("--hb", default="data/uoft/models/mvp/hb_fit.json")
    ap.add_argument("--out", default="data/uoft/models/phase6/phase65")
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_players, test_players = load_test_players(args.splits)

    # ---- train-only priors（泄漏硬断言）----
    priors = load_hb_fit_priors(args.hb, train_players)
    priors["fold_to_3bet"] = fit_fold_to_3bet_prior(args.stats, train_players,
                                                    test_players)
    pop_mu = {f: priors[f]["mu"] for f in priors}

    # ---- 测试玩家 ring 序列 ----
    seq = load_ring_sequences(args.events, player_filter=test_players)
    assert set(seq) <= test_players, "出现非测试玩家（泄漏）"

    results = {"config": {
        "horizons": list(HORIZONS),
        "min_future": MIN_FUTURE,
        "min_opp_default": MIN_OPP_DEFAULT,
        "cbet_min_opp": CBET_MIN_OPP,
        "ps_min_opp": PS_MIN_OPP,
        "low_power_med_opp": LOW_POWER_MED_OPP,
        "bootstrap": {"unit": "player", "paired": True, "n": N_BOOT,
                      "seed": BOOT_SEED},
        "prior_provenance": "train-only: hb_fit.json(fit_on=task_a_train_players)"
                            " + fold_to_3bet 方案 B(train suff stats)",
        "static_profile_used": False,
        "static_profile_note": "6G static_full_history Profile 禁止作为 prediction "
                               "input；每 horizon 由 history-only 数据临时构造画像",
        "cluster": "disabled: 6D 降级 cluster_probs=null，不得重新启用",
        "arms": ["population_prior(B0)", "naive(B1)", "wilson(B2)",
                 "individual_eb_profile(P)"],
        "main_criterion": "N=200 时 P vs B0 ≥3/6 特征显著改善（CI 上界<0）",
        "direction": "negative=improvement",
        "cbet_n_eff_note": "冻结 5B 派生中 cbet 存在 S>n 行（aggressor 下注后弃牌，"
                           "opportunity=0 但 cbet=1）；比率与 Wilson 需 S<=n，"
                           "以 n_eff=max(S,n) 修正（S 事件必然有对应机会，"
                           "不放大成功计数，不改冻结派生）",
    }, "per_horizon": {}}

    for N in HORIZONS:
        per_feature = {}
        for probe in LL_PROBES:
            per_feature[probe] = _eval_ll_probe(seq, priors[probe],
                                                pop_mu[probe], N, probe)
        if N == 50:
            per_feature["position_sensitivity"] = {
                "status": "insufficient_evidence",
                "reason": "N=50 分量 rfi_ep/rfi_lp 历史机会 <5，不参与主判定",
            }
        else:
            per_feature["position_sensitivity"] = _eval_ps(seq, priors, pop_mu, N)
        results["per_horizon"][str(N)] = per_feature

    # ---- 主判定：N=200，P vs B0 ≥3/6 显著 ----
    n200 = results["per_horizon"]["200"]
    improved = [f for f in LL_PROBES
                if n200[f].get("improved_vs_pop") is True]
    ps200 = n200.get("position_sensitivity", {})
    if ps200.get("improved_vs_pop") is True:
        improved.append("position_sensitivity")
    utility_proven = len(improved) >= 3
    results["main_judgment"] = {
        "horizon": 200,
        "improved_features": improved,
        "improved_count": len(improved),
        "threshold": ">=3/6",
        "utility_proven": utility_proven,
        "conclusion": ("Profile 相对无 Profile baseline 提供增量预测价值："
                       "%d/6 特征显著改善" % len(improved) if utility_proven else
                       "未达阈值：仅 %d/6 特征显著改善（如实报告，不调阈值）"
                       % len(improved)),
    }

    (out_dir / "benchmark.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    if not utility_proven:
        entries = [{"component": "phase65_benchmark", "path": "utility_unproven",
                    "reason": f"仅 {len(improved)}/6 特征在 N=200 显著改善",
                    "evidence": {"improved_features": improved}}]
        (out_dir / "DEGRADE_LOG.json").write_text(
            json.dumps({"entries": entries}, ensure_ascii=False, indent=2),
            encoding="utf-8")

    print(json.dumps(results["main_judgment"], ensure_ascii=False))
    for N in HORIZONS:
        d = results["per_horizon"][str(N)]
        line = {f: (bool(v.get("improved_vs_pop")), v.get("low_power", False))
                for f, v in d.items()}
        print(f"N={N}: {json.dumps(line, ensure_ascii=False)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
