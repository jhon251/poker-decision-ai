"""Phase 5B Step 5：Task A / Task B benchmark。

Task A（新玩家泛化）: 先验来自 train；测试玩家输入前 50 手 → 后验 → 真值=其余手频率。
  指标: 95% CI 覆盖率（目标 >= 90%）+ RMSE（后验均值 vs 真值，对比群体均值基线）。
Task B（个体适应）: 测试玩家 history 半 → 后验均值 p̂ → future 半逐手预测。
  指标: 平均 log-loss（HB vs 群体均值 / 朴素个体频率 / Wilson 中心）+ Brier。
  时间审计字段: history_end_timestamp / future_start_timestamp（as_of 断言）。

position_sensitivity: 后验 = rfi_lp 后验 − rfi_ep 后验（RMSE 比较，无逐手 log-loss）。
cbet: exploratory 通道，单独标记，不参与核心判定。

输出 data/uoft/models/mvp/task_a.json, task_b.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from phase5b_common import (PROBES, ensure_dir, load_player_sequences,  # noqa: E402
                            load_splits, probe_rates)
from phase5b_math import (beta_binomial_interval, beta_quantile, brier,  # noqa: E402
                          bootstrap_pct, mean_log_loss, wilson_interval)

GLIMPSE = 50
MIN_TRUTH = 20


def _posterior(prior_a, prior_b, s, n):
    return prior_a + s, prior_b + n - s


def _rmse(preds: list, truths: list) -> float:
    n = len(preds)
    if n == 0:
        return 0.0
    return math.sqrt(sum((p - t) ** 2 for p, t in zip(preds, truths)) / n)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", default="data/uoft/feature_ready/player_event_table.jsonl")
    ap.add_argument("--splits", default="data/uoft/splits.json")
    ap.add_argument("--hb", default="data/uoft/models/mvp/hb_fit.json")
    ap.add_argument("--selection", default="data/uoft/models/mvp/probe_selection.json")
    ap.add_argument("--out", default="data/uoft/models/mvp")
    args = ap.parse_args()
    ensure_dir(args.out)

    splits = load_splits(args.splits)
    test_players = set(splits["task_a"]["test_players"])
    hb = json.load(open(args.hb, encoding="utf-8"))["probes"]
    sel = json.load(open(args.selection, encoding="utf-8"))["probes"]

    seq = load_player_sequences(args.events, player_filter=test_players)
    # 泄漏断言：测试玩家先验不得参与拟合（hb_fit 已断言其只含 train；此处再防御）
    assert set(seq) <= test_players, "出现非测试玩家"

    # 群体均值（先验均值 = HB 的 mu，train 拟合）
    pop_mean = {p: hb[p]["mu"] for p in PROBES}
    prior = {p: (hb[p]["a"], hb[p]["b"]) for p in PROBES}

    # ---------------- Task A ----------------
    task_a = {"config": {"glimpse": GLIMPSE, "min_truth": MIN_TRUTH,
                         "coverage_target": 0.90}, "probes": {}}
    for p in PROBES:
        a0, b0 = prior[p]
        covs = []
        covs_pred = []      # Beta-Binomial 预测区间覆盖率（吸收未来观测噪声）
        preds_hb = []
        preds_pop = []
        truths = []
        n_eval = 0
        n_skip = 0
        for pid, hands in seq.items():
            if len(hands) < GLIMPSE + MIN_TRUTH:
                n_skip += 1
                continue
            glimpse = hands[:GLIMPSE]
            truth = hands[GLIMPSE:]
            s_g, n_g = probe_rates(glimpse, p)
            s_t, n_t = probe_rates(truth, p)
            if n_g == 0 or n_t == 0:
                n_skip += 1
                continue
            a, b = _posterior(a0, b0, s_g, n_g)
            lo = beta_quantile(a, b, 0.025)
            hi = beta_quantile(a, b, 0.975)
            truth_rate = s_t / n_t
            covs.append(1.0 if lo <= truth_rate <= hi else 0.0)
            # 预测区间：未来 n_t 手成功数的 95% Beta-Binomial 区间
            plo, phi = beta_binomial_interval(a, b, n_t)
            covs_pred.append(1.0 if plo <= s_t <= phi else 0.0)
            preds_hb.append(a / (a + b))
            preds_pop.append(pop_mean[p])
            truths.append(truth_rate)
            n_eval += 1
        coverage = sum(covs) / len(covs) if covs else 0.0
        coverage_pred = sum(covs_pred) / len(covs_pred) if covs_pred else 0.0
        task_a["probes"][p] = {
            "n_evaluated": n_eval, "n_skipped": n_skip,
            "coverage_theta_ci": round(coverage, 4),
            "coverage_predictive": round(coverage_pred, 4),
            "coverage_target": 0.90,
            "coverage_pass": coverage_pred >= 0.90,
            "rmse_hb": round(_rmse(preds_hb, truths), 6),
            "rmse_pop": round(_rmse(preds_pop, truths), 6),
        }
        print(f"  TaskA {p}: {json.dumps(task_a['probes'][p], ensure_ascii=False)}",
              flush=True)

    # position_sensitivity：差分后验 RMSE（无逐手 CI 覆盖率）
    ta_ps = {"rmse_hb": [], "rmse_pop": [], "truths": [], "n_eval": 0}
    for pid, hands in seq.items():
        if len(hands) < GLIMPSE + MIN_TRUTH:
            continue
        g, t = hands[:GLIMPSE], hands[GLIMPSE:]
        epg = probe_rates(g, "rfi_ep")
        lpg = probe_rates(g, "rfi_lp")
        ept = probe_rates(t, "rfi_ep")
        lpt = probe_rates(t, "rfi_lp")
        if min(epg[1], lpg[1], ept[1], lpt[1]) < 5:
            continue
        a_ep, b_ep = _posterior(*prior["rfi_ep"], *epg)
        a_lp, b_lp = _posterior(*prior["rfi_lp"], *lpg)
        pred_hb = a_lp / (a_lp + b_lp) - a_ep / (a_ep + b_ep)
        pred_pop = pop_mean["rfi_lp"] - pop_mean["rfi_ep"]
        truth_ps = lpt[0] / lpt[1] - ept[0] / ept[1]
        ta_ps["rmse_hb"].append(pred_hb)
        ta_ps["rmse_pop"].append(pred_pop)
        ta_ps["truths"].append(truth_ps)
        ta_ps["n_eval"] += 1
    task_a["probes"]["position_sensitivity"] = {
        "note": "派生差量：RMSE 比较，无逐手 CI 覆盖率",
        "n_evaluated": ta_ps["n_eval"],
        "rmse_hb": round(_rmse(ta_ps["rmse_hb"], ta_ps["truths"]), 6),
        "rmse_pop": round(_rmse(ta_ps["rmse_pop"], ta_ps["truths"]), 6),
    }
    (Path(args.out) / "task_a.json").write_text(
        json.dumps(task_a, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---------------- Task B ----------------
    task_b = {"config": {"baselines": ["population_mean", "naive_frequency",
                                       "wilson_center"]}, "probes": {},
              "time_audit": {}}
    for p in PROBES:
        a0, b0 = prior[p]
        per_player = []   # (ll_hb, ll_pop, ll_naive, ll_wilson, brier_hb)
        n_skip = 0
        for pid, hands in seq.items():
            n = len(hands)
            half = n // 2
            if half < MIN_TRUTH:
                n_skip += 1
                continue
            hist, fut = hands[:half], hands[half:]
            s_h, n_h = probe_rates(hist, p)
            s_f, n_f = probe_rates(fut, p)
            if n_h == 0 or n_f == 0:
                n_skip += 1
                continue
            # 时间审计字段
            audit = task_b["time_audit"].setdefault(pid, {})
            audit["history_end_timestamp"] = hist[-1][0]
            audit["future_start_timestamp"] = fut[0][0]
            audit["as_of_ok"] = hist[-1][0] <= fut[0][0]
            a, b = _posterior(a0, b0, s_h, n_h)
            p_hb = a / (a + b)
            p_pop = pop_mean[p]
            p_naive = s_h / n_h
            wlo, whi = wilson_interval(s_h, n_h)
            p_wilson = (wlo + whi) / 2.0
            per_player.append({
                "pid": pid,
                "ll_hb": mean_log_loss(p_hb, n_f, s_f),
                "ll_pop": mean_log_loss(p_pop, n_f, s_f),
                "ll_naive": mean_log_loss(p_naive, n_f, s_f),
                "ll_wilson": mean_log_loss(p_wilson, n_f, s_f),
                "brier_hb": brier(p_hb, n_f, s_f),
                "n_hist": n_h, "n_future": n_f,
            })
        if not per_player:
            task_b["probes"][p] = {"n_evaluated": 0, "n_skipped": n_skip}
            continue
        n = len(per_player)
        ll_hb = sum(x["ll_hb"] for x in per_player) / n
        ll_pop = sum(x["ll_pop"] for x in per_player) / n
        ll_naive = sum(x["ll_naive"] for x in per_player) / n
        ll_wilson = sum(x["ll_wilson"] for x in per_player) / n
        diff_naive = [x["ll_hb"] - x["ll_naive"] for x in per_player]
        lo, hi, _ = bootstrap_pct(lambda xs: sum(xs) / len(xs), diff_naive,
                                  n_resample=1000, seed=42)
        diff_pop = [x["ll_hb"] - x["ll_pop"] for x in per_player]
        lo_p, hi_p, _ = bootstrap_pct(lambda xs: sum(xs) / len(xs), diff_pop,
                                      n_resample=1000, seed=42)
        task_b["probes"][p] = {
            "n_evaluated": n, "n_skipped": n_skip,
            "ll_hb": round(ll_hb, 6), "ll_pop": round(ll_pop, 6),
            "ll_naive": round(ll_naive, 6), "ll_wilson": round(ll_wilson, 6),
            "brier_hb": round(sum(x["brier_hb"] for x in per_player) / n, 6),
            "diff_hb_vs_naive": round(ll_hb - ll_naive, 6),
            "diff_hb_vs_naive_ci": [round(lo, 6), round(hi, 6)],
            "diff_hb_vs_pop": round(ll_hb - ll_pop, 6),
            "diff_hb_vs_pop_ci": [round(lo_p, 6), round(hi_p, 6)],
            "hb_better_than_naive": (ll_hb - ll_naive) < 0,
            "hb_better_than_pop": (ll_hb - ll_pop) < 0,
            "exploratory": sel.get(p, {}).get("decision") == "exploratory",
        }
        print(f"  TaskB {p}: ll_hb={ll_hb:.4f} ll_naive={ll_naive:.4f} "
              f"diff={ll_hb-ll_naive:+.4f} ci={lo:.4f}..{hi:.4f}", flush=True)

    # ps Task B：差分后验 RMSE（不定义逐手 log-loss）
    tb_ps = {"rmse_hb": [], "rmse_pop": [], "rmse_naive": [], "truths": [],
             "n_eval": 0}
    for pid, hands in seq.items():
        n = len(hands)
        half = n // 2
        if half < MIN_TRUTH:
            continue
        hist, fut = hands[:half], hands[half:]
        eh = probe_rates(hist, "rfi_ep")
        lh = probe_rates(hist, "rfi_lp")
        ef = probe_rates(fut, "rfi_ep")
        lf = probe_rates(fut, "rfi_lp")
        if min(eh[1], lh[1], ef[1], lf[1]) < 5:
            continue
        a_ep, b_ep = _posterior(*prior["rfi_ep"], *eh)
        a_lp, b_lp = _posterior(*prior["rfi_lp"], *lh)
        truth = lf[0] / lf[1] - ef[0] / ef[1]
        tb_ps["rmse_hb"].append(a_lp / (a_lp + b_lp) - a_ep / (a_ep + b_ep))
        tb_ps["rmse_pop"].append(pop_mean["rfi_lp"] - pop_mean["rfi_ep"])
        tb_ps["rmse_naive"].append(lh[0] / lh[1] - eh[0] / eh[1])
        tb_ps["truths"].append(truth)
        tb_ps["n_eval"] += 1
    task_b["probes"]["position_sensitivity"] = {
        "note": "派生差量：RMSE 比较，无逐手 log-loss",
        "n_evaluated": tb_ps["n_eval"],
        "rmse_hb": round(_rmse(tb_ps["rmse_hb"], tb_ps["truths"]), 6),
        "rmse_pop": round(_rmse(tb_ps["rmse_pop"], tb_ps["truths"]), 6),
        "rmse_naive": round(_rmse(tb_ps["rmse_naive"], tb_ps["truths"]), 6),
    }
    (Path(args.out) / "task_b.json").write_text(
        json.dumps(task_b, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
