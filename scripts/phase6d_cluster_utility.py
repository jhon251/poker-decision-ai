"""Phase 6 Step 6D.3：Cluster Predictive Utility 门控（仅 GMM BIC/ARI 通过后运行）。

严格 as_of 防泄漏：
  - GMM 与簇先验只在「非测试玩家」上拟合（train 集 = T1 − task_a.test_players）
  - 测试玩家 cluster_probs 只用其 history 半特征（z-score 用 train 均值/标准差）
  - 预测 future 半逐手，与单先验 HB（6B global 先验）比较 log-loss
  - 玩家级 bootstrap 95% CI 报告差异；CI 上界 < 0 → 簇提供显著增益

簇先验拟合：软加权经验贝叶斯矩匹配（cluster_probs 软分配，不输出硬标签）。
输出: data/uoft/models/phase6/cluster_utility.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from phase5b_common import load_player_sequences, load_splits, probe_rates  # noqa: E402
from phase5b_math import bootstrap_pct, mean_log_loss  # noqa: E402
from scripts.phase6b_tiers import reliability_rule  # noqa: E402
from scripts.phase6d_gmm import (_gmm_fit, _hard_assign, _log_gaussian,  # noqa: E402
                                 build_matrix)
from scripts.phase6d_pca import PCA_FEATURES, MIN_FEATURES_PRESENT  # noqa: E402

UTILITY_FEATURES = ["vpip", "pfr", "three_bet", "cbet", "rfi_ep", "rfi_lp"]
# 注: fold_to_3bet 不在 event 元组中（5B 冻结），utility 门用 6 个可计算特征
BEST_K = 8
GMM_SEED = 42


def _fit_cluster_priors(probs: np.ndarray, rates: np.ndarray,
                        counts: np.ndarray) -> list:
    """软加权 EB 矩匹配簇先验。probs: (n_players, K)；rates=S/n；counts=n。"""
    K = probs.shape[1]
    priors = []
    for c in range(K):
        w = probs[:, c]
        wsum = w.sum()
        if wsum < 1e-9:
            priors.append({"a": 1.0, "b": 1.0, "mu": 0.5, "wsum": 0.0})
            continue
        mu_c = float((w * rates).sum() / wsum)
        # 软加权玩家间方差（加权 ANOVA 风格）
        var_between = float((w * (rates - mu_c) ** 2).sum() / wsum)
        # 加权组内抽样方差
        var_within = float((w * rates * (1 - rates) / np.maximum(counts, 1.0)).sum() / wsum)
        s2 = max(0.0, var_between - var_within)
        varmax = mu_c * (1 - mu_c)
        if s2 <= 0:
            nu = 1.0
            prot = "sigma2_nonpositive"
        elif varmax / s2 - 1.0 < 0:
            nu = 1.0
            prot = "nu_negative"
        else:
            nu = min(varmax / s2 - 1.0, 100000.0)
            prot = None
        priors.append({"a": mu_c * nu, "b": (1 - mu_c) * nu, "mu": mu_c,
                       "nu": nu, "wsum": float(wsum),
                       "protection": prot})
    return priors


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profiles", default="data/uoft/models/phase6/profiles.jsonl")
    ap.add_argument("--rules", default="data/uoft/models/phase6/tier_reliability_rules.json")
    ap.add_argument("--stats", default="data/uoft/stats/full_pool_suff_stats.jsonl")
    ap.add_argument("--events", default="data/uoft/feature_ready/player_event_table.jsonl")
    ap.add_argument("--splits", default="data/uoft/splits.json")
    ap.add_argument("--priors", default="data/uoft/models/phase6/population_priors.json")
    ap.add_argument("--out", default="data/uoft/models/phase6")
    args = ap.parse_args()

    splits = load_splits(args.splits)
    test_players = set(splits["task_a"]["test_players"])
    rules = json.load(open(args.rules, encoding="utf-8"))
    feat_tier = {k: v["tier"] for k, v in rules["features"].items()}
    allowed = [f for f in PCA_FEATURES if feat_tier.get(f) in ("A", "B")]
    pop_priors = json.load(open(args.priors, encoding="utf-8"))

    # ---- train 玩家（T1 且非测试）的 z-score 矩阵 + 原值 ----
    train_rows = []
    train_ids = []
    train_meta = {}
    for line in open(args.profiles, encoding="utf-8"):
        p = json.loads(line)
        if p["tier"] != "T1" or p["player_id"] in test_players:
            continue
        vec, present = [], 0
        for f in allowed:
            feat = p["features"].get(f)
            if feat and feat["reliability"] == "reliable":
                vec.append(feat["value"]); present += 1
            else:
                vec.append(np.nan)
        if present >= MIN_FEATURES_PRESENT:
            train_rows.append(vec)
            train_ids.append(p["player_id"])
    X = np.array(train_rows, dtype=float)
    mu = np.zeros(X.shape[1])
    sd = np.ones(X.shape[1])
    for j in range(X.shape[1]):
        col = X[:, j]
        finite = ~np.isnan(col)
        mu[j] = col[finite].mean()
        sd[j] = col[finite].std(ddof=0)
        if sd[j] < 1e-12:
            sd[j] = 1.0
    Ztrain = (X - mu) / sd
    Ztrain = np.where(np.isnan(Ztrain), 0.0, Ztrain)

    # ---- 拟合 GMM（train 玩家，k=BEST_K，seed 固定）----
    fit = _gmm_fit(Ztrain, BEST_K, GMM_SEED, n_restarts=3)
    # 每玩家软 cluster_probs（train 与 test 都可用同一 GMM 计算）
    def soft_probs(Z):
        chols = [np.linalg.cholesky(c) for c in fit["covs"]]
        ll = np.stack([np.log(w) + _log_gaussian(Z, fit["means"][j],
                                                 fit["covs"][j], chols[j])
                       for j, w in enumerate(fit["weights"])], axis=1)
        ll = ll - ll.max(axis=1, keepdims=True)
        p = np.exp(ll)
        return p / p.sum(axis=1, keepdims=True)

    train_probs = soft_probs(Ztrain)
    # 硬分配仅用于簇先验拟合定位（不输出）
    train_hard = train_probs.argmax(axis=1)

    # ---- 每特征簇先验（train 玩家 suff stats + 软概率）----
    stats = {}
    for line in open(args.stats, encoding="utf-8"):
        r = json.loads(line)
        if r["player_id"] in train_ids:
            stats[r["player_id"]] = r
    train_idx = {pid: i for i, pid in enumerate(train_ids)}

    cluster_priors = {}
    for f in UTILITY_FEATURES:
        rates = np.array([stats[pid]["features"][f][0]
                          / max(1, stats[pid]["features"][f][1])
                          for pid in train_ids])
        counts = np.array([stats[pid]["features"][f][1] for pid in train_ids])
        cluster_priors[f] = _fit_cluster_priors(train_probs, rates, counts)

    # ---- 测试玩家：history 特征 → 软 cluster_probs → mixture 后验 → future 预测 ----
    seq = load_player_sequences(args.events, player_filter=test_players)
    per_player = []
    for pid in sorted(seq):
        hands = seq[pid]
        n = len(hands)
        half = n // 2
        if half < 20:
            continue
        hist, fut = hands[:half], hands[half:]
        # history 特征值（13 维，缺失=列均值）
        # 注: event 元组仅含 vpip/pfr/three_bet/cbet/rfi_ep/rfi_lp 可计算字段；
        #     其余 7 维（fold_to_*、wtsd/wsd、agg_freq、cold_call）在历史口径
        #     不可计算 → NaN → 列均值填充（z=0），如实记录于 config
        hvec = []
        for f in allowed:
            if f == "position_sensitivity":
                se, ne = probe_rates(hist, "rfi_ep")
                sl, nl = probe_rates(hist, "rfi_lp")
                if ne >= 15 and nl >= 15:
                    hvec.append(sl / nl - se / ne)
                else:
                    hvec.append(np.nan)
                continue
            if f not in ("vpip", "pfr", "three_bet", "cbet", "rfi_ep", "rfi_lp"):
                hvec.append(np.nan)      # 事件元组不可计算 → 列均值
                continue
            s, nn = probe_rates(hist, f)
            if nn >= 15:
                hvec.append(s / nn)
            else:
                hvec.append(np.nan)
        hz = (np.array(hvec, dtype=float) - mu) / sd
        hz = np.where(np.isnan(hz), 0.0, hz)
        probs = soft_probs(hz.reshape(1, -1))[0]
        rec = {"pid": pid, "cluster_probs": [round(float(x), 6) for x in probs]}
        for f in UTILITY_FEATURES:
            s_h, n_h = probe_rates(hist, f)
            s_f, n_f = probe_rates(fut, f)
            if n_h < 10 or n_f < 10:
                continue
            # mixture 后验均值
            mix_mean = 0.0
            for c in range(BEST_K):
                cp = cluster_priors[f][c]
                a, b = cp["a"], cp["b"]
                mix_mean += float(probs[c]) * ((a + s_h) / (a + b + n_h))
            # 单先验 HB（6B global）
            gp = pop_priors["global"][f]
            single_mean = (gp["a"] + s_h) / (gp["a"] + gp["b"] + n_h)
            rec[f] = {
                "ll_mixture": mean_log_loss(min(max(mix_mean, 1e-12), 1 - 1e-12),
                                            n_f, s_f),
                "ll_single": mean_log_loss(min(max(single_mean, 1e-12), 1 - 1e-12),
                                           n_f, s_f),
                "n_future": n_f,
            }
        per_player.append(rec)

    # 汇总 + 玩家级 bootstrap CI
    out = {"config": {"best_k": BEST_K, "gmm_seed": GMM_SEED,
                      "features": UTILITY_FEATURES,
                      "test_players_evaluated": len(per_player)},
           "per_feature": {}, "cluster_priors": cluster_priors}
    any_better = []
    for f in UTILITY_FEATURES:
        pairs = [(r[f]["ll_mixture"], r[f]["ll_single"]) for r in per_player
                 if f in r]
        if len(pairs) < 30:
            out["per_feature"][f] = {"n": len(pairs), "note": "样本不足"}
            continue
        mix = [a for a, _ in pairs]
        single = [b for _, b in pairs]
        diffs = [a - b for a, b in pairs]
        lo, hi, _ = bootstrap_pct(lambda xs: sum(xs) / len(xs), diffs,
                                  n_resample=1000, seed=42)
        mean_mix = sum(mix) / len(mix)
        mean_single = sum(single) / len(single)
        better = (mean_mix - mean_single) < 0 and hi < 0
        any_better.append(better)
        out["per_feature"][f] = {
            "n": len(pairs),
            "ll_mixture": round(mean_mix, 6),
            "ll_single": round(mean_single, 6),
            "diff": round(mean_mix - mean_single, 6),
            "diff_ci": [round(lo, 6), round(hi, 6)],
            "mixture_significantly_better": better,
        }
    gate_utility = any_better and all(
        out["per_feature"][f].get("mixture_significantly_better", False)
        for f in UTILITY_FEATURES
        if "mixture_significantly_better" in out["per_feature"].get(f, {}))
    out["gate_utility"] = gate_utility
    out["conclusion"] = ("簇提供显著预测增益" if gate_utility
                         else "降级：簇未提供显著预测增益（详见 per_feature）")

    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "cluster_utility.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "cluster_priors"},
                     ensure_ascii=False)[:600])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
