"""Phase 6 Step 6D.2：GMM 门控（BIC k=1..8 全记录 + 4-seed ARI 稳定性）。

输入: T1 玩家 z-score 特征矩阵（与 PCA 相同口径，13 维 Tier A+B，缺失=列均值）
流程:
  1. 对 k=1..8 拟合 GMM（全协方差 + 对角线正则 1e-6，kmeans++ 初始化，
     固定 seed 多次重启取最优 log-lik），记录全部 k 的 BIC
  2. 多 seed {42,137,256,512} 重复；取每 seed 的 BIC 最优 k
  3. 各 seed 最优 k 的硬分配（仅 ARI 计算用）两两 ARI
门控判定:
  Gate-BIC: 最优 k == 1 → 降级（数据不支持离散簇）
  Gate-ARI: min 两两 ARI < 0.5 → 降级（簇不稳定）
输出: data/uoft/models/phase6/gmm_gates.json
降级: cluster_probs=null + DEGRADE_LOG.json（由 pipeline 写入）
硬标签禁止: 本脚本输出的分配仅用于 ARI 计算；对外只允许 cluster_probs（软）。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from scripts.phase6d_pca import PCA_FEATURES, MIN_FEATURES_PRESENT  # noqa: E402

SEEDS = (42, 137, 256, 512)
K_MAX = 8
REG = 1e-6


def build_matrix(profiles_path: str, rules_path: str) -> tuple[np.ndarray, list]:
    rules = json.load(open(rules_path, encoding="utf-8"))
    feat_tier = {k: v["tier"] for k, v in rules["features"].items()}
    allowed = [f for f in PCA_FEATURES if feat_tier.get(f) in ("A", "B")]
    rows = []
    for line in open(profiles_path, encoding="utf-8"):
        p = json.loads(line)
        if p["tier"] != "T1":
            continue
        vec, present = [], 0
        for f in allowed:
            feat = p["features"].get(f)
            if feat and feat["reliability"] == "reliable":
                vec.append(feat["value"]); present += 1
            else:
                vec.append(np.nan)
        if present >= MIN_FEATURES_PRESENT:
            rows.append(vec)
    X = np.array(rows, dtype=float)
    mu = np.zeros(X.shape[1])
    sd = np.ones(X.shape[1])
    for j in range(X.shape[1]):
        col = X[:, j]
        finite = ~np.isnan(col)
        mu[j] = col[finite].mean()
        sd[j] = col[finite].std(ddof=0)
        if sd[j] < 1e-12:
            sd[j] = 1.0
    Z = (X - mu) / sd
    return np.where(np.isnan(Z), 0.0, Z), allowed


def _log_gaussian(X: np.ndarray, mean: np.ndarray, cov: np.ndarray,
                  chol: np.ndarray) -> np.ndarray:
    d = X.shape[1]
    diff = X - mean
    sol = np.linalg.solve(chol, diff.T).T            # (n, d)
    quad = np.einsum("ij,ij->i", sol, sol)
    logdet = 2.0 * np.sum(np.log(np.diag(chol)))
    return -0.5 * (d * np.log(2 * np.pi) + logdet + quad)


def _gmm_fit(X: np.ndarray, k: int, seed: int, n_restarts: int = 3,
             max_iter: int = 200, tol: float = 1e-6) -> dict:
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(n_restarts):
        # kmeans++ 初始化
        centers = [X[int(rng.integers(len(X)))]]
        for _ in range(k - 1):
            d2 = np.min([((X - c) ** 2).sum(axis=1) for c in centers], axis=0)
            d2 = d2 / max(d2.sum(), 1e-300)
            centers.append(X[int(rng.choice(len(X), p=d2))])
        means = np.array(centers)
        covs = np.array([np.cov(X, rowvar=False) + REG * np.eye(X.shape[1])
                         for _ in range(k)])
        weights = np.full(k, 1.0 / k)
        loglik = -np.inf
        for _ in range(max_iter):
            chols = [np.linalg.cholesky(c) for c in covs]
            ll = np.stack([np.log(w) + _log_gaussian(X, means[j], covs[j], chols[j])
                           for j, w in enumerate(weights)], axis=1)
            maxll = ll.max(axis=1, keepdims=True)
            lse = maxll[:, 0] + np.log(np.exp(ll - maxll).sum(axis=1))
            resp = np.exp(ll - maxll)
            resp = resp / resp.sum(axis=1, keepdims=True)
            nk = resp.sum(axis=0) + 1e-300
            new_means = resp.T @ X / nk[:, None]
            new_covs = []
            for j in range(k):
                diff = X - new_means[j]
                cov = (resp[:, j, None] * diff).T @ diff / nk[j]
                cov = cov + REG * np.eye(X.shape[1])
                new_covs.append(cov)
            new_weights = nk / nk.sum()
            new_loglik = lse.sum()
            means, covs, weights = new_means, np.array(new_covs), new_weights
            if abs(new_loglik - loglik) < tol * abs(new_loglik):
                loglik = new_loglik
                break
            loglik = new_loglik
        if best is None or loglik > best["loglik"]:
            best = {"means": means, "covs": covs, "weights": weights,
                    "loglik": loglik}
    return best


def _bic(loglik: float, n: int, d: int, k: int) -> float:
    n_params = k * (d + d * (d + 1) / 2) + (k - 1)
    return -2.0 * loglik + n_params * np.log(n)


def _hard_assign(X: np.ndarray, fit: dict) -> np.ndarray:
    chols = [np.linalg.cholesky(c) for c in fit["covs"]]
    ll = np.stack([np.log(w) + _log_gaussian(X, fit["means"][j],
                                             fit["covs"][j], chols[j])
                   for j, w in enumerate(fit["weights"])], axis=1)
    return ll.argmax(axis=1)


def _ari(a: np.ndarray, b: np.ndarray) -> float:
    """Adjusted Rand Index（纯 numpy 混淆矩阵实现）。"""
    n = len(a)
    uniq_a, uniq_b = np.unique(a), np.unique(b)
    cont = np.zeros((len(uniq_a), len(uniq_b)))
    for i, ua in enumerate(uniq_a):
        for j, ub in enumerate(uniq_b):
            cont[i, j] = np.sum((a == ua) & (b == ub))
    sum_a = cont.sum(axis=1)
    sum_b = cont.sum(axis=0)
    n_choose_2 = n * (n - 1) / 2.0
    sum_comb_a = (sum_a * (sum_a - 1) / 2.0).sum()
    sum_comb_b = (sum_b * (sum_b - 1) / 2.0).sum()
    sum_comb_ab = (cont * (cont - 1) / 2.0).sum()
    expected = sum_comb_a * sum_comb_b / n_choose_2 if n_choose_2 > 0 else 0.0
    max_index = 0.5 * (sum_comb_a + sum_comb_b)
    if max_index == 0:
        return 1.0
    return (sum_comb_ab - expected) / (max_index - expected)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profiles", default="data/uoft/models/phase6/profiles.jsonl")
    ap.add_argument("--rules", default="data/uoft/models/phase6/tier_reliability_rules.json")
    ap.add_argument("--out", default="data/uoft/models/phase6")
    args = ap.parse_args()

    X, features = build_matrix(args.profiles, args.rules)
    n, d = X.shape

    # 每 seed: k=1..8 BIC 全记录 + 最优 k
    per_seed = {}
    for seed in SEEDS:
        rng_state = {"seed": seed}
        bics = {}
        fits = {}
        for k in range(1, K_MAX + 1):
            fit = _gmm_fit(X, k, seed + 1000 * k)   # 每 k 独立子 seed
            bics[k] = round(_bic(fit["loglik"], n, d, k), 2)
            fits[k] = fit
        best_k = min(bics, key=bics.get)
        per_seed[seed] = {"bic_by_k": bics, "best_k": best_k,
                          "fits": fits, "rng_state": rng_state}
        print(f"  seed {seed}: best_k={best_k} bics={bics}", flush=True)

    # ARI：各 seed 最优 k 的硬分配（仅稳定性计算）
    assigns = {s: _hard_assign(X, per_seed[s]["fits"][per_seed[s]["best_k"]])
               for s in SEEDS}
    pairs = []
    ari_values = []
    for i in range(len(SEEDS)):
        for j in range(i + 1, len(SEEDS)):
            ari = _ari(assigns[SEEDS[i]], assigns[SEEDS[j]])
            ari_values.append(ari)
            pairs.append({"seed_a": SEEDS[i], "seed_b": SEEDS[j],
                          "ari": round(float(ari), 4)})

    gate_bic = bool(all(per_seed[s]["best_k"] > 1 for s in SEEDS))
    gate_ari = bool(min(ari_values) >= 0.5)
    out = {
        "config": {"k_max": K_MAX, "seeds": list(SEEDS), "reg": REG,
                   "n_players": int(n), "d": int(d), "features": features,
                   "note": "硬分配仅用于 ARI；对外只允许 soft cluster_probs"},
        "per_seed": {str(s): {"bic_by_k": per_seed[s]["bic_by_k"],
                              "best_k": per_seed[s]["best_k"]}
                     for s in SEEDS},
        "ari_pairs": pairs,
        "ari_min": round(float(min(ari_values)), 4),
        "gate_bic": gate_bic,
        "gate_ari": gate_ari,
        "conclusion": ("三门待续" if (gate_bic and gate_ari)
                       else "降级：%s" % (
                           "BIC 最优 k=1" if not gate_bic else
                           "ARI < 0.5")),
    }
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "gmm_gates.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"gate_bic": gate_bic, "gate_ari": gate_ari,
                      "ari_min": out["ari_min"],
                      "best_k_by_seed": {str(s): per_seed[s]["best_k"]
                                         for s in SEEDS}},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
