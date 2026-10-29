"""Phase 6 Step 6D.1：PCA Exploratory Analysis（纯探索，不参与任何门控）。

输入: T1 玩家（n>=200）画像；Tier A + Tier B 特征中 reliability=reliable 的
      z-score 标准化矩阵（unreliable 特征排除在列外，玩家按列缺失容忍度纳入）。
输出: data/uoft/models/phase6/pca.json
  - explained_variance（每 PC）
  - cumulative_explained_variance
  - loading_matrix（features × PCs）
  - top5_loadings_per_pc（每 PC 前 5 个 |loading| 特征 + 符号）
  - interpretation（确定性模板生成，不用 LLM）
结论三形态（如实报告）：低维主结构 / 无显著低维结构 / 无语义。
PCA 结果不影响 6D 后续 GMM 门控（按 spec 禁止人为干预流程）。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

PCA_FEATURES = ["vpip", "pfr", "three_bet", "cbet", "position_sensitivity",
                "fold_to_3bet", "fold_to_cbet", "wtsd", "wsd", "agg_freq",
                "cold_call", "rfi_ep", "rfi_lp"]
MIN_FEATURES_PRESENT = 8


def _interpret(loadings: dict, pc_idx: int, var_frac: float) -> str:
    """确定性模板解释（按 loading 符号生成，无 LLM）。"""
    pos = [(f, w) for f, w in loadings.items() if w > 0.05]
    neg = [(f, w) for f, w in loadings.items() if w < -0.05]
    pos.sort(key=lambda x: -x[1])
    neg.sort(key=lambda x: x[1])
    pos_s = "+".join(f"高{f}({w:.2f})" for f, w in pos[:3]) if pos else "无显著正载荷"
    neg_s = "+".join(f"高{f}({w:.2f})" for f, w in neg[:3]) if neg else "无显著负载荷"
    return f"PC{pc_idx}（解释方差 {var_frac:.1%}）：正极={pos_s}；负极={neg_s}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profiles", default="data/uoft/models/phase6/profiles.jsonl")
    ap.add_argument("--rules", default="data/uoft/models/phase6/tier_reliability_rules.json")
    ap.add_argument("--out", default="data/uoft/models/phase6")
    args = ap.parse_args()

    rules = json.load(open(args.rules, encoding="utf-8"))
    feat_tier = {k: v["tier"] for k, v in rules["features"].items()}
    allowed = [f for f in PCA_FEATURES if feat_tier.get(f) in ("A", "B")]

    rows = []
    for line in open(args.profiles, encoding="utf-8"):
        p = json.loads(line)
        if p["tier"] != "T1":
            continue
        vec, present = [], 0
        for f in allowed:
            feat = p["features"].get(f)
            if feat and feat["reliability"] == "reliable":
                vec.append(feat["value"])
                present += 1
            else:
                vec.append(None)
        if present >= MIN_FEATURES_PRESENT:
            rows.append(vec)
    X = np.array(rows, dtype=float)      # NaN 占位
    n_players, d = X.shape

    # 列均值/标准差（用非缺失玩家），缺失值以均值填充
    mu = np.zeros(d)
    sd = np.ones(d)
    for j in range(d):
        col = X[:, j]
        finite = ~np.isnan(col)
        mu[j] = col[finite].mean()
        sd[j] = col[finite].std(ddof=0)
        if sd[j] < 1e-12:
            sd[j] = 1.0
    Z = (X - mu) / sd
    Z = np.where(np.isnan(Z), 0.0, Z)     # 缺失 → 0（列均值）

    # 协方差特征分解（d <= 13，直接 numpy）
    cov = np.cov(Z, rowvar=False, ddof=0)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(-eigvals)
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    total_var = eigvals.sum()
    expl = eigvals / total_var
    cum = np.cumsum(expl)

    pca = {
        "config": {"features": allowed, "n_players": n_players,
                   "min_features_present": MIN_FEATURES_PRESENT,
                   "standardization": "z-score", "missing": "imputed_to_column_mean",
                   "note": "PCA 仅 exploratory，不参与 GMM 门控"},
        "explained_variance": [round(float(v), 6) for v in expl],
        "cumulative_explained_variance": [round(float(v), 6) for v in cum],
        "loading_matrix": {f: [round(float(eigvecs[j, i]), 4)
                               for i in range(min(d, 8))]
                           for j, f in enumerate(allowed)},
        "top5_per_pc": {},
        "interpretation": {},
        "conclusion": "",
    }
    for i in range(min(d, 8)):
        loads = {f: float(eigvecs[j, i]) for j, f in enumerate(allowed)}
        top = sorted(loads.items(), key=lambda kv: -abs(kv[1]))[:5]
        pca["top5_per_pc"][f"PC{i+1}"] = [
            {"feature": f, "loading": round(w, 4)} for f, w in top]
        pca["interpretation"][f"PC{i+1}"] = _interpret(loads, i + 1, float(expl[i]))

    if cum[min(1, d - 1)] > 0.6:
        conclusion = ("存在低维主结构：前 2 个 PC 解释 %.1f%% 方差"
                      % (100 * cum[min(1, d - 1)]))
    elif expl[0] < 0.3:
        conclusion = "无显著低维结构：方差近似均匀分布（无肘部）"
    else:
        conclusion = "主成分无语义或仅中等集中，仅具数值意义"
    pca["conclusion"] = conclusion

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "pca.json").write_text(json.dumps(pca, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
    print(json.dumps({"n_players": n_players, "features": allowed,
                      "cum2": round(float(cum[min(1, d - 1)]), 4),
                      "conclusion": conclusion}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
