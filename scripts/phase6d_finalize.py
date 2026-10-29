"""Phase 6 Step 6D.4：门控汇总与降级记录（DEGRADE_LOG + cluster_probs 决策）。

三门判定:
  Gate-BIC:     最优 k > 1（数据支持离散结构）
  Gate-ARI:     min 两两 ARI >= 0.5（簇跨 seed 稳定）
  Gate-Utility: cluster mixture 在冻结 Task B 口径下显著优于单先验 HB
                （玩家级 bootstrap 95% CI 上界 < 0）
三门全过 → 输出 cluster_probs.jsonl（软分配）+ DPMM 条件评估
任一门失败 → cluster_probs = null + DEGRADE_LOG.json 记录（如实报告，不硬撑）

输出: data/uoft/models/phase6/cluster_decision.json
      data/uoft/models/phase6/DEGRADE_LOG.json（失败时）
      data/uoft/models/phase6/cluster_probs.jsonl（仅三门全过时）
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gates", default="data/uoft/models/phase6/gmm_gates.json")
    ap.add_argument("--utility", default="data/uoft/models/phase6/cluster_utility.json")
    ap.add_argument("--out", default="data/uoft/models/phase6")
    args = ap.parse_args()

    gates = json.load(open(args.gates, encoding="utf-8"))
    utility = json.load(open(args.utility, encoding="utf-8"))

    gate_bic = gates["gate_bic"]
    gate_ari = gates["gate_ari"]
    gate_utility = utility["gate_utility"]
    all_pass = gate_bic and gate_ari and gate_utility

    # 降级记录（失败路径）
    degradations = []
    if not gate_bic:
        degradations.append({
            "component": "gmm_cluster", "path": "gate_bic",
            "reason": "BIC 最优 k=1（数据不支持离散簇）",
            "evidence": gates["per_seed"],
        })
    if not gate_ari:
        degradations.append({
            "component": "gmm_cluster", "path": "gate_ari",
            "reason": "ARI < 0.5（簇跨 seed 不稳定）",
            "evidence": {"ari_pairs": gates["ari_pairs"],
                         "ari_min": gates["ari_min"]},
        })
    if not gate_utility:
        degradations.append({
            "component": "gmm_cluster", "path": "gate_utility",
            "reason": "cluster mixture 未在冻结 Task B 口径下显著优于单先验 HB "
                     "（差异 bootstrap 95% CI 未全部 < 0）",
            "evidence": utility["per_feature"],
        })

    decision = {
        "gate_bic": gate_bic, "gate_ari": gate_ari,
        "gate_utility": gate_utility,
        "all_pass": all_pass,
        "cluster_probs": "emitted" if all_pass else None,
        "dpmm": "not_started",
        "conclusion": ("GMM 三门通过：输出 cluster_probs（软分配）；DPMM 评估启动条件成立"
                       if all_pass else
                       "降级：cluster_probs=null。玩家行为建模采用纯连续画像"
                       "（单先验 Beta-Binomial HB）。原因与证据见 DEGRADE_LOG.json。"),
    }

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "cluster_decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")

    if not all_pass:
        log = {"entries": degradations}
        (out_dir / "DEGRADE_LOG.json").write_text(
            json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
        # 明确 cluster_probs=null（删除旧文件如有）
        cp = out_dir / "cluster_probs.jsonl"
        if cp.exists():
            cp.unlink()
    print(json.dumps(decision, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
