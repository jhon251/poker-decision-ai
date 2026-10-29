"""M3 — 旋钮扫描矩阵汇总：9 格合并 + CI + 形态判定 + 推荐表。

先逐臂 merge 分段（merge_bench_segments 逻辑同源），再产出：
  docs/m3_opponent/knob_scan.json —— 矩阵（bb/100 + CI95 + 决策分布）
  + 形态判定（对手越软→最优预算越高？）+ 推荐表（按池强度选预算）。
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "docs" / "bench_r1"
SEGMENTS = ["k00", "k01", "k02"]
ARMS = {
    "frequency_bot": ["m3_knob_freqbot_b005", "m3_knob_freqbot_b015",
                      "m3_knob_freqbot_b030"],
    "pool_fish_nl2": ["m3_knob_fish_b005", "m3_knob_fish_b015",
                      "m3_knob_fish_b030"],
    "pool_reg_nl5": ["m3_knob_reg_b005", "m3_knob_reg_b015",
                     "m3_knob_reg_b030"],
}
BUDGETS = [0.05, 0.15, 0.30]


def merge_arm(matchup: str) -> dict | None:
    from external_harness.protocol.bench_match import aggregate
    records: list[dict] = []
    latencies: list[float] = []
    counters: dict[str, int] = {}
    segs: list[str] = []
    for seg in SEGMENTS:
        rec_p = OUT_DIR / f"{matchup}.{seg}.records.jsonl"
        if not rec_p.is_file():
            continue
        segs.append(seg)
        rows = [json.loads(line) for line in
                rec_p.read_text(encoding="utf-8").splitlines() if line.strip()]
        records += rows
        agg_p = OUT_DIR / f"{matchup}.{seg}.json"
        if agg_p.is_file():
            seg_agg = json.loads(agg_p.read_text(encoding="utf-8"))
            latencies += seg_agg.get("latencies_ms") or []
            for k, v in (seg_agg.get("counters") or {}).items():
                counters[k] = counters.get(k, 0) + v
    if not records:
        return None
    agg = aggregate(records, latencies, counters)
    agg["segments"] = segs
    return agg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/m3_opponent/knob_scan.json")
    args = ap.parse_args()

    matrix: dict[str, dict] = {}
    for opp, arms in ARMS.items():
        for bud, arm in zip(BUDGETS, arms):
            agg = merge_arm(arm)
            if agg is None:
                print(f"[summarize] MISSING {arm}", file=sys.stderr)
                continue
            matrix[f"{opp}@{bud}"] = {
                "opponent": opp, "budget_bb": bud, "matchup": arm,
                "n_hands": agg["n_hands_completed"],
                "bb_per_100": agg["bb_per_100"],
                "ci95": agg["ci95_bb_per_100"],
                "illegal": agg["counters"].get("hero_illegal_action", 0),
                "aborts": agg["counters"].get("hand_aborts", 0),
                "deviation_rate": None,
                "action_share": agg.get("action_distribution_share") or {},
                "segments": agg["segments"],
            }
            print(f"{opp}@{bud}: n={agg['n_hands_completed']} "
                  f"bb/100={agg['bb_per_100']} CI={agg['ci95_bb_per_100']} "
                  f"(segments={agg['segments']})")

    # ---- 形态判定：每对手的最优预算档 + 档间差分 ----
    shape: dict = {"per_opponent_best": {}, "notes": []}
    for opp in ARMS:
        cells = [(bud, matrix[f"{opp}@{bud}"]) for bud in BUDGETS
                 if f"{opp}@{bud}" in matrix]
        if len(cells) < 3:
            shape["notes"].append(f"{opp}: incomplete cells")
            continue
        best = max(cells, key=lambda cb: cb[1]["bb_per_100"])
        soft = cells[0][1]
        soft_best = max(range(len(cells)), key=lambda i: cells[i][1]["bb_per_100"])
        shape["per_opponent_best"][opp] = {
            "best_budget": best[0],
            "best_bb100": best[1]["bb_per_100"],
            "bb100_by_budget": {str(b): c["bb_per_100"] for b, c in cells},
        }
        _ = (soft, soft_best)
    # 期望形态：软对手最优预算 ≥ 强对手最优预算
    ops = [o for o in ARMS if o in shape["per_opponent_best"]]
    if len(ops) == 3:
        soft_best = shape["per_opponent_best"]["pool_fish_nl2"]["best_budget"]
        strong_best = shape["per_opponent_best"]["pool_reg_nl5"]["best_budget"]
        shape["supported"] = bool(soft_best >= strong_best)
        shape["verdict"] = (
            f"软端(fish)最优预算={soft_best}bb vs 强端(reg)最优预算="
            f"{strong_best}bb → 形态{'支持' if shape['supported'] else '不支持'}"
            "「对手越软→最优预算越高」")

    out = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"),
           "design": {"budgets_bb": BUDGETS, "opponents": list(ARMS),
                      "hands_per_cell": 10000,
                      "seed_base": 20260930,
                      "note": "同 deal 种子流（offset 分段）；A 组池 bot 参数"
                              "（认证合格）；budget 只改 "
                              "strategy_core.dev_max_ev_loss_bb"},
           "matrix": matrix, "shape": shape}
    # 推荐表（按扫描结果；config 落地由 build 步骤人工确认后写入）
    rec = {}
    for opp in ops:
        b = shape["per_opponent_best"][opp]["best_budget"]
        rec[opp] = {"recommended_budget_bb": b}
    out["recommendation_preview"] = rec
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1),
                        encoding="utf-8")
    print("verdict:", shape.get("verdict"))
    print("->", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
