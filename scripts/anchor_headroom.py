#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P30-9-DO：量化锚定选择的**可回收 regret 上限**（决定是否值得解冻 judge）。

对 r1 库每个 spot 的每手牌：
  * 参考（逐手 EV 最优）= benchmark 的 `best_row`：该手牌 EV 最高的**行**；
  * 锚定实际会选 = 行中 **聚合频率最高者**（anchor_from_solver：raw.chosen 恒空 → 频率 argmax）；
  * 可回收 regret = max(0, ev_hand(best) − ev_hand(anchor_row))。
输出：占比、均值/最大、按动作类别分布，以及"若锚定改为逐手 EV 最优"的理论收益。
"""
from __future__ import annotations

import collections
import json
import pathlib
import statistics
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = pathlib.Path(r"<repo>")
OUT = ROOT / "docs" / "anchor_headroom.json"

rows_all: list[dict] = []
diff = 0
tot = 0
per_spot = collections.Counter()
for f in sorted((ROOT / "data" / "solver" / "spots_r1").glob("*.json")):
    d = json.loads(f.read_text(encoding="utf-8"))
    hands = d.get("private_cards_actor") or []
    acts = d.get("actions") or []
    if not hands or not acts:
        continue
    n = min(len(hands), *(len(a.get("ev_per_hand_bb") or []) for a in acts))
    if n <= 0:
        continue
    # 锚定会选的行：聚合频率最高的动作名 → 该名下行中频率最高者
    anchor_name = str(max(acts, key=lambda a: float(a.get("frequency") or 0)).get("name"))
    anchor_row = max((a for a in acts if str(a.get("name")) == anchor_name),
                      key=lambda a: float(a.get("frequency") or 0))
    for i in range(n):
        evs = []
        for a in acts:
            arr = a.get("ev_per_hand_bb") or []
            v = arr[i] if i < len(arr) else None
            if isinstance(v, (int, float)):
                evs.append((a, v))
        if not evs:
            continue
        best_a, best_v = max(evs, key=lambda kv: kv[1])
        anc_v = next((v for a, v in evs if a is anchor_row), None)
        if anc_v is None:
            continue
        tot += 1
        rec = max(0.0, best_v - anc_v)
        is_diff = best_a is not anchor_row
        if is_diff:
            diff += 1
            per_spot[d["spot_id"]] += 1
        rows_all.append({"spot": d["spot_id"], "hand": str(hands[i]),
                          "anchor": anchor_name, "best": str(best_a.get("name")),
                          "recoverable_bb": round(rec, 4), "diff": is_diff})

rec = [r["recoverable_bb"] for r in rows_all]
rec_diff = [r["recoverable_bb"] for r in rows_all if r["diff"]]
print(f"手牌-决策总数: {tot}（{len(per_spot)} 个 spot 存在差异手牌）")
print(f"逐手最优行 ≠ 锚定行: {diff}/{tot} = {diff/max(tot,1)*100:.1f}%")
print(f"可回收 regret（全部手牌）: 均值 {statistics.fmean(rec):.4f} bb | "
      f"最大 {max(rec):.3f} | >0 的比例 {sum(1 for x in rec if x > 1e-9)/max(tot,1)*100:.1f}%")
if rec_diff:
    print(f"  仅在差异手牌上: 均值 {statistics.fmean(rec_diff):.3f} bb | 最大 {max(rec_diff):.3f}")
    print(f"  >0.05bb（当前预算）的比例: "
          f"{sum(1 for x in rec_diff if x > 0.05)/max(len(rec_diff),1)*100:.1f}%")
print("\n差异最多的 8 个 spot：")
for s, c in per_spot.most_common(8):
    print(f"  {s:<34} {c} 手牌")

OUT.write_text(json.dumps({
    "stage": "P30-9-DO",
    "total_hand_decisions": tot,
    "anchor_vs_per_hand_best_diff_rate": round(diff / max(tot, 1), 4),
    "recoverable_regret_bb": {"mean": round(statistics.fmean(rec), 4),
                               "max": round(max(rec), 4),
                               "positive_rate": round(sum(1 for x in rec if x > 1e-9)
                                                       / max(tot, 1), 4)},
    "on_diff_hands_only": {"mean": (round(statistics.fmean(rec_diff), 4) if rec_diff else None),
                            "max": (round(max(rec_diff), 4) if rec_diff else None),
                            "gt_budget_rate": (round(sum(1 for x in rec_diff if x > 0.05)
                                                      / max(len(rec_diff), 1), 4)
                                               if rec_diff else None)},
    "spots_with_diff": dict(per_spot.most_common(20)),
    "measured_bench_regret_mean_bb": 0.055,
}, ensure_ascii=False, indent=1), encoding="utf-8")
print("\n已写:", OUT)
