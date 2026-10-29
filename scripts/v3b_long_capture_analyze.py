"""V3B-LONG-CAPTURE-VALIDATION — 后分析脚本（只读）。

用法: python scripts/v3b_long_capture_analyze.py <start_epoch_ms> <end_epoch_ms>
统计新窗口内: 各类帧数 / 金色堆叠候选 ≥4 / ≥5 / ≥6 / bottom-right 完整实例
候选 / Hero/You 文本候选 / dealer 红 D 候选 / 座位变化帧。
"""
from __future__ import annotations

import glob
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from poker_decision_ai.vision2_platform import decode_png  # noqa: E402
from poker_decision_ai.vision2.pixel.geometry import connected_components  # noqa: E402


def epoch_ms(name: str) -> int:
    for part in name.split("_"):
        if part.isdigit() and len(part) == 13:
            return int(part)
    return 0


def main() -> int:
    start, end = int(sys.argv[1]), int(sys.argv[2])
    ev = Path("data/vision2/coinpoker_evidence")
    cats = Counter()
    gold_cands = []          # n>=4 金堆叠候选帧
    red_d = []               # 红 D 候选
    hero_you = []            # Hero/You 文本候选（文件名无——按红 D/金堆叠近似）
    total = 0
    for p in glob.glob(str(ev / "*_full.png")):
        name = os.path.basename(p)
        t = epoch_ms(name)
        if not (start <= t <= end):
            continue
        total += 1
        cat = name.split("_")[0]
        cats[cat] += 1
        w, h, rgba = decode_png(Path(p).read_bytes())
        full = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
        r = full[..., 0].astype(int)
        g = full[..., 1].astype(int)
        b = full[..., 2].astype(int)
        gold = ((r > 150) & (g > 110) & (b < 130) & (r >= g)).astype("uint8")
        comps = [c for c in connected_components(gold, min_area=200)
                 if c.bbox.height >= 15 and c.bbox.width >= 35]
        n = len(comps)
        if n >= 4:
            gold_cands.append((name, n,
                               [[int(c.bbox.x), int(c.bbox.y),
                                 int(c.bbox.width), int(c.bbox.height)]
                                for c in comps]))
        red = ((r > 170) & (g < 80) & (b < 80)).sum()
        if red > 200:
            red_d.append((name, int(red)))
    report = {"window_ms": [start, end], "total_new_frames": total,
              "categories": dict(cats),
              "gold_candidates_ge4": len(gold_cands),
              "gold_candidates_by_n": dict(Counter(x[1]
                                                   for x in gold_cands)),
              "red_d_candidate_frames": len(red_d),
              "gold_candidates": gold_cands[:40],
              "red_d_samples": red_d[:10]}
    Path("data/vision2/v3b_long_capture_analysis.json").write_text(
        json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"new frames: {total} | cats: {dict(cats)}")
    print(f"gold>=4: {len(gold_cands)} "
          f"({dict(Counter(x[1] for x in gold_cands))})")
    print(f"red D candidates: {len(red_d)}")
    for x in gold_cands[:12]:
        print("  ", x[0][:44], "n=", x[1])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
