"""P18-B — Occupied 独立分析（READ-ONLY）：dim-gold 颜色分布 + per-seat stack 文本真实位置。"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2.pixel.realclient.seat_position import (  # noqa: E402
    SEAT_ANCHORS_1848)
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402

EV = Path("data/vision2/coinpoker_evidence")
SEATS = ("S1", "S2", "S3", "S4", "S5", "S6")


def main() -> int:
    name = "BUTTON_CHANGED_167_1787992682749_full.png"
    w, h, rgba = decode_png((EV / name).read_bytes())
    arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
    r = arr[..., 0].astype(np.int16)
    g = arr[..., 1].astype(np.int16)
    b = arr[..., 2].astype(np.int16)
    l = r * 0.299 + g * 0.587 + b * 0.114
    gold = (r > 120) & (g > 85) & (r >= g) & (g >= b - 30) & (l > 90)
    # dim-gold 像素颜色分布（全帧）
    gpx = np.where(gold)
    if len(gpx[0]):
        rr = r[gpx]; gg = g[gpx]; bb = b[gpx]
        print(f"dim-gold 像素数={len(gpx[0])}")
        print(f"  R: mean={rr.mean():.0f} min={rr.min()} max={rr.max()}")
        print(f"  G: mean={gg.mean():.0f} min={gg.min()} max={gg.max()}")
        print(f"  B: mean={bb.mean():.0f} min={bb.min()} max={bb.max()}")
    # per-seat gold 文本 y 范围（每座在全帧找最近 gold blob）
    print(f"\n{name} per-seat gold 文本位置:")
    for seat, (ax, ay) in SEAT_ANCHORS_1848.items():
        # 全帧 gold 连通域（列投影分段 + 行投影）
        # 简化：找离锚点最近的 gold 像素簇
        # 先看锚点附近 ±500px 的 gold bbox
        x0, y0 = max(0, ax - 500), max(0, ay - 500)
        x1 = min(w, ax + 500); y1 = min(h, ay + 500)
        sub = gold[y0:y1, x0:x1]
        ys, xs = np.where(sub)
        if len(ys) < 50:
            print(f"  {seat}({ax},{ay}): 附近无 gold")
            continue
        # 最近锚点的 gold 簇（取距锚点最近的一个 blob）
        # 粗：整体 bbox
        print(f"  {seat}({ax},{ay}): gold bbox=({x0+xs.min()},{y0+ys.min()},"
              f"{xs.max()-xs.min()+1},{ys.max()-ys.min()+1}) "
              f"n={len(ys)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
