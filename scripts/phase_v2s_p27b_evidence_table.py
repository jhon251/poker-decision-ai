"""P27-B.0 — 用户帧逐座证据表 + 旧 pipeline 行为记录（READ-ONLY 诊断）。

期望：S4=EMPTY（空标签+幽灵圈+无牌盒+无头像）；5 座 OCCUPIED。
旧 SeatOccupancyDetector（1848 锚点）行为如实记录（不修改）。
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402
from poker_decision_ai.vision2.pixel.realclient.seat_position import (  # noqa: E402
    SeatOccupancyDetector)

FP = Path("data/vision2/coinpoker_evidence/42b553c9468d1d1e55b9801bebad7b4f.png")
w, h, rgba = decode_png(FP.read_bytes())
arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)

STACK = {
    "S1": (165, 413, 360, 485), "S2": (845, 298, 1035, 370),
    "S3": (1520, 413, 1715, 485), "S4": (125, 978, 320, 1050),
    "S5": (1560, 978, 1745, 1050), "S6": (838, 1236, 1040, 1308),
}
print("=== 逐座证据表 ===")
print(f"{'seat':4s} {'panel':>6s} {'name':>5s} {'gold':>5s} {'empty字':>4s} "
      f"{'ghost圈':>5s} {'maroon':>6s}  -> 判")
for seat in sorted(STACK):
    x0, y0, x1, y1 = STACK[seat]
    px0, py0 = max(0, x0 - 10), max(0, y0 - 58)
    px1, py1 = min(w, x1 + 10), min(h, y1 + 12)
    roi = arr[py0:py1, px0:px1]
    r = roi[..., 0].astype(np.int16); g = roi[..., 1].astype(np.int16)
    b = roi[..., 2].astype(np.int16)
    l = r * 0.299 + g * 0.587 + b * 0.114
    panel = ((l > 12) & (l < 62) & (np.abs(r - g) < 30)
             & (np.abs(g - b) < 30)).mean()
    # name
    ln = arr[py0:y0 - 6, x0:x1]
    ln2 = ln[..., 0] * 0.299 + ln[..., 1] * 0.587 + ln[..., 2] * 0.114
    rn = ln[..., 0].astype(np.int16); gn = ln[..., 1].astype(np.int16)
    bn = ln[..., 2].astype(np.int16)
    name_ink = int(((ln2 > 110) & (np.abs(rn - gn) < 28)
                    & (np.abs(gn - bn) < 28)).sum())
    # gold
    rs = arr[y0:y1, x0:x1]
    rg = rs[..., 0].astype(np.int16); gg = rs[..., 1].astype(np.int16)
    bg2 = rs[..., 2].astype(np.int16)
    ls = rg * 0.299 + gg * 0.587 + bg2 * 0.114
    mdn = np.median(ls)
    gold = int(((rg > bg2 + 30) & (gg >= bg2 + 18) & (rg >= gg * 0.9)
                & (ls > mdn * 1.6)).sum())
    # 「空的」候选：座上方 x±120, y0-160..y0-60
    ex0, ey0 = max(0, x0 - 120), max(0, y0 - 170)
    ex1, ey1 = min(w, x0 + 120), min(h, y0 - 60)
    e_roi = arr[ey0:ey1, ex0:ex1]
    er = e_roi[..., 0].astype(np.int16); eg = e_roi[..., 1].astype(np.int16)
    eb = e_roi[..., 2].astype(np.int16)
    el = er * 0.299 + eg * 0.587 + eb * 0.114
    empty_gray = int(((el > 90) & (el < 190) & (np.abs(er - eg) < 25)
                      & (np.abs(eg - eb) < 25)).sum())
    # 幽灵圈/头像：锚点区 maroon + 暗环
    ax, ay = {"S1": (255, 310), "S2": (918, 210), "S3": (1578, 320),
              "S4": (218, 868), "S5": (1618, 868), "S6": (945, 1160)}[seat]
    sub = arr[ay - 90:ay + 90, ax - 100:ax + 100]
    sr = sub[..., 0].astype(np.int16); sg = sub[..., 1].astype(np.int16)
    sb = sub[..., 2].astype(np.int16)
    maroon = int(((sr > sg + 20) & (sr > sb + 10) & (sr > 40)
                  & (sr < 150)).sum())
    sl = sr * 0.299 + sg * 0.587 + sb * 0.114
    ring = float(((sl > 20) & (sl < 50)).mean())
    verdict = ("EMPTY" if empty_gray > 40 and name_ink < 50 and gold < 50
               else ("OCCUPIED" if name_ink >= 50 and gold >= 100
                     else "UNKNOWN"))
    print(f"{seat:4s} {panel:6.2f} {name_ink:5d} {gold:5d} {empty_gray:4d} "
          f"{ring:5.2f} {maroon:6d}  -> {verdict}")

# 旧 pipeline 行为
print("\n=== 旧 SeatOccupancyDetector(1848 锚点) 行为（记录，不修改）===")
old = SeatOccupancyDetector(anchors={
    "S1": (255, 310), "S2": (918, 210), "S3": (1578, 320),
    "S4": (218, 868), "S5": (1618, 868), "S6": (945, 1160)},
    window="1848").detect(arr)
for seat in sorted(old):
    v = old[seat]
    print(f"  {seat}: {v['status']} ({v['reason']})")
