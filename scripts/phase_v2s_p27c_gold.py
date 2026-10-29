"""P27-C — 边界座格精确 gold/name/maroon/panel 测量（P26-B 原语复用）。

目标：确定 OCCUPIED 判据（gold 相对亮度 + name）+ EMPTY 组合判据的阈值，
在 34 占用 / 2 overlay / 1 empty / away / disconnected 上分离。"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402
from poker_decision_ai.vision2.pixel.realclient.coinpoker_stack import (  # noqa: E402
    stack_gold_1876, stack_comps_8conn, SEAT_STACK_ROIS_1876)

EV = Path("data/vision2/coinpoker_evidence")

ANCH = {"S1": (255, 310), "S2": (918, 210), "S3": (1578, 320),
        "S4": (218, 868), "S5": (1618, 868), "S6": (945, 1160)}


def load(fp):
    w, h, rgba = decode_png((EV / fp).read_bytes())
    return np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)


def gold_measure(arr, seat):
    x0, y0, x1, y1 = SEAT_STACK_ROIS_1876[seat]
    roi = arr[y0:y1, x0:x1]
    m, bg = stack_gold_1876(roi)
    comps = stack_comps_8conn(m)
    n = len(comps)
    gold_n = int(m.sum())
    return gold_n, n, bg


def name_measure(arr, seat):
    x0, y0, x1, y1 = SEAT_STACK_ROIS_1876[seat]
    ln = arr[max(0, y0 - 52):y0 - 6, x0:x1]
    ln2 = ln[..., 0] * 0.299 + ln[..., 1] * 0.587 + ln[..., 2] * 0.114
    rn = ln[..., 0].astype(np.int16); gn = ln[..., 1].astype(np.int16)
    bn = ln[..., 2].astype(np.int16)
    return int(((ln2 > 110) & (np.abs(rn - gn) < 28)
                & (np.abs(gn - bn) < 28)).sum())


def maroon_measure(arr, seat):
    ax, ay = ANCH[seat]
    sub = arr[ay - 90:ay + 90, ax - 100:ax + 100]
    sr = sub[..., 0].astype(np.int16); sg = sub[..., 1].astype(np.int16)
    sb = sub[..., 2].astype(np.int16)
    return int(((sr > sg + 20) & (sr > sb + 10) & (sr > 40)
                & (sr < 150)).sum())


# 全部 34 GT 占用 + 2 overlay + 1 empty + 5 空帧占用
rows = []
for ann in json.loads(Path("data/vision2/coinpoker_1876_gt.json")
                      .read_text(encoding="utf-8"))["annotations"]:
    st = (ann.get("fields") or {}).get("stack") or {}
    arr = load(ann["frame_id"])
    for seat in sorted(SEAT_STACK_ROIS_1876):
        gold_n, ncomp, bg = gold_measure(arr, seat)
        nm = name_measure(arr, seat)
        mm = maroon_measure(arr, seat)
        rows.append((ann["frame_id"][:6], seat, "occ", gold_n, ncomp, nm, mm,
                     round(bg, 1)))
# empty 帧
arr = load("42b553c9468d1d1e55b9801bebad7b4f.png")
for seat in sorted(SEAT_STACK_ROIS_1876):
    gold_n, ncomp, bg = gold_measure(arr, seat)
    nm = name_measure(arr, seat)
    mm = maroon_measure(arr, seat)
    state = "empty" if seat == "S4" else "occ"
    rows.append(("42b", seat, state, gold_n, ncomp, nm, mm, round(bg, 1)))

# 统计
occ = [r for r in rows if r[2] == "occ"]
gold_occ = [r[3] for r in occ]
gold_min = min(gold_occ)
gold_min_case = [r for r in occ if r[3] == gold_min][0]
print(f"OCCUPIED gold_n: min={gold_min} (case={gold_min_case[:4]}) "
      f"p10={np.percentile(gold_occ,10):.0f}")
empt = [r for r in rows if r[2] == "empty"]
print(f"EMPTY gold_n: {[r[3] for r in empt]}")
# 空帧 S3/S5 away 的 gold
for r in rows:
    if r[1] in ("S3", "S5") and r[0] == "42b":
        print(f"  42b {r[1]} (away): gold={r[3]} ncomp={r[4]} name={r[5]} "
              f"maroon={r[6]}")
# 全部打印 low-gold 占用（<600）找边界
print("\nlow-gold OCCUPIED (gold<700):")
for r in sorted(rows, key=lambda x: x[3]):
    if r[2] == "occ" and r[3] < 700:
        print(f"  {r[0]} {r[1]}: gold={r[3]} ncomp={r[4]} name={r[5]} "
              f"maroon={r[6]} bg={r[7]}")
