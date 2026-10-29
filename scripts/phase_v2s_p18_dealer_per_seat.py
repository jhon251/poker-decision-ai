"""P18 — Position 1876 红 D per-seat 检测实测（READ-ONLY）。

对 6 个 GT 帧，逐座统计锚点周围红像素（r>150,g<100,b<100），看 max-red 座
是否= GT dealer，判定 per-seat 红 D 检测可行性。
"""
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
GT = json.loads(Path("data/vision2/coinpoker_1876_gt.json").read_text(
    encoding="utf-8"))
SEATS = ("S1", "S2", "S3", "S4", "S5", "S6")


def _per_seat_red(arr, radius=120):
    r = arr[..., 0].astype(np.int16)
    g = arr[..., 1].astype(np.int16)
    b = arr[..., 2].astype(np.int16)
    red = (r > 150) & (g < 100) & (b < 100)
    out = {}
    for seat, (ax, ay) in SEAT_ANCHORS_1848.items():
        x0, y0 = max(0, ax - radius), max(0, ay - radius)
        x1 = min(arr.shape[1], ax + radius)
        y1 = min(arr.shape[0], ay + radius)
        out[seat] = int(red[y0:y1, x0:x1].sum())
    return out


def main() -> int:
    ok = 0
    total = 0
    for ann in GT["annotations"]:
        pos = (ann.get("fields") or {}).get("position") or {}
        gt_dealer = pos.get("dealer")
        if gt_dealer is None:
            continue
        total += 1
        fid = ann["frame_id"]
        w, h, rgba = decode_png((EV / fid).read_bytes())
        arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
        counts = _per_seat_red(arr)
        max_seat = max(counts, key=counts.get)
        hit = max_seat == gt_dealer
        if hit:
            ok += 1
        print(f"{fid}: GT={gt_dealer} per-seat红={counts} max={max_seat} "
              f"{'OK' if hit else 'FAIL'}")
    print(f"\nper-seat 红 D 准确率: {ok}/{total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
