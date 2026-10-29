"""P18-A — occupied dim-gold 独立校验（READ-ONLY，与 dealer 分离）。

对 6 个 GT 帧，逐座统计 dim-gold 掩码 (r>120,g>85,r>=g,g>=b-30,l>90) 的金像素，
判断是否能稳定区分 6 座全 occupied（用户 stack GT 隐含 6 座全有人）。
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


def _per_seat_dim_gold(arr, radius=140):
    r = arr[..., 0].astype(np.int16)
    g = arr[..., 1].astype(np.int16)
    b = arr[..., 2].astype(np.int16)
    l = r * 0.299 + g * 0.587 + b * 0.114
    gold = (r > 120) & (g > 85) & (r >= g) & (g >= b - 30) & (l > 90)
    out = {}
    for seat, (ax, ay) in SEAT_ANCHORS_1848.items():
        x0, y0 = max(0, ax - radius), max(0, ay - radius)
        x1 = min(arr.shape[1], ax + radius)
        y1 = min(arr.shape[0], ay + radius)
        out[seat] = int(gold[y0:y1, x0:x1].sum())
    return out


def main() -> int:
    for ann in GT["annotations"]:
        fid = ann["frame_id"]
        w, h, rgba = decode_png((EV / fid).read_bytes())
        arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
        counts = _per_seat_dim_gold(arr)
        n_occupied = sum(1 for v in counts.values() if v > 200)
        print(f"{fid}: dim-gold per-seat={counts} -> 检出occupied座={n_occupied}/6")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
