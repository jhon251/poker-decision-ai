"""P22-C — 双D NCC v2：座锚点附近 ±350px 有界搜索 + stride 2。"""
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
OUT = Path("data/vision2/coinpoker_assets_v1876dealer")


def _load_tpl(name):
    from poker_decision_ai.vision2_platform.png_decode import decode_png
    w, h, rgba = decode_png((OUT / name).read_bytes())
    arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
    return (arr[..., 0] > 128).astype(np.float64)


def _ncc_bounded(g, t, cx, cy, radius=350):
    """在 (cx,cy) ±radius 搜索（stride 2）。返回 (score, (x,y))。"""
    th, tw = t.shape
    x0, y0 = max(0, cx - radius), max(0, cy - radius)
    x1, y1 = min(g.shape[1], cx + radius), min(g.shape[0], cy + radius)
    sub = g[y0:y1, x0:x1]
    if sub.shape[0] < th or sub.shape[1] < tw:
        return -1.0, None
    wins = np.lib.stride_tricks.sliding_window_view(sub, (th, tw))[::2, ::2]
    wm = wins.mean(axis=(2, 3), keepdims=True)
    wc = wins - wm
    wn = np.sqrt((wc * wc).sum(axis=(2, 3)))
    t0 = t - t.mean()
    tn = float(np.sqrt((t0 * t0).sum()))
    scores = np.abs((wc * t0).sum(axis=(2, 3)) / np.maximum(wn * tn, 1e-9))
    idx = np.unravel_index(np.argmax(scores), scores.shape)
    return float(scores[idx]), (x0 + int(idx[1]) * 2, y0 + int(idx[0]) * 2)


def main() -> int:
    dw = _load_tpl("d_white.png")
    dd = _load_tpl("d_dark.png")
    ok = 0
    total = 0
    for ann in GT["annotations"]:
        pos = (ann.get("fields") or {}).get("position") or {}
        dealer = pos.get("dealer")
        if not isinstance(dealer, dict):
            continue
        gt_seat = dealer.get("seat")
        total += 1
        fid = ann["frame_id"]
        w, h, rgba = decode_png((EV / fid).read_bytes())
        arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
        r = arr[..., 0].astype(np.int16)
        g = arr[..., 1].astype(np.int16)
        b = arr[..., 2].astype(np.int16)
        l = r * 0.299 + g * 0.587 + b * 0.114
        white = (l > 150).astype(np.float64)
        redm = ((r > 40) & (g < 60) & (b < 60) & (r > g * 2)).astype(np.uint8)
        dark = ((l < 120) & (~(redm > 0))).astype(np.float64)
        # 对每个座锚点搜双D，取全帧最佳
        best = None
        for seat, (ax, ay) in SEAT_ANCHORS_1848.items():
            sc_w, pw = _ncc_bounded(white, dw, ax, ay)
            sc_d, pd = _ncc_bounded(dark, dd, ax, ay)
            for sc, pos_, kind in [(sc_w, pw, "W"), (sc_d, pd, "D")]:
                if pos_ is None:
                    continue
                d = float(((pos_[0] - ax) ** 2 + (pos_[1] - ay) ** 2) ** 0.5)
                if d <= 400 and (best is None or sc > best[0]):
                    best = (sc, pos_, kind, seat)
        pred = best[3] if best else None
        hit = pred == gt_seat
        if hit:
            ok += 1
        print(f"{fid[:30]} GT={gt_seat} best={best} {'HIT' if hit else 'MISS'}")
    print(f"\nOK: {ok}/{total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
