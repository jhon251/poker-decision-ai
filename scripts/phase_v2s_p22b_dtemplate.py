"""P22-B — 白 D 字母模板：从帧09按钮提取 → NCC 全帧搜索 → 关联座位。"""
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


def _load():
    """帧09 按钮 (766,982) 提取白 D（红环绕白字）。"""
    w, h, rgba = decode_png((EV /
        "POT_CHANGED_139_1787992592538_full.png").read_bytes())
    arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
    # 全帧找 (766,982) 附近的白 D：白色>180 且在红色包围中
    roi = arr[950:1020, 730:810]
    r = roi[..., 0].astype(np.int16)
    g = roi[..., 1].astype(np.int16)
    b = roi[..., 2].astype(np.int16)
    l = r * 0.299 + g * 0.587 + b * 0.114
    white = l > 180
    ys, xs = np.where(white)
    if len(ys) < 5:
        return None
    # 找白 D（紧致，约 10-25px）
    m = white
    cols = m.sum(axis=0)
    runs, run = [], None
    for i, v in enumerate(cols):
        if v >= 2 and run is None:
            run = i
        elif v < 2 and run is not None:
            runs.append((run, i))
            run = None
    if run is not None:
        runs.append((run, len(cols)))
    for (cs, ce) in runs:
        seg = m[:, cs:ce]
        rys = np.where(seg.sum(axis=1) > 0)[0]
        if len(rys) < 8:
            continue
        hh = rys.max() - rys.min() + 1
        ww = ce - cs
        if 8 <= hh <= 30 and 6 <= ww <= 25:
            gly = white[rys.min():rys.max() + 1, cs:ce]
            return (730 + cs, 950 + rys.min(), gly), arr
    return None, arr


def _ncc(g_full, t):
    th, tw = t.shape
    if th >= g_full.shape[0] or tw >= g_full.shape[1]:
        return -1.0, None
    wins = np.lib.stride_tricks.sliding_window_view(g_full, (th, tw))
    wm = wins.mean(axis=(2, 3), keepdims=True)
    wc = wins - wm
    wn = np.sqrt((wc * wc).sum(axis=(2, 3)))
    t0 = t - t.mean()
    tn = float(np.sqrt((t0 * t0).sum()))
    scores = np.abs((wc * t0).sum(axis=(2, 3)) / np.maximum(wn * tn, 1e-9))
    idx = np.unravel_index(np.argmax(scores), scores.shape)
    return float(scores[idx]), (int(idx[1]), int(idx[0]))


def main() -> int:
    got, _ = _load()
    if got is None:
        print("无法提取 D 模板")
        return 1
    dx, dy, gly = got
    print(f"D 模板: ({dx},{dy}) {gly.shape[1]}x{gly.shape[0]}")
    # 二进制化模板（float）
    d_tpl = gly.astype(np.float64)
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
        l = (arr[..., 0].astype(np.int16) * 0.299
             + arr[..., 1].astype(np.int16) * 0.587
             + arr[..., 2].astype(np.int16) * 0.114)
        g_full = (l > 150).astype(np.float64)     # 白字掩码
        sc, pos = _ncc(g_full, d_tpl)
        if pos is None:
            print(f"{fid[:30]} GT={gt_seat} 无匹配 sc={sc:.2f}")
            continue
        bx, by = pos
        best = None
        for seat, (ax, ay) in SEAT_ANCHORS_1848.items():
            d = float(((bx - ax) ** 2 + (by - ay) ** 2) ** 0.5)
            if d <= 450 and (best is None or d < best[1]):
                best = (seat, d)
        pred = best[0] if best else None
        hit = pred == gt_seat
        if hit:
            ok += 1
        print(f"{fid[:30]} GT={gt_seat} pred={pred} sc={sc:.2f} "
              f"D@({bx},{by}) {'HIT' if hit else 'MISS'}")
    print(f"\nOK: {ok}/{total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
