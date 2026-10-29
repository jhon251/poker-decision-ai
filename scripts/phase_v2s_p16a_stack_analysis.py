"""P16-A — Stack 1876 实测分析（只读）。

目标：定位每座 stack 金额文本的真实 bbox + 字高 + 颜色 + column projection，
找出 P15 0/1074 的根因（锚点偏移 / 金掩码阈值 / 尺度带）。
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2.pixel.realclient.coinpoker_stack import (  # noqa: E402
    SEAT_ANCHORS_1848, StackRecognizer)
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402

SEATS = ("S1", "S2", "S3", "S4", "S5", "S6")


def _blobs(frame, y0, y1, x0, x1, kind="gold_broad", min_w=25, min_h=18):
    r = frame[..., 0].astype(np.int16)
    g = frame[..., 1].astype(np.int16)
    b = frame[..., 2].astype(np.int16)
    l = r * 0.299 + g * 0.587 + b * 0.114
    if kind == "gold_broad":       # StackRecognizer._gold_mask
        M = (r > 120) & (g > 85) & (r >= g) & (g >= b - 30) & (l > 90)
    elif kind == "gold_bright":    # SeatOccupancyDetector._gold_blobs
        M = (r > 150) & (g > 110) & (b < 130) & (r >= g)
    elif kind == "white":          # 白文本
        M = (l > 200) & (np.abs(r - g) < 40) & (np.abs(g - b) < 40)
    else:
        raise ValueError(kind)
    M = M[y0:y1, x0:x1].astype("uint8")
    # 简单连通：逐行投影找 band，再列投影找 run（近似）
    out = []
    rows = M.sum(axis=1)
    bands, run = [], None
    for i, v in enumerate(rows):
        if v > 2 and run is None:
            run = i
        elif v <= 2 and run is not None:
            bands.append((run, i))
            run = None
    if run is not None:
        bands.append((run, len(rows)))
    for bs, be in bands:
        if be - bs < min_h:
            continue
        band = M[bs:be]
        cols = band.sum(axis=0)
        runs, run2 = [], None
        for i, v in enumerate(cols):
            if v > 1 and run2 is None:
                run2 = i
            elif v <= 1 and run2 is not None:
                runs.append((run2, i))
                run2 = None
        if run2 is not None:
            runs.append((run2, len(cols)))
        for cs, ce in runs:
            if ce - cs < min_w:
                continue
            out.append((x0 + cs, y0 + bs, ce - cs, be - bs,
                        int(band[bs:be, cs:ce].sum())))
    return out


def main() -> int:
    # 直接用已知 1876 帧（P15 已确认），避免全量扫描
    names = [
        "BUTTON_CHANGED_167_1787992682749_full.png",
        "POT_CHANGED_12_1787995290794_full.png",
        "POT_CHANGED_102_1787992505794_full.png",
        "BUTTON_CHANGED_598_1787993753587_full.png",
    ]
    ev = Path("data/vision2/coinpoker_evidence")
    for name in names:
        p = ev / name
        if not p.exists():
            print(f"缺帧: {name}")
            continue
        w, h, rgba = decode_png(p.read_bytes())
        if (w, h) != (1876, 1400):
            print(f"非1876: {name} ({w},{h})")
            continue
        arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
        print(f"\n=== {name} ===")
        blobs = _blobs(arr, 1100, 1400, 0, 1876, "gold_broad")
        print(f"  [底带 gold_broad blobs] {len(blobs)} 个")
        for (x, y, bw, bh, ink) in sorted(blobs, key=lambda t: (t[1], t[0]))[:20]:
            print(f"    bbox=({x},{y},{bw},{bh}) ink={ink}")
        for (x, y, bw, bh, ink) in sorted(blobs, key=lambda t: (t[1], t[0]))[:6]:
            px = arr[y:y + bh, x:x + bw].reshape(-1, 4)
            px = px[np.where(px[..., 0] > 100)]
            if len(px) == 0:
                continue
            print(f"    bbox({x},{y},{bw},{bh}) 颜色: "
                  f"R={px[...,0].mean():.0f} G={px[...,1].mean():.0f} "
                  f"B={px[...,2].mean():.0f} luma={ (px[...,0]*0.299+px[...,1]*0.587+px[...,2]*0.114).mean():.0f}")
        allb = _blobs(arr, 0, 1400, 0, 1876, "gold_broad", min_w=30, min_h=20)
        print(f"  [全帧 gold_broad blobs] {len(allb)} 个（w≥30,h≥20）")
        for (x, y, bw, bh, ink) in sorted(allb, key=lambda t: (t[1], t[0]))[:30]:
            near = None
            for seat, (ax, ay) in SEAT_ANCHORS_1848.items():
                d = ((x + bw // 2 - ax) ** 2 + (y + bh // 2 - ay) ** 2) ** 0.5
                if near is None or d < near[1]:
                    near = (seat, round(d, 0))
            print(f"    bbox=({x},{y},{bw},{bh}) ink={ink} 近锚点={near}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
