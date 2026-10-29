"""S6 hero 卡区 presence 扫描（多进程版）。

只读扫描：对每帧解码后裁剪 HEROCARDS_PAIR_ROI，按白卡面门（>=
CARD_PRESENCE_MIN）判定 presence。identity 仍需 image_observed 逐帧确认。
"""
from __future__ import annotations

import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

EV = Path("data/vision2/coinpoker_evidence")
OUT = Path("logs/hero_presence_candidates.json")
PAIR = (795, 1005, 1070, 1195)
MINPX = 8000


def _check(path_str: str):
    from poker_decision_ai.vision2_platform.png_decode import decode_png
    p = Path(path_str)
    try:
        w, h, rgba = decode_png(p.read_bytes())
    except Exception:
        return None
    if (w, h) != (1876, 1400):
        return None
    arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
    x0, y0, x1, y1 = PAIR
    roi = arr[y0:y1, x0:x1]
    r = roi[..., 0].astype(np.int16)
    g = roi[..., 1].astype(np.int16)
    b = roi[..., 2].astype(np.int16)
    l = r * 0.299 + g * 0.587 + b * 0.114
    white = (l > 150) & (r > 140) & (g > 140) & (b > 140)
    n = int(white.sum())
    return {"frame": p.name, "white_px": n} if n >= MINPX else None


def main() -> int:
    frames = sorted(str(p) for p in EV.glob("*.png"))
    print(f"scanning {len(frames)} frames", flush=True)
    cands = []
    with Pool(12) as pool:
        for i, res in enumerate(pool.imap_unordered(
                _check, frames, chunksize=64)):
            if res:
                cands.append(res)
            if (i + 1) % 2000 == 0:
                print(f"[{i + 1}/{len(frames)}] candidates={len(cands)}",
                      flush=True)
    cands.sort(key=lambda c: -c["white_px"])
    OUT.write_text(json.dumps(cands, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print(f"done: {len(cands)} presence-positive frames -> {OUT}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
