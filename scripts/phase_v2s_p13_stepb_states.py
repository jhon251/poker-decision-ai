"""P13-Step B — 底部带状态聚类：828 帧的底部 UI 状态全集。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402

SINCE_FILETIME = 134324456119382219


def load(p: Path) -> np.ndarray:
    w, h, rgba = decode_png(p.read_bytes())
    return np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)


def band_features(arr: np.ndarray) -> tuple:
    r = arr[840:1066, :, 0].astype(np.int16)
    g = arr[840:1066, :, 1].astype(np.int16)
    b = arr[840:1066, :, 2].astype(np.int16)
    lum = r * 0.299 + g * 0.587 + b * 0.114
    R = (r > g + 50) & (r > b + 50) & (r > 110)
    G = (g > r + 40) & (g > b + 40) & (g > 90)
    B = (b > r + 40) & (b > g + 40) & (b > 100)
    Y = (r > 150) & (g > 110) & (b < 90) & (np.abs(r - g) < 90)
    W = (lum > 150) & (np.abs(r - g) < 40) & (np.abs(g - b) < 40)
    # 4 象限计数（量化到 200px 粒度）
    feats = []
    for m in (R, G, B, Y, W):
        for qx in range(0, 1419, 480):
            for qy in range(0, 226, 110):
                blk = m[qy:qy + 110, qx:qx + 480]
                feats.append(int(blk.sum()) // 500)
    return tuple(feats)


def main() -> int:
    since_s = SINCE_FILETIME / 1e7 - 11644473600
    fulls = sorted(Path("data/vision2/coinpoker_evidence").glob("*_full.png"))
    new = [p for p in fulls if p.stat().st_mtime >= since_s]
    clusters: dict = {}
    for path in new:
        arr = load(path)
        key = band_features(arr)
        d = clusters.setdefault(key, {"n": 0, "frames": []})
        d["n"] += 1
        if len(d["frames"]) < 6:
            d["frames"].append(path.stem)
    out = sorted(clusters.values(), key=lambda d: -d["n"])
    print("distinct bottom-band states:", len(out))
    for d in out[:25]:
        print(f"n={d['n']:4d} frames={d['frames']}")
    Path("data/vision2/p13_stepb_states.json").write_text(
        json.dumps([d for d in out], indent=1, ensure_ascii=False),
        encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
