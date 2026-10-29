"""P13-Step B — 底部带 ASCII 结构图 + 按钮内白字测量（指定帧）。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402

FRAMES = {
    "TABLE_CHANGED_398_1787975612766_full.png": "f398",
    "TABLE_CHANGED_403_1787975620299_full.png": "f403",
    "TABLE_CHANGED_404_1787975621874_full.png": "f404",
    "TABLE_CHANGED_481_1787975738203_full.png": "f481",
    "TABLE_CHANGED_545_1787975849475_full.png": "f545",
}


def load(name: str) -> np.ndarray:
    w, h, rgba = decode_png(
        (Path("data/vision2/coinpoker_evidence") / name).read_bytes())
    return np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)


def classify(arr: np.ndarray) -> np.ndarray:
    r = arr[..., 0].astype(np.int16)
    g = arr[..., 1].astype(np.int16)
    b = arr[..., 2].astype(np.int16)
    lum = r * 0.299 + g * 0.587 + b * 0.114
    cls = np.full(lum.shape, ord("."), dtype=np.uint8)
    cls[lum < 35] = ord("#")
    cls[(g > r + 40) & (g > b + 40) & (g > 90)] = ord("G")
    cls[(r > g + 50) & (r > b + 50) & (r > 110)] = ord("R")
    cls[(b > r + 40) & (b > g + 40) & (b > 100)] = ord("B")
    cls[(r > 150) & (g > 110) & (b < 90) & (np.abs(r - g) < 90)] = ord("Y")
    cls[(lum > 150) & (np.abs(r - g) < 40) & (np.abs(g - b) < 40)] = ord("W")
    return cls


def ascii_map(cls: np.ndarray, x0: int, y0: int, x1: int, y1: int,
              scale: int = 8) -> str:
    sub = cls[y0:y1, x0:x1]
    h2 = sub.shape[0] // scale
    w2 = sub.shape[1] // scale
    out = []
    for rr in range(h2):
        line = []
        for cc in range(w2):
            blk = sub[rr * scale:(rr + 1) * scale, cc * scale:(cc + 1) * scale]
            counts = {}
            for ch in np.unique(blk):
                counts[ch] = int((blk == ch).sum())
            picked = "."
            for prio in (ord("R"), ord("B"), ord("G"), ord("Y"), ord("W"),
                         ord("#"), ord(".")):
                if counts.get(prio, 0) >= max(4, scale * scale // 8):
                    picked = chr(prio)
                    break
            line.append(picked)
        out.append(f"{rr * scale + y0:4d} " + "".join(line))
    return "\n".join(out)


def white_in(arr: np.ndarray, bbox: tuple) -> int:
    x, y, w, h = bbox
    sub = arr[y:y + h, x:x + w]
    r = sub[..., 0].astype(np.int16)
    g = sub[..., 1].astype(np.int16)
    b = sub[..., 2].astype(np.int16)
    lum = r * 0.299 + g * 0.587 + b * 0.114
    return int(((lum > 150) & (np.abs(r - g) < 40)
                & (np.abs(g - b) < 40)).sum())


def main() -> int:
    for fn, tag in FRAMES.items():
        p = Path("data/vision2/coinpoker_evidence") / fn
        if not p.exists():
            continue
        arr = load(fn)
        cls = classify(arr)
        print(f"===== {tag} ({fn}) x0-700 y600-1066 scale4 =====")
        print(ascii_map(cls, 0, 600, 700, 1066, scale=4))
        # 暗灰按钮候选：lum 35-120 · sat<45 · 含白字
        x0, y0, x1, y1 = 0, 600, 700, 1066
        sub = arr[y0:y1, x0:x1]
        r = sub[..., 0].astype(np.int16)
        g = sub[..., 1].astype(np.int16)
        b = sub[..., 2].astype(np.int16)
        lum = r * 0.299 + g * 0.587 + b * 0.114
        sat = np.maximum(np.maximum(r, g), b) - np.minimum(np.minimum(r, g), b)
        dark = (lum > 35) & (lum < 120) & (sat < 45)
        rows = dark.sum(axis=1)
        bands, run = [], None
        for i, v in enumerate(rows):
            if v > 100 and run is None:
                run = i
            elif v <= 100 and run is not None:
                bands.append((run, i))
                run = None
        if run is not None:
            bands.append((run, len(rows)))
        for bs, be in bands:
            if not (30 <= be - bs <= 140):
                continue
            band = dark[bs:be]
            cols = band.sum(axis=0)
            runs, run = [], None
            for i, v in enumerate(cols):
                if v > 30 and run is None:
                    run = i
                elif v <= 30 and run is not None:
                    runs.append((run, i))
                    run = None
            if run is not None:
                runs.append((run, len(cols)))
            for cs, ce in runs:
                if not (50 <= ce - cs <= 500):
                    continue
                if band[:, cs:ce].sum() < 3000:
                    continue
                wsub = lum[bs:be, cs:ce]
                rs = r[bs:be, cs:ce]
                gs = g[bs:be, cs:ce]
                bs2 = b[bs:be, cs:ce]
                white = int(((wsub > 150) & (np.abs(rs - gs) < 40)
                             & (np.abs(gs - bs2) < 40)).sum())
                print(f"  DARK_BTN bbox=({cs},{bs + y0},{ce - cs},{be - bs}) "
                      f"area={int(band[:, cs:ce].sum())} white={white}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
