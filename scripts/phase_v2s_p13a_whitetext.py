"""P13-A — 亮白文本带搜索（下桌带 y 500-770），定位 bet pill / 按钮文字。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402


def load_frame(path: Path) -> np.ndarray:
    w, h, rgba = decode_png(path.read_bytes())
    return np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)


def luma_of(arr: np.ndarray) -> np.ndarray:
    r = arr[..., 0].astype(np.int16)
    g = arr[..., 1].astype(np.int16)
    b = arr[..., 2].astype(np.int16)
    return (r * 0.299 + g * 0.587 + b * 0.114).astype(np.int16)


def seg(lum: np.ndarray, thr: int = 150) -> list:
    mask = lum > thr
    rows = mask.sum(axis=1)
    bands, run = [], None
    for i, v in enumerate(rows):
        if v > 2 and run is None:
            run = i
        elif v <= 2 and run is not None:
            bands.append((run, i))
            run = None
    if run is not None:
        bands.append((run, len(rows)))
    out = []
    for bs, be in bands:
        if be - bs < 5:
            continue
        band = mask[bs:be]
        colink = band.sum(axis=0) > 1
        runs, run = [], None
        for i, v in enumerate(colink):
            if v and run is None:
                run = i
            elif not v and run is not None:
                if i - run >= 5:
                    runs.append((run, i))
                run = None
        if run is not None and len(colink) - run >= 5:
            runs.append((run, len(colink)))
        for cs, ce in runs:
            m2 = mask[bs:be, cs:ce]
            ys, xs = np.where(m2)
            if len(ys) < 15:
                continue
            out.append((int(xs.min() + cs), int(ys.min() + bs),
                        int(xs.max() - xs.min() + 1),
                        int(ys.max() - ys.min() + 1)))
    return out


def main() -> int:
    since_s = 134324017101490527 / 1e7 - 11644473600
    fulls = sorted(Path("data/vision2/coinpoker_evidence").glob("*_full.png"))
    new = [p for p in fulls if p.stat().st_mtime >= since_s]
    clusters: dict = {}
    frame_rows: dict = {}
    for path in new:
        arr = load_frame(path)
        h, w = arr.shape[:2]
        lum = luma_of(arr)
        sub = lum  # 全帧
        cands = seg(sub)
        if cands:
            frame_rows[path.stem] = cands
        for (x, y, cw, ch) in cands:
            bx, by = x, y
            key = (bx // 16, by // 16, cw // 16, ch // 8)
            d = clusters.setdefault(key, {"n": 0, "frames": set(),
                                          "sample": [bx, by, cw, ch]})
            d["n"] += 1
            d["frames"].add(path.stem)
    Path("data/vision2/p13a_whitetext.json").write_text(
        json.dumps({"frame_cands": frame_rows,
                    "clusters": [{**{k: v for k, v in d.items()
                                     if k != "frames"},
                                  "distinct_frames": len(d["frames"])}
                                 for d in sorted(
                                     clusters.values(),
                                     key=lambda d: -d["n"])]},
                   indent=1, ensure_ascii=False), encoding="utf-8")
    top = sorted(clusters.values(), key=lambda d: -d["n"])[:40]
    for d in top:
        print("n=%4d fr=%3d bbox=%s" % (d["n"], len(d["frames"]),
                                        d["sample"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
