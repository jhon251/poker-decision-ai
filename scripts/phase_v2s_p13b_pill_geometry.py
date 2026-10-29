"""P13-B — 胶囊（pill）精确几何测量：dark blob 含数字行 → ROI 校准。

只读测量：对代表帧，围绕 A/B 数字行窗口找暗背景（luma<70）连通块并输出
bbox；同时输出数字行内接 bbox。结果用于定义 BET_ROIS_1280。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402

ZONES = {
    "S_TL": ((400, 290, 640, 380), (430, 314, 155, 36)),   # (搜索窗, 数字行窗)
    "S_TR": ((640, 225, 880, 320), (664, 251, 148, 36)),
}
FRAMES = [
    "TABLE_CHANGED_100_1787928276694_full.png",
    "TABLE_CHANGED_101_1787928277940_full.png",
    "TABLE_CHANGED_117_1787928295861_full.png",
    "TABLE_CHANGED_137_1787928327278_full.png",
    "TABLE_CHANGED_185_1787928391857_full.png",
    "TABLE_CHANGED_213_1787928433101_full.png",
    "TABLE_CHANGED_293_1787928544875_full.png",
    "TABLE_CHANGED_413_1787928698111_full.png",
    "TABLE_CHANGED_420_1787928708495_full.png",
]


def load_frame(path: Path) -> np.ndarray:
    w, h, rgba = decode_png(path.read_bytes())
    return np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)


def luma_of(arr: np.ndarray) -> np.ndarray:
    r = arr[..., 0].astype(np.int16)
    g = arr[..., 1].astype(np.int16)
    b = arr[..., 2].astype(np.int16)
    return (r * 0.299 + g * 0.587 + b * 0.114).astype(np.int16)


def pill_bbox(lum: np.ndarray, search: tuple, digit_row: tuple):
    sx0, sy0, sx1, sy1 = search
    dx0, dy0, dw, dh = digit_row
    sub = lum[sy0:sy1, sx0:sx1]
    dark = (sub < 55).astype(np.uint8)
    # 数字行实际白墨 bbox（内部墨水范围，非探测窗）
    dsub = (lum[dy0:dy0 + dh, dx0:dx0 + dw] > 150).astype(np.uint8)
    dys, dxs = np.where(dsub)
    if len(dys) < 20:
        return None
    gy0, gy1 = dys.min() + dy0 - sy0, dys.max() + dy0 - sy0
    gx0, gx1 = dxs.min() + dx0 - sx0, dxs.max() + dx0 - sx0
    dw_row = int(dsub.sum())
    # 行 run：覆盖数字行的暗行带
    rows = dark.sum(axis=1)
    bands, run = [], None
    for i, v in enumerate(rows):
        if v > 20 and run is None:
            run = i
        elif v <= 20 and run is not None:
            bands.append((run, i))
            run = None
    if run is not None:
        bands.append((run, len(rows)))
    row_band = None
    for (bs, be) in bands:
        if bs <= gy0 and be >= gy1:
            row_band = (bs, be)
    if row_band is None:
        return None
    # 列 run：覆盖数字 x 范围的暗列带
    cols = dark[row_band[0]:row_band[1]].sum(axis=0)
    cruns, run = [], None
    for i, v in enumerate(cols):
        if v > 6 and run is None:
            run = i
        elif v <= 6 and run is not None:
            cruns.append((run, i))
            run = None
    if run is not None:
        cruns.append((run, len(cols)))
    col_band = None
    for (cs, ce) in cruns:
        if cs <= gx0 and ce >= gx1:
            col_band = (cs, ce)
    if col_band is None:
        return None
    return {"pill": [int(col_band[0] + sx0), int(row_band[0] + sy0),
                     int(col_band[1] - col_band[0]),
                     int(row_band[1] - row_band[0])],
            "digit_white": int(dw_row)}


def main() -> int:
    out = {}
    for name, (search, row) in ZONES.items():
        out[name] = {"rows": []}
        for fn in FRAMES:
            arr = load_frame(Path("data/vision2/coinpoker_evidence") / fn)
            lum = luma_of(arr)
            r = pill_bbox(lum, search, row)
            if r:
                out[name]["rows"].append({"frame": fn, **r})
    print(json.dumps(out, indent=1, ensure_ascii=False))
    Path("data/vision2/p13b_pill_geometry.json").write_text(
        json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
