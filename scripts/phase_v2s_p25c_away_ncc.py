"""P25-C — away 模板提取（f01 S3 咖啡杯+暂离）+ 全矩阵滑窗 NCC。

模板来源：f01 S3 away 白 bbox (1585,232,63,106)（GT provenance）。
比较：per-seat 期望位置 ± 余量窗口 的滑窗 NCC（亮二值图）。
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402
from poker_decision_ai.vision2_platform.png_encode import encode_rgba_png  # noqa: E402

EV = Path("data/vision2/coinpoker_evidence")
OUT = Path("data/vision2/p25c_crops")

ANCH = {"S1": (255, 310), "S2": (918, 210), "S3": (1578, 320),
        "S4": (218, 868), "S5": (1618, 868), "S6": (945, 1160)}
FRAMES = {
    "f01": "BUTTON_CHANGED_167_1787992682749_full.png",
    "f02": "POT_CHANGED_12_1787995290794_full.png",
    "f03": "POT_CHANGED_102_1787992505794_full.png",
    "f04": "POT_CHANGED_13_1787992279195_full.png",
    "f05": "POT_CHANGED_139_1787992592538_full.png",
    "f06": "BUTTON_CHANGED_598_1787993753587_full.png",
}
GTF = {"f01": {"S3": "away"}, "f02": {"S3": "away"}}


def load(fp):
    w, h, rgba = decode_png((EV / fp).read_bytes())
    return np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)


def white_bin(arr, roi):
    x0, y0, w, h = roi
    sub = arr[y0:y0 + h, x0:x0 + w]
    l = sub[..., 0] * 0.299 + sub[..., 1] * 0.587 + sub[..., 2] * 0.114
    r, g, b = (sub[..., 0].astype(np.int16), sub[..., 1].astype(np.int16),
               sub[..., 2].astype(np.int16))
    return ((l > 140) & (np.abs(r - g) < 30) & (np.abs(g - b) < 30)
            ).astype(np.float64)


# 模板：f01 S3 away = (1585,232,63,106)
tmpl = white_bin(load(FRAMES["f01"]), (1585, 232, 63, 106))
ys, xs = np.where(tmpl > 0)
tmpl = tmpl[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
np.save(str(OUT / "away_tmpl.npy"), tmpl)
print("away tmpl shape:", tmpl.shape)
t0 = tmpl - tmpl.mean()
tn = float(np.sqrt((t0 * t0).sum()))


def slide_ncc(win):
    th, tw = tmpl.shape
    if win.shape[0] < th or win.shape[1] < tw:
        return -1.0
    best = -1.0
    for dy in range(0, win.shape[0] - th + 1, 2):
        for dx in range(0, win.shape[1] - tw + 1, 2):
            sub = win[dy:dy + th, dx:dx + tw]
            s0 = sub - sub.mean()
            dn = float(np.sqrt((s0 * s0).sum()))
            if dn < 1e-9:
                continue
            v = float((s0 * t0).sum() / (dn * tn))
            best = max(best, abs(v))
    return best


# away ROI 每座 = tag ROI + 偏移（f01 S3: tag(1494,325) → away(1585,232)）
TAG = {"S1": (138, 325), "S2": (816, 211), "S3": (1494, 325),
       "S4": (101, 891), "S5": (1531, 891), "S6": (816, 1149)}
AWAY_OFF = (91, -93)
print("=== away NCC matrix ===")
print(f"{'':5s}" + "".join(f"{s:>10s}" for s in ANCH))
for fr, fp in FRAMES.items():
    arr = load(fp)
    row = []
    for seat, (ax, ay) in ANCH.items():
        tx, ty = TAG[seat]
        x0 = tx + AWAY_OFF[0] - 40; y0 = ty + AWAY_OFF[1] - 40
        w, h = 63 + 80, 106 + 80
        x0, y0 = max(0, x0), max(0, y0)
        win = white_bin(arr, (x0, y0, w, h))
        sc = slide_ncc(win)
        f = seat in GTF.get(fr, {})
        row.append(f"{sc:.2f}{'A' if f else '.'}")
    print(f"{fr:5s}" + "".join(f"{v:>10s}" for v in row))
