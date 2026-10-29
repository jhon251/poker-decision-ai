"""P25-B.3 — 全 GT 折叠实例「盖牌」tag 精确测量：
teal bg mask + dark glyph；bbox / offset-vs-anchor / 负例签名计数。
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402
from poker_decision_ai.vision2_platform.png_encode import encode_rgba_png  # noqa: E402

EV = Path("data/vision2/coinpoker_evidence")
OUT = Path("data/vision2/p25b3_crops")

ANCH = {"S1": (255, 310), "S2": (918, 210), "S3": (1578, 320),
        "S4": (218, 868), "S5": (1618, 868), "S6": (945, 1160)}

FRAMES = {
    "f01": ("BUTTON_CHANGED_167_1787992682749_full.png", {}),
    "f02": ("POT_CHANGED_12_1787995290794_full.png", {}),
    "f03": ("POT_CHANGED_102_1787992505794_full.png", {}),
    "f04": ("POT_CHANGED_13_1787992279195_full.png", {}),
    "f05": ("POT_CHANGED_139_1787992592538_full.png", {}),
    "f06": ("BUTTON_CHANGED_598_1787993753587_full.png", {}),
}
FOLDED = {  # GT：seat -> frame
    "S5": "f01", "S6": "f01",
    "S2": "f02", "S4": "f02",
    "S1": "f03", "S3": "f03", "S4": "f03", "S5": "f03",
    "S3": "f04", "S4": "f04", "S5": "f04", "S6": "f04",
    "S2": "f05", "S6": "f05",
    "S3": "f06", "S6": "f06",
}
# 负例（GT 非 folded 的某座/帧）
NEG = [("S1", "f01"), ("S3", "f01"),
       ("S2", "f03"), ("S1", "f04"),
       ("S3", "f05"), ("S1", "f06")]


def load(fp):
    w, h, rgba = decode_png((EV / fp).read_bytes())
    return np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)


def tag_signature(arr, ax, ay):
    """座锚 (ax,ay) 周围搜索 teal-tag：bg mask 连通 + 内部暗字。"""
    r = arr[..., 0].astype(np.int16); g = arr[..., 1].astype(np.int16)
    b = arr[..., 2].astype(np.int16)
    l = r * 0.299 + g * 0.587 + b * 0.114
    x0, x1 = max(0, ax - 280), min(arr.shape[1], ax + 120)
    y0, y1 = max(0, ay - 200), min(arr.shape[0], ay + 260)
    roi_l = l[y0:y1, x0:x1]
    roi_r = r[y0:y1, x0:x1]; roi_g = g[y0:y1, x0:x1]
    roi_b = b[y0:y1, x0:x1]
    # teal bg: luma 50-140, g>r+10, g>=b （muted teal）
    teal = ((roi_l > 50) & (roi_l < 140) & (roi_g > roi_r + 10)
            & (roi_g >= roi_b))
    # 最大连通 bbox（BFS 近似：列/行密度投影）
    dens = teal.sum(axis=0)
    if teal.sum() < 80:
        return None, None
    ys, xs = np.where(teal)
    # 全部分布 bbox
    bx0, by0, bx1, by1 = xs.min(), ys.min(), xs.max(), ys.max()
    # 求"大块"：占 teal 像素 >60% 的行列范围
    rowd = teal.sum(axis=1); cold = teal.sum(axis=0)
    rows = np.where(rowd >= max(6, rowd.max() * 0.4))[0]
    cols = np.where(cold >= max(6, cold.max() * 0.4))[0]
    if len(rows) == 0 or len(cols) == 0:
        bbox = (x0 + bx0, y0 + by0, bx1 - bx0 + 1, by1 - by0 + 1)
    else:
        bbox = (x0 + int(cols.min()), y0 + int(rows.min()),
                int(cols.max()) - int(cols.min()) + 1,
                int(rows.max()) - int(rows.min()) + 1)
    # 内部暗字
    in_l = l[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]
    dark = (in_l < 55).sum()
    return bbox, {"teal_px": int(teal.sum()), "dark_px": int(dark),
                  "dark_ratio": round(float(dark) / max(1, bbox[2] * bbox[3]), 3)}


print("=== 正例（folded）===")
mem: dict = {}
for seat, fr in FOLDED.items():
    fp = FRAMES[fr][0]
    arr = load(fp)
    ax, ay = ANCH[seat]
    bbox, meta = tag_signature(arr, ax, ay)
    if bbox is None:
        print(f"{fr} {seat}: NO teal tag signature")
        continue
    dx = bbox[0] - ax; dy = bbox[1] - ay
    print(f"{fr} {seat}: tag bbox={bbox} offset=({dx:+d},{dy:+d}) "
          f"teal={meta['teal_px']} dark={meta['dark_px']} "
          f"dark_ratio={meta['dark_ratio']}")

print("=== 负例 ===")
for seat, fr in NEG:
    fp = FRAMES[fr][0]
    arr = load(fp)
    ax, ay = ANCH[seat]
    bbox, meta = tag_signature(arr, ax, ay)
    if bbox is None:
        print(f"{fr} {seat}: no-teal-signature (clean)")
    else:
        print(f"{fr} {seat}: teal={meta['teal_px']} bbox={bbox}")
