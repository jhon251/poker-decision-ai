"""P27-A — OCCUPIED 对象级签名在 36 GT 座格上的量化测量：
面板暗框（luma 15-62 中性）+ name 行灰白字 + stack 行 gold 三者存在性。
判断：占用座视觉层级=【座牌盒对象】而非单一场。
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402

EV = Path("data/vision2/coinpoker_evidence")
GT = json.loads(Path("data/vision2/coinpoker_1876_gt.json").read_text(
    encoding="utf-8"))

STACK = {
    "S1": (165, 413, 360, 485), "S2": (845, 298, 1035, 370),
    "S3": (1520, 413, 1715, 485), "S4": (125, 978, 320, 1050),
    "S5": (1560, 978, 1745, 1050), "S6": (838, 1236, 1040, 1308),
}


def load(fp):
    w, h, rgba = decode_png((EV / fp).read_bytes())
    return np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)


def panel_sig(arr, seat):
    x0, y0, x1, y1 = STACK[seat]
    px0, py0 = max(0, x0 - 10), max(0, y0 - 58)
    px1, py1 = min(arr.shape[1], x1 + 10), min(arr.shape[0], y1 + 12)
    roi = arr[py0:py1, px0:px1]
    r = roi[..., 0].astype(np.int16); g = roi[..., 1].astype(np.int16)
    b = roi[..., 2].astype(np.int16)
    l = r * 0.299 + g * 0.587 + b * 0.114
    dark = ((l > 12) & (l < 62) & (np.abs(r - g) < 30)
            & (np.abs(g - b) < 30)).mean()
    # name 行
    nx0, ny0 = px0, py0
    roi_n = arr[max(0, y0 - 52):y0 - 6, x0:x1]
    ln = roi_n[..., 0] * 0.299 + roi_n[..., 1] * 0.587 \
        + roi_n[..., 2] * 0.114
    rn = roi_n[..., 0].astype(np.int16); gn = roi_n[..., 1].astype(np.int16)
    bn = roi_n[..., 2].astype(np.int16)
    name_ink = int(((ln > 110) & (np.abs(rn - gn) < 28)
                    & (np.abs(gn - bn) < 28)).sum())
    # stack 行 gold
    roi_s = arr[y0:y1, x0:x1]
    rs = roi_s[..., 0].astype(np.int16); gs = roi_s[..., 1].astype(np.int16)
    bs = roi_s[..., 2].astype(np.int16)
    ls = rs * 0.299 + gs * 0.587 + bs * 0.114
    bg = np.median(ls)
    gold = int(((rs > bs + 30) & (gs >= bs + 18) & (rs >= gs * 0.9)
                & (ls > bg * 1.6)).sum())
    return round(float(dark), 2), name_ink, gold


print(f"{'seat':4s} {'dark':>5s} {'name':>6s} {'gold':>5s}")
vals = []
for ann in GT["annotations"]:
    arr = load(ann["frame_id"])
    for seat in STACK:
        d, n, g = panel_sig(arr, seat)
        vals.append((d, n, g))
        flag = "OK" if (d >= 0.35 and n >= 50 and g >= 100) else "?"
        print(f"{seat:4s} {d:5.2f} {n:6d} {g:5d} {flag}")
dp = np.array([v[0] for v in vals])
print(f"\n面板: min={dp.min():.2f} p10={np.percentile(dp,10):.2f} "
      f"p50={np.median(dp):.2f} max={dp.max():.2f}  (<0.35 的座格="
      f"{int((dp < 0.35).sum())})")
