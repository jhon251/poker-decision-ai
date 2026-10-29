"""P27-A — 空座实证扫描：座牌盒（dark panel）存在性。
每座 panel 区 = stack ROI 上扩 190px（含 name 行）；签名 = 暗面板占比
（luma 15-60，低饱和）+ 绿毡反证（g>r+20 占比）。
占用座应 panel≈高；若任何帧某座 panel 缺失（绿毡/桌面）→ 空座实例。
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402

EV = Path("data/vision2/coinpoker_evidence")

STACK = {
    "S1": (165, 413, 360, 485), "S2": (845, 298, 1035, 370),
    "S3": (1520, 413, 1715, 485), "S4": (125, 978, 320, 1050),
    "S5": (1560, 978, 1745, 1050), "S6": (838, 1236, 1040, 1308),
}


def panel_sig(arr, seat):
    x0, y0, x1, y1 = STACK[seat]
    px0, py0 = max(0, x0 - 10), max(0, y0 - 60)
    px1, py1 = min(arr.shape[1], x1 + 10), min(arr.shape[0], y1 + 15)
    roi = arr[py0:py1, px0:px1]
    if roi.size == 0:
        return -1.0, -1.0
    r = roi[..., 0].astype(np.int16); g = roi[..., 1].astype(np.int16)
    b = roi[..., 2].astype(np.int16)
    l = r * 0.299 + g * 0.587 + b * 0.114
    dark_panel = ((l > 12) & (l < 62) & (np.abs(r - g) < 30)
                  & (np.abs(g - b) < 30))
    felt = (g > r + 20)
    return float(dark_panel.mean()), float(felt.mean())


# 采样扫描（每 8 帧取 1 + 6 GT 帧必扫）
gt_names = {
    "BUTTON_CHANGED_167_1787992682749_full.png",
    "POT_CHANGED_12_1787995290794_full.png",
    "POT_CHANGED_102_1787992505794_full.png",
    "POT_CHANGED_13_1787992279195_full.png",
    "POT_CHANGED_139_1787992592538_full.png",
    "BUTTON_CHANGED_598_1787993753587_full.png",
}
files = sorted(EV.glob("*_full.png"))
candidates = []
scanned = 0
idx = 0
for fp in files:
    try:
        w, h, rgba = decode_png(fp.read_bytes())
    except Exception:
        continue
    if (w, h) != (1876, 1400):
        continue
    scanned += 1
    idx += 1
    if fp.name not in gt_names and idx % 8 != 0:
        continue
    arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
    rows = {}
    for seat in STACK:
        dp, felt = panel_sig(arr, seat)
        rows[seat] = (round(dp, 2), round(felt, 2))
    # 疑似空座：panel < 0.5 且 felt > 0.1（绿毡入座区）
    susp = [(s, v) for s, v in rows.items() if v[0] < 0.5 and v[1] > 0.1]
    if susp:
        candidates.append((fp.name, rows, susp))
        print(f"{fp.name[:32]} SUSPECT: {susp}")
print(f"\nscanned={scanned} candidates={len(candidates)}")
