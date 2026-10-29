"""P30-2C-A — 在 2 行带帧中找「过牌」vs「跟注N」差异：
过牌组 = X/F + 跟注任何 + 过牌（无金额行）；
跟注组 = X/F + 跟注 N（有金额行）。用金额行存在性（第 2 行带是否 digit）
二分。"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402

EV = Path("data/vision2/coinpoker_evidence")
files = sorted(EV.glob("*.png"))


def bands(fid):
    w, h, rgba = decode_png((EV / fid).read_bytes())
    arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
    roi = arr[1150:1400, 1250:1876]
    r = roi[..., 0].astype(np.int16); g = roi[..., 1].astype(np.int16)
    b = roi[..., 2].astype(np.int16)
    l = r * 0.299 + g * 0.587 + b * 0.114
    text = ((l > 130) & (np.abs(r - g) < 50) & (np.abs(g - b) < 50))
    rowd = text.sum(axis=1)
    bs, run = [], None
    for i, v in enumerate(rowd):
        if v >= 5 and run is None:
            run = [i, i]
        elif v >= 5 and run is not None:
            run[1] = i
        elif run is not None:
            bs.append(run); run = None
    if run:
        bs.append(run)
    return [(a, b) for a, b in bs if b > 40]


check_frames, call_frames = [], []
for fp in files:
    if not fp.name.endswith("_full.png"):
        continue
    try:
        w, h, rgba = decode_png(fp.read_bytes())
    except Exception:
        continue
    if (w, h) != (1876, 1400):
        continue
    bs = bands(fp.name)
    if len(bs) != 2:
        continue
    # 第 2 行带（下带）是否有 digit（金色/白 digit 结构——金额行）
    a2, b2 = bs[1]
    arr = (decode_png((EV / fp.name).read_bytes())[2])
    arr = np.frombuffer(arr, dtype=np.uint8).reshape(1400, 1876, 4)
    roi2 = arr[1150 + a2:1150 + b2 + 1, 1250:1876]
    rr = roi2[..., 0].astype(np.int16); gg = roi2[..., 1].astype(np.int16)
    bb = roi2[..., 2].astype(np.int16)
    l2 = rr * 0.299 + gg * 0.587 + bb * 0.114
    # 金额行：有逗号点结构（文本长度 > 标签行）→ 粗判 digit 存在
    text2 = ((l2 > 130) & (np.abs(rr - gg) < 50) & (np.abs(gg - bb) < 50))
    # digit 行特征：行高 ≤14 且 文本宽度大（含逗号）
    w2 = int(text2.sum())
    if w2 > 250:
        call_frames.append(fp.name)
    else:
        check_frames.append(fp.name)

print(f"跟注类（有金额行）: {len(call_frames)}")
print(f"过牌类（无金额行）: {len(check_frames)}")
print("跟注抽样:", [x[:32] for x in call_frames[:10]])
print("过牌抽样:", [x[:32] for x in check_frames[:5]])
