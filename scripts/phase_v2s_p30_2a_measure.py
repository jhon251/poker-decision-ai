"""P30-2A — header 文本像素测量（bbox）+ SB/BB 徽章 + 金额格式全语料扫描。"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402

EV = Path("data/vision2/coinpoker_evidence")


def load(fp):
    w, h, rgba = decode_png((EV / fp).read_bytes())
    return np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)


# 1) header tight bbox（f01/f03/8bf8/pc110）：亮白文本于 0-500 x 0-60
for fid in ["BUTTON_CHANGED_167_1787992682749_full.png",
            "POT_CHANGED_102_1787992505794_full.png",
            "8bf82f134651db0add626be0304a2ff1.png",
            "POT_CHANGED_110_1787992526057_full.png"]:
    arr = load(fid)
    roi = arr[0:60, 0:520]
    r = roi[..., 0].astype(np.int16); g = roi[..., 1].astype(np.int16)
    b = roi[..., 2].astype(np.int16)
    l = r * 0.299 + g * 0.587 + b * 0.114
    white = (l > 150) & (np.abs(r - g) < 40) & (np.abs(g - b) < 40)
    ys, xs = np.where(white)
    if len(xs):
        print(f"{fid[:24]}: header bbox=({xs.min()},{ys.min()},"
              f"{xs.max()-xs.min()+1},{ys.max()-ys.min()+1}) white={len(xs)}")
    else:
        print(f"{fid[:24]}: no header white")

# 2) 金额格式：stack/sot 值 scan 全语料（找 M/K 后缀）
print("\n=== 金额格式语料扫描（stack 区域 M/K 后缀）===")
mcount = 0
for fp in sorted(EV.glob("*.png")):
    if not fp.name.endswith("_full.png"):
        continue
    try:
        w, h, rgba = decode_png(fp.read_bytes())
    except Exception:
        continue
    if (w, h) != (1876, 1400):
        continue
    arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
    # S1 stack ROI (165,413,360,485)
    roi = arr[413:485, 165:360]
    r = roi[..., 0].astype(np.int16); g = roi[..., 1].astype(np.int16)
    b = roi[..., 2].astype(np.int16)
    l = r * 0.299 + g * 0.587 + b * 0.114
    bg = float(np.median(l))
    gold = ((r > b + 30) & (g >= b + 18) & (r >= g * 0.9)
            & (l > bg * 1.6))
    if int(gold.sum()) > 300:
        mcount += 1
print(f"S1 stack gold 帧数 = {mcount}（无 M/K 判定——格式需模板级区分）")
