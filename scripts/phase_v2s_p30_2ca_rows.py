"""P30-2C-A — 金额行 ROI 精确定位（含 label 行分离）+ 模板参考位图保存。

pc114: 行带 y1330-1385 (h56, label+amount 合)——需分 label/amount；
实际「跟注」标签 y1330-1360，「124,163」y1355-1385。
先分离两行标签。"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402

EV = Path("data/vision2/coinpoker_evidence")


def rowprofile(fid, tag, xa, xb, ya, yb):
    w, h, rgba = decode_png((EV / fid).read_bytes())
    arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
    roi = arr[ya:yb, xa:xb]
    r = roi[..., 0].astype(np.int16); g = roi[..., 1].astype(np.int16)
    b = roi[..., 2].astype(np.int16)
    l = r * 0.299 + g * 0.587 + b * 0.114
    text = ((l > 170) & (np.abs(r - g) < 55) & (np.abs(g - b) < 55))
    rowd = text.sum(axis=1)
    print(f"[{tag}] 行剖面 (ya={ya}):")
    for i in range(0, yb - ya, 3):
        v = int(rowd[i:i + 3].max())
        if v > 0:
            print(f"  y{ya+i}: {v}")


rowprofile("POT_CHANGED_114_1787992536082_full.png", "pc114",
           1400, 1600, 1320, 1395)
