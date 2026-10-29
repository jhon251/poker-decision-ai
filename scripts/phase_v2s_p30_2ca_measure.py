"""P30-2C-A — 金额行 digit 尺寸测量 + 模板提取（GT: 124,163 / 2,000）。

策略：对 GT 文本区域，用已知字符数切分（124,163 = 6 digits+1 comma；
2,000 = 4 digits+1 comma），动态列投影找 digit/comma 段。"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402
from poker_decision_ai.vision2_platform.png_encode import encode_rgba_png  # noqa: E402

EV = Path("data/vision2/coinpoker_evidence")
OUT = Path("data/vision2/coinpoker_assets_v1876tocall")
OUT.mkdir(parents=True, exist_ok=True)


def measure(fid, tag, xa, xb, ya, yb):
    w, h, rgba = decode_png((EV / fid).read_bytes())
    arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
    roi = arr[ya:yb, xa:xb]
    r = roi[..., 0].astype(np.int16); g = roi[..., 1].astype(np.int16)
    b = roi[..., 2].astype(np.int16)
    l = r * 0.299 + g * 0.587 + b * 0.114
    text = ((l > 170) & (np.abs(r - g) < 55) & (np.abs(g - b) < 55))
    # 行投影（金额行）
    rowd = text.sum(axis=1)
    rows = np.where(rowd >= 3)[0]
    if len(rows) == 0:
        print(f"[{tag}] none")
        return
    y0, y1 = rows.min(), rows.max()
    print(f"[{tag}] text band y={ya+y0}-{ya+y1} (h={y1-y0+1})")
    # 列投影（每列墨量，gap 检测）
    cold = text[y0:y1 + 1].sum(axis=0)
    prof = "".join("#" if v >= 1 else "." for v in cold)
    print(f"  col剖面(xa={xa}): {prof}")


measure("POT_CHANGED_114_1787992536082_full.png", "pc114",
        1400, 1600, 1330, 1400)
measure("POT_CHANGED_110_1787992526057_full.png", "pc110",
        1340, 1560, 1290, 1400)
