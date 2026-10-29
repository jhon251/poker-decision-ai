"""P30-2C-A — 精确二分：下带（第 2 行）字符几何。
过牌 = 2 字（过/牌 + ✓）+ 跟注任何；跟注N = 跟注 + 逗号形式金额（4-7 char）。
用下带文本的列段数 + 最大段宽度（digit 段窄 4-14px；"牌" 字宽 ~20px）。"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402

EV = Path("data/vision2/coinpoker_evidence")

# 已知正例金标准
def profile(fid, tag):
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
    low = [b for b in bs if b[1] > 40]
    if not low:
        print(f"[{tag}] no lower band")
        return
    a2, b2 = low[-1]
    roi2 = roi[a2:b2 + 1]
    l2 = r[a2:b2 + 1] * 0.299 + g[a2:b2 + 1] * 0.587 + b[a2:b2 + 1] * 0.114
    t2 = ((l2 > 130) & (np.abs(r[a2:b2+1] - g[a2:b2+1]) < 50)
          & (np.abs(g[a2:b2+1] - b[a2:b2+1]) < 50))
    cold = t2.sum(axis=0)
    segs = []
    run = None
    for i, v in enumerate(cold):
        if v >= 1 and run is None:
            run = [i, i]
        elif v >= 1 and run is not None:
            run[1] = i
        elif run is not None:
            segs.append(run); run = None
    if run:
        segs.append(run)
    ms = (b - a + 1 for a, b in segs)
    print(f"[{tag}] 下带段数={len(segs)} 段宽={sorted(ms, reverse=True)[:6]} "
          f"总墨={int(t2.sum())}")


profile("POT_CHANGED_110_1787992526057_full.png", "pc110 跟注2,000")
profile("POT_CHANGED_177_1787992718375_full.png", "pc177 过牌")
profile("POT_CHANGED_229_1787992854988_full.png", "pc229 过牌")
profile("POT_CHANGED_143_1787992603846_full.png", "pc143 跟注2,000(绿)")
