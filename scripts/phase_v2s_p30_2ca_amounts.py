"""P30-2C-A — 面板金额自动提取：找「跟注 N」按钮（白色文本按钮深底胶囊）。

方法：右下区 (1250-1876, 1150-1400) 找白色文本胶囊（灰色/深底圆角）
→ 文本行分割 → 两行（「跟注」+ 金额）→ 金额行 digit 解析（逗号格式）。
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402

EV = Path("data/vision2/coinpoker_evidence")
files = sorted(EV.glob("*.png"))

results = {}


def extract(fid):
    w, h, rgba = decode_png((EV / fid).read_bytes())
    arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
    roi = arr[1150:1400, 1250:1876]
    r = roi[..., 0].astype(np.int16); g = roi[..., 1].astype(np.int16)
    b = roi[..., 2].astype(np.int16)
    l = r * 0.299 + g * 0.587 + b * 0.114
    # 白文本（含灰色 90-200——「跟注」白字/「X/F」灰字）
    text = ((l > 130) & (np.abs(r - g) < 50) & (np.abs(g - b) < 50))
    rowd = text.sum(axis=1)
    # 行带（文本行）
    bands = []
    run = None
    for i, v in enumerate(rowd):
        if v >= 5 and run is None:
            run = [i, i]
        elif v >= 5 and run is not None:
            run[1] = i
        elif run is not None:
            bands.append(run); run = None
    if run:
        bands.append(run)
    # 底部文本行带（跟注按钮：两行——标签+金额）
    low = [b for b in bands if b[1] > 40]
    if not low:
        return None
    return len(low), [(a, b, b - a + 1) for a, b in low]


for fp in files:
    if not fp.name.endswith("_full.png"):
        continue
    try:
        w, h, rgba = decode_png(fp.read_bytes())
    except Exception:
        continue
    if (w, h) != (1876, 1400):
        continue
    got = extract(fp.name)
    if got:
        key = got[0]
        results.setdefault(key, []).append(fp.name)
print(f"文本行带分布: {[(k, len(v)) for k, v in sorted(results.items())]}")
for k in sorted(results):
    print(f"  {k} 行带 ({len(results[k])} 帧): {[x[:30] for x in results[k][:5]]}")
