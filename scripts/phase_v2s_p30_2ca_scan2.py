"""P30-2C-A Phase 1b — 修正阈值重扫 + 分类抽样（跟注/加注/过牌/红盖牌/无面板）。"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402

EV = Path("data/vision2/coinpoker_evidence")
files = sorted(EV.glob("*.png"))

states = {"red_big": [], "orange": [], "green": [], "gray_text": [],
          "none": []}


def classify(fid):
    w, h, rgba = decode_png((EV / fid).read_bytes())
    arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
    roi = arr[1150:1400, 1250:1876]
    r = roi[..., 0].astype(np.int16); g = roi[..., 1].astype(np.int16)
    b = roi[..., 2].astype(np.int16)
    red = ((r > 110) & (r > g + 40) & (r > b + 40)).sum()
    green = ((g > 90) & (g > r + 25) & (g > b + 25)).sum()
    orange = ((r > 140) & (g > 60) & (g < 190) & (b < 130) & (r > g + 20)).sum()
    l = r * 0.299 + g * 0.587 + b * 0.114
    white = ((l > 150) & (np.abs(r - g) < 40) & (np.abs(g - b) < 40)).sum()
    return red, green, orange, white


for fp in files:
    if not fp.name.endswith("_full.png"):
        continue
    try:
        w, h, rgba = decode_png(fp.read_bytes())
    except Exception:
        continue
    if (w, h) != (1876, 1400):
        continue
    red, green, orange, white = classify(fp.name)
    if red > 5000:
        states["red_big"].append(fp.name)
    elif orange > 5000:
        states["orange"].append(fp.name)
    elif green > 5000:
        states["green"].append(fp.name)
    elif white > 2000:
        states["gray_text"].append(fp.name)
    else:
        states["none"].append(fp.name)

total = sum(len(v) for v in states.values())
print(f"总 {total} 帧")
for k, v in states.items():
    print(f"  {k}: {len(v)}  例: {[x[:30] for x in v[:4]]}")
