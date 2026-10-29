"""P25-B.1 — Status badge 视觉表征探针（READ-ONLY）。

对 4 类状态（folded/away/disconnected/allin）从真实 GT 帧采样：
- 区域 RGB 分布（区分背景/主体/白字）；
- HSV 统计；
- bbox 尺寸。
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402

EV = Path("data/vision2/coinpoker_evidence")


def _hsv(r, g, b):
    r, g, b = r / 255.0, g / 255.0, b / 255.0
    mx, mn = max(r, g, b), min(r, g, b)
    d = mx - mn
    s = d / mx if mx else 0.0
    if d == 0:
        h = 0.0
    elif mx == r:
        h = 60 * (((g - b) / d) % 6)
    elif mx == g:
        h = 60 * ((b - r) / d + 2)
    else:
        h = 60 * ((r - g) / d + 4)
    return h, s * 100, mx * 100


def probe(fid, x0, y0, x1, y1, label):
    w, h, rgba = decode_png((EV / fid).read_bytes())
    arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
    roi = arr[y0:y1, x0:x1].reshape(-1, 4)
    r = roi[..., 0].astype(np.int16)
    g = roi[..., 1].astype(np.int16)
    b = roi[..., 2].astype(np.int16)
    l = r * 0.299 + g * 0.587 + b * 0.114
    print(f"\n[{label}] ({x0},{y0},{x1},{y1})")
    # 亮文字（白/浅）
    bright_m = l > 170
    if bright_m.sum():
        hh, ss, vv = _hsv(r[bright_m].mean(), g[bright_m].mean(),
                          b[bright_m].mean())
        print(f"  亮文字: n={int(bright_m.sum())} "
              f"RGB=({r[bright_m].mean():.0f},{g[bright_m].mean():.0f},"
              f"{b[bright_m].mean():.0f}) HSV=({hh:.0f},{ss:.0f},{vv:.0f})")
    # 中亮（徽章主体：非暗非最亮）
    mid_m = (l > 60) & (l <= 170)
    if mid_m.sum():
        hh, ss, vv = _hsv(r[mid_m].mean(), g[mid_m].mean(), b[mid_m].mean())
        print(f"  徽章主体(中亮): n={int(mid_m.sum())} "
              f"RGB=({r[mid_m].mean():.0f},{g[mid_m].mean():.0f},"
              f"{b[mid_m].mean():.0f}) HSV=({hh:.0f},{ss:.0f},{vv:.0f})")
    # 暗背景
    dark_m = l <= 60
    print(f"  暗背景: n={int(dark_m.sum())}")


probe("BUTTON_CHANGED_167_1787992682749_full.png", 1530, 905, 1650, 950,
      "帧01 S5 folded 盖牌徽章区")
probe("BUTTON_CHANGED_167_1787992682749_full.png", 1600, 290, 1720, 350,
      "帧01 S3 away 暂离+咖啡")
probe("POT_CHANGED_139_1787992592538_full.png", 1710, 120, 1830, 200,
      "帧09 S3 disconnected 断开")
probe("POT_CHANGED_102_1787992505794_full.png", 820, 70, 1030, 220,
      "帧05 S2 allin 亮牌+equity")
