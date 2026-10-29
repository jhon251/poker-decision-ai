"""P18-C — 真实 1876 stack 金额 RGB/HSV 可分离性分析（READ-ONLY）。

定位底部带 stack 金额文本（暗金）与白卡面/灰名牌，测量各自 RGB/HSV，
判断是否可用颜色可靠分离（供 occupied 独立视觉证据判定）。
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402

EV = Path("data/vision2/coinpoker_evidence")


def _rgb_to_hsv(r, g, b):
    r, g, b = r / 255.0, g / 255.0, b / 255.0
    mx, mn = max(r, g, b), min(r, g, b)
    d = mx - mn
    if mx == 0:
        s = 0.0
    else:
        s = d / mx
    if d == 0:
        h = 0.0
    elif mx == r:
        h = 60 * (((g - b) / d) % 6)
    elif mx == g:
        h = 60 * ((b - r) / d + 2)
    else:
        h = 60 * ((r - g) / d + 4)
    return h, s * 100, mx * 100


def main() -> int:
    name = "BUTTON_CHANGED_167_1787992682749_full.png"
    w, h, rgba = decode_png((EV / name).read_bytes())
    arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
    r = arr[..., 0].astype(np.int16)
    g = arr[..., 1].astype(np.int16)
    b = arr[..., 2].astype(np.int16)
    l = r * 0.299 + g * 0.587 + b * 0.114
    # 底部带 y1200-1400 各类像素
    band_r = r[1200:1400]; band_g = g[1200:1400]; band_b = b[1200:1400]
    band_l = l[1200:1400]
    # 暗金候选：r>110, g>80, r>=g, g>=b+25, b<120（比 gray 多 g>b 显著）
    dark_gold = (band_r > 110) & (band_g > 80) & (band_r >= band_g) & \
                (band_g >= band_b + 25) & (band_b < 120)
    gray = (band_r > 130) & (band_r < 200) & (np.abs(band_r - band_g) < 25) & \
           (np.abs(band_g - band_b) < 25)
    white = band_l > 200
    for label, m in [("暗金候选", dark_gold), ("灰名牌", gray), ("白", white)]:
        idx = np.where(m)
        n = len(idx[0])
        if n == 0:
            print(f"{label}: 0 像素")
            continue
        rr = band_r[idx]; gg = band_g[idx]; bb = band_b[idx]
        hh, ss, vv = _rgb_to_hsv(rr.mean(), gg.mean(), bb.mean())
        print(f"{label}: n={n}  R={rr.mean():.0f} G={gg.mean():.0f} "
              f"B={bb.mean():.0f}  H={hh:.0f} S={ss:.0f}% V={vv:.0f}")
        print(f"   R范围={rr.min()}-{rr.max()} G={gg.min()}-{gg.max()} "
              f"B={bb.min()}-{bb.max()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
