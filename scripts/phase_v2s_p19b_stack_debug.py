"""P19-B — 调试：暗金掩码 vs 旧掩码 在各座区域的实际像素计数。"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402

EV = Path("data/vision2/coinpoker_evidence")
REGIONS = {
    "S1": (80, 290, 480, 510), "S2": (740, 170, 1060, 330),
    "S3": (1470, 170, 1840, 510), "S4": (0, 1180, 310, 1390),
    "S5": (1420, 1180, 1870, 1390), "S6": (470, 1180, 1080, 1390),
}


def main() -> int:
    name = "BUTTON_CHANGED_167_1787992682749_full.png"
    w, h, rgba = decode_png((EV / name).read_bytes())
    arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
    r = arr[..., 0].astype(np.int16)
    g = arr[..., 1].astype(np.int16)
    b = arr[..., 2].astype(np.int16)
    l = r * 0.299 + g * 0.587 + b * 0.114
    masks = {
        "bright(r150g110b130)": (r > 150) & (g > 110) & (b < 130) & (r >= g),
        "dim(r120g85)": (r > 120) & (g > 85) & (r >= g) & (g >= b - 30) & (l > 90),
        "darkgold(r110g80g>=b+25)": (r > 110) & (g > 80) & (r >= g) & (g >= b + 25),
    }
    for seat, (x0, y0, x1, y1) in REGIONS.items():
        print(f"{seat} region=({x0},{y0},{x1},{y1})")
        for mname, m in masks.items():
            cnt = int(m[y0:y1, x0:x1].sum())
            print(f"    {mname}: {cnt}")
        # 区域内最亮的像素颜色（看实际有什么）
        sub = arr[y0:y1, x0:x1]
        rr = sub[..., 0].astype(np.int16); gg = sub[..., 1].astype(np.int16)
        bb = sub[..., 2].astype(np.int16); ll = l[y0:y1, x0:x1]
        # 找 r>110 且 g 明显> b 的像素
        goldish = (rr > 110) & (gg > 80) & (rr >= gg) & (gg >= bb + 15)
        if goldish.sum() > 0:
            idx = np.where(goldish)
            print(f"    goldish(g>=b+15) n={len(idx[0])} "
                  f"R={rr[idx].mean():.0f} G={gg[idx].mean():.0f} "
                  f"B={bb[idx].mean():.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
