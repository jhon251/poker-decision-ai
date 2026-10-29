"""P29-A — 极暗 DIM 英雄牌样本查找 + 无牌/牌背负例确认。"""
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


# 1) 全 1876 帧 global bg + hero card 区白量 → 找 "extreme DIM + cards" 帧
files = sorted(EV.glob("*.png"))
found = []
scanned = 0
for fp in files:
    if not fp.name.endswith("_full.png"):
        continue
    try:
        w, h, rgba = decode_png(fp.read_bytes())
    except Exception:
        continue
    if (w, h) != (1876, 1400):
        continue
    scanned += 1
    arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
    # global bg
    ll = arr[..., 0] * 0.299 + arr[..., 1] * 0.587 + arr[..., 2] * 0.114
    bg = float(np.median(ll))
    if bg < 9.0:                      # extreme DIM 候选
        z = arr[990:1195, 780:1090]
        zz = z[..., 0] * 0.299 + z[..., 1] * 0.587 + z[..., 2] * 0.114
        white = int((zz > 60).sum())   # DIM 下调低（>60 即可见）
        if white > 8000:
            found.append((fp.name, round(bg, 1), white))
print(f"scanned={scanned} extreme-DIM+cards={len(found)}")
for n, bg, wc in found[:20]:
    print(f"  {n}: bg={bg} white={wc}")

# 2) 无牌负例确认：f01/f02/f05/f06（hero 卡区应为暗，无白卡）
print("\n=== 无牌负例（GT 帧 hero 卡区）===")
for fp, tag in [("BUTTON_CHANGED_167_1787992682749_full.png", "f01"),
                ("POT_CHANGED_139_1787992592538_full.png", "f05")]:
    arr = load(fp)
    z = arr[990:1195, 780:1090]
    zz = z[..., 0] * 0.299 + z[..., 1] * 0.587 + z[..., 2] * 0.114
    white = int((zz > 150).sum())
    print(f"  {tag}: hero zone white(>150)={white}")
