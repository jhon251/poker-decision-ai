"""P16-C — GT 源截图 004d23eb（1848×1388, K♠9♥）hero 卡实测（READ-ONLY）。

确认 GT 截图 hero 卡 ROI/rank/suit 几何与尺度，供模板提取判定。
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402


def _ink_bbox(roi, mode):
    r = roi[..., 0].astype(np.int16)
    g = roi[..., 1].astype(np.int16)
    b = roi[..., 2].astype(np.int16)
    luma = r * 0.299 + g * 0.587 + b * 0.114
    if mode == "black":
        m = luma < 100
    elif mode == "red":
        m = (r > 100) & (r > g + 40) & (r > b + 40)
    ys, xs = np.where(m)
    if len(ys) < 10:
        return None
    return (xs.min(), ys.min(), xs.max() - xs.min() + 1, ys.max() - ys.min() + 1)


def main() -> int:
    for name in ["004d23eb6e87077fbb4377d5fb63913f.png",
                 "f4652edf931adbdb56579e4c9eb5e971.png"]:
        p = Path("data2") / name
        if not p.exists():
            print(f"缺: {name}")
            continue
        w, h, rgba = decode_png(p.read_bytes())
        arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
        print(f"\n===== {name} ({w}x{h}) =====")
        # hero 区 (802-1043, 1005-1180)
        hc = arr[1005:1180, 802:1043]
        r = hc[..., 0].astype(np.int16)
        g = hc[..., 1].astype(np.int16)
        b = hc[..., 2].astype(np.int16)
        luma = r * 0.299 + g * 0.587 + b * 0.114
        white = (luma > 200).sum()
        red = ((r > 100) & (r > g + 40) & (r > b + 40)).sum()
        print(f"  hero 区白卡={int(white)} 红={int(red)}")
        for ci, (cx0, cx1) in enumerate([(802, 922), (922, 1043)]):
            card = arr[1005:1180, cx0:cx1]
            rank = card[0:70, 0:70]
            suit = card[50:130, 0:80]
            rb = _ink_bbox(rank, "red")
            bb = _ink_bbox(rank, "black")
            srb = _ink_bbox(suit, "red")
            sbb = _ink_bbox(suit, "black")
            print(f"  card{ci}(x{cx0}): rank_red={rb} rank_black={bb}")
            print(f"            suit_red={srb} suit_black={sbb}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
