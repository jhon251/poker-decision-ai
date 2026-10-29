"""P25-C — 细节测量：
1) 过牌 tag 颜色（vs 盖牌 tag / equity pill）
2) disc: 玻璃暗底覆盖 vs 狼毛灰（f06 S1 负例）glass coverage
3) allin: white faces 验证 + band 宽度
"""
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


def stats(arr, bx, by, bw, bh, tag):
    roi = arr[by:by + bh, bx:bx + bw]
    r = roi[..., 0].astype(np.int16); g = roi[..., 1].astype(np.int16)
    b = roi[..., 2].astype(np.int16)
    l = r * 0.299 + g * 0.587 + b * 0.114
    # 1) 过牌 green: 采样多点
    print(f"[{tag}]")
    for dy in (5, bh // 2, bh - 5):
        for dx in (5, bw // 2, bw - 5):
            v = roi[dy, dx][:3]
            print(f"    px({bx+dx},{by+dy}) rgb=({v[0]},{v[1]},{v[2]})")
    gray = ((l >= 90) & (l <= 140) & (np.abs(r - g) < 18)
            & (np.abs(g - b) < 18))
    glass = l < 45
    print(f"    gray={int(gray.sum())} glass({tag})={float(glass.mean()):.2f}")


# 1) 过牌 tag（f02 S1, abs 约 142-242, 327-367）
a = load("POT_CHANGED_12_1787995290794_full.png")
stats(a, 145, 330, 100, 36, "pass_tag_f02S1")
# 对照：盖牌 tag（f02 S2 正例）
stats(a, 816, 211, 103, 40, "fold_tag_f02S2")

# 2) disc: f05 S3 (1537,333,208,60) vs f06 S1 wolf (144,333,209,60)
b = load("POT_CHANGED_139_1787992592538_full.png")
stats(b, 1531, 333, 208, 62, "disc_f05S3")
c = load("BUTTON_CHANGED_598_1787993753587_full.png")
stats(c, 144, 333, 209, 62, "wolf_f06S1")
# 其它负例同位置 glass
for fr, fp, seat, off in (
        ("f03 S6", "POT_CHANGED_102_1787992505794_full.png", "S6", (822, 1149)),
        ("f05 S5", "POT_CHANGED_139_1787992592538_full.png", "S5", (1537, 891))):
    d = load(fp)
    stats(d, off[0], off[1] + 8, 209, 62, f"{fr} off+8")
