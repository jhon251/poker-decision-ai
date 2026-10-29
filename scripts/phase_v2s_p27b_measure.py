"""P27-B.0 — 用户帧 42b 逐座视觉标注（READ-ONLY 测量）。

已知 GT 参考结构（P27-A）：
- OCCUPIED = 座牌盒（暗面板+name+gold stack）+ 头像圈（可选）
- EMPTY = 【待本帧实证】——目视见 S4 区「空的」标签
输出：每座暗面板占比 / name ink / gold / felt / 头像圈（maroon）/ 空标签。"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402
from poker_decision_ai.vision2_platform.png_encode import encode_rgba_png  # noqa: E402

EV = Path("data2/42b553c9468d1d1e55b9801bebad7b4f.png")
OUT = Path("data/vision2/p27b_crops")
w, h, rgba = decode_png(EV.read_bytes())
arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)

# P26-B stack ROI（座牌盒内部件区域参考——仅测量，不推导）
STACK = {
    "S1": (165, 413, 360, 485), "S2": (845, 298, 1035, 370),
    "S3": (1520, 413, 1715, 485), "S4": (125, 978, 320, 1050),
    "S5": (1560, 978, 1745, 1050), "S6": (838, 1236, 1040, 1308),
}


def sig(seat):
    x0, y0, x1, y1 = STACK[seat]
    px0, py0 = max(0, x0 - 10), max(0, y0 - 58)
    px1, py1 = min(w, x1 + 10), min(h, y1 + 12)
    roi = arr[py0:py1, px0:px1]
    r = roi[..., 0].astype(np.int16); g = roi[..., 1].astype(np.int16)
    b = roi[..., 2].astype(np.int16)
    l = r * 0.299 + g * 0.587 + b * 0.114
    dark_panel = ((l > 12) & (l < 62) & (np.abs(r - g) < 30)
                  & (np.abs(g - b) < 30)).mean()
    felt = (g > r + 20).mean()
    # name 行
    ln = arr[py0:y0 - 6, x0:x1]
    ln2 = ln[..., 0] * 0.299 + ln[..., 1] * 0.587 + ln[..., 2] * 0.114
    rn = ln[..., 0].astype(np.int16); gn = ln[..., 1].astype(np.int16)
    bn = ln[..., 2].astype(np.int16)
    name_ink = int(((ln2 > 110) & (np.abs(rn - gn) < 28)
                    & (np.abs(gn - bn) < 28)).sum())
    # gold
    rs = arr[y0:y1, x0:x1]
    rg = rs[..., 0].astype(np.int16); gg = rs[..., 1].astype(np.int16)
    bg2 = rs[..., 2].astype(np.int16)
    ls = rg * 0.299 + gg * 0.587 + bg2 * 0.114
    mdn = np.median(ls)
    gold = int(((rg > bg2 + 30) & (gg >= bg2 + 18) & (rg >= gg * 0.9)
                & (ls > mdn * 1.6)).sum())
    return {"panel": round(float(dark_panel), 2), "felt": round(float(felt), 2),
            "name": name_ink, "gold": gold}


print(f"{'seat':4s} {'panel':>6s} {'felt':>6s} {'name':>5s} {'gold':>5s}")
for seat in sorted(STACK):
    print(f"{seat:4s} {sig(seat)['panel']:6.2f} {sig(seat)['felt']:6.2f} "
          f"{sig(seat)['name']:5d} {sig(seat)['gold']:5d}")

# 空标签「空的」定位：S4 附近（~223,892 估算）低亮 text
# 扫描 x 100-400, y 820-980 找中性灰白字
sub = arr[820:980, 100:400]
l = sub[..., 0] * 0.299 + sub[..., 1] * 0.587 + sub[..., 2] * 0.114
r = sub[..., 0].astype(np.int16); g = sub[..., 1].astype(np.int16)
b = sub[..., 2].astype(np.int16)
gray = ((l > 90) & (l < 180) & (np.abs(r - g) < 25) & (np.abs(g - b) < 25))
ys, xs = np.where(gray)
if len(xs):
    print(f"\n「空的」灰字候选: n={len(xs)} bbox=({100+xs.min()},{820+ys.min()},"
          f"{xs.max()-xs.min()+1},{ys.max()-ys.min()+1})")
# 放大 S4 区
sub2 = arr[780:1050, 40:440]
big = np.repeat(np.repeat(sub2, 2, axis=0), 2, axis=1)
(OUT / "z_user_S4_empty.png").write_bytes(encode_rgba_png(
    big.shape[1], big.shape[0], big.tobytes()))
print("saved z_user_S4_empty.png")
