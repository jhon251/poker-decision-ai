"""P25-B.3 — debug：f01 S5 anchor 周围 mask 直接统计。"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402

EV = Path("data/vision2/coinpoker_evidence")
fp = "BUTTON_CHANGED_167_1787992682749_full.png"
w, h, rgba = decode_png((EV / fp).read_bytes())
arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
r = arr[..., 0].astype(np.int16); g = arr[..., 1].astype(np.int16)
b = arr[..., 2].astype(np.int16)
l = r * 0.299 + g * 0.587 + b * 0.114
# tag 中心 abs ≈ (1580,910)
for (x, y) in [(1535, 895), (1580, 910), (1600, 890), (1560, 920)]:
    print(f"({x},{y}) l={l[y, x]:.1f} r={r[y, x]} g={g[y, x]} b={b[y, x]}")
    cond = ((l[y, x] > 35) & (l[y, x] < 75) & (g[y, x] > r[y, x] + 15)
            & (abs(int(g[y, x]) - int(b[y, x])) <= 18))
    print(f"    mask={bool(cond)}")
# ROI 总统计
x0, x1 = 1418, 1818
y0, y1 = 708, 1128
rl = l[y0:y1, x0:x1]; rr = r[y0:y1, x0:x1]
rg = g[y0:y1, x0:x1]; rb = b[y0:y1, x0:x1]
bg = ((rl > 35) & (rl < 75) & (rg > rr + 15) & (np.abs(rg - rb) <= 18))
print("bg.sum()=", int(bg.sum()))
ys, xs = np.where(bg)
if len(xs):
    print("bg bbox=(", x0 + xs.min(), y0 + ys.min(), xs.max() - xs.min() + 1,
          ys.max() - ys.min() + 1, ")")
