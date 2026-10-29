"""P25-C — f05 S3 disc 徽章精确测量：badge 背景 / 白字位置。"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402

EV = Path("data/vision2/coinpoker_evidence")
fp = "POT_CHANGED_139_1787992592538_full.png"
w, h, rgba = decode_png((EV / fp).read_bytes())
arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)

# 逐像素分析 (1520,310)-(1700,380)
roi = arr[310:380, 1520:1700]
r = roi[..., 0].astype(np.int16); g = roi[..., 1].astype(np.int16)
b = roi[..., 2].astype(np.int16)
l = r * 0.299 + g * 0.587 + b * 0.114
print("luma hist: p1=%.0f p10=%.0f p50=%.0f p90=%.0f p99=%.0f max=%.0f" %
      (np.percentile(l, 1), np.percentile(l, 10), np.percentile(l, 50),
       np.percentile(l, 90), np.percentile(l, 99), l.max()))
for th in (100, 120, 140, 160):
    m = l > th
    if m.sum() < 5:
        print(f"th={th}: {int(m.sum())} px")
        continue
    ys, xs = np.where(m)
    print(f"th={th}: n={int(m.sum())} bbox=({1520+xs.min()},{310+ys.min()},"
          f"{xs.max()-xs.min()+1},{ys.max()-ys.min()+1})")
# 采样文本中心
for (x, y) in [(1560, 340), (1580, 340), (1600, 340), (1620, 340)]:
    v = arr[y, x][:3]
    print(f"({x},{y}) rgb=({v[0]},{v[1]},{v[2]})")
# badge 暗背景估计
sub = arr[320:360, 1530:1660]
sl = sub[..., 0] * 0.299 + sub[..., 1] * 0.587 + sub[..., 2] * 0.114
print("badge zone luma mean=%.0f p25=%.0f" % (sl.mean(),
                                              np.percentile(sl, 25)))
