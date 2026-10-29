"""P25-A — Status badge 视觉表征测量（image-capable + 像素）。"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402

EV = Path("data/vision2/coinpoker_evidence")


def _scan(fid, x0, y0, x1, y1, label):
    w, h, rgba = decode_png((EV / fid).read_bytes())
    arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
    roi = arr[y0:y1, x0:x1]
    r = roi[..., 0].astype(np.int16)
    g = roi[..., 1].astype(np.int16)
    b = roi[..., 2].astype(np.int16)
    l = r * 0.299 + g * 0.587 + b * 0.114
    # 亮/灰文本（徽章文字）
    bright = (l > 140).astype(np.uint8)
    ys, xs = np.where(bright)
    if len(ys) < 10:
        print(f"{label}: 无亮文本")
        return
    # 徽章色（文字近似 + 背景）
    print(f"{label}: 亮文本 bbox=({x0+xs.min()},{y0+ys.min()},"
          f"{xs.max()-xs.min()+1},{ys.max()-ys.min()+1}) n={len(ys)}")
    # 徽章背景（暗/灰区域）
    px = roi[ys.min():ys.max()+1, xs.min():xs.max()+1].reshape(-1, 4)
    px = px[(px[..., 3] > 0)]
    print(f"  色: R={px[:,0].mean():.0f} G={px[:,1].mean():.0f} "
          f"B={px[:,2].mean():.0f}")


# 帧01：S5 盖牌（folded）徽章 ~(1560-1620, 915-945)；S3 暂离（away）~(1620-1700, 300-345)
_scan("BUTTON_CHANGED_167_1787992682749_full.png", 1540, 910, 1650, 950, "帧01 S5 folded(盖牌)")
_scan("BUTTON_CHANGED_167_1787992682749_full.png", 1600, 295, 1720, 350, "帧01 S3 away(暂离+咖啡)")
# 帧09：S3 断开（disconnected）~(1735-1800, 130-190)
_scan("POT_CHANGED_139_1787992592538_full.png", 1720, 125, 1820, 195, "帧09 S3 disconnected(断开)")
