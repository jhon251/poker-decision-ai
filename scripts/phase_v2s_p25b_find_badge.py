"""P25-B — folded 徽章模板提取 + per-seat 检测测试。"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402

EV = Path("data/vision2/coinpoker_evidence")


def main() -> int:
    # 帧01 S5 盖牌徽章（P25-A 观察 ~1550-1615, 915-945）
    w, h, rgba = decode_png((EV /
        "BUTTON_CHANGED_167_1787992682749_full.png").read_bytes())
    arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
    # 扫描 S5 座牌盒左上找青/蓝绿徽章
    roi = arr[905:955, 1530:1650]
    r = roi[..., 0].astype(np.int16)
    g = roi[..., 1].astype(np.int16)
    b = roi[..., 2].astype(np.int16)
    # 青/蓝绿徽章：g 高、r 低、b 中
    teal = (g > 80) & (g > r + 20) & (b > r)
    ys, xs = np.where(teal)
    if len(ys) < 20:
        print("未找到青绿徽章像素")
        return 1
    print(f"青绿徽章: bbox=({1530+xs.min()},{905+ys.min()},"
          f"{xs.max()-xs.min()+1},{ys.max()-ys.min()+1}) n={len(ys)}")
    # 白字（徽章内）
    l = r * 0.299 + g * 0.587 + b * 0.114
    white = (l > 200)
    wyn = int(white[ys.min():ys.max()+1, xs.min():xs.max()+1].sum())
    print(f"  内白字像素: {wyn}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
