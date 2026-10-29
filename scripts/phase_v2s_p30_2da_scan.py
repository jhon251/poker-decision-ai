"""P30-2D-A — 搜索 current_bet 视觉对象：
1) 动作面板中间金额（8bf8 滑块 4,000）——语义 = raise-to/bet 预设；
2) 桌中央 pot 上方/周围金额；
3) 任意「当前下注额」独立显示。
裁剪对照：f01（无行动）/ 8bf8（行动中）/ pc110（跟注中）。"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402
from poker_decision_ai.vision2_platform.png_encode import encode_rgba_png  # noqa: E402

EV = Path("data/vision2/coinpoker_evidence")
OUT = Path("data/vision2/p30_2da_crops")
OUT.mkdir(parents=True, exist_ok=True)


def zoom(fp, x0, y0, x1, y1, name, k=2):
    w, h, rgba = decode_png((EV / fp).read_bytes())
    arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
    y1 = min(y1, h)
    sub = arr[y0:y1, x0:x1]
    if sub.shape[0] == 0 or sub.shape[1] == 0:
        print(name, "empty")
        return
    big = np.repeat(np.repeat(sub, k, axis=0), k, axis=1)
    (OUT / f"{name}.png").write_bytes(encode_rgba_png(
        big.shape[1], big.shape[0], big.tobytes()))
    print(name, big.shape[1], "x", big.shape[0])


# 8bf8 动作面板全带（含滑块 4,000 + 33%/50%/75%/最大 预设 + 2 行按钮）
zoom("8bf82f134651db0add626be0304a2ff1.png", 1250, 1080, 1876, 1400,
     "z_8bf8_panel_full", 2)
# pc110 面板（跟注 2,000 / 跟注任何 / X/F）
zoom("POT_CHANGED_110_1787992526057_full.png", 1250, 1150, 1876, 1400,
     "z_pc110_panel_full", 2)
# 桌中央（pot 胶囊 — 检查 pot 周围是否有当前下注额）
zoom("POT_CHANGED_110_1787992526057_full.png", 700, 500, 1200, 750,
     "z_pc110_center", 2)
