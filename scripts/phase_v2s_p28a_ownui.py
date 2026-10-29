"""P28-A — own-seat UI 对象核查：S6 (hero 约定座) 周边独特 UI。
候选对象：表情反应按钮（💬/😎 pill）、计时器徽章、等待盲注/开始下一局 按钮。
裁剪对比：hero 座 vs 其它座。"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402
from poker_decision_ai.vision2_platform.png_encode import encode_rgba_png  # noqa: E402

EV = Path("data/vision2/coinpoker_evidence")
OUT = Path("data/vision2/p28a_crops")
OUT.mkdir(parents=True, exist_ok=True)


def zoom(fp, x0, y0, x1, y1, name, k=3):
    w, h, rgba = decode_png((EV / fp).read_bytes())
    arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
    sub = arr[y0:y1, x0:x1]
    big = np.repeat(np.repeat(sub, k, axis=0), k, axis=1)
    (OUT / f"{name}.png").write_bytes(encode_rgba_png(
        big.shape[1], big.shape[0], big.tobytes()))
    print(name, big.shape[1], "x", big.shape[0])


# f01: S6 座与左下 emoji pill
zoom("BUTTON_CHANGED_167_1787992682749_full.png", 600, 1050, 1200, 1400,
     "z_f01_S6_ownui", 2)
# f01 左下角控件区（timer⊞/💬/😎）
zoom("BUTTON_CHANGED_167_1787992682749_full.png", 0, 1150, 500, 1400,
     "z_f01_bottomleft", 2)
# 用户空帧：S6 周边（💬😎 + timer 30）
zoom("42b553c9468d1d1e55b9801bebad7b4f.png", 600, 1050, 1200, 1400,
     "z_42b_S6_ownui", 2)
# 用户空帧右下：等待盲注/开始下一局
zoom("42b553c9468d1d1e55b9801bebad7b4f.png", 1250, 1100, 1876, 1400,
     "z_42b_bottomright", 2)
