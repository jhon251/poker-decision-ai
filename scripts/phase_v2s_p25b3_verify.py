"""P25-B.3 — 补充：dim 场景成因核查 + 最强负例内容目视。
f03/f04 是 POT_CHANGED 且有 show/equity overlay → 整桌变暗（全局 p50 15-17 vs 31）。
检查 f04 S1 tag ROI（teal=0.05 最强负例）内容像素；并生成裁剪图。
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402
from poker_decision_ai.vision2_platform.png_encode import encode_rgba_png  # noqa: E402

EV = Path("data/vision2/coinpoker_evidence")
OUT = Path("data/vision2/p25b3_crops")


def load(fp):
    w, h, rgba = decode_png((EV / fp).read_bytes())
    return np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)


def zoom(fp, bx, by, bw, bh, name, k=4):
    arr = load(fp)
    sub = arr[by:by + bh, bx:bx + bw]
    big = np.repeat(np.repeat(sub, k, axis=0), k, axis=1)
    (OUT / f"{name}.png").write_bytes(
        encode_rgba_png(big.shape[1], big.shape[0], big.tobytes()))
    print(f"{name}: {big.shape[1]}x{big.shape[0]}")


# 最强负例 f04 S1 tag ROI
zoom("POT_CHANGED_13_1787992279195_full.png", 138, 325, 104, 41,
     "z_f04_S1_neg")
# 负例 f01 S1（活跃）同 ROI
zoom("BUTTON_CHANGED_167_1787992682749_full.png", 138, 325, 104, 41,
     "z_f01_S1_neg")
# dim 正例 f03 S5 tag（对比正常正例）
zoom("POT_CHANGED_102_1787992505794_full.png", 1531, 891, 103, 40,
     "z_f03_S5_pos_dim")
# 正常正例 f02 S2 tag
zoom("POT_CHANGED_12_1787995290794_full.png", 816, 211, 103, 40,
     "z_f02_S2_pos")
