"""P30-2C-A — 2 行带帧抽样（8 个不同场景）目视读取跟注金额。"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402
from poker_decision_ai.vision2_platform.png_encode import encode_rgba_png  # noqa: E402

EV = Path("data/vision2/coinpoker_evidence")
OUT = Path("data/vision2/p30_2ca_crops")

SAMPLES = [
    ("POT_CHANGED_143_1787992603846_full.png", "s_pc143"),
    ("POT_CHANGED_16_1787995303501_full.png", "s_pc16b"),
    ("POT_CHANGED_177_1787992718375_full.png", "s_pc177"),
    ("POT_CHANGED_229_1787992854988_full.png", "s_pc229"),
    ("POT_CHANGED_252_1787992913615_full.png", "s_pc252"),
    ("POT_CHANGED_290_1787993011134_full.png", "s_pc290"),
    ("POT_CHANGED_332_1787993098695_full.png", "s_pc332"),
    ("BUTTON_CHANGED_598_1787993753587_full.png", "s_f06"),
]


def zoom(fp, name):
    w, h, rgba = decode_png((EV / fp).read_bytes())
    arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
    sub = arr[1150:1400, 1250:1876]
    big = np.repeat(np.repeat(sub, 2, axis=0), 2, axis=1)
    (OUT / f"{name}.png").write_bytes(encode_rgba_png(
        big.shape[1], big.shape[0], big.tobytes()))
    print(name)


for fp, name in SAMPLES:
    zoom(fp, name)
