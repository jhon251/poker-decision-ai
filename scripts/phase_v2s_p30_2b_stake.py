"""P30-2B — 精确 slot 修正：digit_0 定位（前后段排布校准）。

实测段：x167-169(w3)=0? x174-178(w5)=)  x182-184(w3)=?
从 11 段分析：x132-136=( x144-148=3 x152-156=2 x167-169=0? x174-178=)?
详情：1K/2K (320) → 1(69) K(83) /(90) 2(97) K(104) ((132) 3(144) 2(152)
0(167) )(174)... 但 ) 尾段 x182-184 是?? -> 需要目视确认窄段。"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402
from poker_decision_ai.vision2_platform.png_encode import encode_rgba_png  # noqa: E402

EV = Path("data/vision2/coinpoker_evidence")
OUT = Path("data/vision2/p30_2a_crops")


def zoom(fp, x0, y0, x1, y1, name, k=8):
    w, h, rgba = decode_png((EV / fp).read_bytes())
    arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
    sub = arr[y0:y1, x0:x1]
    big = np.repeat(np.repeat(sub, k, axis=0), k, axis=1)
    (OUT / f"{name}.png").write_bytes(encode_rgba_png(
        big.shape[1], big.shape[0], big.tobytes()))
    print(name, big.shape[1], "x", big.shape[0])


# stake 区全段放大（x60-200, y4-26）——确认 "0)" 排布
zoom("BUTTON_CHANGED_167_1787992682749_full.png", 60, 4, 200, 26,
     "z_f01_stake_hi", 8)
