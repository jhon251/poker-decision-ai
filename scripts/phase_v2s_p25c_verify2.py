"""P25-C — 临界负例目视核查：f06 S1 TAG ROI（discGray=199）
与 f02 S1 pill 区（pillRun=18）。"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402
from poker_decision_ai.vision2_platform.png_encode import encode_rgba_png  # noqa: E402

EV = Path("data/vision2/coinpoker_evidence")
OUT = Path("data/vision2/p25c_crops")


def zoom(fp, x0, y0, x1, y1, name, k=4):
    w, h, rgba = decode_png((EV / fp).read_bytes())
    arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
    sub = arr[y0:y1, x0:x1]
    big = np.repeat(np.repeat(sub, k, axis=0), k, axis=1)
    (OUT / f"{name}.png").write_bytes(
        encode_rgba_png(big.shape[1], big.shape[0], big.tobytes()))
    print(name, big.shape[1], "x", big.shape[0])


# f06 S1 tag ROI (138,325,104,41) 及周边
zoom("BUTTON_CHANGED_598_1787993753587_full.png", 100, 280, 400, 430,
     "z_f06_S1_tagroi", 4)
# f05 S3 disc badge（正例）
zoom("POT_CHANGED_139_1787992592538_full.png", 1450, 280, 1720, 420,
     "z_f05_S3_disc_zone", 4)
# f02 S1 pill 区（负例 pillRun=18）
zoom("POT_CHANGED_12_1787995290794_full.png", 105, 160, 405, 480,
     "z_f02_S1_pillzone", 3)
