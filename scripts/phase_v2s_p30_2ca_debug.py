"""P30-2C-A — debug：live amount band vs ref 掩码（为何 NCC 未命中）。"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402
from poker_decision_ai.vision2.pixel.realclient.coinpoker_to_call import (  # noqa: E402
    ToCallRecognizer1876, _text_mask, _ncc, ACTION_PANEL_ROI)

EV = Path("data/vision2/coinpoker_evidence")
REF = Path("data/vision2/coinpoker_assets_v1876tocall")

ref124 = np.load(REF / "amt_124163.npy").astype(np.float64)
ref2k = np.load(REF / "amt_2000.npy").astype(np.float64)

w, h, rgba = decode_png((EV / "POT_CHANGED_114_1787992536082_full.png")
                        .read_bytes())
arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
x0, y0, x1, y1 = ACTION_PANEL_ROI
panel = arr[y0:y1, x0:x1]
mask = _text_mask(panel)
REC = ToCallRecognizer1876()
bands = REC._row_bands(mask)
print("bands:", bands)
if bands:
    a2, b2 = bands[-1]
    print(f"lower band: a={a2} b={b2}")
    sub = mask[a2:b2 + 1]
    ys = np.where(sub.sum(axis=1) > 0)[0]
    row_mask = sub[ys.min():ys.max() + 1]
    print(f"row_mask shape: {row_mask.shape} vs ref124 {ref124.shape}")
    print(f"NCC vs ref124: {_ncc(row_mask, ref124):.4f}")
    print(f"NCC vs ref2k:  {_ncc(row_mask, ref2k):.4f}")
    # 列投影
    cold = mask[a2:b2 + 1].sum(axis=0)
    print("cold profile:", [int(v) for v in cold[::5]])
