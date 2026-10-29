"""P30-2C-A — pc143 (2,000 green) + pc177 (check) 调试。"""
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
REC = ToCallRecognizer1876()
ref2k = np.load(REF / "amt_2000.npy").astype(np.float64)
ref124 = np.load(REF / "amt_124163.npy").astype(np.float64)

for fid, tag in [("POT_CHANGED_143_1787992603846_full.png", "pc143"),
                 ("POT_CHANGED_177_1787992718375_full.png", "pc177")]:
    w, h, rgba = decode_png((EV / fid).read_bytes())
    arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
    x0, y0, x1, y1 = ACTION_PANEL_ROI
    panel = arr[y0:y1, x0:x1]
    mask = _text_mask(panel)
    bands = REC._row_bands(mask)
    a2, b2 = bands[-1]
    groups = REC._col_groups(mask, a2, b2)
    print(f"[{tag}] bands={bands}")
    for ga, gb in groups:
        tight = REC._tight_amount(mask, a2, b2, ga, gb)
        if tight is None:
            continue
        print(f"  group({ga},{gb}) tight={tight.shape} "
              f"NCC2k={_ncc(tight, ref2k):.4f} "
              f"NCC124={_ncc(tight, ref124):.4f}")
