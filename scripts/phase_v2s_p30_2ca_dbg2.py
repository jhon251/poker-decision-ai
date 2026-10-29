"""P30-2C-A — debug：各状态列组宽度分布（区分 金额 vs 过牌）。"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402
from poker_decision_ai.vision2.pixel.realclient.coinpoker_to_call import (  # noqa: E402
    ToCallRecognizer1876, _text_mask, ACTION_PANEL_ROI)

EV = Path("data/vision2/coinpoker_evidence")
REC = ToCallRecognizer1876()

for fid, tag in [
        ("POT_CHANGED_114_1787992536082_full.png", "pc114 amo"),
        ("POT_CHANGED_110_1787992526057_full.png", "pc110 amo"),
        ("POT_CHANGED_177_1787992718375_full.png", "pc177 chk"),
        ("POT_CHANGED_229_1787992854988_full.png", "pc229 chk"),
        ("POT_CHANGED_143_1787992603846_full.png", "pc143 amo(green)")]:
    w, h, rgba = decode_png((EV / fid).read_bytes())
    arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
    x0, y0, x1, y1 = ACTION_PANEL_ROI
    panel = arr[y0:y1, x0:x1]
    mask = _text_mask(panel)
    bands = REC._row_bands(mask)
    if not bands:
        print(f"[{tag}] no bands")
        continue
    a2, b2 = bands[-1]
    cold = mask[a2:b2 + 1].sum(axis=0)
    runs, run = [], None
    for i, v in enumerate(cold):
        if v >= 1 and run is None:
            run = [i, i]
        elif v >= 1 and run is not None:
            run[1] = i
        elif run is not None:
            runs.append(run); run = None
    if run:
        runs.append(run)
    merged = []
    for rr in runs:
        if merged and rr[0] - merged[-1][1] >= 8:
            merged.append(rr)
        elif merged:
            merged[-1][1] = rr[1]
        else:
            merged.append(rr)
    widths = [b_ - a_ + 1 for a_, b_ in merged if b_ - a_ + 1 >= 20]
    print(f"[{tag}] last band y{a2}-{b2} groups widths={widths}")
