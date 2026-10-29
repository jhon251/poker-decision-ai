"""P30-2C-A — check (pc177) wide group 结构检查：为何被当金额。
group(233,295)=63px —— 具体字符构成。"""
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

w, h, rgba = decode_png((EV / "POT_CHANGED_177_1787992718375_full.png")
                        .read_bytes())
arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
x0, y0, x1, y1 = ACTION_PANEL_ROI
panel = arr[y0:y1, x0:x1]
mask = _text_mask(panel)
bands = REC._row_bands(mask)
a2, b2 = bands[-1]
groups = REC._col_groups(mask, a2, b2)
print(f"bands={bands} groups={groups}")
for ga, gb in groups:
    sub = mask[a2:b2 + 1, ga:gb + 1]
    ys = np.where(sub.sum(axis=1) > 0)[0]
    print(f"  group({ga},{gb}) w={gb-ga+1} rows={ys.max()-ys.min()+1} "
          f"ink={int(sub.sum())}")
