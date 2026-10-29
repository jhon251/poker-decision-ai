"""P29-B — 实测验证角标 offset 候选项（对 3 GT 帧每卡 rank/suit 直接预测）。"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402
from poker_decision_ai.vision2.pixel.realclient.coinpoker_herocards1876 import (  # noqa: E402
    _load_templates, _match_glyph)

EV = Path("data/vision2/coinpoker_evidence")
C1 = (813, 1011, 121, 178)
C2 = (837, 1011, 122, 178)
prep = _load_templates()


def load(fp):
    w, h, rgba = decode_png((EV / fp).read_bytes())
    return np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)


def glyph(arr, card, dx, dy, w, h):
    roi = arr[card[1] + dy:card[1] + dy + h, card[0] + dx:card[0] + dx + w]
    r = roi[..., 0].astype(np.int16); g = roi[..., 1].astype(np.int16)
    b = roi[..., 2].astype(np.int16)
    l = r * 0.299 + g * 0.587 + b * 0.114
    return (l < 120).astype(np.float64)


# 候选 offset：
# card1 rank: (8,5,55,58) suit: (10,60,42,60)
# card2 rank: (8,18,55,45) suit: (10,72,42,55)  (右卡下移/旋转)
CAND = {
    "c1_rank": (8, 5, 55, 58), "c1_suit": (10, 60, 42, 60),
    "c2_rank": (8, 18, 50, 45), "c2_suit": (10, 70, 40, 55),
}
for fp, gt in [
        ("8bf82f134651db0add626be0304a2ff1.png", ("5c", "3c")),
        ("POT_CHANGED_110_1787992526057_full.png", ("6c", "3h")),
        ("POT_CHANGED_102_1787992505794_full.png", ("9c", "3d"))]:
    arr = load(fp)
    print(f"--- {fp[:28]} GT={gt} ---")
    for slot, card in (("c1", C1), ("c2", C2)):
        for part, off in (("rank", CAND[f"{slot}_rank"]),
                          ("suit", CAND[f"{slot}_suit"])):
            g = glyph(arr, card, *off)
            if g.sum() < 15:
                print(f"  {slot}_{part}: ink={int(g.sum())} (<15)")
                continue
            ch, sc, mg = _match_glyph(g, prep, None)
            print(f"  {slot}_{part}: ink={int(g.sum())} -> ch={ch} "
                  f"score={sc:.3f} margin={mg:.3f}")
