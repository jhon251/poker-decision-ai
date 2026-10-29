"""P29-B — debug：绝对窗 ink + match 分（8bf8）。"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402
from poker_decision_ai.vision2.pixel.realclient.coinpoker_herocards1876 import (  # noqa: E402
    RANK_WINDOWS, SUIT_WINDOWS, _corner_mask, _match_glyph, _load_templates)

EV = Path("data/vision2/coinpoker_evidence")


def load(fp):
    w, h, rgba = decode_png((EV / fp).read_bytes())
    return np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)


prep = _load_templates()
arr = load("8bf82f134651db0add626be0304a2ff1.png")
for key in ("card1", "card2"):
    r_ink = _corner_mask(arr, RANK_WINDOWS[key])
    s_ink = _corner_mask(arr, SUIT_WINDOWS[key])
    print(f"{key}: rank_ink={int(r_ink.sum())} suit_ink={int(s_ink.sum())}")
    ch, sc, mg = _match_glyph(r_ink, prep, None)
    print(f"  rank -> {ch} score={sc} margin={mg}")
    ch2, sc2, mg2 = _match_glyph(s_ink, prep, None)
    print(f"  suit -> {ch2} score={sc2} margin={mg2}")
