"""P29-B — 单卡匹配调试：corner ROI 内容 / 墨量 / NCC 分数。"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402
from poker_decision_ai.vision2.pixel.realclient.coinpoker_herocards1876 import (  # noqa: E402
    _corner_mask, _match_glyph, _load_templates, HEROCARDS_CARD1,
    HEROCARDS_CARD2, RANK_OFF, SUIT_OFF)

EV = Path("data/vision2/coinpoker_evidence")


def load(fp):
    w, h, rgba = decode_png((EV / fp).read_bytes())
    return np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)


prep = _load_templates()
arr = load("8bf82f134651db0add626be0304a2ff1.png")
# card1 rank corner
r_ink = _corner_mask(arr, HEROCARDS_CARD1, RANK_OFF)
print("card1 rank ink sum:", int(r_ink.sum()), "shape:", r_ink.shape)
ch, sc, mg = _match_glyph(r_ink, prep, None)
print(f"card1 rank match: ch={ch} score={sc} margin={mg}")
s_ink = _corner_mask(arr, HEROCARDS_CARD1, SUIT_OFF)
print("card1 suit ink sum:", int(s_ink.sum()))
ch2, sc2, mg2 = _match_glyph(s_ink, prep, None)
print(f"card1 suit match: ch={ch2} score={sc2} margin={mg2}")
# 模板尺寸
print("模板尺寸:", {k: [(t.shape) for t, _, _ in v] for k, v in prep.items()})
