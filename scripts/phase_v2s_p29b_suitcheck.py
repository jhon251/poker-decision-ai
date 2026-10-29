"""P29-B — 诚实评估：f03 c2 (3d) 用现有模板（c/h）匹配会得到什么？
目的：确认是否会把 d 误认成 h（若会——红黑校验可能兜底，但要测）。"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402
from poker_decision_ai.vision2.pixel.realclient.coinpoker_herocards1876 import (  # noqa: E402
    _load_templates, _match_glyph, _corner_mask, SUIT_OFFS,
    HEROCARDS_CARD2)

EV = Path("data/vision2/coinpoker_evidence")


def load(fp):
    w, h, rgba = decode_png((EV / fp).read_bytes())
    return np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)


prep = _load_templates()
print("模板:", {k: [(t.shape) for t, _, _ in v] for k, v in prep.items()})
arr = load("POT_CHANGED_102_1787992505794_full.png")
# c2 suit 角标（实测位置 833-875, 1080-1120）
roi = arr[1080:1120, 826:875]
r = roi[..., 0].astype(np.int16); g = roi[..., 1].astype(np.int16)
b = roi[..., 2].astype(np.int16)
l = r * 0.299 + g * 0.587 + b * 0.114
print(f"f03 c2 suit ink: {int((l<120).sum())} red_frac: "
      f"{float(((r>110)&(r>g+30)&(r>b+30))[l<120].mean()):.2f}")
