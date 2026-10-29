"""P30-2B — 统一掩码：live/ref 都 luma>120（同 f01 提取方式）→ NCC 重测。"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402
from poker_decision_ai.vision2.pixel.realclient.coinpoker_betting_header import (  # noqa: E402
    _ncc, STAKE_ROI)

EV = Path("data/vision2/coinpoker_evidence")
OUT = Path("data/vision2/coinpoker_assets_v1876betting_header")
ref = np.load(OUT / "stake_mask_f01.npy").astype(np.float64)

for fid in ["BUTTON_CHANGED_167_1787992682749_full.png",
            "8bf82f134651db0add626be0304a2ff1.png",
            "POT_CHANGED_110_1787992526057_full.png",
            "POT_CHANGED_102_1787992505794_full.png"]:
    w, h, rgba = decode_png((EV / fid).read_bytes())
    arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
    x0, y0, x1, y1 = STAKE_ROI
    sub = arr[y0:y1, x0:x1]
    l = sub[..., 0] * 0.299 + sub[..., 1] * 0.587 + sub[..., 2] * 0.114
    sm = (l > 120).astype(np.float64)
    print(f"{fid[:30]}: NCC={_ncc(sm, ref):.4f} ink={int(sm.sum())}")
