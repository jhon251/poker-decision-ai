"""P30-2B — debug：live stake mask vs ref NCC 分数（为何 <0.90 或 char fail）。"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402
from poker_decision_ai.vision2.pixel.realclient.coinpoker_betting_header import (  # noqa: E402
    _ncc, _text_mask, STAKE_ROI)

EV = Path("data/vision2/coinpoker_evidence")
OUT = Path("data/vision2/coinpoker_assets_v1876betting_header")
ref = np.load(OUT / "stake_mask_f01.npy").astype(np.float64)

for fid in ["BUTTON_CHANGED_167_1787992682749_full.png",
            "8bf82f134651db0add626be0304a2ff1.png"]:
    w, h, rgba = decode_png((EV / fid).read_bytes())
    arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
    sm = _text_mask(arr, STAKE_ROI)
    print(f"\n{fid[:30]}:")
    print(f"  live mask shape={sm.shape} ink={int(sm.sum())} ref ink={int(ref.sum())}")
    print(f"  NCC = {_ncc(sm, ref):.4f}")
    # 形状对齐检查
    print(f"  ref shape={ref.shape}")
    # rowwise ink
    print("  live rowd:", [int(v) for v in sm.sum(axis=1)][:26])
    print("  ref  rowd:", [int(v) for v in ref.sum(axis=1)][:26])
