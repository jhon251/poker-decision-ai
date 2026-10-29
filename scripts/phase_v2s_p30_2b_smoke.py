"""P30-2B — detector 冒烟：4 帧真值 + save stake mask（f01 → asset）。"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402
from poker_decision_ai.vision2.pixel.realclient.coinpoker_betting_header import (  # noqa: E402
    BettingHeaderRecognizer1876)

EV = Path("data/vision2/coinpoker_evidence")
OUT = Path("data/vision2/coinpoker_assets_v1876betting_header")

# 保存 f01 stake mask（detector 引用参考）
w, h, rgba = decode_png((EV / "BUTTON_CHANGED_167_1787992682749_full.png")
                        .read_bytes())
arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
l = arr[..., 0] * 0.299 + arr[..., 1] * 0.587 + arr[..., 2] * 0.114
mask = (l > 120)[0:26, 90:200].astype(np.uint8)
np.save(OUT / "stake_mask_f01.npy", mask)
print("saved stake_mask_f01.npy", mask.shape, int(mask.sum()))

REC = BettingHeaderRecognizer1876()
print("ref_mask loaded:", REC.ref_mask is not None)
print("assets:", list(REC.assets.keys()))

for fid in ["BUTTON_CHANGED_167_1787992682749_full.png",
            "8bf82f134651db0add626be0304a2ff1.png",
            "POT_CHANGED_110_1787992526057_full.png",
            "POT_CHANGED_102_1787992505794_full.png"]:
    w, h, rgba = decode_png((EV / fid).read_bytes())
    arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
    res = REC.recognize(arr)
    print(f"{fid[:28]}: sb={res['sb']} bb={res['bb']} ante={res['ante']} "
          f"status={res['status']} ({res['reason']}) conf={res['confidence']}")
