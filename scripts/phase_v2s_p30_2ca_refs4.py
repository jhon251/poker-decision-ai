"""P30-2C-A — 按 detector 视角提取 green ref（紧贴 _tight_amount 输出）。"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402
from poker_decision_ai.vision2.pixel.realclient.coinpoker_to_call import (  # noqa: E402
    ToCallRecognizer1876, _text_mask, ACTION_PANEL_ROI)

EV = Path("data/vision2/coinpoker_evidence")
OUT = Path("data/vision2/coinpoker_assets_v1876tocall")
REC = ToCallRecognizer1876()

w, h, rgba = decode_png((EV / "POT_CHANGED_143_1787992603846_full.png")
                        .read_bytes())
arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
x0, y0, x1, y1 = ACTION_PANEL_ROI
panel = arr[y0:y1, x0:x1]
mask = _text_mask(panel)
bands = REC._row_bands(mask)
a2, b2 = bands[-1]
groups = REC._col_groups(mask, a2, b2)
for ga, gb in groups:
    if gb - ga + 1 >= 90:
        tight = REC._tight_amount(mask, a2, b2, ga, gb)
        print(f"green amount group ({ga},{gb}) tight={tight.shape}")
        np.save(OUT / "amt_2000_green.npy", tight.astype(np.uint8))
        # 更新 manifest
        mp = Path(OUT) / "manifest.json"
        mani = json.loads(mp.read_text(encoding="utf-8"))
        for a in mani["assets"]:
            if a["label"] == "amt_2000_green":
                a["bbox"] = [x0 + ga, y0 + a2, gb - ga + 1, b2 - a2 + 1]
        mp.write_text(json.dumps(mani, indent=2, ensure_ascii=False),
                      encoding="utf-8")
        print("green ref = detector-view tight; manifest updated")
