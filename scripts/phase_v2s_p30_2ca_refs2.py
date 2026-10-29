"""P30-2C-A — 2,000 ref 重建（金额行 y1334-1364）；pc114 ref 复核。"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402

EV = Path("data/vision2/coinpoker_evidence")
OUT = Path("data/vision2/coinpoker_assets_v1876tocall")


def extract_mask(fid, xa, ya, xb, yb):
    w, h, rgba = decode_png((EV / fid).read_bytes())
    arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
    roi = arr[ya:yb, xa:xb]
    r = roi[..., 0].astype(np.int16); g = roi[..., 1].astype(np.int16)
    b = roi[..., 2].astype(np.int16)
    l = r * 0.299 + g * 0.587 + b * 0.114
    mask = ((l > 170) & (np.abs(r - g) < 55) & (np.abs(g - b) < 55))
    ys = np.where(mask.sum(axis=1) > 0)[0]
    xs = np.where(mask.sum(axis=0) > 0)[0]
    if len(ys) == 0 or len(xs) == 0:
        return None
    return mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


# pc110 金额 2,000：金额行 y1334-1364（abs），x 1474-1554
m2k = extract_mask("POT_CHANGED_110_1787992526057_full.png",
                   1440, 1330, 1600, 1370)
print("2,000 ref:", m2k.shape if m2k is not None else None,
      int(m2k.sum()) if m2k is not None else 0)
np.save(OUT / "amt_2000.npy", m2k.astype(np.uint8))

# pc114 复核（金额行 y1350-1383）
m124 = extract_mask("POT_CHANGED_114_1787992536082_full.png",
                    1400, 1345, 1600, 1390)
print("124,163 ref:", m124.shape if m124 is not None else None,
      int(m124.sum()) if m124 is not None else 0)
np.save(OUT / "amt_124163.npy", m124.astype(np.uint8))

# 更新 manifest
mp = Path(OUT) / "manifest.json"
m = json.loads(mp.read_text(encoding="utf-8"))
for a in m["assets"]:
    if "amt_2000" in a["label"]:
        a["bbox"] = [1440, 1330, 160, 40]
    elif "amt_124163" in a["label"]:
        a["bbox"] = [1400, 1345, 200, 45]
mp.write_text(json.dumps(m, indent=2, ensure_ascii=False), encoding="utf-8")
print("manifest updated")
