"""P30-2C-A — pc143 (green) 2,000 变体 ref 提取（GT=2000）+ 同步 refs。"""
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


# pc143 金额行（面板带 (144,234)，金额组 panel x 192-289 → abs x 1442-1539）
# 行剖面定位金额行 y：先粗测带内底部
m = extract_mask("POT_CHANGED_143_1787992603846_full.png",
                 1440, 1330, 1550, 1390)
print("pc143 2,000 green ref:", m.shape if m is not None else None,
      int(m.sum()) if m is not None else 0)
np.save(OUT / "amt_2000_green.npy", m.astype(np.uint8))

mp = Path(OUT) / "manifest.json"
mani = json.loads(mp.read_text(encoding="utf-8"))
mani["assets"].append({
    "label": "amt_2000_green", "crop_ref": "amt_2000_green.npy",
    "bbox": [1440, 1330, 110, 60], "source_frame_id":
    "POT_CHANGED_143_1787992603846_full.png", "gt_text": "2,000",
    "gt_value": 2000,
    "provenance": "P30-2C-A image-observed (V4 Flash)",
    "kind": "to_call_amount_1876", "variant": "green_button"})
mp.write_text(json.dumps(mani, indent=2, ensure_ascii=False),
              encoding="utf-8")
print("manifest updated (green variant)")
