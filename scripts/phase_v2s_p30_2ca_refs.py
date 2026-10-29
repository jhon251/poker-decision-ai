"""P30-2C-A — 金额行参考位图提取（GT: pc114=124,163 / pc110=2,000）+ manifest。"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402
from poker_decision_ai.vision2_platform.png_encode import encode_rgba_png  # noqa: E402

EV = Path("data/vision2/coinpoker_evidence")
OUT = Path("data/vision2/coinpoker_assets_v1876tocall")
OUT.mkdir(parents=True, exist_ok=True)

# 金额行 ROI（abs；文字紧框后保存）
REF = [
    ("POT_CHANGED_114_1787992536082_full.png", 1400, 1345, 1600, 1390,
     "amt_124163"),
    ("POT_CHANGED_110_1787992526057_full.png", 1340, 1296, 1560, 1368,
     "amt_2000"),
]


def tight(a, b):
    """a,b 文本掩码 → 紧框。"""
    rows = np.where(a.sum(axis=1) > 0)[0]
    cols = np.where(a.sum(axis=0) > 0)[0]
    if len(rows) == 0 or len(cols) == 0:
        return None
    return a[rows.min():rows.max() + 1, cols.min():cols.max() + 1], b


assets = []
for fid, xa, ya, xb, yb, name in REF:
    w, h, rgba = decode_png((EV / fid).read_bytes())
    arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
    roi = arr[ya:yb, xa:xb]
    r = roi[..., 0].astype(np.int16); g = roi[..., 1].astype(np.int16)
    b = roi[..., 2].astype(np.int16)
    l = r * 0.299 + g * 0.587 + b * 0.114
    mask = ((l > 170) & (np.abs(r - g) < 55) & (np.abs(g - b) < 55))
    t = tight(mask, None)
    if t is None:
        print(f"!! {name}: empty")
        continue
    g_, _ = t
    np.save(OUT / f"{name}.npy", g_.astype(np.uint8))
    png = np.zeros((*g_.shape, 4), dtype=np.uint8)
    png[..., :3] = 255
    png[..., 3] = (g_ * 255).astype(np.uint8)
    (OUT / f"{name}.png").write_bytes(encode_rgba_png(
        g_.shape[1], g_.shape[0], png.tobytes()))
    gt_val = 124163 if "124163" in name else 2000
    assets.append({"label": name, "crop_ref": f"{name}.npy",
                   "bbox": [xa, ya, xb - xa, yb - ya],
                   "source_frame_id": fid,
                   "gt_text": "124,163" if "124163" in name else "2,000",
                   "gt_value": gt_val,
                   "provenance": "P30-2C-A image-observed (V4 Flash)",
                   "kind": "to_call_amount_1876"})
    print(f"  {name}: tight={g_.shape} ink={int(g_.sum())}")

manifest = {"version": "v1876tocall", "note": ("to_call 金额行参考位图"
             "（GT 目视确认；字符级分割不可靠——整行模板 NCC）。"),
             "assets": assets}
(OUT / "manifest.json").write_text(json.dumps(manifest, indent=2,
                                              ensure_ascii=False),
                                   encoding="utf-8")
print(f"assets={len(assets)}")
