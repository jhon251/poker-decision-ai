"""P30-2B — slot 模板提取（GT 文本 1K/2K (320)，每字符紧框 bbox）。

Slot 窗口按 GT 目视位置（f01 实测段位）：
  "1" @66-80, "K" @81-89, "/" @90-96, "2" @97-101, "K" @102-110,
  "(" @128-140, "3" @141-149, "2" @150-158, "0" @159-166, ")" @167-190
每 slot 内取 tight ink bbox → 模板。之后验证 NCC 判别力。"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402
from poker_decision_ai.vision2_platform.png_encode import encode_rgba_png  # noqa: E402

EV = Path("data/vision2/coinpoker_evidence")
OUT = Path("data/vision2/coinpoker_assets_v1876betting_header")
OUT.mkdir(parents=True, exist_ok=True)

FP = "BUTTON_CHANGED_167_1787992682749_full.png"   # GT: ...- 1K/2K (320)
w, h, rgba = decode_png((EV / FP).read_bytes())
arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
roi = arr[0:40, 0:520]
r = roi[..., 0].astype(np.int16); g = roi[..., 1].astype(np.int16)
b = roi[..., 2].astype(np.int16)
l = r * 0.299 + g * 0.587 + b * 0.114
mask = (l > 110) & (np.abs(r - g) < 60) & (np.abs(g - b) < 60)

SLOTS = [
    ("digit_1", 66, 80), ("K1", 81, 89), ("slash", 90, 96),
    ("digit_2", 97, 101), ("K2", 102, 110),
    ("paren_open", 128, 140), ("digit_3", 141, 149),
    ("digit_2b", 150, 158), ("digit_0", 159, 166),
    ("paren_close", 167, 190),
]
assets = []
for lab, xa, xb in SLOTS:
    sub = mask[:, xa:xb + 1]
    rows = np.where(sub.sum(axis=1) > 0)[0]
    cols = np.where(sub.sum(axis=0) > 0)[0]
    if len(rows) == 0 or len(cols) == 0:
        print(f"!! {lab}: idlesh slot ({xa}-{xb})")
        continue
    g = sub[rows.min():rows.max() + 1, cols.min():cols.max() + 1]
    fname = f"hdr_{lab}.png"
    rgba_out = np.zeros((g.shape[0], g.shape[1], 4), dtype=np.uint8)
    rgba_out[..., :3] = 255
    rgba_out[..., 3] = (g * 255).astype(np.uint8)
    (OUT / fname).write_bytes(encode_rgba_png(
        g.shape[1], g.shape[0], rgba_out.tobytes()))
    base = "digit" if lab.startswith("digit") else lab
    kind = base if lab in ("K1", "K2") else (
        "digit" if lab.startswith("digit") else lab)
    assets.append({"label": ("digit_1" if lab == "digit_1" else
                             ("digit_2" if lab == "digit_2" else
                              ("digit_2" if lab == "digit_2b" else
                               ("digit_3" if lab == "digit_3" else
                                ("digit_0" if lab == "digit_0" else
                                 ("K" if lab in ("K1", "K2") else lab)))))),
                   "crop_ref": fname, "slot": [xa, xb],
                   "bbox": [xa + int(cols.min()), int(rows.min()),
                            int(cols.max() - cols.min() + 1),
                            int(rows.max() - rows.min() + 1)],
                   "source_frame_id": FP, "gt_text": "1K/2K (320)",
                   "provenance": "P30-2A image-observed (V4 Flash)",
                   "kind": "betting_header_1876"})
    print(f"  {lab}: tight_bbox={assets[-1]['bbox']}")

manifest = {"client": "coinpoker", "version": "v1876bettingheader",
            "gt_text": "1K/2K (320)",
            "note": ("P30-2B header glyph 模板（GT 目视确认）。NOT_COVERED: "
                     "digit{4,5,6,7,8,9}、小数、M 后缀（header 未出现）→ "
                     "遇即 UNKNOWN。"),
            "assets": assets}
(OUT / "manifest.json").write_text(json.dumps(manifest, indent=2,
                                              ensure_ascii=False),
                                   encoding="utf-8")
print(f"\nassets = {len(assets)}")
