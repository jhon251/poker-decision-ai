"""P30-2B — header glyph 模板提取（f01 GT：1K/2K (320)，位置标注）。

仅提取 stake 区（x66+，排除 logo/hand_id/prefix）：
「1K/2K (320)」→ 字符序: 1 K / 2 K ( 3 2 0 )
模板标签（GT provenance = P30-2A image-observed）：
  digit_1, K, slash, digit_2, K, paren_open, digit_3, digit_2, digit_0, paren_close
未覆盖 digit {5,9,4,6,7,8} → NOT_COVERED → 遇到即 UNKNOWN。
"""
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

FP = "BUTTON_CHANGED_167_1787992682749_full.png"  # GT: NLH 1218466 - 1K/2K (320)
w, h, rgba = decode_png((EV / FP).read_bytes())
arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
roi = arr[0:40, 0:520]
r = roi[..., 0].astype(np.int16); g = roi[..., 1].astype(np.int16)
b = roi[..., 2].astype(np.int16)
l = r * 0.299 + g * 0.587 + b * 0.114
# 高掩码：亮白（bg=27 → luma>>120 为文本；抗锯齿边缘亦含）
mask = (l > 110) & (np.abs(r - g) < 60) & (np.abs(g - b) < 60)

cold = mask.sum(axis=0)
runs, run = [], None
for i, v in enumerate(cold):
    if v >= 1 and run is None:
        run = [i, i]
    elif v >= 1 and run is not None:
        run[1] = i
    elif run is not None:
        runs.append(run); run = None
if run:
    runs.append(run)
merged = []
for rr in runs:
    if merged and rr[0] - merged[-1][1] <= 3:
        merged[-1][1] = rr[1]
    else:
        merged.append(rr)

# stake 区（x≥66；"1K/2K (320)" 从 x69 开始）
stake_segs = [s for s in merged if s[0] >= 66]
print(f"stake 字符段数 = {len(stake_segs)}（期望 10）")
LABELS = ["digit_1", "K", "slash", "digit_2", "K2", "paren_open",
          "digit_3", "digit_2", "digit_0", "paren_close"]
assets = []
if len(stake_segs) == 10:
    for lab, (a, bb) in zip(LABELS, stake_segs):
        sub = mask[:, a:bb + 1]
        rows = np.where(sub.sum(axis=1) > 0)[0]
        if len(rows) == 0:
            continue
        gy0, gy1 = int(rows.min()), int(rows.max()) + 1
        glyph = sub[gy0:gy1, :]
        fname = f"hdr_{lab}.png"
        rgba_out = np.zeros((glyph.shape[0], glyph.shape[1], 4),
                            dtype=np.uint8)
        rgba_out[..., :3] = 255
        rgba_out[..., 3] = (glyph * 255).astype(np.uint8)
        (OUT / fname).write_bytes(encode_rgba_png(
            glyph.shape[1], glyph.shape[0], rgba_out.tobytes()))
        assets.append({"label": lab, "crop_ref": fname,
                       "bbox": [a, gy0, bb - a + 1, gy1 - gy0],
                       "source_frame_id": FP, "gt_text": "1K/2K (320)",
                       "provenance": "P30-2A image-observed (V4 Flash)",
                       "kind": "betting_header_1876"})
        print(f"  {lab}: ({a},{gy0},{bb-a+1},{gy1-gy0})")
else:
    print("!! stake segs != 10, 需要检查")

manifest = {"client": "coinpoker", "version": "v1876bettingheader",
            "note": ("P30-2B header glyph 模板（GT: 1K/2K (320) 目视确认）。"
                     "未覆盖 digit {4,5,6,7,8,9} 及小数/M 后缀 → NOT_COVERED"
                     "→ UNKNOWN。"),
            "assets": assets}
(OUT / "manifest.json").write_text(json.dumps(manifest, indent=2,
                                              ensure_ascii=False),
                                   encoding="utf-8")
print(f"\nassets = {len(assets)}")
