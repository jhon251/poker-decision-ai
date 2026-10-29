"""V3F-P5R — 从真实 Corpus 提取 Stack 数字字形（一次性资产构建，只读源帧）。

提取源（真实帧 × 已知数值，按字符顺序对齐 label——非伪造模板）：
  SCENE_188: S1 "201,000" S4 "197,360"   SETTLED_961: S1 "2.28M" S3 "67,360"
  STATUS_246: S1 "388,080"   SCENE_60(?): 见下方失败样本亦可
每字符：金色掩码 → 列投影分割 → 紧致裁剪 → 存 glyph + manifest（provenance 完整）。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from poker_decision_ai.vision2_platform import decode_png, encode_rgba_png  # noqa: E402
from poker_decision_ai.vision2.pixel.geometry import connected_components  # noqa: E402

OUT = Path("data/vision2/coinpoker_assets_v2p5r")
OUT.mkdir(parents=True, exist_ok=True)

# (frame, seat, 锚点区域[x0,y0,x1,y1], 期望字符串)
SOURCES = [
    ("SCENE_CHANGED_188_1787489345211_full.png", "S1", (170, 330, 480, 480),
     "201,000"),
    ("SCENE_CHANGED_188_1787489345211_full.png", "S4", (120, 900, 480,
                                                        1060), "197,360"),
    ("SETTLED_BOARD_CHECK_961_1787653653866_full.png", "S1", (200, 380,
                                                              460, 470),
     "2.28M"),
    ("SETTLED_BOARD_CHECK_961_1787653653866_full.png", "S3", (880, 60,
                                                              1140, 180),
     "67,360"),
    ("STATUS_UI_CHANGED_246_1787488634077_full.png", "S1", (150, 300,
                                                            480, 470),
     "388,080"),
    ("SCENE_CHANGED_60_1787489266339_full.png", "S4", (120, 900, 480,
                                                       1060), "438,490"),
]


def gold_mask(arr: np.ndarray) -> np.ndarray:
    r = arr[..., 0].astype(np.int16)
    g = arr[..., 1].astype(np.int16)
    b = arr[..., 2].astype(np.int16)
    return ((r > 130) & (g > 95) & (b < 145) & (r >= g) & (g >= b - 20)
            & (~((r > 240) & (g > 240) & (b > 240)))).astype(np.uint8)


manifest = {"client": "coinpoker", "version": "v2p5r",
            "ground_truth_status": "visually_labeled",
            "provenance": {"annotation": "V3F-P5R real-frame glyph extraction "
                                          "(known-value ordered labeling)",
                           "label_source": "agent_vision_annotation"},
            "assets": []}
for fname, seat, (x0, y0, x1, y1), text in SOURCES:
    w, h, rgba = decode_png(Path(
        f"data/vision2/coinpoker_evidence/{fname}").read_bytes())
    frame = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
    roi = frame[y0:y1, x0:x1]
    gm = gold_mask(roi)
    if gm.sum() < 50:
        print(f"{seat}@{fname[:22]} gold=0 skip")
        continue
    ys, xs = np.where(gm > 0)
    inner = gm[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    # 列投影分割（字符间隙 ≥2 列）
    cols = inner.sum(axis=0)
    comps = []
    in_run = False
    for i, c in enumerate(cols):
        if c > 0 and not in_run:
            start, in_run = i, True
        elif c == 0 and in_run:
            comps.append((start, i))
            in_run = False
    if in_run:
        comps.append((start, len(cols)))
    if len(comps) != len(text):
        print(f"{seat}@{fname[:22]} split {len(comps)} != {len(text)} skip")
        continue
    for (cs, ce), ch in zip(comps, text):
        seg = inner[:, cs:ce]
        rows = (seg > 0).sum(axis=1)
        sub = seg[rows > 0][rows > 0] if False else seg[np.where(
            seg.sum(axis=1) > 0)[0][0]:np.where(seg.sum(axis=1) > 0)[0][-1]
                                              + 1]
        sub = sub[(sub.sum(axis=1) > 0)]
        chh, cww = sub.shape
        if cww < 3 or chh < 8 or chh > 64:
            continue
        crop = roi[ys.min() + np.where(seg.sum(axis=1) > 0)[0][0]:
                   ys.min() + np.where(seg.sum(axis=1) > 0)[0][-1] + 1,
                   xs.min() + cs:xs.min() + ce]
        p = OUT / f"g_{ch}_{fname[7:20]}_{seat}.png"
        p.write_bytes(encode_rgba_png(crop.shape[1], crop.shape[0],
                                      crop.tobytes()))
        manifest["assets"].append({
            "label": ch, "kind": "stack_digit_glyph", "crop_ref": str(p),
            "source_frame_id": fname, "seat": seat,
            "bbox": [x0 + xs.min() + cs, y0 + ys.min(),
                     ce - cs, crop.shape[0]],
            "coordinate_system": "full_png_1419x1066",
            "ground_truth_status": "visually_labeled",
            "confidence": 0.95,
            "provenance": {"annotation": "ordered labeling of known value",
                           "label_source": "agent_vision_annotation"}})
    print(f"{seat}@{fname[:22]} '{text}' → {len(comps)} glyphs")

(OUT / "manifest.json").write_text(json.dumps(
    manifest, indent=1, ensure_ascii=False), encoding="utf-8")
print("manifest assets:", len(manifest["assets"]))
