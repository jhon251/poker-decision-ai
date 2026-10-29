"""V3F-P5S — 校准窗提取（目视校准 ROI，仅真实文本）。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from poker_decision_ai.vision2_platform import decode_png, encode_rgba_png  # noqa: E402

OUT = Path("data/vision2/coinpoker_assets_v2p5r")
OUT.mkdir(parents=True, exist_ok=True)
manifest = {}
mp = OUT / "manifest.json"
if mp.exists():
    manifest = json.loads(mp.read_text(encoding="utf-8"))
manifest.setdefault("assets", [])

# (frame, 文本窗 full bbox, 期望值) —— 目视校准（P5S-2）
SOURCES = [
    ("SETTLED_BOARD_CHECK_1328_1787713028674_full.png",
     (231, 985, 400, 1047), "318,871"),
    ("SETTLED_BOARD_CHECK_1328_1787713028674_full.png",
     (250, 425, 415, 475), "953,909"),
    ("SETTLED_BOARD_CHECK_1328_1787713028674_full.png",
     (935, 296, 1050, 350), "55,352"),
    ("SETTLED_BOARD_CHECK_961_1787653653866_full.png",
     (944, 100, 1070, 168), "67,360"),
    ("SETTLED_BOARD_CHECK_961_1787653653866_full.png",
     (275, 415, 368, 470), "2.28M"),
    ("SCENE_CHANGED_60_1787489266339_full.png",
     (223, 970, 408, 1017), "434,504"),
    ("STATUS_UI_CHANGED_246_1787488634077_full.png",
     (208, 375, 423, 455), "388,080"),
]


def gold_mask(roi: np.ndarray) -> np.ndarray:
    r = roi[..., 0].astype(np.int16)
    g = roi[..., 1].astype(np.int16)
    b = roi[..., 2].astype(np.int16)
    l = r * 0.299 + g * 0.587 + b * 0.114
    return ((r > 120) & (g > 85) & (r >= g) & (g >= b - 30)
            & (l > 90) & (~((r > 240) & (g > 240) & (b > 240))
                         | (r > 150))).astype(np.uint8)


got, skipped = [], []
for fname, (x0, y0, x1, y1), text in SOURCES:
    w, h, rgba = decode_png(Path(
        f"data/vision2/coinpoker_evidence/{fname}").read_bytes())
    frame = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
    roi = frame[y0:y1, x0:x1]
    gm = gold_mask(roi)
    if gm.sum() < 40:
        skipped.append((fname[:20], text, "no_gold"))
        continue
    ys, xs = np.where(gm > 0)
    inner = gm[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    cols = inner.sum(axis=0)
    comps, run = [], None
    for i, c in enumerate(cols):
        if c > 0 and run is None:
            run = i
        elif c == 0 and run is not None:
            comps.append((run, i))
            run = None
    if run is not None:
        comps.append((run, len(cols)))
    if len(comps) != len(text):
        skipped.append((fname[:20], text, f"split {len(comps)}"))
        continue
    for (cs, ce), ch in zip(comps, text):
        seg = inner[:, cs:ce]
        rws = np.where(seg.sum(axis=1) > 0)[0]
        crop = roi[ys.min() + rws[0]:ys.min() + rws[-1] + 1,
                   xs.min() + cs:xs.min() + ce]
        p = OUT / f"g_{ch}_{text.replace(',', 'x')}_{fname[:6]}.png"
        p.write_bytes(encode_rgba_png(crop.shape[1], crop.shape[0],
                                      crop.tobytes()))
        manifest["assets"].append({
            "label": ch, "kind": "stack_digit_glyph", "crop_ref": str(p),
            "source_frame_id": fname, "text_window": [x0, y0, x1, y1],
            "bbox": [int(x0 + xs.min() + cs), int(y0 + ys.min() + rws[0]),
                     int(ce - cs), int(crop.shape[0])],
            "coordinate_system": "full_png_1419x1066",
            "ground_truth_status": "visually_labeled", "confidence": 0.95,
            "provenance": {"annotation": "P5S calibrated window, "
                                          f"value={text}",
                           "label_source": "agent_vision_annotation"}})
    got.append((fname[:20], text, len(comps)))

(OUT / "manifest.json").write_text(json.dumps(
    manifest, indent=1, ensure_ascii=False), encoding="utf-8")
print("extracted:", got)
print("skipped:", skipped)
labels = sorted({a["label"] for a in manifest["assets"]})
print("glyph labels:", labels)
