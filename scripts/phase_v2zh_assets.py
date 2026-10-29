"""Phase V2ZH — Board Card 字形重提取（hero 同款：卡角 ROI + 内容 bbox + 验证）。

GT 帧（视觉标注真值）：187 (2c Ac 8d 9c Ks)、556 (5h Td 7s)。
验证门：crop 尺寸 ≥150px、ink_ratio ≥0.10、颜色族记录——碎片不入库。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from poker_decision_ai.vision2.pixel.geometry import connected_components  # noqa: E402
from poker_decision_ai.vision2_platform import (  # noqa: E402
    decode_png, encode_rgba_png)

SLOT_X0, SLOT_Y0, SLOT_W, SLOT_H = 664, 551, 140, 177
FRAMES = {
    "SCENE_CHANGED_188_1787489345211_full.png":
        ["2c", "Ac", "8d", "9c", "Ks"],
    "SCENE_CHANGED_556_1787489611440_full.png": ["5h", "Td", "7s"],
}


def ink_mask(card):
    r = card[..., 0].astype(int)
    g = card[..., 1].astype(int)
    b = card[..., 2].astype(int)
    luma = r * 0.299 + g * 0.587 + b * 0.114
    dark = luma < 115
    red = (r > 100) & (r > g + 40) & (r > b + 40)
    return (dark | red).astype("uint8")


def glyphs(card, label):
    rankc, suitc = label[0], label[1]
    corner = card[6:88, 6:74]
    mask = ink_mask(corner)[3:-3, 3:-3]
    cw, ch = corner.shape[1], corner.shape[0]
    comps = [c for c in connected_components(mask, min_area=12)
             if c.bbox.width < 0.8 * cw and c.bbox.height < 0.7 * ch]
    comps.sort(key=lambda c: (c.bbox.y, c.bbox.x))
    out = []
    for j, c in enumerate(comps[:2]):
        by, bx = int(c.bbox.y) + 3, int(c.bbox.x) + 3
        crop = corner[by - 1:by + int(c.bbox.height) + 4,
                      bx - 1:bx + int(c.bbox.width) + 4]
        r = crop[..., 0].astype(int)
        g = crop[..., 1].astype(int)
        b = crop[..., 2].astype(int)
        luma = r * 0.299 + g * 0.587 + b * 0.114
        ink = float(((luma < 115) | ((r > 100) & (r > g + 40)
                                     & (r > b + 40))).mean())
        red_ratio = float(((r > 100) & (r > g + 40)
                           & (r > b + 40)).mean())
        if crop.size < 150 or ink < 0.10:
            continue                                 # 碎片 → 不入库
        out.append((crop, rankc if j == 0 else suitc,
                    [6 + bx, 6 + by, crop.shape[1], crop.shape[0]],
                    round(ink, 3), "red" if red_ratio > 0.10 else "black"))
    return out


def main() -> int:
    sys.argv = sys.argv[:1]
    base = Path("data/vision2/coinpoker_evidence")
    out = Path("data/vision2/coinpoker_assets_v2zh")
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"client": "coinpoker", "version": "v2zh",
                "coordinate_system": "full_png_1419x1066",
                "ground_truth_status": "visually_labeled", "assets": []}
    for fname, labels in FRAMES.items():
        w, h, rgba = decode_png((base / fname).read_bytes())
        full = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
        for i, label in enumerate(labels):
            slot = full[SLOT_Y0:SLOT_Y0 + SLOT_H,
                        SLOT_X0 + i * SLOT_W:SLOT_X0 + (i + 1) * SLOT_W]
            for j, (crop, lbl, bbox, ink, fam) in enumerate(
                    glyphs(slot, label)):
                p = out / f"board_{label}_{j}.png"
                p.write_bytes(encode_rgba_png(crop.shape[1], crop.shape[0],
                                              crop.tobytes()))
                manifest["assets"].append({
                    "label": lbl, "kind": "card_glyph",
                    "bbox": [SLOT_X0 + i * SLOT_W + bbox[0],
                             SLOT_Y0 + bbox[1], bbox[2],
                             bbox[3]],
                    "crop_ref": str(p),
                    "coordinate_system": "full_png",
                    "source_frame_id": fname[:23],
                    "ink_ratio": ink, "color_family": fam,
                    "ground_truth_status": "visually_labeled",
                    "confidence": 0.92,
                    "provenance": {"annotation": "V2V/V2W visual reading",
                                   "label_source": "agent_vision_annotation",
                                   "method": "corner_roi_content_bbox"}})
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")
    from collections import Counter
    assets = manifest["assets"]
    print("assets:", len(assets))
    print("labels:", dict(sorted(Counter(a["label"]
                                         for a in assets).items())))
    ranks = sorted({a["label"] for a in assets if a["label"] in "AKQJT98765432"})
    suits = sorted({a["label"] for a in assets if a["label"] in "shdc"})
    print("rank coverage:", ranks)
    print("suit coverage:", suits)
    print("families:", dict(Counter(a["color_family"] for a in assets)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
