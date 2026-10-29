"""Phase V2ZJ — 中心大花色模板提取（视觉确认 + 最大组件分离）。

- 视觉确认（2c 卡）：中心大 ♣ @ card-rel (38,95,77,82)；角标花色 (8,62,40,53)；
- 提取：中心区 card_rel (25,82,100,95) 内 ink 掩码 → **最大组件**（中心大
  花色自动与角标/雕花分离）→ 内容 bbox 紧字形 → 验证入库。
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
CENTER_REGION = (25, 82, 100, 95)      # card-relative


def ink_mask(arr):
    r = arr[..., 0].astype(int)
    g = arr[..., 1].astype(int)
    b = arr[..., 2].astype(int)
    luma = r * 0.299 + g * 0.587 + b * 0.114
    dark = luma < 115
    red = (r > 100) & (r > g + 40) & (r > b + 40)
    return (dark | red).astype("uint8")


def main() -> int:
    sys.argv = sys.argv[:1]
    base = Path("data/vision2/coinpoker_evidence")
    out = Path("data/vision2/coinpoker_assets_v2zj")
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"client": "coinpoker", "version": "v2zj",
                "coordinate_system": "full_png_1419x1066",
                "ground_truth_status": "visually_labeled", "assets": []}
    missing = []
    for fname, labels in FRAMES.items():
        w, h, rgba = decode_png((base / fname).read_bytes())
        full = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
        for i, label in enumerate(labels):
            suitc = label[1]
            cx = SLOT_X0 + i * SLOT_W + CENTER_REGION[0]
            cy = SLOT_Y0 + CENTER_REGION[1]
            region = full[cy:cy + CENTER_REGION[3],
                          cx:cx + CENTER_REGION[2]]
            if region.size < 100:
                missing.append(label)
                continue
            mask = ink_mask(region)
            comps = connected_components(mask, min_area=60)
            if not comps:
                missing.append(label)
                continue
            big = max(comps, key=lambda c: c.area)
            b = big.bbox
            x0 = max(0, int(b.x) - 2)
            y0 = max(0, int(b.y) - 2)
            crop = region[y0:y0 + int(b.height) + 4,
                          x0:x0 + int(b.width) + 4]
            r = crop[..., 0].astype(int)
            g = crop[..., 1].astype(int)
            b_ = crop[..., 2].astype(int)
            luma = r * 0.299 + g * 0.587 + b_ * 0.114
            ink = float(((luma < 115) | ((r > 100) & (r > g + 40)
                                         & (r > b_ + 40))).mean())
            red_r = float(((r > 100) & (r > g + 40)
                           & (r > b_ + 40)).mean())
            if crop.size < 400 or ink < 0.12:
                missing.append(label)
                continue
            p = out / f"center_suit_{label}.png"
            p.write_bytes(encode_rgba_png(crop.shape[1], crop.shape[0],
                                          crop.tobytes()))
            manifest["assets"].append({
                "label": suitc, "kind": "card_glyph",
                "bbox": [cx + x0, cy + y0, crop.shape[1], crop.shape[0]],
                "crop_ref": str(p),
                "coordinate_system": "full_png",
                "source_frame_id": fname[:23],
                "ink_ratio": round(ink, 3),
                "color_family": "red" if red_r > 0.10 else "black",
                "ground_truth_status": "visually_labeled",
                "confidence": 0.92,
                "provenance": {"annotation": "V2ZJ visual confirmation",
                               "label_source": "agent_vision_annotation",
                               "method": "center_region_largest_component"}})
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=1, ensure_ascii=False),
        encoding="utf-8")
    from collections import Counter
    assets = manifest["assets"]
    print("assets:", len(assets))
    print("suits:", dict(sorted(Counter(a["label"]
                                        for a in assets).items())))
    print("families:", dict(Counter(a["color_family"] for a in assets)))
    if missing:
        print("missing (null):", missing)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
