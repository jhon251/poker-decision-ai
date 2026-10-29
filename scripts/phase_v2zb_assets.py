"""Phase V2ZB — 按 V2ZA 已验证 PNG 坐标重提取（hero Qs/9d + pot 多值）。

- hero：(108,740,202,175) 全 PNG 坐标 → 双卡 → corner-ROI 提取 rank/suit；
- pot：B(×1.539)→PNG 坐标（25k/640/263,520/280,220）亮字字符提取（anchor 方案）；
- board：既有资产坐标 provenance 审计。
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

S = 1419.0 / 922.0
HERO = ("SCENE_CHANGED_188_1787489345211_full.png", (108, 740, 202, 175),
        ["Qs", "9d"])
POTS = [
    ("SCENE_CHANGED_188_1787489345211_full.png",
     (578 * S, 480 * S, 120 * S, 33 * S), "25,000"),
    ("SCENE_CHANGED_283_1787489414967_full.png",
     (585 * S, 300 * S, 125 * S, 45 * S), "640"),
    ("SCENE_CHANGED_452_1787489541883_full.png",
     (585 * S, 450 * S, 120 * S, 33 * S), "263,520"),
    ("SCENE_CHANGED_556_1787489611440_full.png",
     (590 * S, 445 * S, 140 * S, 33 * S), "280,220"),
]


def ink_mask(card):
    r = card[..., 0].astype(int)
    g = card[..., 1].astype(int)
    b = card[..., 2].astype(int)
    luma = r * 0.299 + g * 0.587 + b * 0.114
    dark = luma < 115
    red = (r > 110) & (g < 100) & (b < 100)
    return (dark | red).astype("uint8")


def corner_glyphs(card, label):
    rankc, suitc = label[0], label[1]
    ch, cwd = card.shape[0], card.shape[1]
    x0, y0, x1, y1 = int(cwd * 0.02), int(ch * 0.02), int(cwd * 0.55), \
        int(ch * 0.42)
    corner = card[y0:y1, x0:x1]
    mask = ink_mask(corner)[3:-3, 3:-3]
    comps = connected_components(mask, min_area=8)
    comps = [c for c in comps
             if c.bbox.width < 0.75 * (x1 - x0)
             and c.bbox.height < 0.6 * (y1 - y0)]
    comps.sort(key=lambda c: (c.bbox.y, c.bbox.x))
    out = []
    for j, c in enumerate(comps[:2]):
        bx, by = int(c.bbox.x) + 3, int(c.bbox.y) + 3
        bw, bh = int(c.bbox.width), int(c.bbox.height)
        out.append((corner[by - 1:by + bh + 1, bx - 1:bx + bw + 1],
                    rankc if j == 0 else suitc,
                    [x0 + bx, y0 + by, bw, bh]))
    return out


def pot_chars(sub, text):
    r = sub[..., 0].astype(int)
    g = sub[..., 1].astype(int)
    b = sub[..., 2].astype(int)
    luma = r * 0.299 + g * 0.587 + b * 0.114
    mask = luma > 150
    col_on = mask.sum(axis=0) > 2
    runs = []
    x = 0
    while x < len(col_on):
        if not col_on[x]:
            x += 1
            continue
        sx = x
        while x < len(col_on) and col_on[x]:
            x += 1
        runs.append((sx, x))
    chars = []
    for (sx, x1) in runs:
        ys = np.where(mask[:, sx:x1].any(axis=1))[0]
        if len(ys) < 6:
            continue
        chars.append((sx, int(ys.min()), x1 - sx,
                      int(ys.max() - ys.min() + 1)))
    out = []
    if len(chars) != len(text):
        return out
    for j, (cx, cy, cw, chh) in enumerate(chars):
        out.append((sub[max(0, cy - 1):cy + chh + 1,
                        max(0, cx - 1):cx + cw + 1], text[j],
                    [cx, cy, cw, chh]))
    return out


def main() -> int:
    sys.argv = sys.argv[:1]
    base = Path("data/vision2/coinpoker_evidence")
    out = Path("data/vision2/coinpoker_assets_v2zb")
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"client": "coinpoker", "version": "v2zb",
                "coordinate_system": "full_png_1419x1066",
                "ground_truth_status": "visually_labeled", "assets": []}

    def save(crop, name, label, frame_id, bbox, roi, kind):
        p = out / f"{name}.png"
        p.write_bytes(encode_rgba_png(crop.shape[1], crop.shape[0],
                                      crop.tobytes()))
        manifest["assets"].append({
            "template_id": name, "label": label, "kind": kind,
            "source_frame_id": frame_id, "bbox": [int(v) for v in bbox],
            "crop_ref": str(p), "roi": roi, "confidence": 0.92,
            "coordinate_system": "full_png",
            "provenance": {"annotation": "V2V/V2W visual reading",
                           "coordinate_fix": "V2ZA audit",
                           "ground_truth_status": "visually_labeled",
                           "label_source": "agent_vision_annotation"}})

    # P0 hero（A 坐标）
    fname, (x0, y0, cw, chh), labels = HERO
    w, h, rgba = decode_png((base / fname).read_bytes())
    full = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
    for i, label in enumerate(labels):
        card = full[int(y0):int(y0 + chh), int(x0 + i * cw / 2):
                    int(x0 + (i + 1) * cw / 2)]
        for j, (crop, lbl, b) in enumerate(corner_glyphs(card, label)):
            save(crop, f"hero_{label}_{j}", lbl, fname[:23],
                 [x0 + i * cw / 2 + b[0], y0 + b[1], b[2], b[3]],
                 "hero", "card_glyph")
    # P1 pot（B→PNG 坐标）
    for fname, (px, py, pw, ph), text in POTS:
        w, h, rgba = decode_png((base / fname).read_bytes())
        full = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
        sub = full[int(py):int(py + ph), int(px):int(px + pw)]
        for j, (crop, lbl, b) in enumerate(pot_chars(sub, text)):
            save(crop, f"pot_{fname[:19]}_x{j}", lbl, fname[:23],
                 [px + b[0], py + b[1], b[2], b[3]], "pot_text",
                 "digit_glyph")
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=1, ensure_ascii=False),
        encoding="utf-8")
    from collections import Counter
    print("assets:", len(manifest["assets"]))
    print("by kind:", dict(Counter(a["kind"] for a in manifest["assets"])))
    print("labels:", dict(sorted(Counter(a["label"]
                                         for a in manifest["assets"]).items())))
    print("hero assets:", sum(1 for a in manifest["assets"]
                              if a["roi"] == "hero"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
