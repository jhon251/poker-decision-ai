"""Phase V2Z — CoinPoker 卡牌字形提取稳定化（rank_corner ROI 法）。

针对 V2Y 根因：
  - 不再用整卡 connected component（雕花/边框与角标合并）；
  - 改为 卡角 ROI（左上 55%×42%）内提取 → rank 在上、角标花色在下；
  - 红色系（HSV 色相扩展）+ 黑色系双掩码；
  - hero 专项：槽位重新探测 + 极性统计 + Q/9 可提取性输出；
  - pot：640 / 263,520 / 280,220 多位置数字 + anchor 验证。
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
BOARDS = {
    "SCENE_CHANGED_188_1787489345211_full.png": (435, 525, 93, 115,
                                                 ["2c", "Ac", "8d", "9c",
                                                  "Ks"]),
    "SCENE_CHANGED_452_1787489541883_full.png": (435, 525, 93, 115,
                                                 ["5s", "6c", "4s", "5h",
                                                  "6h"]),
    "SCENE_CHANGED_556_1787489611440_full.png": (435, 525, 93, 115,
                                                 ["5h", "Td", "7s"]),
}
HERO = {
    "SCENE_CHANGED_188_1787489345211_full.png": (108, 740, 102, 88,
                                                 ["Qs", "9d"]),
}
POTS = {
    "SCENE_CHANGED_283_1787489414967_full.png": ((585, 300, 125, 45), "640"),
    "SCENE_CHANGED_452_1787489541883_full.png": ((585, 450, 120, 33),
                                                 "263,520"),
    "SCENE_CHANGED_556_1787489611440_full.png": ((590, 445, 140, 33),
                                                 "280,220"),
}


def ink_mask(card: np.ndarray) -> np.ndarray:
    r = card[..., 0].astype(int)
    g = card[..., 1].astype(int)
    b = card[..., 2].astype(int)
    luma = r * 0.299 + g * 0.587 + b * 0.114
    dark = luma < 115
    # 红色系（HSV 色相扩展：H<20 或 H>335）
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    dlt = mx - mn
    h = np.zeros_like(r, dtype=float)
    m = dlt > 20
    h[m] = 60.0 * (((g[m] - b[m]) / dlt[m]) % 6) if False else \
        60.0 * np.where(r[m] == mx[m],
                        ((g[m] - b[m]) / dlt[m]) % 6,
                        np.where(g[m] == mx[m],
                                 (b[m] - r[m]) / dlt[m] + 2,
                                 (r[m] - g[m]) / dlt[m] + 4))
    red = (mx > 90) & (dlt > 40) & ((h < 20) | (h > 335))
    return (dark | red).astype("uint8")


def corner_glyphs(card: np.ndarray, label: str) -> list:
    """卡角 ROI（左上 55%×42%）→ [(crop, glyph_label, bbox_card_rel)]。"""
    rankc, suitc = label[0], label[1]
    ch, cwd = card.shape[0], card.shape[1]
    x0, y0, x1, y1 = int(cwd * 0.02), int(ch * 0.02), \
        int(cwd * 0.55), int(ch * 0.42)
    corner = card[y0:y1, x0:x1]
    mask = ink_mask(corner)[3:-3, 3:-3]          # 去内边框 3px
    comps = connected_components(mask, min_area=8)
    comps = [c for c in comps
             if c.bbox.width < 0.75 * (x1 - x0)
             and c.bbox.height < 0.6 * (y1 - y0)]
    comps.sort(key=lambda c: (c.bbox.y, c.bbox.x))
    out = []
    for j, c in enumerate(comps[:2]):
        lbl = rankc if j == 0 else suitc
        bx, by = int(c.bbox.x) + 3, int(c.bbox.y) + 3
        bw, bh = int(c.bbox.width), int(c.bbox.height)
        crop = corner[by - 1:by + bh + 1, bx - 1:bx + bw + 1]
        out.append((crop, lbl, [x0 + bx, y0 + by, bw, bh]))
    return out


def pot_chars(sub: np.ndarray, text: str) -> list:
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
    out = Path("data/vision2/coinpoker_assets_v2z")
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"client": "coinpoker", "version": "v2z",
                "ground_truth_status": "visually_labeled", "assets": []}

    def save(crop, name, label, frame_id, bbox, roi):
        p = out / f"{name}.png"
        p.write_bytes(encode_rgba_png(crop.shape[1], crop.shape[0],
                                      crop.tobytes()))
        manifest["assets"].append({
            "template_id": name, "label": label,
            "kind": "card_glyph" if roi in ("board", "hero")
            else "digit_glyph",
            "source_frame_id": frame_id, "bbox": list(bbox),
            "crop_ref": str(p), "roi": roi, "confidence": 0.92,
            "provenance": {"annotation": "V2V/V2W visual reading",
                           "ground_truth_status": "visually_labeled",
                           "label_source": "agent_vision_annotation"}})

    for fname, (vx, vy, vw, vh, cards) in BOARDS.items():
        w, h, rgba = decode_png((base / fname).read_bytes())
        full = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
        for i, label in enumerate(cards):
            card = full[int(vy * S):int((vy + vh) * S),
                        int((vx + i * vw) * S):int((vx + (i + 1) * vw) * S)]
            for j, (crop, lbl, b) in enumerate(corner_glyphs(card, label)):
                save(crop, f"board_{label}_{j}", lbl, fname[:23],
                     [int((vx + i * vw) * S) + b[0], int(vy * S) + b[1],
                      b[2], b[3]], "board")
    hero_report = {}
    for fname, (vx, vy, vw, vh, cards) in HERO.items():
        w, h, rgba = decode_png((base / fname).read_bytes())
        full = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
        for i, label in enumerate(cards):
            card = full[int(vy * S):int((vy + vh) * S),
                        int((vx + i * vw) * S):int((vx + (i + 1) * vw) * S)]
            if card.size < 4:
                hero_report[label] = {"extracted": False,
                                      "reason": "empty_crop"}
                continue
            bright = (card[..., 0].astype(int) * 0.299
                      + card[..., 1].astype(int) * 0.587
                      + card[..., 2].astype(int) * 0.114) > 200
            dark = (card[..., 0].astype(int) * 0.299
                    + card[..., 1].astype(int) * 0.587
                    + card[..., 2].astype(int) * 0.114) < 80
            glyphs = corner_glyphs(card, label)
            hero_report[label] = {
                "extracted": len(glyphs) > 0, "glyph_count": len(glyphs),
                "card_size": [card.shape[1], card.shape[0]],
                "white_px_ratio": round(float(bright.mean()), 3),
                "dark_px_ratio": round(float(dark.mean()), 3)}
            for j, (crop, lbl, b) in enumerate(glyphs):
                save(crop, f"hero_{label}_{j}", lbl, fname[:23],
                     [int((vx + i * vw) * S) + b[0], int(vy * S) + b[1],
                      b[2], b[3]], "hero")
    for fname, (slot, text) in POTS.items():
        w, h, rgba = decode_png((base / fname).read_bytes())
        full = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
        (vx, vy, vw, vh) = slot
        sub = full[int(vy * S):int((vy + vh) * S),
                   int(vx * S):int((vx + vw) * S)]
        chars = pot_chars(sub, text)
        for j, (crop, lbl, b) in enumerate(chars):
            save(crop, f"pot_{fname[:19]}_{j}", lbl, fname[:23],
                 [int(vx * S) + b[0], int(vy * S) + b[1], b[2], b[3]],
                 "pot_text")
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=1, ensure_ascii=False),
        encoding="utf-8")
    from collections import Counter
    print("assets:", len(manifest["assets"]))
    print("by kind:", dict(Counter(a["kind"] for a in manifest["assets"])))
    by_lbl = Counter(a["label"] for a in manifest["assets"])
    print("labels:", dict(sorted(by_lbl.items())))
    ranks = sorted(set(a["label"] for a in manifest["assets"]
                       if a["kind"] == "card_glyph"
                       and a["label"] in "AKQJT98765432"))
    suits = sorted(set(a["label"] for a in manifest["assets"]
                       if a["kind"] == "card_glyph"
                       and a["label"] in "shdc"))
    print("card rank coverage:", ranks)
    print("card suit coverage:", suits)
    print("hero probe:", json.dumps(hero_report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
