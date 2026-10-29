"""Phase V2Y — CoinPoker 视觉资产补齐（多帧逐卡槽提取 + pot 多位置 + hero）。

GT 来源：V2V/V2W 视觉读取真值：
  188: 板 2c Ac 8d 9c Ks + 明牌 Qs 9d + pot 25,000
  452: 板 5s 6c 4s 5h 6h + pot 263,520
  556: 板 5h Td 7s + pot 280,220
  283: pot 640
提取规则（确定性）：白卡 → 墨色组件 → 边框环过滤 →
  rank = y 最小且 x < 45% 卡宽；角标花色 = 紧随 rank 的小组件；
  中心大花色 = 最大组件（同花色标签）；pot = 亮字字符切分。
每资产：source_frame_id/bbox/crop_ref/label/confidence/GT=visually_labeled。
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
FRAMES = {
    "SCENE_CHANGED_188_1787489345211_full.png": {
        "board": {"slot": (435, 525, 93, 115),
                  "cards": ["2c", "Ac", "8d", "9c", "Ks"]},
        "hero": {"slot": (108, 740, 102, 88),
                 "cards": ["Qs", "9d"]},
        "pots": [((578, 480, 120, 33), "25,000")]},
    "SCENE_CHANGED_452_1787489541883_full.png": {
        "board": {"slot": (435, 525, 93, 115),
                  "cards": ["5s", "6c", "4s", "5h", "6h"]},
        "pots": [((585, 450, 120, 33), "263,520")]},
    "SCENE_CHANGED_556_1787489611440_full.png": {
        "board": {"slot": (435, 525, 93, 115),
                  "cards": ["5h", "Td", "7s"]},
        "pots": [((590, 445, 140, 33), "280,220")]},
}
POT_EXTRA = {}


def card_glyphs(card: np.ndarray, label: str) -> list:
    """单卡 → [(crop, glyph_label, bbox)]（rank + 角标花色 + 中心花色）。

    先裁卡内芯（去 6px 边距）再提取——角标字形常触卡边框，
    直接提取会与边框环合并而被误过滤。
    """
    rankc, suitc = label[0], label[1]
    core = card[6:-6, 6:-6]
    offset = 6
    r = core[..., 0].astype(int)
    g = core[..., 1].astype(int)
    b = core[..., 2].astype(int)
    luma = r * 0.299 + g * 0.587 + b * 0.114
    dark = luma < 115
    red = (r > 110) & (g < 100) & (b < 100)      # 红/粉花色
    mask = (dark | red).astype("uint8")
    ch, cwd = core.shape[0], core.shape[1]
    comps = [c for c in connected_components(mask, min_area=12)
             if not (c.bbox.width >= 0.85 * cwd
                     and c.bbox.height >= 0.85 * ch)]
    if not comps:
        return []
    corner = [c for c in comps
              if c.bbox.x < 0.45 * cwd and c.bbox.y < 0.35 * ch]
    center = [c for c in comps if c.bbox.x >= 0.45 * cwd
              or c.bbox.y >= 0.35 * ch]
    corner.sort(key=lambda c: (c.bbox.y, c.bbox.x))
    out = []
    for j, c in enumerate(corner[:2]):
        lbl = rankc if j == 0 else suitc
        b = c.bbox
        out.append((core[max(0, int(b.y) - 2):int(b.y) + int(b.height) + 2,
                         max(0, int(b.x) - 2):int(b.x) + int(b.width) + 2],
                    lbl, [int(b.x) + offset, int(b.y) + offset,
                          int(b.width), int(b.height)]))
    if center:
        big = max(center, key=lambda c: c.area)
        b = big.bbox
        out.append((core[max(0, int(b.y) - 2):int(b.y) + int(b.height) + 2,
                         max(0, int(b.x) - 2):int(b.x) + int(b.width) + 2],
                    suitc, [int(b.x) + offset, int(b.y) + offset,
                            int(b.width), int(b.height)]))
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
    out = Path("data/vision2/coinpoker_assets_v2y")
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"client": "coinpoker", "version": "v2y",
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

    for fname, spec in FRAMES.items():
        w, h, rgba = decode_png((base / fname).read_bytes())
        full = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
        fid = fname[:23]
        if "board" in spec:
            (vx, vy, vw, vh) = spec["board"]["slot"]
            for i, label in enumerate(spec["board"]["cards"]):
                card = full[int(vy * S):int((vy + vh) * S),
                            int((vx + i * vw) * S):int((vx + (i + 1) * vw) * S)]
                for j, (crop, lbl, b) in enumerate(card_glyphs(card, label)):
                    save(crop, f"board_{label}_{j}", lbl, fid,
                         [int((vx + i * vw) * S) + b[0],
                          int(vy * S) + b[1], b[2], b[3]], "board")
        if "hero" in spec:
            (vx, vy, vw, vh) = spec["hero"]["slot"]
            for i, label in enumerate(spec["hero"]["cards"]):
                card = full[int(vy * S):int((vy + vh) * S),
                            int((vx + i * vw) * S):int((vx + (i + 1) * vw) * S)]
                for j, (crop, lbl, b) in enumerate(card_glyphs(card, label)):
                    save(crop, f"hero_{label}_{j}", lbl, fid,
                         [int((vx + i * vw) * S) + b[0],
                          int(vy * S) + b[1], b[2], b[3]], "hero")
        for (slot, text) in spec.get("pots", []):
            (vx, vy, vw, vh) = slot
            sub = full[int(vy * S):int((vy + vh) * S),
                       int(vx * S):int((vx + vw) * S)]
            for j, (crop, lbl, b) in enumerate(pot_chars(sub, text)):
                save(crop, f"pot_{fid}_{text.replace('.', 'x')}_{j}", lbl,
                     fid, [int(vx * S) + b[0], int(vy * S) + b[1],
                           b[2], b[3]], "pot_text")
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
