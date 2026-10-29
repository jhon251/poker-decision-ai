"""Phase V2X — CoinPoker GT → Runtime 视觉资产（P0/P1/P2 字形提取）。

标注来源：V2V/V2W 视觉读取真值（frame 188/452/556）。
方法：白卡区域 → 墨色组件（rank 角标/角标花色/中心大花色）→ 按标注值
逐卡赋值标签；pot 白字字符切分 → 数字标签。
每个资产：source_frame_id/bbox/crop_ref/label/confidence/provenance/
ground_truth_status=visually_labeled。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from poker_decision_ai.vision2_platform import (  # noqa: E402
    decode_png, encode_rgba_png)

S = 1419.0 / 922.0          # 视图 → PNG 缩放（标注 bbox 为视图坐标）

#: 帧 → (board 卡坐标视图, 牌面值)（V2V/V2W 视觉真值）
BOARDS = {
    "SCENE_CHANGED_188_1787489345211_full.png": (
        (435, 525, 93, 115), ["2c", "Ac", "8d", "9c", "Ks"]),
    "SCENE_CHANGED_556_1787489611440_full.png": (
        (435, 525, 93, 115), ["5h", "Td", "7s"]),
}
HERO = {
    "SCENE_CHANGED_188_1787489345211_full.png": ((108, 740, 102, 88),
                                                 ["Qs", "9d"]),
}
POTS = {
    "SCENE_CHANGED_188_1787489345211_full.png": ((578, 480, 120, 33),
                                                 "25,000"),
}


def _comps(sub: np.ndarray, darkness: bool = True, min_area: int = 30):
    """墨色（或亮色）连通组件（8 连通 BFS 简化为 numpy 标记）。"""
    r = sub[..., 0].astype(int)
    g = sub[..., 1].astype(int)
    b = sub[..., 2].astype(int)
    luma = (r * 0.299 + g * 0.587 + b * 0.114)
    mask = luma < 110 if darkness else luma > 150
    from poker_decision_ai.vision2.pixel.geometry import connected_components
    m = mask.astype("uint8")
    comps = connected_components(m, min_area=min_area)
    return [c.bbox for c in comps]


def main() -> int:
    sys.argv = sys.argv[:1]
    base = Path("data/vision2/coinpoker_evidence")
    out = Path("data/vision2/coinpoker_assets_v2x")
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"client": "coinpoker", "version": "v2x",
                "ground_truth_status": "visually_labeled",
                "assets": []}

    def save(crop, name, label, frame_id, bbox, conf, roi):
        p = out / f"{name}.png"
        p.write_bytes(encode_rgba_png(crop.shape[1], crop.shape[0],
                                      crop.tobytes()))
        manifest["assets"].append({
            "template_id": name, "label": label, "kind": "card_glyph"
            if roi in ("board", "hero") else "digit_glyph",
            "source_frame_id": frame_id, "bbox": list(bbox),
            "crop_ref": str(p), "roi": roi, "confidence": conf,
            "provenance": {"annotation": "V2V/V2W visual reading",
                           "ground_truth_status": "visually_labeled",
                           "label_source": "agent_vision_annotation"}})

    for fname, (slot, cards) in BOARDS.items():
        w, h, rgba = decode_png((base / fname).read_bytes())
        full = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
        (vx, vy, vw, vh) = slot
        x0 = int(vx * S)
        for i, label in enumerate(cards):
            card = full[int(vy * S):int((vy + vh) * S),
                        int((vx + i * vw) * S):int((vx + (i + 1) * vw) * S)]
            if card.size < 4:
                continue
            rankc, suitc = label[0], label[1]
            # 墨色组件（过滤边框环：bbox 覆盖 ≥85% 卡面 → 跳过）
            cands = _comps(card)
            ch, cwd = card.shape[0], card.shape[1]
            keeps = [b for b in cands
                     if not (b.width >= 0.85 * cwd
                             and b.height >= 0.85 * ch)]
            comps = sorted(keeps, key=lambda b: (b.y * 1000 + b.x))
            for j, b in enumerate(comps[:3]):
                cy = max(0, int(b.y) - 2)
                cx = max(0, int(b.x) - 2)
                crop = card[cy:cy + int(b.height) + 4,
                            cx:cx + int(b.width) + 4]
                lbl = rankc if j == 0 else (suitc if j == 1 else suitc)
                save(crop, f"board_{label}_{j}", lbl, fname[:23],
                     [int(x0 + cx), int(vy * S + cy),
                      int(b.width) + 4, int(b.height) + 4], 0.9, "board")
    for fname, (slot, cards) in HERO.items():
        w, h, rgba = decode_png((base / fname).read_bytes())
        full = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
        (vx, vy, vw, vh) = slot
        for i, label in enumerate(cards):
            card = full[int(vy * S):int((vy + vh) * S),
                        int((vx + i * vw) * S):int((vx + (i + 1) * vw) * S)]
            if card.size < 4:
                continue
            rankc, suitc = label[0], label[1]
            comps = sorted(_comps(card), key=lambda b: (b.y * 1000 + b.x))
            for j, b in enumerate(comps[:3]):
                cy = max(0, int(b.y) - 2)
                cx = max(0, int(b.x) - 2)
                crop = card[cy:cy + int(b.height) + 4, cx:cx + int(b.width) + 4]
                lbl = rankc if j == 0 else suitc
                save(crop, f"hero_{label}_{j}", lbl, fname[:23],
                     [int(vx * S + cx), int(vy * S + cy), int(b.width) + 4,
                      int(b.height) + 4], 0.9, "hero")
    for fname, (slot, text) in POTS.items():
        w, h, rgba = decode_png((base / fname).read_bytes())
        full = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
        (vx, vy, vw, vh) = slot
        sub = full[int(vy * S):int((vy + vh) * S),
                   int(vx * S):int((vx + vw) * S)]
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
        if len(chars) == len(text):
            for j, (cx, cy, cw, chh) in enumerate(chars):
                crop = sub[cy - 1:cy + chh + 1, max(0, cx - 1):cx + cw + 1]
                save(crop, f"pot_{fname[:19]}_{j}", text[j], fname[:23],
                     [int(vx * S + cx), int(vy * S + cy), cw, chh], 0.9,
                     "pot_text")
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")
    print("assets:", len(manifest["assets"]))
    from collections import Counter
    print("by kind:", dict(Counter(a["kind"] for a in manifest["assets"])))
    print("labels:", sorted({a["label"] for a in manifest["assets"]}))
    print("saved:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
