"""V2ZV — 从 V2ZT 真实 GT 帧提取 FN 模板（rank 3/6/Q/黑T + center-c）。

- 961: 3c(slot0) 6h(slot1) Ac(slot3) → rank 3/6 + center c（A♣ 中心大花色）
- 1331: Ts(slot0) Qc(slot1) → rank 黑T + Q
- 1523: 6s(slot0) 7s(slot1) → rank 6（复用 961 的 6）+（s 已存在）
方法：V2ZH 卡角 rank 区内容 bbox + 中心区最大组件（V2ZJ）。
黑 T 以 color_variant=black 注册（与红 T 共存）。
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
    "SETTLED_BOARD_CHECK_961_1787653653866_full.png":
        ["3c", "6h", "Th", "Ac", "5h"],
    "SETTLED_BOARD_CHECK_1331_1787653932447_full.png": ["Ts", "Qc", "Ks"],
    "SETTLED_BOARD_CHECK_1523_1787654061677_full.png": ["6s", "7s", "5h"],
}
WANT_RANKS = {"3": "3c", "6": "6h", "Q": "Qc", "T@black": "Ts"}


def ink_mask(arr):
    r = arr[..., 0].astype(int)
    g = arr[..., 1].astype(int)
    b = arr[..., 2].astype(int)
    luma = r * 0.299 + g * 0.587 + b * 0.114
    dark = luma < 115
    red = (r > 100) & (r > g + 40) & (r > b + 40)
    return (dark | red).astype("uint8")


def rank_glyph(slot):
    zone = slot[6:6 + 57, 6:6 + 49]
    m = ink_mask(zone)[3:-3, 3:-3]
    ys, xs = np.where(m)
    if len(xs) < 40:
        return None, None
    x0, x1, y0, y1 = xs.min() + 3, xs.max() + 3, ys.min() + 3, ys.max() + 3
    crop = zone[y0 - 1:y1 + 3, x0 - 1:x1 + 3]
    return crop, (int(x0) - 1, int(y0) - 1, int(crop.shape[1]), int(crop.shape[0]))


def center_suit(slot):
    region = slot[25 + 82:25 + 82 + 95, 25:25 + 100]
    m = ink_mask(region)
    comps = connected_components(m, min_area=60)
    if not comps:
        return None, None
    big = max(comps, key=lambda c: c.area)
    b = big.bbox
    bx, by = int(b.x), int(b.y)
    crop = region[by - 1:by + int(b.height) + 3,
                  bx - 1:bx + int(b.width) + 3]
    return crop, (int(bx) - 1, int(by) - 1, int(crop.shape[1]), int(crop.shape[0]))


def main() -> int:
    sys.argv = sys.argv[:1]
    base = Path("data/vision2/coinpoker_evidence")
    out = Path("data/vision2/coinpoker_assets_v2zv")
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"client": "coinpoker", "version": "v2zv",
                "coordinate_system": "full_png_1419x1066",
                "coordinate_fix": "V2ZA/V2ZH 实测坐标（settled 帧复用）",
                "ground_truth_status": "visually_labeled", "assets": []}
    for fname, labels in FRAMES.items():
        w, h, rgba = decode_png((base / fname).read_bytes())
        full = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
        for i, label in enumerate(labels):
            x0 = SLOT_X0 + i * SLOT_W
            slot = full[SLOT_Y0:SLOT_Y0 + SLOT_H, x0:x0 + SLOT_W]
            rankc = label[0]
            suitc = label[1]
            if label in ("3c", "6h", "Qc", "Ts"):
                crop, b = rank_glyph(slot)
                if crop is None or crop.size < 150:
                    print(f"rank extract fail {label}")
                    continue
                r = crop[..., 0].astype(int)
                g = crop[..., 1].astype(int)
                b_ = crop[..., 2].astype(int)
                luma = r * 0.299 + g * 0.587 + b_ * 0.114
                ink = float(((luma < 115) | ((r > 100) & (r > g + 40)
                                             & (r > b_ + 40))).mean())
                red_r = float(((r > 100) & (r > g + 40)
                               & (r > b_ + 40)).mean())
                if ink < 0.10:
                    print(f"rank ink low {label}")
                    continue
                lbl = "T@black" if label == "Ts" else rankc
                name = f"rank_{label.replace('/', 'x')}"
                p = out / f"{name}.png"
                p.write_bytes(encode_rgba_png(crop.shape[1], crop.shape[0],
                                              crop.tobytes()))
                manifest["assets"].append({
                    "label": lbl if lbl != "T@black" else "T",
                    "color_variant": "black" if lbl == "T@black" else None,
                    "kind": "card_glyph",
                    "bbox": [x0 + b[0], SLOT_Y0 + b[1], b[2], b[3]],
                    "crop_ref": str(p), "coordinate_system": "full_png",
                    "coordinate_fix": "V2ZA/V2ZH",
                    "source_frame_id": fname[:19],
                    "ink_ratio": round(ink, 3),
                    "color_family": "black" if red_r <= 0.10 else "red",
                    "ground_truth_status": "visually_labeled",
                    "confidence": 0.92,
                    "provenance": {"annotation": "V2ZT visual GT",
                                   "label_source": "agent_vision_annotation",
                                   "method": "rank_zone_content_bbox"}})
            if label == "Ac":
                crop, b = center_suit(slot)
                if crop is not None and crop.size >= 400:
                    p = out / "center_c_Ac.png"
                    p.write_bytes(encode_rgba_png(crop.shape[1],
                                                  crop.shape[0],
                                                  crop.tobytes()))
                    manifest["assets"].append({
                        "label": "c", "kind": "card_glyph",
                        "bbox": [x0 + 25 + b[0], SLOT_Y0 + 82 + b[1],
                                 b[2], b[3]],
                        "crop_ref": str(p),
                        "coordinate_system": "full_png",
                        "coordinate_fix": "V2ZA/V2ZH",
                        "source_frame_id": fname[:19],
                        "ink_ratio": 0.5, "color_family": "black",
                        "ground_truth_status": "visually_labeled",
                        "confidence": 0.9,
                        "provenance": {"annotation":
                                       "V2ZT visual GT (Ac center club)",
                                       "method": "center_largest_component"}})
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=1, ensure_ascii=False),
        encoding="utf-8")
    print("assets:", len(manifest["assets"]))
    for a in manifest["assets"]:
        print("  ", a["label"], a.get("color_variant"), a["bbox"],
              a["color_family"], "ink", a.get("ink_ratio"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
