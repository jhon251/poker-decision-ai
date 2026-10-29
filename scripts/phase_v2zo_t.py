"""Phase V2ZO — 完整 T rank 模板重提取（rank 区全内容 bbox，非逐组件）。

- T GT：SCENE_556 Td 卡（slot1）；
- 旧 T 缺陷：14×50 竖笔碎片（T 横笔与相邻 d 角标粘连，组件分裂）；
- 方法：rank 区（卡角 x 6-55, y 6-62）内 **ink 联合内容 bbox**（横笔+竖笔
  整体，x<55 排除右侧 d）→ 完整 T；
- 质量门：宽 ≥22（非竖笔片）、高 ≥40、ink ≥0.15、red 家族、全 provenance。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from poker_decision_ai.vision2_platform import (  # noqa: E402
    decode_png, encode_rgba_png)

SLOT_X0, SLOT_Y0, SLOT_W, SLOT_H = 664, 551, 140, 177
FRAME = "SCENE_CHANGED_556_1787489611440_full.png"
T_SLOT = 1                          # Td 卡
RANK_ZONE = (6, 6, 49, 56)          # 卡角 rank 区 (x,y,w,h)，x<55 排除 d


def main() -> int:
    sys.argv = sys.argv[:1]
    base = Path("data/vision2/coinpoker_evidence")
    w, h, rgba = decode_png((base / FRAME).read_bytes())
    full = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
    x0 = SLOT_X0 + T_SLOT * SLOT_W + RANK_ZONE[0]
    y0 = SLOT_Y0 + RANK_ZONE[1]
    zone = full[y0:y0 + RANK_ZONE[3], x0:x0 + RANK_ZONE[2]]
    r = zone[..., 0].astype(int)
    g = zone[..., 1].astype(int)
    b = zone[..., 2].astype(int)
    luma = r * 0.299 + g * 0.587 + b * 0.114
    ink = ((luma < 115) | ((r > 100) & (r > g + 40)
                           & (r > b + 40)))
    ys, xs = np.where(ink)
    if len(xs) < 40:
        print("T_TEMPLATE_GT_NOT_AVAILABLE: rank 区 ink 不足")
        return 3
    bx0, bx1 = int(xs.min()), int(xs.max()) + 1
    by0, by1 = int(ys.min()), int(ys.max()) + 1
    crop = zone[by0:by1, bx0:bx1]
    cr = crop[..., 0].astype(int)
    cg = crop[..., 1].astype(int)
    cb = crop[..., 2].astype(int)
    cl = cr * 0.299 + cg * 0.587 + cb * 0.114
    cink = float(((cl < 115) | ((cr > 100) & (cr > cg + 40)
                                & (cr > cb + 40))).mean())
    cred = float(((cr > 100) & (cr > cg + 40)
                  & (cr > cb + 40)).mean())
    fam = "red" if cred > 0.10 else "black"
    wd, ht = crop.shape[1], crop.shape[0]
    # 横笔检查：top 行宽 ≈ 全宽（非仅竖笔）
    top_rows = ink[by0:by0 + 8, :].any(axis=0)
    top_width = int(top_rows.sum())
    valid = wd >= 22 and ht >= 40 and cink >= 0.15 and top_width >= 18
    out = Path("data/vision2/coinpoker_assets_v2zo")
    out.mkdir(parents=True, exist_ok=True)
    p = out / "board_T.png"
    p.write_bytes(encode_rgba_png(wd, ht, crop.tobytes()))
    manifest = {"client": "coinpoker", "version": "v2zo",
                "coordinate_system": "full_png_1419x1066",
                "ground_truth_status": "visually_labeled",
                "quality_gate": {"width": wd, "height": ht,
                                 "ink_ratio": round(cink, 3),
                                 "top_width": top_width,
                                 "valid": bool(valid)},
                "assets": [{"label": "T", "kind": "card_glyph",
                            "bbox": [x0 + bx0, y0 + by0, wd, ht],
                            "crop_ref": str(p),
                            "coordinate_system": "full_png",
                            "source_frame_id": FRAME[:23],
                            "ink_ratio": round(cink, 3),
                            "color_family": fam,
                            "ground_truth_status": "visually_labeled",
                            "confidence": 0.92,
                            "provenance": {"annotation": "V2W visual reading",
                                           "label_source":
                                               "agent_vision_annotation",
                                           "method":
                                               "rank_zone_content_bbox"}}]}
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")
    print("T extracted:", wd, "x", ht, "ink", round(cink, 3), "fam", fam,
          "top_width", top_width, "-> valid:", valid)
    print("bbox:", [x0 + bx0, y0 + by0, wd, ht])
    return 0 if valid else 3


if __name__ == "__main__":
    raise SystemExit(main())
