"""Phase V2ZA — CoinPoker 坐标系统对账（实证裁剪校验，零猜测）。

对 V2V/V2W 标注 ROI 逐一验证：
  - 解释 A：标注坐标 = 全尺寸 PNG 坐标（1419×1066）；
  - 解释 B：标注坐标 = 显示缩放坐标（×1.5391 → PNG）。
每候选裁剪 → 检查：in-bounds / 非空 / 内容签名（pot=亮字、board/hero=白卡
+墨色、D=红、徽章=文字块）。通过者即有效坐标系。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from poker_decision_ai.vision2_platform import decode_png  # noqa: E402

S = 1419.0 / 922.0          # 显示 → PNG
PNG = (1419.0, 1066.0)

#: 候选 ROI（来源标注；原坐标 + 内容签名期望）
CANDIDATES = [
    # (name, source_frame, bbox, signature)
    ("hero_cards", "SCENE_CHANGED_188_1787489345211_full.png",
     [108, 740, 202, 175], "hero"),
    ("pot_25k", "SCENE_CHANGED_188_1787489345211_full.png",
     [578, 480, 177, 40], "text"),
    ("board_5cards", "SCENE_CHANGED_188_1787489345211_full.png",
     [435, 525, 465, 190], "board"),
    ("pot_640", "SCENE_CHANGED_283_1787489414967_full.png",
     [585, 300, 125, 45], "text"),
    ("pot_263k", "SCENE_CHANGED_452_1787489541883_full.png",
     [585, 450, 120, 33], "text"),
    ("pot_280k", "SCENE_CHANGED_556_1787489611440_full.png",
     [590, 445, 140, 33], "text"),
    ("flop_3cards", "SCENE_CHANGED_556_1787489611440_full.png",
     [435, 525, 265, 190], "board"),
]


def signature(arr: np.ndarray, kind: str) -> dict:
    if arr.size < 4:
        return {"ok": False, "reason": "empty"}
    r = arr[..., 0].astype(int)
    g = arr[..., 1].astype(int)
    b = arr[..., 2].astype(int)
    luma = r * 0.299 + g * 0.587 + b * 0.114
    bright = float((luma > 200).mean())
    ink = float(((luma < 115) | ((r > 110) & (g < 100) & (b < 100))).mean())
    if kind == "text":
        ok = 0.02 <= bright <= 0.9 and (luma > 150).sum() > 20
    elif kind == "board":
        # 白卡矩形计数（区分 3/5 卡板区 vs 错位区域）
        from poker_decision_ai.vision2.pixel.geometry import (
            connected_components)
        m = (luma > 200).astype("uint8")
        comps = connected_components(m, min_area=4000)
        cards = sum(1 for c in comps
                    if 0.4 <= c.aspect_ratio <= 1.3
                    and c.fill_ratio >= 0.55)
        ok = cards >= 3 and ink > 0.01
        return {"ok": bool(ok), "bright_ratio": round(bright, 3),
                "ink_ratio": round(ink, 3), "white_cards": cards}
    elif kind == "hero":
        ok = 0.3 <= bright <= 0.95 and ink > 0.01
    else:
        ok = bright < 0.5
    return {"ok": bool(ok), "bright_ratio": round(bright, 3),
            "ink_ratio": round(ink, 3)}


def main() -> int:
    sys.argv = sys.argv[:1]
    base = Path("data/vision2/coinpoker_evidence")
    report = {"source_image_size": list(PNG),
              "runtime_capture_size": list(PNG),
              "annotation_candidates": [], "concluded": {}}
    for name, fname, bbox, kind in CANDIDATES:
        w, h, rgba = decode_png((base / fname).read_bytes())
        full = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
        entry = {"name": name, "source_frame": fname, "bbox": bbox,
                 "coordinate_system": "unknown",
                 "candidates": []}
        for label, scale in (("A: as_full_png", 1.0),
                             ("B: display_x1.539", S)):
            x0 = int(bbox[0] * scale)
            y0 = int(bbox[1] * scale)
            x1 = int((bbox[0] + bbox[2]) * scale)
            y1 = int((bbox[1] + bbox[3]) * scale)
            if x0 < 0 or y0 < 0 or x1 > w or y1 > h or x1 <= x0 or y1 <= y0:
                entry["candidates"].append(
                    {"interpretation": label, "converted": [x0, y0, x1 - x0,
                                                            y1 - y0],
                     "valid": False, "reason": "out_of_bounds"})
                continue
            crop = full[y0:y1, x0:x1]
            sig = signature(crop, kind)
            entry["candidates"].append(
                {"interpretation": label,
                 "converted": [x0, y0, x1 - x0, y1 - y0],
                 "valid": sig["ok"], "signature": sig})
        ok_interps = [c["interpretation"] for c in entry["candidates"]
                      if c.get("valid")]
        entry["concluded_system"] = ok_interps[0] if ok_interps else None
        report["annotation_candidates"].append(entry)
        report["concluded"][name] = entry["concluded_system"]
    Path("data/vision2/coinpoker_coordinate_audit.json").write_text(
        json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report["concluded"], indent=1, ensure_ascii=False))
    for e in report["annotation_candidates"]:
        print(f"  {e['name']}: " + " | ".join(
            f"{c['interpretation']}={c['valid']}"
            for c in e["candidates"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
