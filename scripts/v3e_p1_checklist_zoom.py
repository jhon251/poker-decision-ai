"""V3E-P1 — 确认清单角标放大图（只读分析；仅写 gt_confirm_v3e_p1/）。

对清单帧提取 rank/suit 角标窗（rebuild 提取窗），最近邻放大 ×6 拼接，
供人工视觉裁决 ♣/♠ 与目标 rank。不做任何推断，不改任何冻结资产。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poker_decision_ai.vision2_platform.png_decode import decode_png
from poker_decision_ai.vision2_platform.png_encode import encode_rgba_png

EV = Path("data/vision2/coinpoker_evidence")
OUT = Path("gt_confirm_v3e_p1")

# 与 p33_hero_gt_adjudicate.py 一致的 rebuild 提取窗（含上下文余量）
CROP = {1: {"rank": (818, 1016, 856, 1076), "suit": (818, 1070, 868, 1130)},
        2: {"rank": (934, 1010, 974, 1070), "suit": (928, 1066, 982, 1122)}}
SCALE = 6


def main() -> int:
    manifest = json.loads(
        Path("gt_confirm_v3e_p1/checklist_manifest.json")
        .read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)
    for item in manifest["items"]:
        f = item["frame"]
        w, h, rgba = decode_png((EV / f).read_bytes())
        arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
        tiles = []
        for ci in (1, 2):
            for kind in ("rank", "suit"):
                if f"card{ci}_{kind}" not in item.get("show", []):
                    continue
                x0, y0, x1, y1 = CROP[ci][kind]
                roi = arr[y0:y1, x0:x1].copy()
                big = np.repeat(np.repeat(roi, SCALE, axis=0), SCALE, axis=1)
                tiles.append((f"c{ci}_{kind}", big))
        if not tiles:
            continue
        W = max(t.shape[1] for _, t in tiles) + 16
        H = sum(t.shape[0] + 16 for _, t in tiles) + 16
        sheet = np.zeros((H, W, 4), dtype=np.uint8)
        sheet[..., :3] = 30
        sheet[..., 3] = 255
        yy = 16
        for _name, t in tiles:
            sheet[yy:yy + t.shape[0], 8:8 + t.shape[1]] = t
            yy += t.shape[0] + 16
        out = OUT / f"zoom__{Path(f).stem}.png"
        out.write_bytes(encode_rgba_png(sheet.shape[1], sheet.shape[0],
                                        sheet.tobytes()))
        print(out.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
