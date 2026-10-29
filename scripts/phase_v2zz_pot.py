"""V2ZZ P0 — pot 数字健壮分割 + 补缺模板（3/8/1/7/9 从真实 GT 帧）。

- 分割：亮字掩码（luma>140）+ 列投影 gap-run（细阈值抗粘连）；
- GT 值映射：逐字符位置左→右对齐视觉 GT 串；
- 严格：分割数 != 串长 → 该帧跳过（不猜）；缺模板保持 null。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from poker_decision_ai.vision2_platform import (  # noqa: E402
    decode_png, encode_rgba_png)

#: (帧, GT pot 串, 数字区 full_png bbox 粗定位)
CASES = [
    ("SETTLED_BOARD_CHECK_961_1787653653866_full.png", "23,280",
     (860, 730, 420, 80)),
    ("SETTLED_BOARD_CHECK_1331_1787653932447_full.png", "6,600",
     (860, 730, 420, 80)),
    ("SETTLED_BOARD_CHECK_1523_1787654061677_full.png", "385,960",
     (860, 720, 420, 90)),
]


def seg_chars(sub, thr=140):
    r = sub[..., 0].astype(int)
    g = sub[..., 1].astype(int)
    b = sub[..., 2].astype(int)
    luma = r * 0.299 + g * 0.587 + b * 0.114
    col = (luma > thr).sum(axis=0) > 2
    # gap-run：列开启→关闭；粘连字符用更细 gap（1 列断开即分段）
    runs = []
    x = 0
    while x < len(col):
        if not col[x]:
            x += 1
            continue
        s = x
        while x < len(col) and col[x]:
            x += 1
        runs.append((s, x))
    return runs


def main() -> int:
    sys.argv = sys.argv[:1]
    base = Path("data/vision2/coinpoker_evidence")
    out = Path("data/vision2/coinpoker_assets_v2zy")
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"client": "coinpoker", "version": "v2zy",
                "coordinate_system": "full_png_1419x1066",
                "ground_truth_status": "visually_labeled", "assets": []}
    for fname, text, (bx, by, bw, bh) in CASES:
        w, h, rgba = decode_png((base / fname).read_bytes())
        full = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
        sub = full[by:by + bh, bx:bx + bw]
        runs = seg_chars(sub)
        if len(runs) != len(text):
            print(f"{fname[:22]}: seg {len(runs)} != {len(text)} {runs}")
            continue
        for j, (s, e) in enumerate(runs):
            crop = sub[:, s:e]
            if text[j] in "123789" and text[j] not in \
                    {a["label"] for a in manifest["assets"]}:
                p = out / f"pot_{text[j]}_{fname[:6]}.png"
                p.write_bytes(encode_rgba_png(crop.shape[1], crop.shape[0],
                                              crop.tobytes()))
                manifest["assets"].append({
                    "label": text[j], "kind": "digit_glyph",
                    "bbox": [bx + s, by, e - s, bh],
                    "crop_ref": str(p), "coordinate_system": "full_png",
                    "coordinate_fix": "V2ZZ band detect",
                    "source_frame_id": fname[:19],
                    "ink_ratio": round(float(
                        (crop[..., 0].astype(int) * 0.299
                         + crop[..., 1].astype(int) * 0.587
                         + crop[..., 2].astype(int) * 0.114 > 150).mean()),
                        3),
                    "color_family": "white",
                    "ground_truth_status": "visually_labeled",
                    "confidence": 0.9,
                    "provenance": {"annotation": "V2ZT visual GT pot",
                                   "label_source":
                                       "agent_vision_annotation"}})
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")
    print("pot assets:", len(manifest["assets"]))
    for a in manifest["assets"]:
        print("  ", a["label"], a["bbox"], a["source_frame_id"][:18])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
