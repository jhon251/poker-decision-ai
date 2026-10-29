"""V2ZY — 提取 pot 缺失数字（3/8）与 status 徽章模板（folded/away/all-in）。

- pot 3/8：961 帧 "底池 23,280"（digit 区 951-1261 full 坐标）逐字符切分；
- status 徽章：folded『盖牌』(187)、away『暂离』(580)、all-in『全押』(1523)。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from poker_decision_ai.vision2_platform import (  # noqa: E402
    decode_png, encode_rgba_png)

S = 1419.0 / 922.0


def seg_bright(sub, min_w=4):
    r = sub[..., 0].astype(int)
    g = sub[..., 1].astype(int)
    b = sub[..., 2].astype(int)
    luma = r * 0.299 + g * 0.587 + b * 0.114
    col = (luma > 190).sum(axis=0) > 2
    runs = []
    x = 0
    while x < len(col):
        if not col[x]:
            x += 1
            continue
        s = x
        while x < len(col) and col[x]:
            x += 1
        if x - s >= min_w:
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

    def save(crop, name, label, fname, bbox, kind):
        p = out / f"{name}.png"
        p.write_bytes(encode_rgba_png(crop.shape[1], crop.shape[0],
                                      crop.tobytes()))
        r = crop[..., 0].astype(int)
        g = crop[..., 1].astype(int)
        b = crop[..., 2].astype(int)
        luma = r * 0.299 + g * 0.587 + b * 0.114
        ink = float((luma > 150).mean())
        manifest["assets"].append({
            "label": label, "kind": kind,
            "bbox": [int(v) for v in bbox], "crop_ref": str(p),
            "coordinate_system": "full_png", "source_frame_id": fname[:19],
            "ink_ratio": round(ink, 3), "color_family": "white",
            "ground_truth_status": "visually_labeled", "confidence": 0.9,
            "provenance": {"annotation": "V2ZT/V2V visual reading",
                           "label_source": "agent_vision_annotation"}})

    # pot digits from 961 "底池 23,280" (digit region full x 951-1261, y 723-815)
    w, h, rgba = decode_png((base / "SETTLED_BOARD_CHECK_961_1787653653866_full.png").read_bytes())
    full = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
    sub = full[723:815, 951:1261]
    runs = seg_bright(sub)
    text = "23,280"
    if len(runs) == len(text):
        for j, (s, e) in enumerate(runs):
            crop = sub[:, s:e]
            save(crop, f"pot_{text[j]}_{j}", text[j], "SETTLED_961",
                 [951 + s, 723, e - s, 92], "digit_glyph")
    else:
        print("pot seg mismatch:", len(runs), "vs", len(text), runs)

    # status badges
    badges = [
        ("SETTLED_BOARD_CHECK_187_1787489344228_full.png",
         (870, 218, 105, 37), "folded", "盖牌"),
        ("SETTLED_BOARD_CHECK_580_1787489625695_full.png",
         (610, 140, 90, 50), "away", "暂离"),
        ("SETTLED_BOARD_CHECK_1523_1787654061677_full.png",
         (400, 240, 90, 40), "allin", "全押"),
    ]
    for fname, (bx, by, bw, bh), label, _txt in badges:
        w, h, rgba = decode_png((base / fname).read_bytes())
        full = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
        if bx + bw > w or by + bh > h:
            print(f"badge {label} bbox oob for {fname[:20]}")
            continue
        crop = full[by:by + bh, bx:bx + bw]
        save(crop, f"badge_{label}", label, fname, [bx, by, bw, bh],
             "status_badge")
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")
    print("assets:", len(manifest["assets"]))
    for a in manifest["assets"]:
        print("  ", a["label"], a["kind"], a["bbox"], "ink", a["ink_ratio"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
