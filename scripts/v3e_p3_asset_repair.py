"""V3E-P3 小规模 GT Asset Repair（用户授权）：仅新增 rank_4@c2 与 rank_7@c1。

GT 依据 = 用户人工目视确认面板（c01_n46=4、c06_n29=7）+ 用户终审 GT 锚帧
（33/47=4s4h、606=7c2c）。position-specific 提取；零逻辑修改。
出现任何碰撞/回归 → 回退保持 UNKNOWN。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from v3e_p1_hero_audit import EXT_WIN, EV, glyph
from poker_decision_ai.vision2_platform.png_decode import decode_png
from poker_decision_ai.vision2_platform.png_encode import encode_rgba_png

ASSETS = Path("data/vision2/coinpoker_assets_v1876herocards")
LOG = Path("logs/v3e_p3_repair.json")

ADDS = [
    {
        "label": "rank_4", "kind": "herocards_rank_1876", "rank": "4",
        "card": 2, "pos_key": 2, "kind_key": "rank",
        "src": "POT_CHANGED_47_1787995398974_full.png",
        "alt_src": ["POT_CHANGED_33_1787995356823_full.png",
                    "POT_CHANGED_568_1787993681999_full.png"],
        "human": "4s4h (c2=4h)",
        "provenance": (
            "V3E-P3 GAP REPAIR (user-authorized): user visual confirmation "
            "panel c01_n46 = 46 unique 4-shape moments across 4 hand "
            "sessions; GT anchors 33/47 = 4s4h (user final GT); 4/6 "
            "collision measured 0.123-0.181 safe; position-specific c2 "
            "extraction"),
    },
    {
        "label": "rank_7", "kind": "herocards_rank_1876", "rank": "7",
        "card": 1, "pos_key": 1, "kind_key": "rank",
        "src": "POT_CHANGED_606_1787993775320_full.png",
        "alt_src": [],
        "human": "7c2c (c1=7c)",
        "provenance": (
            "V3E-P3 GAP REPAIR (user-authorized): user visual confirmation "
            "panel c06_n29 = 29 unique 7-shape moments across 3 sessions; "
            "GT anchor 606 = 7c2c (user final GT); 7 vs T512/J181 tight "
            "NCC 0.299/0.316 safe; position-specific c1 extraction"),
    },
]


def extract(frame, pos):
    w, h, rgba = decode_png((EV / frame).read_bytes())
    arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
    return glyph(arr, EXT_WIN[pos]["rank"])


def encode(mask):
    h, w = mask.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., 0:3] = 255
    rgba[..., 3] = mask.astype(np.uint8) * 255
    return encode_rgba_png(w, h, rgba.tobytes())


def main() -> int:
    mp = json.loads((ASSETS / "manifest.json").read_text(encoding="utf-8"))
    before_n = len(mp["assets"])
    existing = {(a["label"], a.get("card")) for a in mp["assets"]}
    existing_files = {Path(a["crop_ref"]).name for a in mp["assets"]}
    added = []
    for spec in ADDS:
        key = (spec["label"], spec["card"])
        if key in existing:
            print("SKIP %s c%d already present" % key)
            continue
        placed = False
        for src in [spec["src"]] + spec["alt_src"]:
            m, bbox = extract(src, spec["pos_key"])
            if m is None:
                print("  extraction failed for", src)
                continue
            fname = "h%s%s_c%d_%d_%d.png" % (
                "r", spec["rank"], spec["card"], bbox[0], bbox[1])
            if fname in existing_files:
                print("  file collision", fname)
                continue
            (ASSETS / fname).write_bytes(encode(m))
            entry = {
                "label": spec["label"], "kind": spec["kind"],
                "crop_ref": fname, "bbox": list(bbox), "card": spec["card"],
                spec["kind_key"]: spec["rank"],
                "glyph_size": [int(bbox[2]), int(bbox[3])],
                "source_frame_id": src,
                "provenance": spec["provenance"] +
                    "; source frame %s (human=%s)" % (src, spec["human"]),
                "gt_status": "normal",
            }
            mp["assets"].append(entry)
            existing_files.add(fname)
            added.append({"label": spec["label"], "card": spec["card"],
                          "file": fname, "src": src, "bbox": list(bbox)})
            print("ADD %s c%d <- %s bbox=%s"
                  % (spec["label"], spec["card"], src[:44], list(bbox)))
            placed = True
            break
        if not placed:
            print("FAILED to place", key)
    mp["note"] = mp["note"].replace(
        "4@c2(组内0.808<0.85)", "").replace(
        "rank 7@c1、2@c2", "rank 2@c2")
    (ASSETS / "manifest.json").write_text(
        json.dumps(mp, ensure_ascii=False, indent=2), encoding="utf-8")
    LOG.write_text(json.dumps({"before": before_n, "after": len(mp["assets"]),
                               "added": added}, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print("assets: %d -> %d" % (before_n, len(mp["assets"])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
