"""P24-A — S5 候选胶囊逐读测试。"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2.pixel.realclient.coinpoker_bet import (  # noqa: E402
    gray_luma, detect_bet_capsule, segment_bet_glyphs, classify_bet_glyphs,
    load_bet_templates)
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402

EV = Path("data/vision2/coinpoker_evidence")
tpls = load_bet_templates()

CASES = [
    ("POT_CHANGED_12_1787995290794_full.png", "89600",
     [(1492, 753, 48, 26), (1531, 954, 73, 19), (1679, 947, 98, 25),
      (1579, 994, 68, 33), (1661, 994, 66, 34)]),
    ("BUTTON_CHANGED_598_1787993753587_full.png", "34500",
     [(1586, 958, 13, 3), (1661, 954, 61, 18), (1680, 998, 28, 23)]),
]


def main() -> int:
    for fid, gt, caps in CASES:
        w, h, rgba = decode_png((EV / fid).read_bytes())
        arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
        lum = gray_luma(arr)
        print(f"\n=== {fid[:28]} GT={gt} ===")
        for (bx, by, bw, bh) in caps:
            # 用胶囊 bbox 作为 ROI 重跑（+padding）
            cap = detect_bet_capsule(lum, (max(0, bx - 10), max(0, by - 8),
                                           bw + 20, bh + 16))
            if cap is None:
                print(f"  ({bx},{by},{bw},{bh}): 重新检测失败")
                continue
            bx2, by2, bw2, bh2 = cap["bbox"]
            glyphs = segment_bet_glyphs(cap["inner"])
            if glyphs is None:
                print(f"  ({bx},{by}): seg 失败")
                continue
            labels = classify_bet_glyphs(glyphs, tpls)
            text = "".join(labels[0]) if labels else "?"
            print(f"  ({bx},{by},{bw},{bh}): tokens={len(glyphs)} "
                  f"读={text!r} {'HIT' if text == gt or text.replace(',','')==gt else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
