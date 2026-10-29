"""P24-A — 3 个失败 GT 的胶囊精确定位 + 结构诊断。"""
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
    ("BUTTON_CHANGED_167_1787992682749_full.png", "S6", (850, 780, 110, 40)),
    ("POT_CHANGED_12_1787995290794_full.png", "S5", (1300, 885, 150, 50)),
    ("BUTTON_CHANGED_598_1787993753587_full.png", "S5", (1300, 885, 150, 50)),
]


def main() -> int:
    for fid, seat, (x0, y0, w, h) in CASES:
        ww, hh, rgba = decode_png((EV / fid).read_bytes())
        arr = np.frombuffer(rgba, dtype=np.uint8).reshape(hh, ww, 4)
        lum = gray_luma(arr)
        print(f"\n=== {fid[:28]} {seat} ===")
        # 细搜索：区域内滑动 150x40 找胶囊
        found = None
        for yy in range(max(0, y0 - 30), min(hh - 40, y0 + 30), 10):
            for xx in range(max(0, x0 - 40), min(ww - 150, x0 + 40), 10):
                cap = detect_bet_capsule(lum, (xx, yy, 150, 40))
                if cap is None:
                    continue
                bx, by, bw, bh = cap["bbox"]
                found = (bx, by, bw, bh, cap)
        if found is None:
            print("  区域内无胶囊检出")
        else:
            bx, by, bw, bh, cap = found
            print(f"  胶囊 bbox=({bx},{by},{bw},{bh}) dark={cap['dark_ratio']:.2f}")
            glyphs = segment_bet_glyphs(cap["inner"])
            if glyphs is None:
                print("  分割失败（segment 返回 None）")
            else:
                print(f"  token 数={len(glyphs)}")
                for gl in glyphs:
                    print(f"    token bbox=({gl['x']},{gl['y']}) "
                          f"w={gl['w']} h={gl['h']}")
                labels = classify_bet_glyphs(glyphs, tpls)
                if labels is None:
                    print("  分类失败")
                else:
                    print(f"  读={''.join(labels[0])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
