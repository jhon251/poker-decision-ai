"""P16-E — Bet 1876 字形尺度 + v2bet1280 复用实测 v2（READ-ONLY）。

用粗网格搜索找胶囊（已证明有效），再逐胶囊做 segment+classify。
"""
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


def _capsules(lum):
    found = []
    for y0 in range(60, 1340, 60):
        for x0 in range(0, 1876 - 260, 60):
            cap = detect_bet_capsule(lum, (x0, y0, 260, 80))
            if cap is None:
                continue
            bx, by, bw, bh = cap["bbox"]
            dup = any(abs(f[0] - bx) < 40 and abs(f[1] - by) < 30 for f in found)
            if not dup:
                found.append((bx, by, bw, bh, cap))
    return found


def main() -> int:
    tpls = load_bet_templates()
    names = [
        "BUTTON_CHANGED_167_1787992682749_full.png",
        "POT_CHANGED_12_1787995290794_full.png",
        "POT_CHANGED_102_1787992505794_full.png",
        "BUTTON_CHANGED_598_1787993753587_full.png",
    ]
    for name in names:
        p = EV / name
        w, h, rgba = decode_png(p.read_bytes())
        if (w, h) != (1876, 1400):
            continue
        arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
        lum = gray_luma(arr)
        print(f"\n===== {name} =====")
        for (bx, by, bw, bh, cap) in _capsules(lum):
            glyphs = segment_bet_glyphs(cap["inner"])
            if glyphs is None:
                continue
            gh = [gl["h"] for gl in glyphs]
            gw = [gl["w"] for gl in glyphs]
            labels = classify_bet_glyphs(glyphs, tpls)
            if labels is None:
                print(f"  胶囊({bx},{by},{bw},{bh}) h={gh} w={gw} -> 分类失败")
            else:
                print(f"  胶囊({bx},{by},{bw},{bh}) h={gh} w={gw} "
                      f"-> {''.join(labels[0])} margins={labels[1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
