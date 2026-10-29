"""P16-E — Bet 1876 READ-ONLY：胶囊搜索（E1/E2）。

在全帧滑动 detect_bet_capsule 逻辑（亮墨行带 + 暗底胶囊校验），找 1876
帧中所有 bet-capsule 候选（位置/尺寸/暗底比），并与 1848 座锚点对照。
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2.pixel.realclient.coinpoker_bet import (  # noqa: E402
    gray_luma, detect_bet_capsule)
from poker_decision_ai.vision2.pixel.realclient.coinpoker_stack import (  # noqa: E402
    SEAT_ANCHORS_1848)
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402

EV = Path("data/vision2/coinpoker_evidence")


def main() -> int:
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
        found = []
        # 滑动搜索：60px 步长，ROI 260×80
        for y0 in range(60, 1340, 60):
            for x0 in range(0, 1876 - 260, 60):
                cap = detect_bet_capsule(lum, (x0, y0, 260, 80))
                if cap is None:
                    continue
                bx, by, bw, bh = cap["bbox"]
                # 近锚点
                near = None
                for seat, (ax, ay) in SEAT_ANCHORS_1848.items():
                    d = ((bx + bw // 2 - ax) ** 2 + (by + bh // 2 - ay) ** 2) ** 0.5
                    if near is None or d < near[1]:
                        near = (seat, round(d))
                # 去重（邻近）
                dup = False
                for (fx, fy, fw2, fh2, *_) in found:
                    if abs(fx - bx) < 40 and abs(fy - by) < 30:
                        dup = True
                        break
                if not dup:
                    found.append((bx, by, bw, bh, round(cap["dark_ratio"], 2), near))
        for (bx, by, bw, bh, dr, near) in sorted(found, key=lambda t: (t[1], t[0])):
            print(f"  胶囊 bbox=({bx},{by},{bw},{bh}) dark={dr} 近锚点={near}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
