"""P16-A — Stack 1876 根因诊断（只读）：blob 检测 → 关联 → 读取逐段失败点。"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2.pixel.realclient.coinpoker_stack import (  # noqa: E402
    SEAT_ANCHORS_1848, StackRecognizer, SeatLikeGoldBlobs)
from poker_decision_ai.vision2.pixel.realclient.seat_position import (  # noqa: E402
    SeatOccupancyDetector)
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402

EV = Path("data/vision2/coinpoker_evidence")


def main() -> int:
    rec = StackRecognizer(window="1848")
    for name in ["POT_CHANGED_12_1787995290794_full.png",
                 "BUTTON_CHANGED_167_1787992682749_full.png"]:
        p = EV / name
        w, h, rgba = decode_png(p.read_bytes())
        arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
        print(f"\n=== {name} ===")
        # 1. bright gold blobs（recognize 实际用的）
        golds = SeatLikeGoldBlobs(arr)
        print(f"bright gold blob 中心: {golds}")
        # 2. 关联
        for seat, (ax, ay) in SEAT_ANCHORS_1848.items():
            best = None
            for (bx, by) in golds:
                d = float(((bx - ax) ** 2 + (by - ay) ** 2) ** 0.5)
                if d <= 320.0 and (best is None or d < best[1]):
                    best = ((bx, by), d)
            print(f"  {seat} anchor({ax},{ay}) -> 关联={best}")
            if best is not None:
                read = rec._read_stack_text(arr, best[0][0], best[0][1])
                print(f"      read -> raw={read.get('raw')} status={read.get('status')}")
        # 3. 直接对 hero 区域读 stack（绕过关联，验证模板可读）
        #    hero 区底带 x 700-900, y 1210-1270
        roi = arr[1210:1270, 700:900]
        gold = rec._gold_mask(roi)
        ys, xs = np.where(gold > 0)
        print(f"hero 区(700-900,1210-1270) gold 像素={len(ys)}")
        if len(ys) >= 30:
            inner = gold[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
            read = rec._chars_from_mask(roi, inner, ys, xs)
            print(f"  hero 区直读 -> raw={read.get('raw')} status={read.get('status')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
