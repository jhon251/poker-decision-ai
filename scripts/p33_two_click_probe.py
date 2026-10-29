"""双击不复用验证：同帧→异帧→同帧交替 3 次 analyze。

每次 analyze 入口 vision_cache_new_epoch() → 缓存作用域仅一次调用；
异帧结果必须不同（不得用上一帧冒充当前帧）；每次均真实重算（pipeline_ms）。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poker_decision_ai.vision2_platform.png_decode import decode_png
from cash_game_ui.runner import CashGameRunner

EV = Path("data/vision2/coinpoker_evidence")
FRAME_A = EV / "8bf82f134651db0add626be0304a2ff1.png"      # 有手牌 5c3c
FRAME_B = EV / "BUTTON_CHANGED_167_1787992682749_full.png"  # 无手牌


def main() -> int:
    fa = FRAME_A.read_bytes()
    fb = FRAME_B.read_bytes()
    wa, ha, ra = decode_png(fa)
    wb, hb, rb = decode_png(fb)
    runner = CashGameRunner()
    seq = [("A(8bf8 有牌)", ra, wa, ha), ("B(无牌帧)", rb, wb, hb),
           ("A 再来一次", ra, wa, ha)]
    ok = True
    for i, (name, rgba, w, h) in enumerate(seq, 1):
        t0 = time.perf_counter()
        d = runner.analyze(rgba, w, h).to_dict()
        dt = (time.perf_counter() - t0) * 1000
        vis = d.get("vision") or {}
        print(f"click{i} {name}: status={d.get('status')} "
              f"action={d.get('decision_action')} "
              f"hero_cards={vis.get('hero_cards')} "
              f"hero_seat={vis.get('hero_seat')} pot={vis.get('pot')} "
              f"total={dt:.0f}ms")
        if i == 1 and vis.get("hero_cards") != ["5c", "3c"]:
            ok = False
        if i == 2 and vis.get("hero_cards") is not None:
            print("  FAIL: 异帧不应复用 A 的手牌!")
            ok = False
        if i == 3 and vis.get("hero_cards") != ["5c", "3c"]:
            print("  FAIL: 第三击未重新识别!")
            ok = False
    print("TWO_CLICK_NO_REUSE:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
