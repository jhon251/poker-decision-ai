"""cProfile 钻取：vision bridge 与 decide tick 的热点（诊断脚本）。"""
from __future__ import annotations

import cProfile
import io
import pstats
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

FRAME = "data/vision2/coinpoker_evidence/8bf82f134651db0add626be0304a2ff1.png"

from poker_decision_ai.vision2_platform.png_decode import decode_png
from cash_game_ui.runner import CashGameRunner


def top(p: pstats.Stats, n: int = 14) -> str:
    s = io.StringIO()
    p.stream = s
    p.sort_stats("tottime").print_stats(n)
    out = s.getvalue()
    s2 = io.StringIO()
    p.stream = s2
    p.sort_stats("cumulative").print_stats(n)
    out += "\n--- by cumulative ---\n" + s2.getvalue()
    return out


def main() -> int:
    w, h, rgba = decode_png(open(FRAME, "rb").read())
    runner = CashGameRunner()
    runner._get_runtime()  # 预构建 runtime，隔离一次构建成本

    import poker_decision_ai.vision2.pixel.realclient.production_bridge as pb
    print("=== profile: frame_to_decision_request (vision bridge) ===")
    pr = cProfile.Profile()
    pr.enable()
    pb.frame_to_decision_request(rgba, w, h)
    pr.disable()
    print(top(pstats.Stats(pr), 14))

    import poker_decision_ai.production_decision_loop as pdl
    req = pb.frame_to_decision_request(rgba, w, h)
    loop = pdl.ProductionDecisionLoop(runner._get_runtime(), throttle_s=0.0)
    print("=== profile: ProductionDecisionLoop.tick (decide) ===")
    pr2 = cProfile.Profile()
    pr2.enable()
    loop.tick(200, rgba, w, h)
    pr2.disable()
    print(top(pstats.Stats(pr2), 14))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
