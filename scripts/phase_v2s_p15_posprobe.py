"""P15 — Position 1876 识别探针（只读，抽样）。"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2.pixel.realclient.seat_position import (  # noqa: E402
    PositionObserver)
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402


def main() -> int:
    obs = PositionObserver()
    frames = sorted(Path("data/vision2/coinpoker_evidence").glob("*_full.png"))
    n = 0
    rec = 0
    reasons = {}
    samples = []
    for p in frames:
        w, h, rgba = decode_png(p.read_bytes())
        if (w, h) != (1876, 1400):
            continue
        n += 1
        if n > 40:                     # 抽样（快）
            break
        out = obs.observe(rgba, w, h)
        u = out["runtime_result"]["uncertainty"]
        r = out["runtime_result"].get("reason")
        reasons[r] = reasons.get(r, 0) + 1
        if u == "RECOGNIZED":
            rec += 1
            if len(samples) < 4:
                samples.append((p.name, out.get("hero_position")))
    print(f"1876 帧: {n}, RECOGNIZED: {rec}")
    print("reasons:", reasons)
    print("samples:", samples)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
