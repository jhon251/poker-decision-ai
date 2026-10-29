"""P30-3-A — 真实帧全链：8bf8 是否可产生决策（进入 Judge 需要什么）。"""
from __future__ import annotations
import asyncio
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402
from poker_decision_ai.vision2.pixel.realclient.production_bridge import (  # noqa: E402
    frame_to_decision_request)

EV = Path("data/vision2/coinpoker_evidence")

# 全真实帧枚举 bridge 状态
for fid in ["8bf82f134651db0add626be0304a2ff1.png",
            "POT_CHANGED_110_1787992526057_full.png",
            "BUTTON_CHANGED_167_1787992682749_full.png"]:
    w, h, rgba = decode_png((EV / fid).read_bytes())
    res = frame_to_decision_request(rgba, w, h)
    br = res["bridge"]
    print(f"{fid[:30]}: status={res['status']} reason={res['reason']}")
    if res["status"] == "UNAVAILABLE":
        print("  → 生产接线返回 request=None（decide() 不可喂——UNKNOWN-first 正确）")
