"""GUI 链路耗时分解（诊断脚本——只读测量，不改产品）。

阶段边界：
  build_runtime  — runner 首次构建（一次性）
  capture        — BitBlt 屏幕捕获（仅 live 模式）
  vision_bridge  — frame_to_decision_request（V3F 观察器 + Position）
  decide         — ProductionDecisionLoop.tick（State/Equity/Solver/Judge 全链）
  total          — runner.analyze 全程
另解析 EventLogger events-*.jsonl 中本次新增的 agent_invocation latency。

用法: python scripts/p33_gui_timing.py [frame] [runs]
"""
from __future__ import annotations

import json
import sys
import time
import unittest.mock as mock
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

FRAME = sys.argv[1] if len(sys.argv) > 1 else \
    "data/vision2/coinpoker_evidence/8bf82f134651db0add626be0304a2ff1.png"
RUNS = int(sys.argv[2]) if len(sys.argv) > 2 else 3

from poker_decision_ai.vision2_platform.png_decode import decode_png
from cash_game_ui.runner import CashGameRunner
from cash_game_ui.preview_gui import capture_frame_smart
import poker_decision_ai.vision2.pixel.realclient.production_bridge as pb
import poker_decision_ai.production_decision_loop as pdl

timings = {"bridge": [], "tick": [], "total": []}
orig_bridge = pb.frame_to_decision_request
orig_tick = pdl.ProductionDecisionLoop.tick


def timed_bridge(*a, **k):
    t0 = time.perf_counter()
    r = orig_bridge(*a, **k)
    timings["bridge"].append((time.perf_counter() - t0) * 1000)
    return r


def timed_tick(self, *a, **k):
    t0 = time.perf_counter()
    r = orig_tick(self, *a, **k)
    timings["tick"].append((time.perf_counter() - t0) * 1000)
    return r


def main() -> int:
    t0 = time.perf_counter()
    w, h, rgba = decode_png(open(FRAME, "rb").read())
    decode_ms = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    runner = CashGameRunner()
    build_ms = (time.perf_counter() - t0) * 1000

    # capture 单测（live BitBlt 路径）
    t0 = time.perf_counter()
    capture_frame_smart()
    capture_ms = (time.perf_counter() - t0) * 1000

    log_dir = Path("logs")
    ev_before = {p.name: p.stat().st_size
                 for p in log_dir.glob("events-*.jsonl")} \
        if log_dir.exists() else {}

    last = None
    with mock.patch.object(pb, "frame_to_decision_request", timed_bridge), \
         mock.patch.object(pdl.ProductionDecisionLoop, "tick", timed_tick):
        for i in range(RUNS):
            t0 = time.perf_counter()
            d = runner.analyze(rgba, w, h).to_dict()
            dt = (time.perf_counter() - t0) * 1000
            timings["total"].append(dt)
            vc = (d.get("vision") or {})
            last = d
            print("run%d: total=%.0fms  bridge=%.0fms  tick=%.0fms  "
                  "status=%s action=%s hero_cards=%s"
                  % (i + 1, dt, timings["bridge"][-1], timings["tick"][-1],
                     d.get("status"), d.get("decision_action"),
                     vc.get("hero_cards")))

    # 本次新增 agent_invocation 延迟
    ev_after = {p.name: p.stat().st_size for p in log_dir.glob("events-*.jsonl")}
    lat = []
    for name, size in ev_after.items():
        before = ev_before.get(name, 0)
        if size > before:
            with open(log_dir / name, encoding="utf-8") as fh:
                fh.seek(before)
                for line in fh:
                    try:
                        e = json.loads(line)
                    except Exception:
                        continue
                    if e.get("event") == "agent_invocation":
                        lat.append((str(e.get("agent")),
                                    e.get("latency_ms") or 0,
                                    e.get("status")))
    lat.sort(key=lambda x: -x[1])
    print("decode=%.0fms build_runtime=%.0fms capture(BitBlt)=%.0fms"
          % (decode_ms, build_ms, capture_ms))
    print("agent_invocation (desc latency, this batch):")
    for a, ms, st in lat[:14]:
        print("  %-20s %8.0fms  %s" % (a, ms, st))
    ps = last.get("provider_status") or {}
    print("providers:", {k: (ps.get(k) or {}).get("status")
                         for k in ("llm", "vision", "solver")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
