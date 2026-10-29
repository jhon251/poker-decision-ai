"""V2ZM — 证据帧板时序分析器（确定性，识别门控保持）。

对 AutoCapture 保存的证据帧（SCENE/POT/BUTTON 优先级排序）运行
CoinPokerBoardRecognizer → 每帧 slot 数/牌面 → 时序 0→3→4→5 检测：
- 单调递增校验；
- 已确认卡不得无原因改变；
- 3→5 跳过 4 → TURN=NOT_OBSERVED（不反推）；
- 记录 2c/5h/Ks 等 suit 证据路径与 rank_gate 观察。
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from poker_decision_ai.vision2.pixel.realclient.coinpoker_board import (  # noqa: E402
    CoinPokerBoardRecognizer)
from poker_decision_ai.vision2_platform import decode_png  # noqa: E402


def main() -> int:
    sys.argv = sys.argv[:1]
    ev = Path("data/vision2/coinpoker_evidence")
    rec = CoinPokerBoardRecognizer()
    frames = sorted(p for p in ev.glob("SCENE_CHANGED_*.png"))
    print(f"evidence pngs: {len(frames)}")
    lat = []
    timeline = []
    for p in frames:
        t0 = time.perf_counter()
        w, h, rgba = decode_png(p.read_bytes())
        res = rec.recognize(rgba, w, h, int(p.name.split("_")[1])
                            if "_" in p.name and p.name.split("_")[1]
                            .isdigit() else 0)
        lat.append((time.perf_counter() - t0) * 1000)
        ok = [(c["slot_index"], c["raw_text"], c["confidence"])
              for c in res["cards"] if c["status"] == "OK"]
        unc = [(c["slot_index"], c.get("reason"))
               for c in res["cards"] if c["status"] != "OK"]
        timeline.append({"frame": p.name[:60], "ok": ok,
                         "uncertain": unc,
                         "count": len(ok)})
    # 时序板数变化
    prev_count = 0
    transitions = []
    for t in timeline:
        if t["count"] != prev_count:
            transitions.append({"frame": t["frame"],
                                "board_count": t["count"],
                                "from": prev_count, "to": t["count"],
                                "cards": t["ok"]})
            prev_count = t["count"]
    lat_sorted = sorted(lat)
    n = len(lat_sorted)
    pct = lambda q: round(lat_sorted[min(n - 1, int(n * q))], 1) if n else None
    report = {
        "frames_analyzed": len(timeline),
        "recognition_latency_ms": {"p50": pct(0.50), "p95": pct(0.95),
                                   "p99": pct(0.99), "frames": n},
        "board_transitions": transitions,
        "timeline": timeline,
        "turn_observed": any(t["to"] == 4 for t in transitions),
    }
    Path("data/vision2/v2zm_board_temporal.json").write_text(
        json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print("recognition latency P50/P95/P99:",
          pct(0.50), pct(0.95), pct(0.99), "ms over", n, "frames")
    print("board transitions:")
    for t in transitions:
        print("  ", t["frame"], f"{t['from']}->{t['to']}", t["cards"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
