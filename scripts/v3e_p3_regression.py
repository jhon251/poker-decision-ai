"""V3E-P3 修复后 43 帧 BEFORE/AFTER 决策级回归 dump + diff。

对 data/vision2/coinpoker_evidence/ 的 43 帧逐一跑生产桥
（production_bridge.frame_to_decision_request），与 P1 修复后基线
（logs/v3e_p1_repair_after.json = 本轮修复前状态）逐字段比对。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poker_decision_ai.vision2.pixel.realclient.production_bridge import (
    frame_to_decision_request)
from poker_decision_ai.vision2_platform.png_decode import decode_png

EV = Path("data/vision2/coinpoker_evidence")
BASE = Path("logs/v3e_p1_repair_after.json")   # 修复前基线（P1 后未变过资产）
OUT = Path("logs/v3e_p3_repair_after.json")


def dump() -> dict:
    import time
    out = {}
    baseline = json.loads(BASE.read_text(encoding="utf-8"))
    frames = sorted(baseline)          # 严格限定 = P1 基线 43 帧
    t0 = time.time()
    for i, name in enumerate(frames, 1):
        w, h, rgba = decode_png((EV / name).read_bytes())
        r = frame_to_decision_request(rgba, w, h)
        br = r["bridge"]            # public 包装：bridge 为内层结果
        # 与 P1 基线完全一致的投影（bridge 结果 + bridge_status）
        out[name] = {
            "status": br["status"], "reason": br.get("reason"),
            "observation": br.get("observation"),
            "bridge_status": br["status"],
            "field_confidences": br.get("field_confidences"),
            "blocking_fields": br.get("blocking_fields"),
            "unknown_fields": br.get("unknown_fields"),
        }
        print("[%d/%d] %s %s %.1fs" % (i, len(frames), name[:16],
                                       br["status"], time.time() - t0),
              flush=True)
    return out


def main() -> int:
    after = dump()
    OUT.write_text(json.dumps(after, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    before = json.loads(BASE.read_text(encoding="utf-8"))
    assert set(before) == set(after), (
        "frame sets differ: %d vs %d" % (len(before), len(after)))
    diffs = {}
    for f in sorted(before):
        b, a = before[f], after[f]
        if b != a:
            fields = [k for k in sorted(set(b) | set(a))
                      if b.get(k) != a.get(k)]
            diffs[f] = fields
            print("CHANGED", f)
            for k in fields:
                print("   %-22s %s -> %s" % (
                    k, json.dumps(b.get(k), ensure_ascii=False)[:150],
                    json.dumps(a.get(k), ensure_ascii=False)[:150]))
    print("frames total:", len(before), "changed:", len(diffs))
    if not diffs:
        print("IDENTICAL - 43-frame regression: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
