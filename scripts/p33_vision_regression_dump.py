"""视觉回归基线 dump：对测试引用的 evidence 帧跑完整 bridge，落 JSON。

用途：检测器数值等价改造前后的逐字段对比（dealer/hero/pot/stack/status/
street/board/to_call + field_confidences）。缓存默认关闭——每帧真实重算。
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("logs/vision_baseline.json")

from poker_decision_ai.vision2_platform.png_decode import decode_png
from poker_decision_ai.vision2.pixel.realclient.production_bridge import (
    frame_to_decision_request)

EV = Path("data/vision2/coinpoker_evidence")


def main() -> int:
    refs = set()
    for p in Path("tests").rglob("*.py"):
        t = p.read_text(encoding="utf-8", errors="ignore")
        refs.update(re.findall(r"([0-9a-f]{32}\.png|\w+_full\.png)", t))
    frames = sorted(r for r in refs if (EV / r).exists())
    print(f"frames: {len(frames)}", flush=True)
    out = {}
    for i, name in enumerate(frames):
        w, h, rgba = decode_png((EV / name).read_bytes())
        try:
            got = frame_to_decision_request(rgba, w, h)
            br = got.get("bridge") or {}
            out[name] = {
                "status": got.get("status"),
                "reason": got.get("reason"),
                "bridge_status": br.get("status"),
                "field_confidences": br.get("field_confidences"),
                "blocking_fields": br.get("blocking_fields"),
                "unknown_fields": br.get("unknown_fields"),
                "observation": br.get("observation"),
            }
        except Exception as e:  # noqa: BLE001 — 如实记录
            out[name] = {"error": f"{type(e).__name__}: {e}"}
        print(f"[{i + 1}/{len(frames)}] {name} -> "
              f"{out[name].get('status', out[name].get('error'))}",
              flush=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print(f"wrote {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
