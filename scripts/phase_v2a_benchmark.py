"""Phase V2A Pixel Core 基准（实测 P50/P95/P99；不编造）。

用法:
    python scripts/phase_v2a_benchmark.py [--iterations 200] [--warmup 10]
                                          [--kind synthetic|real|both]

- synthetic：确定性场景循环；real：picture data 真实截图循环（只读）；
- 输出控制台表格 + data/vision2/pixel_v2a_benchmark.json。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from poker_decision_ai.vision2.pixel.benchmark import (  # noqa: E402
    run_pixel_benchmark)


def main() -> int:
    ap = argparse.ArgumentParser(description="V2A pixel benchmark")
    ap.add_argument("--iterations", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--kind", type=str, default="both",
                    choices=("synthetic", "real", "both"))
    args = ap.parse_args()

    out: dict = {}
    for kind in (("synthetic", "real") if args.kind == "both" else (args.kind,)):
        print(f"=== V2A Pixel Core benchmark（{kind} 输入） ===")
        try:
            res = run_pixel_benchmark(args.iterations, args.warmup, kind)
            print(res.render_table())
            out[kind] = res.to_dict()
        except FileNotFoundError as e:
            print(f"[skip] {e}")

    path = Path("data/vision2/pixel_v2a_benchmark.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False),
                    encoding="utf-8")
    print(f"\n结果已写入 {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
