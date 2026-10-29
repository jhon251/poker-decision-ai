"""Phase V1 感知层性能基准（实测 P50/P95/P99，不编造数字）。

用法：
    python scripts/phase_v1_benchmark.py [--iterations 200] [--warmup 10]
                                          [--fps 10] [--out PATH]

- 输入全部为 synthetic fixture（确定性场景），输出 JSON + 控制台表格；
- 真实截图/OCR/多模态路径的 latency 未经实测，报告只陈述 synthetic 数字；
- 可复现：同一环境、同一参数 → 相同场景序列（耗时随机器波动属预期）。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from poker_decision_ai.vision2.realtime.benchmark import (  # noqa: E402
    run_benchmark)


def main() -> int:
    ap = argparse.ArgumentParser(description="Vision V1 benchmark harness")
    ap.add_argument("--iterations", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--fps", type=float, default=10.0,
                    choices=(1, 2, 5, 10))
    ap.add_argument("--out", type=str,
                    default="data/vision2/benchmark_v1.json")
    args = ap.parse_args()

    result = run_benchmark(args.iterations, args.warmup, args.fps)

    print("=== Vision V1 fast path（synthetic 输入，实测） ===")
    print(result.render_table())
    print("\n=== 环境 ===")
    print(json.dumps(result.environment, indent=2, ensure_ascii=False))

    out = Path(args.out)
    result.to_json(str(out))
    print(f"\n结果已写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
