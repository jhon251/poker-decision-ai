"""M4 — pytest 分块全绿协议（ENVIRONMENT：本机会外杀长跑 python → 分块）。

按目录/主题拆块，每块独立 `pytest -q`；全部 EXIT=0 即视为全绿。
单块失败先单跑该文件排除 deadline 争抢抖动再定论。

用法：
  <repo>/.venv/Scripts/python.exe scripts/run_m4_pytest_chunks.py [--chunk i]
"""
from __future__ import annotations

import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / ".tmp"

CHUNKS: dict[str, list[str]] = {
    "chunk0_unit": ["tests/unit"],
    "chunk1_m4": ["tests/test_m4_gate.py", "tests/test_m3_opponent_model.py",
                  "tests/test_solver2_strategy_anchor.py",
                  "tests/test_solver1_integration.py"],
    "chunk2_loop": ["tests/test_bench1_loop.py", "tests/test_tournament1_loop.py",
                    "tests/test_frequency_prior.py",
                    "tests/test_domain_math_audit.py"],
    "chunk3_phase5_6": ["tests/test_phase5a_uoft.py", "tests/test_phase5b_math.py",
                        "tests/test_phase5b_mvp.py",
                        "tests/test_phase65_benchmark.py",
                        "tests/test_phase6b_schema_tiers.py",
                        "tests/test_phase6c_profiles.py",
                        "tests/test_phase6d_gmm.py",
                        "tests/test_phase6e_stability.py",
                        "tests/test_phase6f_utility.py",
                        "tests/test_phase6g_interfaces.py"],
    "chunk4_phase7_8": ["tests/test_phase7a_decision.py",
                        "tests/test_phase7b_validation.py",
                        "tests/test_phase7c_recommendation.py",
                        "tests/test_phase8a_replay.py",
                        "tests/test_phase8b_baseline_response.py",
                        "tests/test_phase8c_synthetic.py"],
    "chunk5_vision": ["tests/test_vision2.py", "tests/test_vision2_platform.py",
                      "tests/test_vision2_realtime.py"],
    "chunk6_root_rest": [p.name for p in pathlib.Path(ROOT / "tests").glob("test_*.py")
                         if p.name not in {
                             "test_m4_gate.py", "test_m3_opponent_model.py",
                             "test_solver2_strategy_anchor.py",
                             "test_solver1_integration.py",
                             "test_bench1_loop.py", "test_tournament1_loop.py",
                             "test_frequency_prior.py",
                             "test_domain_math_audit.py",
                             "test_phase5a_uoft.py", "test_phase5b_math.py",
                             "test_phase5b_mvp.py", "test_phase65_benchmark.py",
                             "test_phase6b_schema_tiers.py",
                             "test_phase6c_profiles.py", "test_phase6d_gmm.py",
                             "test_phase6e_stability.py",
                             "test_phase6f_utility.py",
                             "test_phase6g_interfaces.py",
                             "test_phase7a_decision.py",
                             "test_phase7b_validation.py",
                             "test_phase7c_recommendation.py",
                             "test_phase8a_replay.py",
                             "test_phase8b_baseline_response.py",
                             "test_phase8c_synthetic.py",
                             "test_vision2.py", "test_vision2_platform.py",
                             "test_vision2_realtime.py"}],
}


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", default=None)
    args = ap.parse_args()
    names = [args.chunk] if args.chunk else list(CHUNKS)
    LOG_DIR.mkdir(exist_ok=True)
    failures = []
    for name in names:
        paths = CHUNKS[name]
        log_path = LOG_DIR / f"m4_{name}.log"
        cmd = [sys.executable, "-m", "pytest", "-q", *paths]
        print(f"[pytest] {name}: {' '.join(paths)[:110]}…", flush=True)
        t0 = time.perf_counter()
        with open(log_path, "w", encoding="utf-8") as lf:
            r = subprocess.run(cmd, cwd=str(ROOT), stdout=lf,
                               stderr=subprocess.STDOUT, timeout=7200)
        dt = round((time.perf_counter() - t0) / 60, 1)
        tail = log_path.read_text(encoding="utf-8").strip().splitlines()[-2:]
        for ln in tail:
            print(f"  | {ln}", flush=True)
        print(f"[pytest] {name}: EXIT={r.returncode} ({dt}min)", flush=True)
        if r.returncode != 0:
            failures.append(name)
    print("FAILED chunks:", failures if failures else "none — ALL GREEN")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
