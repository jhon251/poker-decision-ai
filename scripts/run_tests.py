"""P58/P59 — 权威测试入口（单实例锁；smoke/serial/parallel；进度与摘要）。

用法:
  python scripts/run_tests.py --mode smoke    # 快速健康检查（开发反馈，
                                              # 不替代完整验收）
  python scripts/run_tests.py                 # serial = 权威完整测试
                                              # （2,194 项，~45-83 min）
  python scripts/run_tests.py --mode parallel # EXPERIMENTAL——默认拒绝，
                                              # 需 --force-parallel 显式启用

P58 审计结论（Windows/CPython 3.13 + xdist 3.8.0 + 本套件）：
  - parallel 存在不可安全绕过的稳定性限制：v3.1 compute_ev 路径在
    xdist worker 中偶发 Windows access violation（faulthandler 抓到
    原生栈：evaluator kickers/evaluate5 段），worker 硬死亡
    （"node down: Not properly terminated"）；
  - --max-worker-restart 恢复后 execnet 网关再次挂起（收尾等待超时）；
  - 结论：serial 为唯一权威口径；parallel 保留为 EXPERIMENTAL。

P59 交付：
  - 单实例锁（多实例互抢 CPU 是定位过的小时级拖慢根因）；
  - 运行全程输出 start / 阶段进度 / elapsed / 最终数量 / PASS-FAIL /
    exit code；
  - 日志落 .tmp/testrun_*.log（进度: tail -f）；
  - smoke 模式：固定快速不变量子集，仅开发反馈用。

不修改任何测试语义/断言；不跳过任何测试；不修改生产/策略代码。
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / ".tmp"
LOCK = ROOT / ".tmp" / ".pytest_singleton.lock"

# 受控默认：16 逻辑核机器上 6 worker（约 1/3 核数）——留余量给
# IDE/其他进程；不盲目吃满全部核心（P58 决议）。
DEFAULT_WORKERS = 6

# P59 smoke：固定的快速不变量子集（确定性、秒级~分钟级）——
# 仅开发过程快速反馈；完整验收必须用 serial。
SMOKE_NODEIDS = [
    "tests/unit/test_p41_strategy_core.py::test_call_ev_formula",
    "tests/unit/test_p41_strategy_core.py::test_check_ev_formula",
    "tests/unit/test_p41_strategy_core.py::test_check_better_when_eq_below_half",
    "tests/unit/test_p56_strategy_continuation.py::test_i1_river_bit_frozen",
    "tests/unit/test_p56_strategy_continuation.py::test_i1_preflop_no_cv",
    "tests/unit/test_p56_strategy_continuation.py::test_i5_cash_utility_bit_equal_with_cv",
    "tests/unit/test_p56_strategy_continuation.py::test_a8_realization_direction",
    "tests/unit/test_p57_robustness.py::test_boundaries_all",
    "tests/unit/test_p57_robustness.py::test_a3v2_shift_never_positive",
]


def other_pytest_running() -> int | None:
    """返回其他 pytest 进程 PID（无则 None）。"""
    out = subprocess.run(
        ["powershell", "-Command",
         "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
         "Where-Object { $_.CommandLine -match 'pytest' } | "
         "Select-Object -ExpandProperty ProcessId"],
        capture_output=True, text=True)
    pids = [int(x) for x in out.stdout.split() if x.strip().isdigit()]
    return pids[0] if pids else None


def build_cmd(mode: str, workers: int, extra: list[str]) -> list[str]:
    py = str(ROOT / ".venv" / "Scripts" / "python.exe")
    # 注意：pyproject addopts 已含 -q；再传一次 -q 会升级为 quieter，
    # 抑制最终 'N passed' 摘要行（P59 验收发现的日志缺陷）。
    cmd = [py, "-m", "pytest"]
    if mode == "smoke":
        cmd += SMOKE_NODEIDS
    elif mode == "parallel":
        cmd += ["tests/", "-n", str(workers), "--dist", "loadfile",
                "--max-worker-restart=2",
                "-o", "faulthandler_timeout=240"]
    else:
        cmd += ["tests/"]
    cmd += extra
    return cmd


def _progress_from_log(log: Path) -> str:
    """从 pytest -q 日志提取最后一个进度标记（如 [ 65%]）。"""
    try:
        text = log.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return "collecting..."
    marks = re.findall(r"\[\s*\d+%\]", text)
    return marks[-1] if marks else "collecting..."


def _summary_from_log(log: Path) -> dict:
    out = {"passed": None, "failed": None, "errors": None}
    try:
        text = log.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return out
    m = re.search(r"(\d+) passed", text)
    if m:
        out["passed"] = int(m.group(1))
    m = re.search(r"(\d+) failed", text)
    if m:
        out["failed"] = int(m.group(1))
    m = re.search(r"(\d+) error", text)
    if m:
        out["errors"] = int(m.group(1))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("smoke", "serial", "parallel"),
                    default="serial")
    ap.add_argument("--force-parallel", action="store_true",
                    help="EXPERIMENTAL：显式启用并行（存在 access "
                         "violation worker 崩溃风险，见文件头审计结论）")
    ap.add_argument("-w", "--workers", type=int, default=DEFAULT_WORKERS)
    ap.add_argument("--extra", nargs="*", default=[],
                    help='额外 pytest 参数，如 --extra "-k p56"')
    args = ap.parse_args()
    if args.mode == "parallel" and not args.force_parallel:
        print("[run_tests] parallel=EXPERIMENTAL：当前环境存在 xdist worker "
              "access violation（P58 审计）；使用 --force-parallel 显式启用。"
              "本次拒绝。")
        return 2

    if LOCK.exists():
        stale = LOCK.read_text(encoding="utf-8").strip()
        print(f"[run_tests] 单实例锁命中（另一 runner pid={stale}）。"
              "若确认已结束请删除 .tmp/.pytest_singleton.lock。拒绝启动。")
        return 2
    busy = other_pytest_running()
    if busy:
        print(f"[run_tests] 检测到 pytest 进程 pid={busy} 正在运行——"
              "拒绝并发启动（多实例互抢 CPU 是定位过的小时级拖慢根因）。")
        return 2
    LOCK.write_text(str(os.getpid()), encoding="utf-8")

    LOG_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log = LOG_DIR / f"testrun_{ts}_{args.mode}.log"
    cmd = build_cmd(args.mode, args.workers, args.extra)
    print(f"[run_tests] START mode={args.mode} "
          f"workers={args.workers if args.mode == 'parallel' else 1}"
          f"{' (EXPERIMENTAL)' if args.mode == 'parallel' else ''}")
    print(f"[run_tests] log: {log}（进度: tail -f {log}）")
    print(f"[run_tests] cmd: {' '.join(cmd[1:])}")
    sys.stdout.flush()

    t0 = time.perf_counter()
    with open(log, "w", encoding="utf-8") as fh:
        proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT,
                                cwd=str(ROOT))
        # 进度心跳（每 60s 打印 elapsed + 最新进度标记——长任务可见性）
        while proc.poll() is None:
            time.sleep(60)
            if proc.poll() is None:
                elapsed = round((time.perf_counter() - t0) / 60, 1)
                print(f"[run_tests] elapsed={elapsed}min "
                      f"progress={_progress_from_log(log)}")
                sys.stdout.flush()
    wall = time.perf_counter() - t0
    LOCK.unlink(missing_ok=True)

    summary = _summary_from_log(log)
    verdict = "PASS" if proc.returncode == 0 else "FAIL"
    print(f"[run_tests] END wall={round(wall / 60, 2)}min "
          f"passed={summary['passed']} failed={summary['failed']} "
          f"errors={summary['errors']} → {verdict} "
          f"(pytest exit={proc.returncode})")
    print(f"[run_tests] 完整日志: {log}")
    if proc.returncode != 0:
        print(f"[run_tests] 存在失败——检查日志定位: {log}")
    return 0 if proc.returncode == 0 else proc.returncode


if __name__ == "__main__":
    sys.exit(main())
