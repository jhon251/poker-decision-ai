#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P30-9-FN：**一条命令的总验证**（全部守卫与度量）。

用法：
    python scripts/verify_all.py              # 快速：dist 同步 + 轻量基准门 + 真实帧 + 视觉门 + 单测
    python scripts/verify_all.py --6max       # 追加 6-max 保真度（4 offset 聚合，约 10 分钟）
    python scripts/verify_all.py --fb         # 追加 fb（TV 距离）

每项打印 PASS/FAIL 与关键数值；任一项失败 ⇒ 非零退出。结果写 docs/verify_all.json。
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import subprocess
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = pathlib.Path(__file__).resolve().parents[1]
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")


def _run(argv: list[str], timeout: int = 5400) -> tuple[int, str]:
    p = subprocess.run([PY, *argv], cwd=str(ROOT), capture_output=True, text=True,
                        encoding="utf-8", errors="replace", timeout=timeout)
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--6max", dest="sixmax", action="store_true")
    ap.add_argument("--fb", action="store_true")
    ap.add_argument("--holdout", action="store_true",
                     help="追加 6-max 留出验证（非循环口径，约 5 分钟）")
    ap.add_argument("--mech", action="store_true",
                     help="追加四机制汇总门禁（补读/多人告知/现场真解/缓存，约 1 分钟）")
    ap.add_argument("--real", action="store_true",
                     help="追加**真实会话真解率**门禁（确定性、秒级；回退即 FAIL）")
    args = ap.parse_args()

    results: list[dict] = []
    fails: list[str] = []

    def step(name: str, argv: list[str], timeout: int = 5400) -> str:
        rc, out = _run(argv, timeout)
        ok = rc == 0
        results.append({"step": name, "ok": ok, "rc": rc})
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
        if not ok:
            fails.append(name)
            for line in out.splitlines()[-6:]:
                print("    ", line.strip()[:150])
        return out

    # 1) 发布包一致性
    step("dist 同步一致性", ["scripts/check_dist_sync.py"])
    # 1a) 真实帧产品路径 —— **必须在审计之前**（否则审计读到上一轮陈旧 JSON ⇒ 假 FAIL ✗）
    out = step("真实帧均衡知情率", ["scripts/coverage_real_frames.py"])
    m = re.search(r"均衡知情率（真实解参与）: (\d+)/(\d+)", out)
    if m:
        print(f"     {m.group(1)}/{m.group(2)}")
    # 2) 轻量基准门（r1-flop / r1-turn·river，可选 fb）
    gate_argv = ["scripts/bench_gate_check.py"] + (["--fb"] if args.fb else [])
    out = step("基准门（r1/fb）", gate_argv)
    m = re.search(r"\[r1-flop\] 当前: (\{.*?\})", out)
    if m:
        print("     r1-flop:", m.group(1)[:120])
    # 2c) 四机制汇总门禁（本目标新增能力自检）
    if args.mech:
        out = step("四机制汇总门禁", ["scripts/mechanisms_gate.py"])
        for line in out.splitlines():
            if line.startswith("[PASS]") or line.startswith("[FAIL]"):
                print("    ", line[:120])
    # 2b) 真实会话真解率（确定性；防止"真实覆盖"悄悄掉回 0%）
    if args.real:
        out = step("真实会话真解率门禁", ["scripts/real_coverage_gate.py", "--gate"])
        m = re.search(r"真实可评估局面 (\d+) \| exact (\d+)", out)
        if m:
            print(f"     真实可评估 {m.group(1)} | exact {m.group(2)}")
    # 3) 6-max 保真度（可选）
    if args.sixmax:
        out = step("6-max 保真度（4 offset）",
                    ["scripts/bench_6max_suite.py", "--offsets", "0,24,48,72",
                     "--nodes", "24", "--hands", "8"])
        m = re.search(r"合并 (\d+) 决策：支持 ([0-9.]+)% \| 主选 ([0-9.]+)% \| 锚定 ([0-9.]+)%", out)
        if m:
            print(f"     {m.group(1)} 决策：支持 {m.group(2)}% / 主选 {m.group(3)}% / 锚定 {m.group(4)}%")
    # 3b) 6-max 留出验证（非循环口径）
    if args.holdout:
        for lvl in ("board", "family"):
            out = step(f"6-max 留出（{lvl}）",
                        ["scripts/holdout_6max.py", "--hands", "6", "--level", lvl,
                         "--max-nodes", "24"])
            m = re.search(r"支持 ([0-9.]+) \| 主选 ([0-9.]+) \| 频率质量 ([0-9.]+)", out)
            if m:
                print(f"     支持 {m.group(1)} / 主选 {m.group(2)} / 质量 {m.group(3)}")

    # 4a) 会话级采集清单（数据积累后自动更新 ✓）
    step("会话级采集清单", ["scripts/session_manifest.py"])
    # 4b) 诚实看板（聚合既有实测数字；不做 EV/盈利声明 ✓）
    step("诚实看板（聚合）", ["scripts/honest_dashboard.py"])
    # 4c) 目标逐项终检（每条目标 → 可核查证据 → 诚实判定 ✓）
    step("目标逐项终检", ["scripts/goal_completion_audit.py"])
    # 9) **审计一致性放在最后**（P30-9-HN：此前放前面会读到未刷新的测量/上一轮结果 ⇒ 假 FAIL ✗）
    step("审计一致性（记录↔实测）", ["scripts/audit_consistency.py"])
    # 5) 视觉门
    out = step("视觉语料门", ["scripts/vision_score.py"])
    m = re.search(r"覆盖率: (\d+/\d+)", out)
    if m:
        print("     ", m.group(1))
    # 6) 单测（守卫相关集）
    step("单测（守卫集）", ["-m", "pytest", "tests/unit/test_spot_table_6max_board_exact.py",
                            "tests/unit/test_spot_table_6max_actor.py",
                            "tests/unit/test_spot_table_6max.py",
                            "tests/unit/test_spot_table_6max_size_factor.py",
                            "tests/unit/test_sixmax_multiway_guard.py",
                            "tests/unit/test_per_hand_deviation.py",
                            "tests/unit/test_hu_spot_actor.py",
                            "tests/unit/test_judge.py", "-q", "-p", "no:cacheprovider"])

    out_path = ROOT / "docs" / "verify_all.json"
    out_path.write_text(json.dumps({"results": results, "fails": fails,
                                     "sixmax": args.sixmax, "fb": args.fb},
                                    ensure_ascii=False, indent=1), encoding="utf-8")
    if fails:
        print(f"\n[FAIL] 未通过项：{fails}（详见 docs/verify_all.json）")
        return 1
    print("\n[PASS] 全部验证通过（dist 同步 / 基准门 / 真实帧 / 视觉门 / 单测"
          + (" / 6-max" if args.sixmax else "") + "）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
