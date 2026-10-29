#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P30-9-FO：**审计一致性核对**（记录值 ↔ 当前实测）。

目的：防止"报告写一套、实测另一套"的漂移。核对项：
  * r1-flop regret_mean；r1-turn/river regret_mean；fb TV 距离；
  * 6-max 支持/主选/锚定（4 offset 聚合）；
  * 真实帧均衡知情率；发布包一致性。
数据来源：docs/bench_regression_gate.json（基线）、docs/bench_6max_suite.json、
docs/verify_all.json、docs/coverage_real_frames.json、docs/dist_sync.json。
"""
from __future__ import annotations

import json
import pathlib
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = pathlib.Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"


def _load(name: str) -> dict | None:
    p = DOCS / name
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:                                        # noqa: BLE001
        return None


def main() -> int:
    gate = _load("bench_regression_gate.json") or {}
    runs = gate.get("runs", [])
    base_flop = next((r["metrics"] for r in reversed(runs)
                       if str(r.get("suite")) == "r1-flop"), {})
    base_tr = next((r["metrics"] for r in reversed(runs)
                     if str(r.get("suite")) == "r1-turn/river"), {})
    base_fb = next((r["metrics"] for r in reversed(runs)
                     if str(r.get("suite")) == "fb"), {})
    suite6 = (_load("bench_6max_suite.json") or {}).get("aggregate", {})
    verify = _load("verify_all.json") or {}
    frames = _load("coverage_real_frames.json") or []
    dist = _load("dist_sync.json") or {}

    checks: list[tuple[str, object, object, bool]] = []
    def _num(v, default=9.0):
        """注意：0.0 是合法值 ⇒ 不可用 `v or default`（假值陷阱）。"""
        return default if v is None else float(v)

    checks.append(("r1-flop regret_mean（基线）", base_flop.get("regret_mean"), 0.002,
                    abs(_num(base_flop.get("regret_mean")) - 0.002) < 1e-6))
    checks.append(("r1-turn/river regret_mean（基线）", base_tr.get("regret_mean"), 0.0,
                    _num(base_tr.get("regret_mean")) <= 1e-6))
    checks.append(("fb TV（基线）", base_fb.get("tv_distance"), 0.114,
                    abs(_num(base_fb.get("tv_distance")) - 0.114) < 1e-6))
    checks.append(("6-max 支持率（聚合）", suite6.get("support"), "≥0.99",
                    _num(suite6.get("support"), 0.0) >= 0.99))
    checks.append(("6-max 主选一致（聚合）", suite6.get("primary"), "≥0.91",
                    _num(suite6.get("primary"), 0.0) >= 0.91))
    checks.append(("6-max 锚定率（聚合）", suite6.get("anchor"), 1.0,
                    _num(suite6.get("anchor"), 0.0) >= 0.999))
    n_frames = len(frames)
    n_informed = sum(1 for r in frames if r.get("informed"))
    # P30-9-HP：非确定性**已修** ✓（请求纪元门控 ⇒ 同一请求内缓存写入不可见 ✓；
    #   实测连跑 5 次全部 5/7 ✓）⇒ 判据**收紧回精确 5/7** ✓。
    checks.append(("真实帧均衡知情率（纪元门控后已确定 ✓）",
                    f"{n_informed}/{n_frames}", "5/7",
                    n_frames == 7 and n_informed == 5))
    checks.append(("发布包一致性（问题数）", len(dist.get("problems") or []), 0,
                    len(dist.get("problems") or []) == 0))
    if verify:
        # P30-9-GX：**去掉自我引用** ✗ —— 原先直接把"上一次 verify 的失败项"当作判据，
        # 而"审计一致性"本身就在那次失败项里 ⇒ 死锁：一旦失败就永远清不掉 ✗✗。
        # P30-9-HZ：**再修陈旧状态陷阱** ✗✗ —— `verify_all.json` 在本轮**结束时**才写，
        #   所以本步骤读到的是**上一轮**的失败项 ⇒ 上一轮若失败过，本轮即使全绿也会被误判 FAIL ✗
        #   （实测：上一轮 `四机制汇总门禁` FAIL ⇒ 本轮审计误报 ✗）。
        #   ⇒ 该项**不是一致性检查**（是别人的运行摘要 ✓）⇒ 降级为**仅供参考，不计入判定** ✓。
        _fails = [f for f in (verify.get("fails") or [])
                   if "审计一致性" not in str(f)]
        checks.append(("上一次 verify 的失败项（仅供参考，不计入判定 ✓）", _fails, "（不计判定）",
                        True))

    bad = 0
    print(f"{'检查项':<28} {'实测':<14} {'期望':<10} 结论")
    for name, got, want, ok in checks:
        print(f"{name:<28} {str(got):<14} {str(want):<10} {'✓' if ok else '✗'}")
        bad += 0 if ok else 1
    (DOCS / "audit_consistency.json").write_text(
        json.dumps({"checks": [{"name": n, "got": str(g), "want": str(w), "ok": o}
                                for n, g, w, o in checks], "failed": bad},
                    ensure_ascii=False, indent=1), encoding="utf-8")
    if bad:
        print(f"\n[FAIL] {bad} 项不一致（详见 docs/audit_consistency.json）")
        return 1
    print("\n[PASS] 审计记录与实测一致")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
