#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P30-9-GU/HQ：**补读与输入缺口证据统计**（读 `logs/board_fallback.jsonl`）。

口径修正（P30-9-HQ，修正两处报告缺陷 ✗）：
  1. **恢复率只在补读事件上有意义** ✓：分母 = `fallback_ok + fallback_failed`；
     若两者皆为 0 ⇒ 如实报告"**不可读**（无补读事件）" ✓，**不再打出 0%** ✗（此前分母为空却给数字 ✗）；
  2. `to_call_unknown` 属**另一类输入缺口** ⇒ **单独统计** ✓，不混进补读恢复率 ✓；
  3. 按**来源**分组（`runtime` = 真实对局 ✓ / `frames_replay` / `gate` 等）✓，
     只有 `runtime` 事件才用于"真实恢复率" ✓。
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = pathlib.Path(__file__).resolve().parents[1]
CANDIDATES = [pathlib.Path("logs") / "board_fallback.jsonl",
               ROOT / "logs" / "board_fallback.jsonl",
               ROOT / "dist" / "logs" / "board_fallback.jsonl"]
FALLBACK_KINDS = ("fallback_ok", "fallback_failed")


def load() -> list[dict]:
    for p in CANDIDATES:
        if p.is_file():
            out = []
            for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
                if line.strip():
                    try:
                        out.append(json.loads(line))
                    except Exception:                            # noqa: BLE001
                        continue
            return out
    return []


def _rate(rows: list[dict]) -> float | None:
    """补读恢复率；**无补读事件 ⇒ None（不可读）** ✓。"""
    ok = sum(1 for r in rows if r.get("kind") == "fallback_ok")
    bad = sum(1 for r in rows if r.get("kind") == "fallback_failed")
    return (ok / (ok + bad)) if (ok + bad) else None


def summarize(rows: list[dict]) -> dict:
    """纯函数：给定事件行 ⇒ 结构化统计（便于单测 ✓）。"""
    by_kind = collections.Counter(r.get("kind") for r in rows)
    by_source = collections.Counter(r.get("source") or "runtime" for r in rows)
    real = [r for r in rows if (r.get("source") or "runtime") == "runtime"]
    real_fb = [r for r in real if r.get("kind") in FALLBACK_KINDS]
    ok = sum(1 for r in rows if r.get("kind") == "fallback_ok")
    bad = sum(1 for r in rows if r.get("kind") == "fallback_failed")
    return {
        "events": len(rows),
        "by_kind": dict(by_kind),
        "by_source": dict(by_source),
        "fallback_ok": ok,
        "fallback_failed": bad,
        "recovery_rate_all_sources": _rate(rows),
        "real_fallback_events": len(real_fb),
        "recovery_rate_real": _rate(real_fb),
        "to_call_unknown_total": by_kind.get("to_call_unknown", 0),
        "to_call_unknown_real": sum(
            1 for r in real if r.get("kind") == "to_call_unknown"),
        "note": "恢复率只在补读事件上有意义；无补读事件 ⇒ None（不可读 ✓）",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true", help="输出 JSON")
    args = ap.parse_args()
    rows = load()
    s = summarize(rows)
    if not rows:
        msg = ("尚无证据事件（logs/board_fallback.jsonl 不存在或为空）——"
                "下次真实出现残缺牌面或 to_call 未知时会自动留痕 ✓（不猜测 ✓）")
        print(msg)
        if args.json:
            print(json.dumps({"events": 0, **s}, ensure_ascii=False))
        return

    if s["recovery_rate_all_sources"] is None:
        print(f"证据事件 {len(rows)}：**其中无补读事件** ⇒ 补读恢复率**不可读** ✓"
              f"（此前会误打 0% ✗）")
    else:
        print(f"补读事件（全部来源）{s['fallback_ok'] + s['fallback_failed']}："
              f"成功 {s['fallback_ok']} / 失败 {s['fallback_failed']} ⇒ "
              f"恢复率 **{s['recovery_rate_all_sources'] * 100:.0f}%**")
    print(f"  按来源: {s['by_source']}")
    if s["recovery_rate_real"] is None:
        print(f"  **真实**补读事件: {s['real_fallback_events']} 条 ⇒ 真实恢复率**不可读** ✓"
              f"（现有事件来自帧回放/门禁 ⇒ 不冒充真实数据 ✓）")
    else:
        print(f"  **真实**恢复率: {s['recovery_rate_real'] * 100:.0f}%"
              f"（{s['real_fallback_events']} 条补读事件）")
    print(f"  `to_call` UNKNOWN（另一类输入缺口，单独统计 ✓）: "
          f"合计 {s['to_call_unknown_total']} 次（其中真实 {s['to_call_unknown_real']} 次）")
    for kind, n in sorted(s["by_kind"].items()):
        print(f"    {str(kind):<18} × {n}")
    ex = [r for r in rows if r.get("kind") == "fallback_ok"][:3]
    for r in ex:
        print(f"  例: {r.get('ts')} street={r.get('street')} "
              f"v3f={r.get('v3f_board')} → local={r.get('local_board')}")
    if args.json:
        print(json.dumps(s, ensure_ascii=False))


if __name__ == "__main__":
    main()
