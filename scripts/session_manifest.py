#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P30-9-HL：**会话级采集清单**（按真实会话统计各项指标，供数据积累后自动收敛 ✓）。

输出 `docs/session_manifest.json` + 终端表格。每会话给出：
  翻后请求 / 完整牌面 / `to_call` 可知 / 两人池 / **三者兼备** / 唯一牌面数 / **精确命中** ✓
口径与 `real_coverage_gate.py` 一致（确定性、不走引擎 ✓）；并对"三者兼备"给出小样本提示 ✓。
"""
from __future__ import annotations

import collections
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = pathlib.Path(__file__).resolve().parents[1]
LOG = ROOT / "dist" / "logs"
LIB = ROOT / "data" / "solver" / "spots_6max"
NEED = {"flop": 3, "turn": 4, "river": 5}


def observations() -> list[tuple[str, dict]]:
    """返回 [(会话日, 观测)] —— 会话日取事件日期（文件名/ts 均可 ✓）。"""
    out: list[tuple[str, dict]] = []
    for f in sorted(LOG.glob("events-*.jsonl")):
        day = f.name.replace("events-", "").replace(".jsonl", "")[:8]
        seen: dict[str, dict] = {}
        for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                d = json.loads(line)
            except Exception:                                    # noqa: BLE001
                continue
            if d.get("event") != "agent_invocation" or d.get("agent") != "vision":
                continue
            vo = ((d.get("input") or {}).get("request") or {}).get("vision_observation")
            if vo and vo.get("street"):
                seen[str(d.get("request_id") or len(seen))] = vo
        for rid, vo in seen.items():
            out.append((day, vo))
    return out


def keys_from(vo: dict) -> dict | None:
    from scripts.real_coverage_gate import keys_from as _kf
    return _kf(vo)


def main() -> None:
    from poker_decision_ai.core.providers.spot_table_6max import SpotTable6max
    table = SpotTable6max(str(LIB))
    table._ensure_loaded()

    rows = observations()
    by_day: dict[str, list[dict]] = collections.defaultdict(list)
    for day, vo in rows:
        by_day[day].append(vo)

    manifest: dict = {"sessions": {}, "note":
                       "按真实会话统计（确定性口径，不走引擎 ✓）；"
                       "三者兼备 = 牌面完整 + to_call 可知 + 两人池 ✓"}
    print(f"{'会话':<10} {'翻后':>4} {'完整':>4} {'tc知':>4} {'2人':>4} {'兼备':>4} "
          f"{'唯一':>4} {'命中':>4}")
    tot = collections.Counter()
    for day in sorted(by_day):
        post = [v for v in by_day[day] if str(v.get("street") or "") in NEED]
        if not post:
            continue
        cb = sum(1 for v in post
                  if len([c for c in (v.get("board") or []) if c]) == NEED[str(v["street"])])
        tk = sum(1 for v in post if v.get("to_call") is not None)
        two = sum(1 for v in post
                   if len([s for s in (v.get("seats") or []) if s.get("in_hand")]) == 2)
        full = [v for v in post
                 if len([c for c in (v.get("board") or []) if c]) == NEED[str(v["street"])]
                 and v.get("to_call") is not None
                 and len([s for s in (v.get("seats") or []) if s.get("in_hand")]) == 2]
        ks = [k for k in (keys_from(v) for v in full) if k]
        exact = 0
        for k in ks:
            for fam in ("3bp", "srp"):
                h = table.lookup({**k, "family": fam})
                if h is not None and not h.get("is_approximate") \
                        and not (h.get("raw") or {}).get("consensus_n"):
                    exact += 1
                    break
        uniq = len({"".join(str(c) for c in (v.get("board") or [])) for v in post})
        manifest["sessions"][day] = {"postflop": len(post), "board_complete": cb,
                                       "to_call_known": tk, "two_player": two,
                                       "fully_evaluable": len(full), "unique_boards": uniq,
                                       "exact_hits": exact,
                                       "exact_rate": (round(exact / len(ks), 4) if ks else None)}
        tot.update({"postflop": len(post), "board_complete": cb, "to_call_known": tk,
                     "two_player": two, "fully_evaluable": len(full), "exact": exact,
                     "keys": len(ks)})
        print(f"{day:<10} {len(post):>4} {cb:>4} {tk:>4} {two:>4} {len(full):>4} "
              f"{uniq:>4} {exact}/{len(ks)}")
    manifest["totals"] = dict(tot)
    if tot.get("keys"):
        manifest["totals"]["exact_rate"] = round(tot["exact"] / tot["keys"], 4)
    print(f"\n合计: 翻后 {tot['postflop']} | 三者兼备 {tot['fully_evaluable']} "
          f"| 精确命中 {tot['exact']}/{tot['keys']}"
          + (f" = {tot['exact'] / tot['keys'] * 100:.0f}%" if tot.get("keys") else ""))
    if tot.get("keys", 0) < 30:
        print("⚠️ 样本仍很小（<30）⇒ 比例不可作能力结论 ✓（如实标注 ✓）")
    (ROOT / "docs" / "session_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print("已写 docs/session_manifest.json")


if __name__ == "__main__":
    main()
