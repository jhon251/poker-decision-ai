#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""建议统计：从 logs/attempts.jsonl（优先）或 logs/gui-runs.log 计算成功率。

用法（在 EXE 目录或项目根）：
    python scripts/advice_stats.py [--dir <含 logs/ 的目录>]
输出：总尝试数 / 成功率 / 各状态分布 / 阻断字段分布 / 按小时趋势。
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import re
import sys


def _from_jsonl(path: pathlib.Path) -> list[dict]:
    out = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    except OSError:
        return []
    return out


def _from_runlog(path: pathlib.Path) -> list[dict]:
    out = []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    pat = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}):\d{2}:\d{2} #\d+ status=(\w+)"
                     r"(?:.*?blocking=\[([^\]]*)\])?")
    for line in text.splitlines():
        m = pat.match(line)
        if not m:
            continue
        blocking = [b.strip().strip("'") for b in (m.group(3) or "").split(",")
                    if b.strip()]
        out.append({"ts": m.group(1), "status": m.group(2), "blocking": blocking})
    return out


def main() -> int:
    try:                      # Windows 控制台默认代码页 ≠ UTF-8 ⇒ 强制
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:                             # noqa: BLE001
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=".", help="含 logs/ 的目录（默认当前目录）")
    a = ap.parse_args()
    base = pathlib.Path(a.dir)
    recs = _from_jsonl(base / "logs" / "attempts.jsonl")
    src = "logs/attempts.jsonl"
    if not recs:
        recs = _from_runlog(base / "logs" / "gui-runs.log")
        src = "logs/gui-runs.log（旧的纯文本格式，blocking 可能缺失）"
    if not recs:
        print(f"无记录：{base/'logs'}（先点几次「识别当前牌局」）")
        return 1
    total = len(recs)
    by_status = collections.Counter(r.get("status") for r in recs)
    ok = by_status.get("OK", 0) + by_status.get("ADVICE_REFERENCE", 0)
    blocking = collections.Counter()
    for r in recs:
        for b in (r.get("blocking") or []):
            blocking[b] += 1
    print(f"记录来源: {src}")
    print(f"总尝试  : {total}")
    print(f"决策成功: {by_status.get('OK', 0)}  "
          f"({100.0 * by_status.get('OK', 0) / total:.1f}%)")
    print(f"含参考建议: {by_status.get('ADVICE_REFERENCE', 0)}  "
          f"(OK+参考 = {100.0 * ok / total:.1f}%)")
    print("\n状态分布:")
    for k, v in by_status.most_common():
        print(f"  {k:24s} {v}")
    if blocking:
        print("\n阻断字段分布:")
        for k, v in blocking.most_common():
            print(f"  {k:24s} {v}")
    hourly = collections.Counter(str(r.get("ts"))[:13] for r in recs)
    print("\n按小时:")
    for k in sorted(hourly):
        sub = [r for r in recs if str(r.get("ts")).startswith(k)]
        o = sum(1 for r in sub if r.get("status") in ("OK", "ADVICE_REFERENCE"))
        print(f"  {k}  尝试 {len(sub):3d}  成功 {o:3d}  "
              f"({100.0 * o / len(sub):.0f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
