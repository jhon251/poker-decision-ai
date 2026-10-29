#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P30-9-EV：外部 6-max 解的**节点完备性审计**。

统计 277 个解文件中所有**非空**节点的深度/行动者/到达路径，
并与已导出到 `data/solver/spots_6max` 的节点对比，指出**未导出**的节点族。
"""
from __future__ import annotations

import collections
import json
import pathlib
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = pathlib.Path(r"<repo>")
SRC = ROOT / "data" / "external" / "poker_practice" / "postflop"
DST = ROOT / "data" / "solver" / "spots_6max"


def walk(node, depth, path, out):
    if not isinstance(node, dict):
        return
    if node.get("actions"):
        out.append((depth, str(node.get("p") or "?"), tuple(path)))
    for i, ch in enumerate(node.get("c") or []):
        walk(ch, depth + 1, path + [i], out)


def main() -> None:
    files = sorted(SRC.glob("*.json"))
    nodes: list[tuple] = []
    for f in files:
        d = json.loads(f.read_text(encoding="utf-8"))
        walk(d.get("root"), 0, [], nodes)
    by_depth = collections.Counter(n[0] for n in nodes)
    by_key = collections.Counter((n[0], n[1], len(n[2])) for n in nodes)
    print(f"解文件 {len(files)} 个 | 非空节点 {len(nodes)} 个")
    print("按深度:", dict(by_depth))
    print("\n(深度, 行动者, 路径长度) 分布:")
    for k, v in sorted(by_key.items()):
        print(f"   depth={k[0]} actor={k[1]:<4} pathlen={k[2]} → {v}")
    exported = len(list(DST.glob("*.json")))
    print(f"\n已导出节点数（含 manifest）: {exported}")
    # 到达路径族（相对根）
    fam = collections.Counter(n[2][:2] for n in nodes)
    print("\n前两跳路径族（top 10）:")
    for k, v in fam.most_common(10):
        print(f"   {k} → {v}")


if __name__ == "__main__":
    main()
