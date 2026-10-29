#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P30-9-EW：未导出的深层 6-max 节点能带来多少**可用新桶**。

对已导出（spots_6max）与未导出（外部树深层）节点，统一计算
`(node, actor_position, pot_class, texture, bet_bucket)`（facing 时含尺寸桶）
⇒ 统计未导出节点新增的桶数量与样例。
"""
from __future__ import annotations

import collections
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from poker_decision_ai.core.providers.spot_table import _pot_class, _texture_class  # noqa: E402
from poker_decision_ai.core.providers.spot_table_6max import norm_pos          # noqa: E402

ROOT = pathlib.Path(r"<repo>")
SRC = ROOT / "data" / "external" / "poker_practice" / "postflop"
DST = ROOT / "data" / "solver" / "spots_6max"
ACT = {"X": "check", "B": "bet", "AI": "allin", "C": "call", "F": "fold", "R": "raise"}


def bucket_of(node_kind: str, actor: str, pot: float, board: list, ratio: float | None) -> tuple:
    bb = None
    if ratio is not None:
        bb = "large" if ratio > 0.8 else ("mid" if ratio >= 0.45 else "small")
    return (node_kind, actor, str(_pot_class(pot)), str(_texture_class(board)), str(bb))


def main() -> None:
    have = collections.Counter()
    for f in DST.glob("*.json"):
        r = json.loads(f.read_text(encoding="utf-8"))
        ratio = r.get("bet_ratio")
        have[bucket_of(str(r.get("node")), norm_pos(r.get("actor_position")),
                        float(r.get("pot_bb") or 0), list(r.get("board") or []), ratio)] += 1
    print(f"已导出桶数: {len(have)}")

    new = collections.defaultdict(list)
    all_b = collections.Counter()
    for f in sorted(SRC.glob("*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        board = list(d.get("board") or [])
        stack = float(d.get("stack") or 0)
        spot = str(d.get("spot") or "")
        parts = spot.split("-")
        pos = {"oop": parts[1] if len(parts) > 1 else "?", "ip": parts[2] if len(parts) > 2 else "?"}

        def walk(node, pot, path):
            if not isinstance(node, dict):
                return
            if node.get("actions"):
                actor = pos.get(str(node.get("p") or "oop"), "?")
                acts = node.get("actions") or []
                facing = None
                if path:                       # 非根节点 ⇒ 面对上一跳的下注
                    pass
                for a in acts:
                    name = ACT.get(str(a.get("a")), str(a.get("a") or "").lower())
                    size = a.get("bb")
                    if name in ("bet", "allin") and size is not None:
                        # 子节点视角：对手下注 ⇒ 子节点为 facing
                        pass
                # 本节点自身的 facing 判定：由父动作给出（下面递归时传入）
                ratio = node.get("_ratio")
                b = bucket_of("facing_bet" if ratio is not None else "first_in",
                               norm_pos(actor), pot, board, ratio)
                all_b[b] += 1
                if b not in have:
                    new[b].append((f.name, tuple(path)))
            for i, ch in enumerate(node.get("c") or []):
                if not isinstance(ch, dict):
                    continue
                a = (node.get("actions") or [])[i] if i < len(node.get("actions") or []) else {}
                nm = ACT.get(str(a.get("a")), "")
                sz = a.get("bb")
                nxt_pot = pot + (float(sz) if (nm in ("bet", "call") and sz is not None) else 0.0)
                if isinstance(ch, dict) and nm in ("bet", "allin") and sz is not None:
                    ch = {**ch, "_ratio": round(float(sz) / max(pot, 1e-9), 4)}
                walk(ch, nxt_pot, path + [i])

        walk(d.get("root"), float(d.get("pot") or 0), [])

    print(f"外部全树桶数: {len(all_b)}")
    print(f"未导出节点可带来的**新桶**: {len(new)}")
    for b, ex in sorted(new.items(), key=lambda kv: -len(kv[1]))[:16]:
        print(f"   {b} → {len(ex)} 节点，样例 {ex[0][0]}")


if __name__ == "__main__":
    main()
