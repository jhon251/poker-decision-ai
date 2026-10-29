#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P30-9-FV：**可信覆盖率估计**（修正上一版的两个错误）。

上一版错误：
  1. 按 `request_id` 去重，但同一局面被重复分析（`Qh7d2c` 出现 19 次）⇒ 样本被虚增；
  2. 未剔除**牌面残缺**（`flop Ac`、`flop 4c2d`）⇒ 残缺牌面必然 miss，混淆了覆盖率。

本版：
  A. 按**唯一局面**（街+完整牌面+在池人数+底池+跟注额）去重，并统计残缺牌面比例（产品缺陷指标）；
  B. 用**随机牌面**在真实常见配置下做覆盖率估计（样本量足够）：
     SRP(BTN-BB) / 3BP(BTN-BB) × first_in/facing × 若干尺寸 ⇒ 报告
     `exact / 桶内共识 / 桶代表近似 / miss` 分布。
"""
from __future__ import annotations

import collections
import json
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from poker_decision_ai.core.providers.spot_table_6max import SpotTable6max  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
LOG = ROOT / "dist" / "logs"
NEED = {"flop": 3, "turn": 4, "river": 5}
RANKS = "AKQJT98765432"
SUITS = "shdc"


def real_situations() -> list[dict]:
    seen: dict[tuple, dict] = {}
    raw = 0
    for f in sorted(LOG.glob("events-*.jsonl")):
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
            if not vo or vo.get("street") not in NEED:
                continue
            raw += 1
            n_in = len([s for s in (vo.get("seats") or []) if s.get("in_hand")])
            key = (vo.get("street"), "".join(vo.get("board") or []), n_in,
                    vo.get("pot"), vo.get("to_call"))
            seen.setdefault(key, vo)
    return [v | {"_raw": raw} for v in seen.values()], raw


def coverage_random(table: SpotTable6max, n: int = 200, seed: int = 42) -> dict:
    rnd = random.Random(seed)
    deck = [r + s for r in RANKS for s in SUITS]
    configs = [
        # (family, pot_bb, stack_bb, positions, node, ratio)
        # P30-9-GN：按**真实形态**（第 7 轮提取）：SB vs BTN / pot 40 / 栈 100 为主线
        ("3bp", 40.0, 100.0, {"oop": "SB", "ip": "BTN"}, "first_in", None),
        ("3bp", 40.0, 100.0, {"oop": "BB", "ip": "BTN"}, "first_in", None),
        ("3bp", 25.5, 87.5, {"oop": "BB", "ip": "BTN"}, "first_in", None),
        ("srp", 3.5, 96.5, {"oop": "BB", "ip": "BTN"}, "first_in", None),
    ]
    stat: dict[str, collections.Counter] = {f"{c[0]}/{c[4]}/{c[3]["oop"]}-{c[3]["ip"]}/pot{c[1]:g}": collections.Counter()
                                             for c in configs}
    for _ in range(n):
        board = rnd.sample(deck, 3)
        for fam, pot, stack, pos, node, ratio in configs:
            q = {"street": "flop", "board": board, "pot_bb": pot, "node": node,
                  "positions": pos, "actor": pos["oop"], "family": fam, "stack_bb": stack,
                  "bet_ratio": ratio,
                  "to_call_bb": (ratio * pot / (1 + ratio) if ratio else None)}
            h = table.lookup(q)
            if h is None:
                kind = "miss"
            elif (h.get("raw") or {}).get("consensus_n"):
                kind = "consensus"
            elif h.get("is_approximate"):
                kind = "approx"
            else:
                kind = "exact"
            stat[f"{fam}/{node}/{pos["oop"]}-{pos["ip"]}/pot{pot:g}"][kind] += 1
    out = {}
    for k, c in stat.items():
        tot = sum(c.values())
        out[k] = {"n": tot, **{kk: c.get(kk, 0) for kk in ("exact", "consensus", "approx", "miss")},
                   "exact_rate": round(c.get("exact", 0) / max(tot, 1), 4),
                   "any_hit_rate": round((tot - c.get("miss", 0)) / max(tot, 1), 4)}
    return out


def main() -> None:
    sits, raw = real_situations()
    print(f"真实翻后调用 {raw} 次 ⇒ **唯一局面 {len(sits)}**")
    incomplete = [s for s in sits
                   if len("".join(s.get("board") or "")) // 2 != NEED.get(str(s.get("street")), 0)]
    print(f"  其中牌面**残缺** {len(incomplete)}/{len(sits)} = "
          f"{len(incomplete)/max(len(sits),1)*100:.0f}%（产品/日志缺陷指标 ✗）")
    print(f"  完整牌面唯一局面 {len(sits)-len(incomplete)} 个")

    table = SpotTable6max(str(ROOT / "data" / "solver" / "spots_6max"))
    table._ensure_loaded()
    cov = coverage_random(table, n=200)
    print("\n随机牌面覆盖率（每配置 200 个随机翻牌）：")
    for k, v in cov.items():
        print(f"  {k:<16} exact {v['exact']:>3} ({v['exact_rate']*100:.0f}%) | "
              f"共识 {v['consensus']:>3} | 近似 {v['approx']:>3} | miss {v['miss']:>3} | "
              f"任一命中 {v['any_hit_rate']*100:.0f}%")

    (ROOT / "docs" / "coverage_estimate.json").write_text(
        json.dumps({"real": {"raw_calls": raw, "unique_situations": len(sits),
                              "incomplete_board": len(incomplete),
                              "complete_board": len(sits) - len(incomplete)},
                     "random_board_coverage": cov},
                    ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n已写 docs/coverage_estimate.json")


if __name__ == "__main__":
    main()
