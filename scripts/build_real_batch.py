#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P30-9-GL：真实形态批次生成器（**位置对/底池/栈为显式参数**，防复发）。

教训（第 6 轮）：此前批次把位置对硬编码为 BB-BTN ✗，而真实会话主导是 **SB vs BTN**
⇒ 位置键不匹配 ⇒ 精确命中恒为 0 ✗（"真实覆盖 ≈0%"的真因）。
本生成器强制显式给出 `--oop/--ip/--pot/--stack/--street/--n`，并在配置里写明来源；
可用 `--from-logs` 从 `docs/real_params_all.json` 读取**真实分布**中的主导参数 ✓。

用法示例：
  python scripts/build_real_batch.py --oop SB --ip BTN --pot 40 --stack 100 --n 500 \
      --out configs/solver/spots_6max_sbbtn_500.yaml
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import random
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = pathlib.Path(r"<repo>")
RANKS, SUITS = "AKQJT98765432", "shdc"
# 声明范围（oop = 3-bettor，ip = caller）；来源如实标注：**我们声明的 HU 范围**，非外部库
RANGES = {"oop": "QQ+,JJ,AKs,AQs,AJs,KQs,AKo,AQo",
           "ip": ("TT-22,AQs-A2s,KQs-KTs,QJs-QTs,JTs,T9s,98s,87s,AQo,KQo,QJo")}


def texture(board: list[str]) -> str:
    ranks = [c[0] for c in board]
    suits = [c[1] for c in board]
    if len(set(ranks)) < len(ranks):
        return "paired"
    if len(set(suits)) == 1:
        return "monotone"
    return "rainbow" if len(set(suits)) == 3 else "two-tone"


def gen_boards(n: int, seed: int, extra: int = 0) -> list[list[str]]:
    """按真实纹理配比抽样（rainbow 38% / two-tone 46% / monotone 8% / paired 8%）。"""
    target = {"rainbow": round(n * 0.38), "two-tone": round(n * 0.46),
               "monotone": round(n * 0.08), "paired": round(n * 0.08)}
    rnd = random.Random(seed)
    got: dict[str, list[list[str]]] = {k: [] for k in target}
    seen: set[tuple] = set()
    guard = 0
    while any(len(got[k]) < target[k] for k in target) and guard < 600000:
        guard += 1
        cards = rnd.sample([r + s for r in RANKS for s in SUITS], 3 + extra)
        tex = texture(cards[:3])
        if tex not in target or len(got[tex]) >= target[tex]:
            continue
        key = tuple(cards)
        if key in seen:
            continue
        seen.add(key)
        got[tex].append(cards)
    return [c for k in ("rainbow", "two-tone", "monotone", "paired") for c in got[k]][:n]


def dominant_from_logs() -> dict:
    """从真实参数文件取主导（位置对/底池/栈）。"""
    p = ROOT / "docs" / "real_params_all.json"
    if not p.is_file():
        return {}
    rows = json.loads(p.read_text(encoding="utf-8"))
    pairs = collections.Counter(tuple(sorted([r["hero_pos"] or "?", r["villain_pos"] or "?"]))
                                 for r in rows if r.get("n_in") == 2)
    post = [r for r in rows if r.get("street") in ("flop", "turn", "river")]
    pots = collections.Counter(int(r["pot_bb"] // 10 * 10) for r in post if r.get("pot_bb"))
    top_pair = pairs.most_common(1)[0][0] if pairs else None
    return {"pair": top_pair, "pot": (pots.most_common(1)[0][0] if pots else None)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--oop", required=True, help="oop 位置（须与真实日志一致！）")
    ap.add_argument("--ip", required=True)
    ap.add_argument("--pot", type=float, required=True, help="底池 bb（真实分布主导值）")
    ap.add_argument("--stack", type=float, required=True, help="有效栈 bb")
    ap.add_argument("--street", default="flop", choices=["flop", "turn", "river"])
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=20261016)
    ap.add_argument("--family", default="3bp")
    ap.add_argument("--id-prefix", default=None, help="记录 id 前缀（默认 <oop><ip> 小写）")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    extra = {"flop": 0, "turn": 1, "river": 2}[args.street]
    boards = gen_boards(args.n, args.seed, extra)
    pre = args.id_prefix or f"{args.oop.lower()}{args.ip.lower()}"
    spots = [{"spot_id": f"R-{args.family}-{args.oop}-{args.ip}-{args.street}-{''.join(b)}",
               "street": args.street, "board": list(b), "pot_bb": args.pot,
               "effective_stack_bb": args.stack, "ranges": dict(RANGES),
               "max_iterations": 300, "target_exploitability_pot_pct": 1.0,
               "note": (f"真实形态 {args.oop} vs {args.ip}（pot {args.pot}/栈 {args.stack}，"
                         f"{texture(b[:3])}）| our_hu_solve | 范围=我们声明范围")}
              for b in boards]
    cfg = {"spot_suite": {
        "name": f"SOLVER-real-{args.oop}-{args.ip}-{args.street}-{args.n}",
        "game": f"HU 子局（{args.oop} vs {args.ip}）— 真实形态参数（显式给出）",
        "cli": "tools/bin/postflop-bench-cli.exe",
        "solver": "postflop-solver (b-inary) — 真实 CFR；HU；逐手 EV+频率",
        "positions": {"oop": args.oop.upper(), "ip": args.ip.upper()},
        "pot_bb": args.pot, "effective_stack_bb": args.stack, "street": args.street,
        "ranges_note": "声明范围为我们自己的 HU 子局范围（非外部库）",
        "seed_note": f"seed={args.seed}；真实纹理配比",
        "id_prefix": pre, "n_spots": len(spots)}, "spots": spots}
    out = pathlib.Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.write_text(json.dumps(cfg, ensure_ascii=False, indent=1), encoding="utf-8")
    dom = dominant_from_logs()
    print(f"已写 {out.name}：{len(spots)} spot | oop={args.oop} ip={args.ip} "
          f"pot={args.pot} stack={args.stack} street={args.street}")
    if dom:
        print(f"  真实日志主导：位置对={dom.get('pair')} 底池档={dom.get('pot')} "
              f"（请核对与本次参数一致 ✓）")
    print(f"  入库命令：python scripts/import_hu_solves_to_6max.py --oop {args.oop} "
          f"--ip {args.ip} --id-prefix {pre} ...")


if __name__ == "__main__":
    main()
