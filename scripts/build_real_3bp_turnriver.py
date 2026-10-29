"""P30-9-GB：生成 **turn / river 子局**批量（真实形态 3BP BTN-BB）。

语义（诚实标注，两个节点族）：
  A. 翻牌**过牌-过牌** → turn/river：pot 25.5bb、栈 87.5bb *(双方翻牌未投入)*
  B. 翻牌 **33% 下注-跟注** → turn/river：pot 42.3bb、栈 79.1bb *(翻牌各投 8.4bb)*

范围沿用 3BP 的声明范围（BB 3-bettor / BTN caller）；牌面按纹理配比抽样、确定性 seed。
产出：`configs/solver/spots_6max_real3bp_turnriver.yaml`
"""
from __future__ import annotations

import json
import pathlib
import random
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = pathlib.Path(r"<repo>")
N_EACH = 150
SEED = 20261013
RANKS, SUITS = "AKQJT98765432", "shdc"
RANGES = {"oop": "QQ+,JJ,AKs,AQs,AJs,KQs,AKo,AQo",
           "ip": ("TT-22,AQs-A2s,KQs-KTs,QJs-QTs,JTs,T9s,98s,87s,"
                   "AQo,KQo,QJo")}


def flip_texture(cards: list[str]) -> str:
    ranks = [c[0] for c in cards]
    suits = [c[1] for c in cards]
    if len(set(ranks)) < len(ranks):
        return "paired"
    if len(set(suits)) == 1:
        return "monotone"
    return "rainbow" if len(set(suits)) == len(cards) else "two-tone"


def gen_runouts(n: int, seed: int, street: str) -> list[list[str]]:
    """按 flop 纹理配比抽样完整 runout（flop 3 张 + turn [+ river]）。"""
    extra = 1 if street == "turn" else 2
    target = {"rainbow": round(n * 0.38), "two-tone": round(n * 0.46),
               "monotone": round(n * 0.08), "paired": round(n * 0.08)}
    rnd = random.Random(seed)
    got: dict[str, list[list[str]]] = {k: [] for k in target}
    seen: set[tuple] = set()
    guard = 0
    while any(len(got[k]) < target[k] for k in target) and guard < 400000:
        guard += 1
        cards = rnd.sample([r + s for r in RANKS for s in SUITS], 3 + extra)
        tex = flip_texture(cards[:3])
        if tex not in target or len(got[tex]) >= target[tex]:
            continue
        key = tuple(cards)
        if key in seen:
            continue
        seen.add(key)
        got[tex].append(cards)
    return [c for k in ("rainbow", "two-tone", "monotone", "paired") for c in got[k]][:n]


def main() -> None:
    spots = []
    for street in ("turn", "river"):
        for fam, pot, stack, tag in (("checkcheck", 25.5, 87.5, "xx"),
                                      ("betcall", 42.3, 79.1, "bc")):
            for runout in gen_runouts(N_EACH, SEED + hash(street + tag) % 1000, street):
                spots.append({
                    "spot_id": f"R6-3BP-BB-BTN-{street}-{tag}-{''.join(runout)}",
                    "street": street, "board": list(runout),
                    "pot_bb": pot, "effective_stack_bb": stack,
                    "ranges": dict(RANGES),
                    "max_iterations": 300, "target_exploitability_pot_pct": 1.0,
                    "note": (f"真实形态 turn/river 补覆盖（{flip_texture(runout[:3])}，{tag}）"
                              "| our_hu_solve"),
                })
    cfg = {"spot_suite": {
        "name": "SOLVER-6-real-3bp-turnriver",
        "game": "HU 子局（BB 3bet / BTN 跟）— turn/river 真实形态补覆盖",
        "cli": "tools/bin/postflop-bench-cli.exe",
        "solver": "postflop-solver (b-inary) — 真实 CFR；HU；单街起树；逐手 EV+频率",
        "n_spots": len(spots),
        "nodes_note": "xx=翻牌过牌-过牌后；bc=翻牌 33% 下注-跟注后（pot/stack 已按此设）",
        "seed_note": f"seed={SEED}；flop 纹理配比同前",
    }, "spots": spots}
    out = ROOT / "configs" / "solver" / "spots_6max_real3bp_turnriver.yaml"
    out.write_text(json.dumps(cfg, ensure_ascii=False, indent=1), encoding="utf-8")
    from collections import Counter
    print(f"已写 {out.name}：{len(spots)} 个 spot")
    print("  按街/节点:", dict(Counter((s["street"], s["spot_id"].split("-")[4]) for s in spots)))


if __name__ == "__main__":
    main()
