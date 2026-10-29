"""P30-9-FW：生成**真实形态 3BP 翻牌**批量（提高真实翻牌 exact 覆盖）。

形态与库内 `3bp-BTN-BB` 对齐（BTN 开 2.5 / BB 3bet / BTN 跟）：
  * oop = BB（3-bettor，强线性范围）；ip = BTN（caller，宽范围）；
  * pot 25.5bb、有效栈 87.5bb、street flop；
  * 牌面按**真实纹理配比**抽样（rainbow ~38% / two-tone ~46% / monotone ~8% / paired ~8%），
    确定性种子 ⇒ 可复现。

产出：`configs/solver/spots_6max_real3bp.yaml`（N 个 HU 子局 spot，供 solve_spots.py 求解）
"""
from __future__ import annotations

import json
import pathlib
import random
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = pathlib.Path(r"<repo>")
N_BOARDS = 48
SEED = 20261011

RANKS = "AKQJT98765432"
SUITS = "shdc"
# 声明范围（我们自己的 HU 子局范围；来源如实标注，勿与外部库混同）
RANGES = {
    "oop": "QQ+,JJ,AKs,AQs,AJs,KQs,AKo,AQo",                      # BB 3-bettor
    "ip": ("TT-22,AQs-A2s,KQs-KTs,QJs-QTs,JTs,T9s,98s,87s,"
            "AQo,KQo,QJo"),                                        # BTN caller
}


def texture(board: list[str]) -> str:
    ranks = [c[0] for c in board]
    suits = [c[1] for c in board]
    paired = len(set(ranks)) < len(ranks)
    mono = len(set(suits)) == 1
    if paired:
        return "paired"
    if mono:
        return "monotone"
    return "rainbow" if len(set(suits)) == 3 else "two-tone"


def gen_boards(n: int, seed: int) -> list[list[str]]:
    """按目标纹理配比抽样牌面（去重、互不相同）。"""
    target = {"rainbow": round(n * 0.38), "two-tone": round(n * 0.46),
               "monotone": round(n * 0.08), "paired": round(n * 0.08)}
    rnd = random.Random(seed)
    got: dict[str, list[list[str]]] = {k: [] for k in target}
    seen: set[tuple] = set()
    guard = 0
    while any(len(got[k]) < target[k] for k in target) and guard < 200000:
        guard += 1
        board = rnd.sample([r + s for r in RANKS for s in SUITS], 3)
        tex = texture(board)
        if tex not in target or len(got[tex]) >= target[tex]:
            continue
        key = tuple(sorted(board))
        if key in seen:
            continue
        seen.add(key)
        got[tex].append(board)
    out = [b for k in ("rainbow", "two-tone", "monotone", "paired") for b in got[k]]
    return out[:n]


def main() -> None:
    boards = gen_boards(N_BOARDS, SEED)
    spots = []
    for i, board in enumerate(boards, 1):
        spots.append({
            "spot_id": f"R4-3BP-BB-BTN-flop-{''.join(board)}",
            "street": "flop", "board": list(board),
            "pot_bb": 25.5, "effective_stack_bb": 87.5,
            "ranges": dict(RANGES),
            "max_iterations": 300, "target_exploitability_pot_pct": 1.0,
            "note": f"真实形态补覆盖（{texture(board)}）| 来源：我们自解 HU 子局（BB 3bet/BTN 跟）",
        })
    cfg = {"spot_suite": {
        "name": "SOLVER-4-real-3bp-flops",
        "game": "HU 子局（BTN 开 2.5 / BB 3bet / BTN 跟）— 真实形态翻牌补覆盖",
        "cli": "tools/bin/postflop-bench-cli.exe",
        "solver": "postflop-solver (b-inary) — 真实 CFR；HU；从当前街起树；逐手 EV+频率导出",
        "n_spots": len(spots),
        "ranges_note": "声明范围为我们自己的 HU 子局范围（非外部库范围）⇒ 入库时 provenance 如实标注",
        "seed_note": f"牌面抽样确定性 seed={SEED}；纹理配比 rainbow/two-tone/monotone/paired",
    }, "spots": spots}
    out = ROOT / "configs" / "solver" / "spots_6max_real3bp.yaml"
    out.write_text(json.dumps(cfg, ensure_ascii=False, indent=1), encoding="utf-8")
    tex_count: dict[str, int] = {}
    for b in boards:
        tex_count[texture(b)] = tex_count.get(texture(b), 0) + 1
    print(f"已写 {out.name}：{len(spots)} 个牌面 | 纹理分布 {tex_count}")
    print("样例:", ["".join(b) for b in boards[:8]])


if __name__ == "__main__":
    main()
