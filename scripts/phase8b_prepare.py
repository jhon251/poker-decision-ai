"""Phase 8B Step 1：预注册 hand-level split（60/20/20，seed=42，按 table 分层）。

一次性只读扫描 Pluribus 语料 → 8A 冻结资格判定（replayable 集合完整继承）→
写入 8B 工作文件（hands_replayable.jsonl + splits.json）。此后 PAM 拟合与实验
均只读这两个工作文件，不再重扫 parquet。

冻结边界：只调用 8A 冻结函数（derive_position_map / replay_hand / mapping），
不修改任何 Phase ≤8A 产物。
"""
from __future__ import annotations

import glob
import json
import random
import sys
from dataclasses import asdict
from pathlib import Path

_SCRIPT_DIR = Path(__file__).parent
_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_SCRIPT_DIR))

import pyarrow.parquet as pq  # noqa: E402

import phase8a_replay as p8  # noqa: E402
from scripts.uoft_mapping import map_row_to_canonical  # noqa: E402

DATASET = r"C:\PokerDataset\UofT_CPRG\data"
OUT_DIR = _ROOT / "data/uoft/models/phase8/phase8b"
SPLIT_SEED = 42
SPLIT_RATIO = (0.6, 0.2, 0.2)


def main() -> int:
    files = sorted(glob.glob(f"{DATASET}\\*.parquet"))
    hands = []
    n_excluded = {"position_ambiguous": 0, "engine_error": 0, "map_failed": 0,
                  "non_pluribus": 0}
    seen = set()
    for fp in files:
        t = pq.read_table(fp)
        for row in t.to_pylist():
            sf = (row.get("source_file") or "").lower()
            if "pluribus" not in sf:
                continue
            hand, _ = map_row_to_canonical(row, 0)
            if hand is None:
                n_excluded["map_failed"] += 1
                continue
            pids = [str(p.player_id) for p in hand.players]
            pos = p8.derive_position_map(pids, row["seats"],
                                         row["blinds_or_straddles"],
                                         hand.preflop_actions)
            if pos is None:
                n_excluded["position_ambiguous"] += 1
                continue
            hero = pids[0]
            trace = p8.replay_hand(hand, hero, pos, seed=SPLIT_SEED)
            if trace["excluded"]:
                n_excluded["engine_error"] += 1
                continue
            hk = f"pluribus|{sf}|{hand.hand_id}"
            if hk in seen:
                continue                       # 防重复（不应发生）
            seen.add(hk)
            table = (row.get("source_file") or "").rsplit("/", 1)[0]
            hands.append({
                "hand_key": hk, "table": table, "hand_id": hand.hand_id,
                "players": [{"pid": str(p.player_id),
                             "stack_start": p.stack_start,
                             "stack_end": p.stack_end,
                             "hole_cards": list(p.hole_cards)}
                            for p in hand.players],
                "sb": hand.sb, "bb": hand.bb,
                "positions": pos["labels"], "orbit": pos["orbit"],
                "actions": {"preflop": [asdict(a) for a in hand.preflop_actions],
                            "flop": [asdict(a) for a in hand.flop_actions],
                            "turn": [asdict(a) for a in hand.turn_actions],
                            "river": [asdict(a) for a in hand.river_actions]},
                "flop": list(hand.flop or []), "turn": list(hand.turn or []),
                "river": list(hand.river or []),
                "all_in": list(hand.all_in or []),
            })

    # ---- 预注册 split：按 table 分层 60/20/20（每表内确定性 shuffle）----
    by_table: dict[str, list] = {}
    for h in hands:
        by_table.setdefault(h["table"], []).append(h)
    splits = {"train": [], "cal": [], "eval": []}
    for table, hs in sorted(by_table.items()):
        rng = random.Random(f"8b-split-{table}")
        order = hs[:]
        rng.shuffle(order)
        n = len(order)
        n_tr = int(n * SPLIT_RATIO[0])
        n_ca = int(n * SPLIT_RATIO[1])
        for i, h in enumerate(order):
            part = "train" if i < n_tr else ("cal" if i < n_tr + n_ca else "eval")
            h["split"] = part
            splits[part].append(h["hand_key"])
    n = {k: len(v) for k, v in splits.items()}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "hands_replayable.jsonl", "w", encoding="utf-8") as w:
        for h in hands:
            w.write(json.dumps(h, ensure_ascii=False) + "\n")
    (OUT_DIR / "splits.json").write_text(json.dumps({
        "seed": SPLIT_SEED, "ratio": list(SPLIT_RATIO),
        "strategy": "per-table stratified (60/20/20, deterministic shuffle)",
        "n": n, "sets": splits,
        "excluded": n_excluded,
        "total_replayable": len(hands),
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"replayable={len(hands)} splits={n} excluded={n_excluded}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
