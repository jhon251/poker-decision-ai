"""Phase 5A Step 4：Task A / Task B 数据划分。

Task A（New Player Generalization）：玩家级 80/20（按 venue×stakes_bucket×ring 分层）
  训练玩家 → 拟合群体/聚类先验；测试玩家（~1000 人）完全不可见，只用于评估。
Task B（Existing Player Adaptation）：玩家内时间对半（按 started_at 升序）
  前半手牌 → 更新后验；后半手牌 → 评估。as_of 强制（前半时间戳 <= 后半）。

输出 data/uoft/splits.json：
  { task_a: {train_players, test_players},
    task_b: {player_id: {history_hand_ids, future_hand_ids, ...}} }

只读 event table；不修改冻结文件（as_of 语义在本脚本内实现）。
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", default="data/uoft/feature_ready/player_event_table.jsonl")
    ap.add_argument("--selection", default="data/uoft/sampling/selected_players.json")
    ap.add_argument("--out", default="data/uoft/splits.json")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train-frac", type=float, default=0.8)
    args = ap.parse_args()

    sel = json.load(open(args.selection, encoding="utf-8"))
    # 玩家主属性（分层用，取自 sampling 输出）
    player_attrs = {pl["player_id"]: pl for pl in sel}

    # 逐玩家收集手牌（hand_id, ts）并按时间排序
    by_player: dict[str, list] = collections.defaultdict(list)
    with open(args.events, encoding="utf-8") as f:
        for line in f:
            ev = json.loads(line)
            by_player[ev["player_id"]].append((ev["ts"], ev["hand_id"]))

    for pid in by_player:
        by_player[pid].sort(key=lambda x: (x[0] or "", x[1]))

    # ---- Task A：玩家级分层 80/20 ----
    strata: dict[tuple, list] = collections.defaultdict(list)
    for pid, pl in player_attrs.items():
        key = (pl["venue"], pl["stakes_bucket"], pl["ring"])
        strata[key].append(pid)
    train_players: list[str] = []
    test_players: list[str] = []
    rng = random.Random(args.seed)
    for key in sorted(strata):
        members = sorted(strata[key])
        rng.shuffle(members)
        k = max(1, int(round(len(members) * args.train_frac)))
        train_players.extend(members[:k])
        test_players.extend(members[k:])
    train_players.sort()
    test_players.sort()
    assert not (set(train_players) & set(test_players)), "train/test 玩家重叠"
    assert len(train_players) + len(test_players) == len(sel)

    # ---- Task B：玩家内时间对半 ----
    task_b = {}
    for pid, hands in by_player.items():
        n = len(hands)
        half = n // 2
        history = [h for _, h in hands[:half]]
        future = [h for _, h in hands[half:]]
        # as_of 不变量：history 时间戳 <= future 时间戳
        hist_ts = [t for t, _ in hands[:half]]
        fut_ts = [t for t, _ in hands[half:]]
        assert not hist_ts or not fut_ts or (max(hist_ts) <= min(fut_ts)), \
            f"{pid}: history/future 时间重叠"
        task_b[pid] = {
            "n_hands": n,
            "n_history": len(history),
            "n_future": len(future),
            "history_hand_ids": history,
            "future_hand_ids": future,
        }

    splits = {
        "task_a": {
            "n_train_players": len(train_players),
            "n_test_players": len(test_players),
            "train_players": train_players,
            "test_players": test_players,
        },
        "task_b": task_b,
    }
    Path(args.out).write_text(json.dumps(splits, ensure_ascii=False), encoding="utf-8")

    # 摘要
    fut_sum = sum(v["n_future"] for v in task_b.values())
    print(f"Task A: train={len(train_players)} test={len(test_players)} "
          f"(overlap={len(set(train_players)&set(test_players))})")
    print(f"Task B: players={len(task_b)} future_events_total={fut_sum}")
    # 测试玩家最小 future 手数
    test_fut = [task_b[p]["n_future"] for p in test_players]
    print(f"test players future hands: min={min(test_fut)} median="
          f"{sorted(test_fut)[len(test_fut)//2]} max={max(test_fut)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
