"""示例种子数据：向仓库注入演示玩家档案/群体段位/历史手牌。

用法: .\\.venv\\Scripts\\python.exe scripts/seed_demo_data.py
（写入 data/repository/*.jsonl；storage.backend=json 时生效）
"""
from __future__ import annotations

import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from poker_decision_ai.core.orchestration import build_runtime  # noqa: E402

RNG = random.Random(42)
NOW = time.time()


def make_hand(pid: str, i: int, vpip: bool, pfr: bool, cbet: bool,
              fold_to_cbet: bool, agg: float, net_bb: float) -> dict:
    return {
        "hand_id": f"seed-{pid}-{i}",
        "player_id": pid,
        "game": "cash", "stakes": "NL10", "n_players": 6, "position": "CO",
        "vpip": vpip, "pfr": pfr, "three_bet": False, "faced_3bet": False,
        "folded_to_3bet": None, "four_bet": False,
        "cbet_opportunity": cbet or (i % 3 == 0), "cbet": cbet,
        "faced_cbet": True, "folded_to_cbet": fold_to_cbet,
        "saw_flop": vpip, "went_to_showdown": i % 4 == 0,
        "won_at_showdown": i % 2 == 0 if i % 4 == 0 else None,
        "won_hand": net_bb > 0, "net_bb": net_bb,
        "bets": int(agg), "calls": int(3 - agg), "folds": 2,
        "ts": NOW - 60 * 86400 + i * 43200,   # 最近 60 天，每半天一手
    }


def _now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def main() -> None:
    runtime = build_runtime()
    repos = runtime.repos

    # ---- 玩家档案 ----
    repos.players.upsert({
        "player_id": "Bob", "name": "Bob",
        "feature_vector": {
            "tightness": 0.72, "aggression": 0.61, "bluff_tendency": 0.18,
            "calling_tendency": 0.42, "folding_tendency": 0.55,
            "overbet_tendency": 0.12, "check_raise_tendency": 0.2,
            "tilt_risk": 0.34, "icm_sensitivity": 0.67, "stage_adaptivity": 0.5,
        },
        "behavior_statistics": {"vpip": {"value": 0.24, "samples": 1200},
                                "pfr": {"value": 0.19, "samples": 1200},
                                "fold_to_cbet": {"value": 0.48, "samples": 320}},
        "sample_size": 1200, "last_seen": _now_iso(),
    })
    repos.players.upsert({
        "player_id": "Eve", "name": "Eve",
        "feature_vector": {
            "tightness": 0.28, "aggression": 0.35, "bluff_tendency": 0.10,
            "calling_tendency": 0.81, "folding_tendency": 0.20,
            "overbet_tendency": 0.05, "check_raise_tendency": 0.05,
            "tilt_risk": 0.55, "icm_sensitivity": 0.3, "stage_adaptivity": 0.3,
        },
        "behavior_statistics": {"vpip": {"value": 0.58, "samples": 300},
                                "pfr": {"value": 0.08, "samples": 300},
                                "fold_to_cbet": {"value": 0.18, "samples": 80}},
        "sample_size": 300, "last_seen": _now_iso(),
    })

    # ---- 群体段位 ----
    repos.populations.upsert_segment("NL10:cash:6max", {
        "vpip": {"value": 0.26, "samples": 45000},
        "pfr": {"value": 0.18, "samples": 45000},
        "three_bet": {"value": 0.07, "samples": 21000},
        "fold_to_cbet": {"value": 0.52, "samples": 12000},
        "cbet": {"value": 0.62, "samples": 9000},
        "aggression_frequency": {"value": 0.38, "samples": 30000},
        "sample_size": 45000,
    })
    repos.populations.upsert_segment("mtt:bubble:6max", {
        "vpip": {"value": 0.18, "samples": 8000},
        "fold_to_cbet": {"value": 0.66, "samples": 2100},
        "three_bet": {"value": 0.04, "samples": 3100},
        "sample_size": 8000,
    })

    # ---- 历史手牌（统计/画像/情绪用） ----
    for i in range(120):
        repos.hands.save(make_hand("Bob", i, vpip=i % 4 != 0, pfr=i % 5 == 0,
                                   cbet=i % 3 == 0, fold_to_cbet=i % 2 == 0,
                                   agg=2, net_bb=RNG.uniform(-8, 10)))
    for i in range(60):
        repos.hands.save(make_hand("Eve", i, vpip=i % 3 != 0, pfr=False,
                                   cbet=False, fold_to_cbet=i % 4 != 0,
                                   agg=1, net_bb=RNG.uniform(-6, 4)))
    for i in range(8):
        # 最近的 8 手 = 连败 + bad beat（用于情绪/策略演化演示）
        repos.hands.save({**make_hand("Bob", 1000 + i, vpip=False, pfr=False, cbet=False,
                                      fold_to_cbet=True, agg=0, net_bb=-4.0),
                          "ts": NOW - (8 - i) * 900,
                          "equity_at_decision": 0.72 if i < 3 else None,
                          "allin": True, "bad_beat": i < 3})

    print(f"players: {len(repos.players.all())}")
    print(f"population segments: {len(repos.populations.all_segments())}")
    print(f"hands: {repos.hands.count()}")
    print("种子数据已写入", runtime.config.paths()["data_dir"])


if __name__ == "__main__":
    main()
