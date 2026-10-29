"""Phase 8B Step 2：PAM（Population Action Model）拟合（train-only）。

PAM = 经验条件频率模型，同时充当 Baseline A（hero 策略）与 Response Model
（villain 响应）。冻结约束：
  - 只消费公开信息（street/position/spot/下注档），禁止任何隐藏/结果字段
    （拟合只读 actions/positions；终局/手牌/胜者/净结果一律不读——源码级保证）
  - 只在 8B train split 手（60%）上拟合；cal/eval 手零参与（hand_key 交集 = ∅ 硬断言）
  - 平滑伪计数 = 1（预注册常量）；最小样本阶梯：cell<50 → street×spot → street
  - 不是 ML；不新增任何模型；deterministic（同输入 → 同产物）

输出 data/uoft/models/phase8/phase8b/pam.json + fit_report.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).parent
_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_SCRIPT_DIR))

OUT_DIR = _ROOT / "data/uoft/models/phase8/phase8b"
ACTIONS = ("fold", "check", "call", "bet", "raise")
SIZE_BUCKETS = (("s1", 1 / 3.0), ("s2", 2 / 3.0), ("s3", 1.0), ("s4", float("inf")))
BET_BUCKETS = (("b1", 0.5), ("b2", 1.5), ("b3", float("inf")))
PSEUDO_COUNT = 1.0
MIN_CELL = 50


def _bet_bucket(ratio: float) -> str:
    for name, hi in BET_BUCKETS:
        if ratio <= hi:
            return name
    return "b3"


def _size_bucket(ratio: float) -> str:
    for name, hi in SIZE_BUCKETS:
        if ratio <= hi:
            return name
    return "s4"


def enumerate_actions(hand: dict) -> list[dict]:
    """逐行动枚举公开 cell（PAM 拟合口径）：每 token → (cell, action, size_bucket)。

    与 8A 口径一致：commitment 跟踪、pot = 全部贡献、facing = current_bet − 已投入。
    只读 hand["actions"]/["positions"]；不读手牌/终局筹码/胜者字段。
    """
    out = []
    pos = hand["positions"]
    committed: dict[str, float] = {}
    pot = 0.0
    voluntary = False
    folded: set = set()
    for street in ("preflop", "flop", "turn", "river"):
        current_bet = 0.0
        for a in hand["actions"].get(street, []):
            pid = a["player_id"]
            kind = a.get("action", "")
            if kind == "post":
                amt = float(a.get("amount") or 0.0)
                pot += amt
                committed[pid] = committed.get(pid, 0.0) + amt
                if street == "preflop" and hand.get("bb"):
                    current_bet = max(current_bet, float(hand["bb"]))
                continue
            if pid in folded:
                continue
            if kind not in ACTIONS:
                continue
            position = pos.get(pid)
            if position is None:
                continue                       # 不应发生（位置已推导）；保守跳过
            pot_before = pot
            facing = current_bet - committed.get(pid, 0.0)
            if facing < -1e-9:
                continue                       # 数据异常：保守跳过（计入 skip）
            if street == "preflop":
                if voluntary:
                    spot = "facing_bet" if facing > 1e-9 else "checked_to"
                else:
                    spot = "unopened"
            else:
                spot = "facing_bet" if facing > 1e-9 else "checked_to"
            bucket = _bet_bucket(facing / max(pot_before, 1e-9)) \
                if spot == "facing_bet" else "b0"
            size_b = None
            if kind in ("bet", "raise"):
                amt = float(a.get("amount") or 0.0)
                size_b = _size_bucket(amt / max(pot_before, 1e-9))
            out.append({"street": street, "position": position, "spot": spot,
                        "bet_bucket": bucket, "action": kind,
                        "size_bucket": size_b, "pid": pid})
            # ---- 状态更新 ----
            if kind == "fold":
                folded.add(pid)
            elif kind == "check":
                pass
            elif kind == "call":
                amt = float(a.get("amount") or 0.0)
                pot += amt
                committed[pid] = committed.get(pid, 0.0) + amt
                if street == "preflop":
                    voluntary = True
            else:                              # bet / raise
                to = float(a.get("to") or 0.0)
                amt = float(a.get("amount") or 0.0)
                prior = committed.get(pid, 0.0)
                contrib = (to - prior) if to > 0 else amt
                if contrib < 0:
                    contrib = 0.0
                pot += contrib
                committed[pid] = prior + contrib
                current_bet = max(current_bet, committed[pid])
                if street == "preflop":
                    voluntary = True
    return out


def fit(hands: list[dict]) -> dict:
    cell_counts: dict[str, dict[str, int]] = {}
    street_spot_counts: dict[str, dict[str, int]] = {}
    street_counts: dict[str, dict[str, int]] = {}
    size_counts: dict[str, dict[str, int]] = {}
    size_ss_counts: dict[str, dict[str, int]] = {}
    for h in hands:
        for tok in enumerate_actions(h):
            key = f"{tok['street']}|{tok['position']}|{tok['spot']}|{tok['bet_bucket']}"
            cell_counts.setdefault(key, {})
            cell_counts[key][tok["action"]] = cell_counts[key].get(tok["action"], 0) + 1
            ss = f"{tok['street']}|{tok['spot']}"
            street_spot_counts.setdefault(ss, {})
            street_spot_counts[ss][tok["action"]] = \
                street_spot_counts[ss].get(tok["action"], 0) + 1
            street_counts.setdefault(tok["street"], {})
            street_counts[tok["street"]][tok["action"]] = \
                street_counts[tok["street"]].get(tok["action"], 0) + 1
            if tok["size_bucket"] is not None:
                sk = f"{tok['street']}|{tok['spot']}|{tok['action']}"
                size_counts.setdefault(sk, {})
                size_counts[sk][tok["size_bucket"]] = \
                    size_counts[sk].get(tok["size_bucket"], 0) + 1
                ss2 = f"{tok['street']}|{tok['action']}"
                size_ss_counts.setdefault(ss2, {})
                size_ss_counts[ss2][tok["size_bucket"]] = \
                    size_ss_counts[ss2].get(tok["size_bucket"], 0) + 1

    action_probs: dict[str, dict[str, float]] = {}
    size_probs: dict[str, dict[str, float]] = {}
    fallback = {"cell": 0, "street_spot": 0, "street": 0,
                "size_cell": 0, "size_street": 0}
    for key, counts in cell_counts.items():
        street = key.split("|")[0]
        ss = key.split("|")[0] + "|" + key.split("|")[2]
        if sum(counts.values()) >= MIN_CELL:
            src = counts
        elif sum(street_spot_counts.get(ss, {}).values()) >= MIN_CELL:
            src = street_spot_counts[ss]
            fallback["street_spot"] += 1
        else:
            src = street_counts[street]
            fallback["street"] += 1
        total = sum(src.values())
        action_probs[key] = {a: round((src.get(a, 0) + PSEUDO_COUNT)
                                      / (total + PSEUDO_COUNT * len(ACTIONS)), 6)
                             for a in ACTIONS}
    for sk, counts in size_counts.items():
        street, action = sk.split("|")[0], sk.split("|")[2]
        if sum(counts.values()) >= MIN_CELL:
            src = counts
        elif sum(size_ss_counts.get(f"{street}|{action}", {}).values()) >= MIN_CELL:
            src = size_ss_counts[f"{street}|{action}"]
            fallback["size_street"] += 1
        else:
            continue                        # 不足：该 (street,action) 尺寸分布不可用
        total = sum(src.values())
        size_probs[sk] = {b: round((src.get(b, 0) + PSEUDO_COUNT)
                                   / (total + PSEUDO_COUNT * len(SIZE_BUCKETS)), 6)
                          for b, _ in SIZE_BUCKETS}
    return {"action_probs": action_probs, "size_probs": size_probs,
            "n_cells": len(action_probs),
            "n_size_cells": len(size_probs),
            "fallback": fallback,
            "cell_counts": {k: dict(v) for k, v in cell_counts.items()}}


def main() -> int:
    splits = json.load(open(OUT_DIR / "splits.json", encoding="utf-8"))
    train_set = set(splits["sets"]["train"])
    cal_set = set(splits["sets"]["cal"])
    eval_set = set(splits["sets"]["eval"])
    assert not (train_set & (cal_set | eval_set)), "split 交集非空（泄漏）"

    train_hands = []
    seen = set()
    for line in open(OUT_DIR / "hands_replayable.jsonl", encoding="utf-8"):
        h = json.loads(line)
        if h["hand_key"] in train_set:
            assert h["hand_key"] not in seen
            seen.add(h["hand_key"])
            train_hands.append(h)
    assert len(train_hands) == splits["n"]["train"], "train 手数与 split 不符"

    pam = fit(train_hands)
    artifact = {
        "phase": "8B", "model": "PAM (Population Action Model)",
        "role": "Baseline A policy + villain response model（同一张表）",
        "nature": "empirical conditional frequency model（经验频率表 + Laplace 平滑）；"
                  "不是 ML；不含 hidden info（street/position/spot/bet-bucket 仅公开字段）",
        "config": {
            "actions": list(ACTIONS), "size_buckets": [list(b) for b in SIZE_BUCKETS],
            "bet_buckets": [list(b) for b in BET_BUCKETS],
            "pseudo_count": PSEUDO_COUNT, "min_cell": MIN_CELL,
            "fallback_ladder": "cell → (street×spot) → street（同阶梯用于尺寸分布）",
            "fit_set": "8B train split (60%)，hand_key 与 cal/eval 交集 = ∅ 硬断言",
            "prohibited_fields": ["对手手牌", "终局筹码", "胜者", "摊牌",
                                  "未来行动", "净结果"],
        },
        "train_hands": len(train_hands),
        "n_cells": pam["n_cells"], "n_size_cells": pam["n_size_cells"],
        "fallback": pam["fallback"],
        "action_probs": pam["action_probs"],
        "size_probs": pam["size_probs"],
    }
    (OUT_DIR / "pam.json").write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / "fit_report.json").write_text(json.dumps({
        "phase": "8B", "model": "PAM", "train_hands": len(train_hands),
        "n_cells": pam["n_cells"], "n_size_cells": pam["n_size_cells"],
        "fallback": pam["fallback"],
        "cell_counts": pam["cell_counts"],
        "note": "本模型仅证明在 frozen response model 下的 simulated advantage；"
                "不构成现实玩家行为的真实性证明",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"PAM fitted: train_hands={len(train_hands)} cells={pam['n_cells']} "
          f"fallback={pam['fallback']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
