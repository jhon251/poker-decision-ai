"""Phase 8C 共享：合成手模拟器 + 偏差策略 + 特征率测量 + V0 μ 校准。

冻结边界：
  - 只读 8B 冻结 PAM artifact；冻结 train-only priors（hb_fit/方案 B 口径）
  - 校准 = 一次性、预实验、seed=42、**结果无关**：不读取任何 validation/eval
    outcome、detection accuracy、ΔEV、recommendation verdict；不修改冻结 μ、
    7A/7B/7C 任何阈值、δ=0.02、action mapping；不按 replay 结果重拟合
  - 潜伏参数（type/feature/direction/δ）只存在于生成器与 admin evaluator；
    观测（事件行）只有公开字段
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import sys

_SCRIPT_DIR = Path(__file__).parent
_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_SCRIPT_DIR))

from phase8b_arms import ACTIONS, PAMPolicy, SIZE_CENTERS  # noqa: E402
from phase8b_pam import _bet_bucket, _size_bucket  # noqa: E402
import phase8c_spec as spec  # noqa: E402

PAM_PATH = _ROOT / "data/uoft/models/phase8/phase8b/pam.json"
CAL_PATH = spec.OUT_DIR / "villains_calibration.json"
LATENT_PATH = spec.OUT_DIR / "villains_latent.json"

RANKS = "23456789TJQKA"
SUITS = "cdhs"


def _deck() -> list[str]:
    return [r + s for r in RANKS for s in SUITS]


def _pam_probs(pam: dict, street: str, position: str, spot: str, bucket: str,
               legal: list[str]) -> dict[str, float]:
    probs = pam["action_probs"]
    src = probs.get(f"{street}|{position}|{spot}|{bucket}")
    if src is None:
        src = probs.get(f"{street}|NA|{spot}|b1") or probs.get(
            f"{street}|NA|{spot}|b0")
    if src is None:
        cand = [v for k, v in probs.items() if k.startswith(street + "|")]
        agg = {a: sum(c.get(a, 0.0) for c in cand) for a in ACTIONS}
        total = sum(agg.values()) or 1.0
        src = {a: v / total for a, v in agg.items()}
    out = {a: src.get(a, 0.0) for a in legal}
    total = sum(out.values())
    if total <= 0:
        return {a: 1.0 / len(legal) for a in legal}
    return {a: v / total for a, v in out.items()}


def _sample(probs: dict, rng: random.Random) -> str:
    r = rng.random()
    acc = 0.0
    for a in sorted(probs):
        acc += probs[a]
        if r <= acc:
            return a
    return sorted(probs)[-1]


class PlayerPolicy:
    """PAM × 每特征乘子 × (V3) donk 乘子 —— 合成玩家行为策略。

    multipliers: {"vpip": m, "pfr": m, "three_bet": m, "cbet": m, "fold_to_3bet": m}
    donk_mult: 非翻前加注者 postflop bet 概率乘子（V3 专用，其余 1.0）。
    """

    def __init__(self, pam: dict, multipliers: dict, donk_mult: float = 1.0):
        self.pam = pam
        self.m = dict(multipliers)
        self.donk_mult = donk_mult

    def probs(self, street: str, position: str, spot: str, bucket: str,
              legal: list[str], raise_count: int, preflop_aggressor: str | None,
              self_id: str) -> dict[str, float]:
        base = _pam_probs(self.pam, street, position, spot, bucket, legal)
        adj = dict(base)
        if street == "preflop":
            for a in ("call", "raise"):
                if a in adj:
                    adj[a] *= self.m.get("vpip", 1.0)
            if "raise" in adj:
                adj["raise"] *= self.m.get("pfr", 1.0)
                if raise_count == 1:                     # 其加注 = 第 2 次加注
                    adj["raise"] *= self.m.get("three_bet", 1.0)
            if raise_count >= 2 and "fold" in adj:       # 面对 3bet
                adj["fold"] *= self.m.get("fold_to_3bet", 1.0)
        else:
            if "bet" in adj:
                if preflop_aggressor == self_id:
                    adj["bet"] *= self.m.get("cbet", 1.0)
                else:
                    adj["bet"] *= self.donk_mult
        total = sum(adj.values())
        if total <= 0:
            return {a: 1.0 / len(legal) for a in legal}
        return {a: v / total for a, v in adj.items()}


def simulate_hands(pam: dict, policies: dict[str, PlayerPolicy], n_hands: int,
                   seed: int) -> list[dict]:
    """生成 n 手 6-max 合成手（8B canonical-like dict + 每玩家事件行）。

    BTN 按手轮转；cards 由 seed 确定性牌堆；全部玩家按各自策略行动至终局。
    """
    seats = [f"S{i}" for i in range(1, spec.N_SEATS + 1)]
    hands = []
    for h in range(n_hands):
        # 每手独立 RNG 流（牌堆/行动随机互不跨手污染——V3 正交性等需要
        # postflop 差异不得改变后续手牌的牌堆与行动流）
        deck_rng = random.Random(f"8c-deck-{seed}-{h}")
        act_rng = random.Random(f"8c-act-{seed}-{h}")
        deck = _deck()
        deck_rng.shuffle(deck)
        btn = seats[h % spec.N_SEATS]
        orbit = seats                               # 固定顺时针序
        positions = {}
        for d, s in enumerate(seats):
            dist = (d - seats.index(btn)) % spec.N_SEATS
            if dist == 0:
                positions[s] = "BTN"
            elif dist == 1:
                positions[s] = "SB"
            elif dist == 2:
                positions[s] = "BB"
            elif dist == spec.N_SEATS - 1:
                positions[s] = "LP"
            else:
                positions[s] = "EP"
        stacks = {s: spec.STACK for s in seats}
        total_committed = {s: 0.0 for s in seats}
        hole = {s: [deck.pop(), deck.pop()] for s in seats}
        board = []
        actions = {"preflop": [], "flop": [], "turn": [], "river": []}
        seq = 0
        folded: set = set()
        acted: set = set()
        street_committed: dict[str, float] = {}
        current_bet = 0.0
        pot = 0.0
        street = "preflop"
        last_aggressor: str | None = None
        preflop_aggressor: str | None = None
        raise_count = 0
        # 盲注
        sb_seat = seats[(seats.index(btn) + 1) % spec.N_SEATS]
        bb_seat = seats[(seats.index(btn) + 2) % spec.N_SEATS]
        for s, amt in ((sb_seat, spec.SB), (bb_seat, spec.BB)):
            pot += amt
            street_committed[s] = street_committed.get(s, 0.0) + amt
            total_committed[s] += amt
            actions["preflop"].append({"street": "preflop", "seq": seq,
                                       "player_id": s, "action": "post",
                                       "amount": amt})
            seq += 1
        current_bet = spec.BB
        # 行动循环
        board_seq: list[list[str]] = []
        pointer = (seats.index(btn) + 3) % spec.N_SEATS   # UTG 起
        for _step in range(400):
            active = [s for s in seats if s not in folded]
            if len(active) <= 1:
                break
            actable = [s for s in active
                       if stacks[s] - total_committed[s] > 1e-9]
            if len(actable) <= 1:
                while len(board) < 5:
                    board.append(deck.pop())
                break
            # 街完成检测
            done = True
            for s in active:
                rem = stacks[s] - total_committed[s]
                if street_committed.get(s, 0.0) + 1e-9 < current_bet and rem > 1e-9:
                    done = False
                if s not in acted and rem > 1e-9:
                    done = False
            if done:
                if street == "river":
                    break
                if street == "preflop":
                    if raise_count:
                        preflop_aggressor = last_aggressor
                    street = "flop"
                    for _ in range(3):
                        board.append(deck.pop())
                elif street == "flop":
                    street = "turn"
                    board.append(deck.pop())
                else:
                    street = "river"
                    board.append(deck.pop())
                street_committed = {}
                current_bet = 0.0
                acted = set()
                pointer = (seats.index(bb_seat) + 1) % spec.N_SEATS
                continue
            s = seats[pointer % spec.N_SEATS]
            if s in folded:
                pointer += 1
                continue
            rem = stacks[s] - total_committed[s]
            if rem <= 1e-9:
                acted.add(s)
                pointer += 1
                continue
            facing = current_bet - street_committed.get(s, 0.0)
            if street == "preflop" and raise_count == 0:
                spot = "unopened"
            elif facing > 1e-9:
                spot = "facing_bet"
            else:
                spot = "checked_to"
            bucket = "b0"
            if spot == "facing_bet":
                r = facing / max(pot, 1e-9)
                bucket = _bet_bucket(r)
            legal = (["fold", "call"] + (["raise"] if rem > facing else [])
                     if facing > 1e-9 else ["check", "bet"])
            pol = policies[s]
            probs = pol.probs(street, positions[s], spot, bucket, legal,
                              raise_count, preflop_aggressor, s)
            action = _sample(probs, act_rng)
            acted.add(s)
            if action == "fold":
                folded.add(s)
            elif action == "check":
                pass
            elif action == "call":
                amt = min(facing, rem)
                pot += amt
                street_committed[s] = street_committed.get(s, 0.0) + amt
                total_committed[s] += amt
                if street == "preflop":
                    raise_count = max(raise_count, 1)  # limper 不计加注数
            else:  # bet / raise
                size = 0.5 * pot
                if street == "preflop":
                    target = street_committed.get(s, 0.0) + max(size, spec.BB)
                else:
                    target = street_committed.get(s, 0.0) + max(size, spec.BB)
                target = min(target, street_committed.get(s, 0.0) + rem)
                contrib = max(target - street_committed.get(s, 0.0), 0.0)
                pot += contrib
                street_committed[s] = street_committed.get(s, 0.0) + contrib
                total_committed[s] += contrib
                current_bet = max(current_bet, street_committed[s])
                last_aggressor = s
                if street == "preflop":
                    raise_count += 1
            actions[street].append({"street": street, "seq": seq,
                                    "player_id": s, "action": action,
                                    "amount": round(
                                        street_committed.get(s, 0.0), 4),
                                    "to": round(street_committed.get(s, 0.0), 4)
                                    if action in ("bet", "raise") else None})
            seq += 1
            pointer += 1
        hands.append({
            "hand_key": f"8c-{seed}-{h}", "table": "8c", "hand_id": f"8c-{seed}-{h}",
            "players": [{"pid": s, "stack_start": spec.STACK, "stack_end": None,
                         "hole_cards": hole[s]} for s in seats],
            "sb": spec.SB, "bb": spec.BB,
            "positions": positions, "orbit": orbit,
            "actions": actions,
            "flop": board[:3] if len(board) >= 3 else [],
            "turn": board[3:4] if len(board) >= 4 else [],
            "river": board[4:5] if len(board) >= 5 else [],
            "all_in": [],
            "btn": btn, "ts": f"2020-01-01T00:{h // 60:02d}:{h % 60:02d}",
        })
    return hands


def feature_rows(hands: list[dict], target: str) -> list[dict]:
    """每手目标玩家的决策特征事件行（7A feature_suff 所需字段 + ts）。

    口径与冻结 derive_player_events 一致：vpip/pfr/three_bet 全手分母；
    faced_3bet = 轮到目标玩家时恰好面对第 2 次加注（>0）；folded_to_3bet 同机会弃牌；
    cbet_opportunity = 目标为翻前最后加注者且翻牌到达。
    """
    rows = []
    for hand in hands:
        pre = hand["actions"]["preflop"]
        committed: dict[str, float] = {}
        current_bet = 0.0
        raise_count = 0
        last_raiser: str | None = None
        vpip = pfr = tb = False
        faced3 = fold3 = False
        for a in pre:
            who, kind = a["player_id"], a["action"]
            if kind == "post":
                committed[who] = committed.get(who, 0.0) + float(a.get("amount") or 0)
                current_bet = max(current_bet, spec.BB)
                continue
            facing = current_bet - committed.get(target, 0.0)
            if kind == "fold":
                if who == target and raise_count == 2 and facing > 1e-9:
                    faced3 = True
                    fold3 = True
            elif kind == "check":
                pass
            elif kind == "call":
                if who == target:
                    vpip = True
                    if raise_count == 2 and facing > 1e-9:
                        faced3 = True
                    committed[target] = current_bet
                else:
                    committed[who] = current_bet
            else:  # raise
                raise_count += 1
                if who == target:
                    vpip = True
                    pfr = True
                    if raise_count == 2:
                        tb = True
                    if raise_count >= 3 and facing > 1e-9:
                        faced3 = True           # 面对 3bet 后 4bet
                    committed[target] = float(a.get("to") or 0.0)
                    current_bet = max(current_bet, committed[target])
                    last_raiser = target
                else:
                    committed[who] = float(a.get("to") or 0.0)
                    current_bet = max(current_bet, committed[who])
                    last_raiser = who
        flop_bet = any(a["player_id"] == target and a["action"] == "bet"
                       for a in hand["actions"]["flop"])
        cbet_opp = (last_raiser == target) and bool(hand["actions"]["flop"])
        rows.append({
            "player_id": target, "ts": hand["ts"], "table_type": "6max",
            "vpip": vpip, "pfr": pfr, "three_bet": tb,
            "faced_3bet": faced3, "folded_to_3bet": fold3,
            "cbet_opportunity": cbet_opp, "cbet": cbet_opp and flop_bet,
        })
    return rows


def measure_rates(hands: list[dict], target: str) -> dict:
    rows = feature_rows(hands, target)
    out = {}
    n = len(rows)
    out["vpip"] = sum(r["vpip"] for r in rows) / n
    out["pfr"] = sum(r["pfr"] for r in rows) / n
    out["three_bet"] = sum(r["three_bet"] for r in rows) / n
    n_f = sum(r["faced_3bet"] for r in rows)
    out["fold_to_3bet"] = (sum(r["folded_to_3bet"] for r in rows) / n_f) if n_f else None
    n_c = sum(r["cbet_opportunity"] for r in rows)
    out["cbet"] = (sum(r["cbet"] for r in rows) / n_c) if n_c else None
    return out


def calibrate(pam: dict, targets: dict, seed: int = spec.SEED) -> dict:
    """V0 式 μ 校准：确定性乘子搜索（一次、预实验、结果无关）。

    targets = {feature: rate}；返回 {multipliers, realized, iterations, seed}。
    只读取 PAM 与自身模拟；不读取任何 outcome/accuracy/ΔEV/verdict。
    """
    mult = {"vpip": 1.0, "pfr": 1.0, "three_bet": 1.0, "cbet": 1.0,
            "fold_to_3bet": 1.0}
    realized = {}
    infeasible = [f for f, t in targets.items() if t < 0.0 or t > 1.0]
    used_iters = 0
    for it in range(spec.CAL_MAX_ITERS):
        used_iters = it + 1
        pol = PlayerPolicy(pam, mult)
        policies = {f"S{i}": pol for i in range(1, spec.N_SEATS + 1)}
        hands = simulate_hands(pam, policies, spec.CAL_SIM_HANDS, seed=seed + it)
        realized = measure_rates(hands, spec.VILLAIN_SEAT)
        converged = True
        for f, target in targets.items():
            if f in infeasible:
                continue                 # target 不可行（<0 或 >1）：不更新，标记
            r = realized.get(f)
            if r is None or r <= 0:
                continue
            if abs(r - target) > spec.CAL_TOL:
                # 阻尼乘法更新（√ 步长）：耦合特征下防振荡（预注册算法）
                mult[f] = max(0.1, min(10.0, mult[f] * (target / r) ** 0.5))
                converged = False
        if converged:
            break
    return {"multipliers": {k: round(v, 6) for k, v in mult.items()},
            "realized": {k: (round(v, 6) if v is not None else None)
                         for k, v in realized.items()},
            "iterations": used_iters,
            "infeasible_features": infeasible,
            "seed": seed, "targets": targets}
