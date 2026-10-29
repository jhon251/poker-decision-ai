"""Phase 8B Step 4：Replay / exploit mechanism validation（simulated EV，feasibility）。

严格边界（8B Step 0 + 用户确认要求）：
  - 只验证 exploit arm 在 frozen response model 下的 simulated advantage；**不是**
    真实 EV / 真实盈利 / ROI / P2 proven
  - PAM 仅 train split 拟合（8B Step 2）；本模块只读 pam.json + eval 手
  - 两臂：同 hand / 同 hero / 同 RNG 种子流 / 同环境 / 同 response model；
    唯一差异 = exploit signal 注入（合成 rec，预注册）
  - 主臂 (k=1.0, ε=1.0)；全臂报告；禁止 replay 后选最优臂
  - primary = paired ΔEV（bb/100），hand-level paired bootstrap（n=1000, seed=42）
    + table-block sensitivity；统计显著（CI 下界>0）与实践显著（≥1 bb/100，
    预注册 convention）分开报告
  - 所有 EV 输出标注 "simulated EV under frozen response model"
  - 8B 不产生 P2 verdict；任何 BLOCK 触发 → 停止并写 DEGRADE_LOG

输出 data/uoft/models/phase8/phase8b/experiment.json
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).parent
_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_SCRIPT_DIR))

from poker_decision_ai.domain.poker.cards import parse_cards  # noqa: E402
from poker_decision_ai.domain.poker.evaluator import evaluate7  # noqa: E402
from phase5b_math import bootstrap_pct  # noqa: E402
from phase8a_replay import FROZEN_EPS, FROZEN_K, PRIMARY_ARM, BaselineArm, \
    ExploitArm  # noqa: E402
from phase8b_arms import ACTIONS, PAMPolicy, SIZE_CENTERS, \
    SyntheticRecommendationProvider  # noqa: E402

OUT_DIR = _ROOT / "data/uoft/models/phase8/phase8b"
N_ROLLOUTS = 50
BOOT_N, BOOT_SEED = 1000, 42
PRACTICAL_BB100 = 1.0        # convention（预注册；非统计判据）
EV_LABEL = "simulated EV under frozen response model"


# ---------------------------------------------------------------- 状态重建与 DP
def walk_dps(hand: dict, hero: str) -> list[dict]:
    """canonical 行动流 → hero 决策点状态列表（8B 预注册判定：轮到 hero 行动）。

    每个 token **先应用**再判定下一行动者；facing = 应用后的 current_bet − hero 已投入。
    只读 actions/positions/sb/bb/stack_start（公开字段）。
    """
    orbit = hand["orbit"]
    committed: dict[str, float] = {}
    total_committed: dict[str, float] = {}
    folded: set = set()
    acted: set = set()
    pot = 0.0
    voluntary = False
    history: list[dict] = []
    board: list[str] = []
    street = "preflop"
    current_bet = 0.0
    last_aggressor: str | None = None
    stacks = {p["pid"]: float(p["stack_start"] or 0.0) for p in hand["players"]}
    dps: list[dict] = []

    def snapshot(token_idx: int, acting: str) -> dict:
        return {
            "street": street, "board": list(board), "pot": pot,
            "committed": dict(committed), "total_committed": dict(total_committed),
            "folded": set(folded), "acted": set(acted),
            "current_bet": current_bet, "last_aggressor": last_aggressor,
            "history": [dict(h) for h in history], "token_idx": token_idx,
            "acting": acting, "voluntary": voluntary,
        }

    token_idx = 0
    for st in ("preflop", "flop", "turn", "river"):
        street = st
        if st == "flop":
            board = list(hand["flop"])
        elif st == "turn":
            board = list(hand["flop"]) + list(hand["turn"])
        elif st == "river":
            board = list(hand["flop"]) + list(hand["turn"]) + list(hand["river"])
        committed = {}
        current_bet = 0.0
        acted = set()
        for a in hand["actions"].get(st, []):
            pid = a["player_id"]
            kind = a.get("action", "")
            if kind == "post":
                amt = float(a.get("amount") or 0.0)
                pot += amt
                committed[pid] = committed.get(pid, 0.0) + amt
                total_committed[pid] = total_committed.get(pid, 0.0) + amt
                if st == "preflop":
                    current_bet = max(current_bet, float(hand.get("bb") or 0.0))
                token_idx += 1
                continue
            if kind not in ACTIONS:
                continue
            # ---- 先应用当前 token ----
            if kind == "fold":
                folded.add(pid)
                acted.add(pid)
            elif kind == "check":
                acted.add(pid)
            elif kind == "call":
                amt = float(a.get("amount") or 0.0)
                pot += amt
                committed[pid] = committed.get(pid, 0.0) + amt
                total_committed[pid] = total_committed.get(pid, 0.0) + amt
                acted.add(pid)
                if st == "preflop":
                    voluntary = True
            else:  # bet / raise
                to = float(a.get("to") or 0.0)
                amt = float(a.get("amount") or 0.0)
                prior = committed.get(pid, 0.0)
                contrib = (to - prior) if to > 0 else amt
                contrib = max(contrib, 0.0)
                pot += contrib
                committed[pid] = prior + contrib
                total_committed[pid] = total_committed.get(pid, 0.0) + contrib
                current_bet = max(current_bet, committed[pid])
                last_aggressor = pid
                acted.add(pid)
                if st == "preflop":
                    voluntary = True
            history.append({"street": st, "action": kind, "player_id": pid,
                            "amount": float(a.get("amount") or 0.0),
                            "to": float(a.get("to") or 0.0)})
            token_idx += 1
            # ---- DP 判定：应用后轮到 hero ----
            if pid == hero:
                continue
            n = len(orbit)
            base = orbit.index(pid)
            nxt = None
            for k in range(1, n + 1):
                cand = orbit[(base + k) % n]
                if cand not in folded:
                    nxt = cand
                    break
            if nxt != hero:
                continue
            if sum(1 for p in orbit if p not in folded) <= 1:
                continue                          # 只剩一人：手已结束，无决策
            hero_remaining = stacks.get(hero, 0.0) - total_committed.get(hero, 0.0)
            if hero_remaining <= 1e-9:
                continue                          # hero 已 all-in，无决策
            facing = current_bet - committed.get(hero, 0.0)
            if facing > 1e-9 or hero not in acted:
                dps.append(snapshot(token_idx, hero))
    return dps


# ---------------------------------------------------------------- rollout
def _villain_probs(pam: dict, street: str, position: str, spot: str,
                   bucket: str, legal: list[str]) -> dict[str, float]:
    probs = pam["action_probs"]
    key = f"{street}|{position}|{spot}|{bucket}"
    src = probs.get(key)
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


def _sample(probs: dict[str, float], rng: random.Random) -> str:
    r = rng.random()
    acc = 0.0
    for a in sorted(probs):
        acc += probs[a]
        if r <= acc:
            return a
    return sorted(probs)[-1]


def _sample_size(pam: dict, street: str, spot: str, action: str,
                 pot: float, rng: random.Random) -> float:
    probs = pam.get("size_probs", {}).get(f"{street}|{spot}|{action}")
    if not probs:
        return 0.5 * pot
    names = sorted(probs)
    r = rng.random()
    acc = 0.0
    for b in names:
        acc += probs[b]
        if r <= acc:
            return SIZE_CENTERS.get(b, 0.5) * pot
    return SIZE_CENTERS.get(names[-1], 0.5) * pot


def _obs_for(st: dict, hand: dict, hero: str) -> dict:
    facing = st["current_bet"] - st["committed"].get(hero, 0.0)
    # hero 剩余筹码 = 开局筹码 − 累计投入（公开信息）
    hero_stack_start = next(p["stack_start"] for p in hand["players"]
                            if p["pid"] == hero)
    hero_remaining = hero_stack_start - st["total_committed"].get(hero, 0.0)
    if facing > 1e-9:
        legal = ["fold", "call"] + (["raise"] if hero_remaining > facing else [])
    else:
        legal = ["check", "bet"] if hero_remaining > 0 else ["check"]
    return {
        "hand_id": hand["hand_id"], "game_type": "cash",
        "street": st["street"], "board": list(st["board"]),
        "hero_hole_cards": next(p["hole_cards"] for p in hand["players"]
                                if p["pid"] == hero),
        "pot": round(st["pot"], 4),
        "current_bet": round(st["current_bet"], 4),
        "to_call": round(max(facing, 0.0), 4),
        "min_raise_to": round(max(st["current_bet"] * 2, hand.get("bb") or 0.0), 4),
        "blinds": {"sb": hand.get("sb") or 0.0, "bb": hand.get("bb") or 0.0,
                   "ante": 0.0},
        "hero_stack": round(hero_remaining, 4),
        "hero_position": hand["positions"].get(hero),
        "action_history": [dict(h) for h in st["history"]],
        "available_actions": legal,
    }


def _deck() -> list[str]:
    ranks = "23456789TJQKA"
    suits = "cdhs"
    return [r + s for r in ranks for s in suits]


def _board_continuation(hand: dict, board_seed: str) -> list[list[str]]:
    """5 张公共牌 = 真实已发牌 + 合成延续（仅当反事实路径超过真实发牌时）。

    规则（8B 预注册，环境声明）：残余牌堆（52 − 已知手牌 − 真实公共牌）以
    board_seed 确定性洗牌，补足到 5 张；两臂共享同一 board_seed → 公共牌完全一致
    （环境齐性）；已标注 "synthesized board continuation（环境的一部分）"。
    """
    real = list(hand["flop"]) + list(hand["turn"]) + list(hand["river"])
    if len(real) >= 5:
        return [real[:3], real[:4], real[:5]]
    known = set(real)
    for p in hand["players"]:
        known.update(p.get("hole_cards") or [])
    residual = [c for c in _deck() if c not in known]
    rng = random.Random(board_seed)
    rng.shuffle(residual)
    full = real + residual[:5 - len(real)]
    return [full[:3], full[:4], full[:5]]


def rollout(hand: dict, hero: str, dp: dict, hero_decide, villain_decide,
            pam: dict, rng: random.Random, board_seed: str = "8b-board") -> float:
    """从 DP 状态自对弈到终局；返回 hero 相对该 DP 的净筹码（pot 收益 − DP 后投入）。"""
    st = {
        "street": dp["street"], "board": list(dp["board"]), "pot": dp["pot"],
        "committed": dict(dp["committed"]),
        "total_committed": dict(dp["total_committed"]),
        "folded": set(dp["folded"]), "acted": set(dp["acted"]),
        "current_bet": dp["current_bet"],
        "last_aggressor": dp["last_aggressor"],
        "history": [dict(h) for h in dp["history"]],
        "voluntary": dp["voluntary"],
    }
    orbit = hand["orbit"]
    bb = float(hand.get("bb") or 0.0)
    stacks = {p["pid"]: float(p["stack_start"] or 0.0) for p in hand["players"]}
    hero_contrib_post = 0.0
    board_ptr = 0
    board_seq = _board_continuation(hand, board_seed)

    def active():
        return [p for p in orbit if p not in st["folded"]]

    def street_done():
        for p in active():
            if st["committed"].get(p, 0.0) + 1e-9 < st["current_bet"]:
                if stacks[p] - st["total_committed"].get(p, 0.0) > 1e-9:
                    return False          # 非 all-in 且未匹配
            if p not in st["acted"] and stacks[p] - st["total_committed"].get(
                    p, 0.0) > 1e-9:
                return False
        return True

    actors = active()
    if not actors:
        return 0.0
    pointer = 0
    if dp.get("acting") in actors:
        pointer = actors.index(dp["acting"])

    for _ in range(400):                    # 安全上限（预注册；正常 < 40 步）
        actors = active()
        if len(actors) <= 1:
            winner = actors[0] if actors else None
            won = (winner == hero)
            return (st["pot"] if won else 0.0) - hero_contrib_post
        # 只剩 ≤1 名可行动者（其余 all-in）→ 无更多下注，直接推进到摊牌
        actable = [p for p in actors
                   if stacks[p] - st["total_committed"].get(p, 0.0) > 1e-9]
        if len(actable) <= 1:
            while st["street"] != "river" and board_ptr < 3:
                st["street"] = ("flop", "turn", "river")[board_ptr]
                st["board"] = list(board_seq[board_ptr])
                board_ptr += 1
            st["street"] = "river"
            break
        if street_done():
            if st["street"] == "river":
                break                       # 摊牌
            # 下一街
            if board_ptr >= 3:
                break                       # 数据异常：强制摊牌
            st["street"] = ("flop", "turn", "river")[board_ptr]
            st["board"] = list(board_seq[board_ptr])
            board_ptr += 1
            st["committed"] = {}
            st["current_bet"] = 0.0
            st["acted"] = set()
            # 新街首行动者 = BB 后第一个 active（positions 中 label==BB）
            bbs = [p for p in orbit if hand["positions"].get(p) == "BB"]
            start = (orbit.index(bbs[0]) + 1) % len(orbit) if bbs else 0
            while orbit[start] not in actors:
                start = (start + 1) % len(orbit)
            pointer = actors.index(orbit[start])
            continue
        pid = actors[pointer % len(actors)]
        facing = st["current_bet"] - st["committed"].get(pid, 0.0)
        remaining = stacks[pid] - st["total_committed"].get(pid, 0.0)
        if remaining <= 1e-9:               # all-in：不能再行动
            st["acted"].add(pid)
            pointer += 1
            continue
        spot = "facing_bet" if facing > 1e-9 else "checked_to"
        if st["street"] == "preflop" and not st["voluntary"]:
            spot = "unopened"
        bucket = "b0"
        if spot == "facing_bet":
            r = facing / max(st["pot"], 1e-9)
            bucket = "b1" if r <= 0.5 else ("b2" if r <= 1.5 else "b3")
        if pid == hero:
            obs = _obs_for(st, hand, hero)
            probs = hero_decide(obs)          # 臂策略（含 exploit 乘子）
            action = _sample(probs, rng)
        else:
            legal = (["fold", "call"] + (["raise"] if remaining > facing else [])
                     if facing > 1e-9
                     else ["check", "bet"])
            probs = villain_decide(pam, st["street"], hand["positions"].get(pid),
                                   spot, bucket, legal)
            action = _sample(probs, rng)
        # ---- 应用行动 ----
        st["acted"].add(pid)
        if action == "fold":
            st["folded"].add(pid)
        elif action == "check":
            pass
        elif action == "call":
            amt = min(facing, remaining)
            st["pot"] += amt
            st["committed"][pid] = st["committed"].get(pid, 0.0) + amt
            st["total_committed"][pid] = st["total_committed"].get(pid, 0.0) + amt
            if pid == hero:
                hero_contrib_post += amt
            if st["street"] == "preflop":
                st["voluntary"] = True
        elif action in ("bet", "raise"):
            size = _sample_size(pam, st["street"], spot, action, st["pot"], rng)
            if action == "raise":
                target = st["committed"].get(pid, 0.0) + max(
                    size, max(st["current_bet"], bb))
            else:
                target = max(size, bb)
            target = min(target, st["committed"].get(pid, 0.0) + remaining)
            contrib = target - st["committed"].get(pid, 0.0)
            contrib = max(contrib, 0.0)
            st["pot"] += contrib
            st["committed"][pid] = st["committed"].get(pid, 0.0) + contrib
            st["total_committed"][pid] = st["total_committed"].get(pid, 0.0) + contrib
            st["current_bet"] = max(st["current_bet"], st["committed"][pid])
            st["last_aggressor"] = pid
            if pid == hero:
                hero_contrib_post += contrib
            if st["street"] == "preflop":
                st["voluntary"] = True
        st["history"].append({"street": st["street"], "action": action,
                              "player_id": pid})
        pointer += 1

    # 摊牌：真实手牌 + 冻结 evaluator（admin 上下文，绝不进策略/模型）
    actors = active()
    if len(actors) >= 2:
        best, winners = -1, []
        for p in actors:
            cards = next(pl["hole_cards"] for pl in hand["players"]
                         if pl["pid"] == p)
            score = evaluate7(parse_cards(cards + st["board"]))
            if score > best:
                best, winners = score, [p]
            elif score == best:
                winners.append(p)
        won = hero in winners
        share = st["pot"] / len(winners) if winners else 0.0
        return (share if won else 0.0) - hero_contrib_post
    winner = actors[0] if actors else st["last_aggressor"]
    won = winner == hero
    return (st["pot"] if won else 0.0) - hero_contrib_post


# ---------------------------------------------------------------- 主实验
def main() -> int:
    splits = json.load(open(OUT_DIR / "splits.json", encoding="utf-8"))
    pam = json.load(open(OUT_DIR / "pam.json", encoding="utf-8"))
    eval_set = set(splits["sets"]["eval"])
    hands = {}
    for line in open(OUT_DIR / "hands_replayable.jsonl", encoding="utf-8"):
        h = json.loads(line)
        if h["hand_key"] in eval_set:
            hands[h["hand_key"]] = h
    assert len(hands) == splits["n"]["eval"]
    # ---- hero = 固定玩家 "Pluribus"（预注册主座位）----
    # 说明：8B Step 0 写 "players[0] 即 Pluribus 座位"；数据探针证实 players 列表
    # 顺序按桌旋转（120 个元组均为同一 6 人组的不同旋转），players[0] 并非固定玩家。
    # 此处实现审计原意：hero = 固定玩家 Pluribus（数据中每手均在场，硬断言）。
    hero = "Pluribus"
    for h in hands.values():
        assert any(p["pid"] == hero for p in h["players"]), \
            f"hero={hero} 不在 hand {h['hand_key']} 中（BLOCK：hero 座位假设失效）"

    arms_grid = [(k, e) for k in FROZEN_K for e in FROZEN_EPS]
    results = {f"k{k}_eps{e}": {"per_hand_delta_bb": [], "n_dp": 0}
               for k, e in arms_grid}
    divergence = {f"k{k}_eps{e}": 0 for k, e in arms_grid}
    total_dp = 0
    n_excluded = 0

    for hk, hand in hands.items():
        dps = walk_dps(hand, hero)
        hand_deltas = {f"k{k}_eps{e}": 0.0 for k, e in arms_grid}
        for dp in dps:
            total_dp += 1
            obs0 = _obs_for(dp, hand, hero)
            # 两臂首行动分歧（同 seed 流）
            for k, e in arms_grid:
                seed = f"8b-{hk}-{total_dp}-0"
                base = BaselineArm(random.Random(seed), policy=PAMPolicy(pam))
                expl = ExploitArm(random.Random(seed), k=k, eps=e,
                                  recommendation_provider=SyntheticRecommendationProvider(hk),
                                  policy=PAMPolicy(pam))
                if base.decide(obs0) != expl.decide(obs0):
                    divergence[f"k{k}_eps{e}"] += 1
            # 两臂 rollout EV（同种子流：base/expl 同一 seed → 公共随机数；
            # 分歧仅来自 exploit 乘子改变行动后下游流自然分叉）
            base_list = []
            for r in range(N_ROLLOUTS):
                seed = f"8b-{hk}-{total_dp}-{r}"
                rng_seed = seed + "-rng"
                base = BaselineArm(random.Random(seed), policy=PAMPolicy(pam))
                base_list.append(rollout(
                    hand, hero, dp, base.action_probs,
                    lambda pam_, s, p, sp, b, legal: _villain_probs(
                        pam_, s, p, sp, b, legal),
                    pam, random.Random(rng_seed),
                    board_seed=f"8b-board-{hk}-{total_dp}-{r}"))
            base_mean = sum(base_list) / N_ROLLOUTS
            for k, e in arms_grid:
                key = f"k{k}_eps{e}"
                expl_list = []
                for r in range(N_ROLLOUTS):
                    seed = f"8b-{hk}-{total_dp}-{r}"
                    rng_seed = seed + "-rng"
                    expl = ExploitArm(random.Random(seed), k=k, eps=e,
                                      recommendation_provider=SyntheticRecommendationProvider(hk),
                                      policy=PAMPolicy(pam))
                    expl_list.append(rollout(
                        hand, hero, dp, expl.action_probs,
                        lambda pam_, s, p, sp, b, legal: _villain_probs(
                            pam_, s, p, sp, b, legal),
                        pam, random.Random(rng_seed),
                        board_seed=f"8b-board-{hk}-{total_dp}-{r}"))
                delta = sum(expl_list) / N_ROLLOUTS - base_mean
                results[key]["n_dp"] += 1
                hand_deltas[key] += delta
        for k, e in arms_grid:
            key = f"k{k}_eps{e}"
            results[key]["per_hand_delta_bb"].append(
                round(hand_deltas[key] / float(hand.get("bb") or 100.0), 6))

    # ---- 聚合 + hand-level paired bootstrap + table-block sensitivity ----
    out_arms = {}
    hands_order = list(hands.keys())
    for k, e in arms_grid:
        key = f"k{k}_eps{e}"
        deltas = results[key]["per_hand_delta_bb"]
        assert len(deltas) == len(hands_order)
        mean_bb_hand = sum(deltas) / len(deltas) if deltas else 0.0
        lo, hi, _ = bootstrap_pct(lambda xs: sum(xs) / len(xs), deltas,
                                  n_resample=BOOT_N, seed=BOOT_SEED)
        # table-block sensitivity：先按表聚合，再重采样表（92 表）
        idx = {hk: i for i, hk in enumerate(hands_order)}
        by_table: dict[str, list] = {}
        for hk in hands_order:
            by_table.setdefault(hands[hk]["table"], []).append(deltas[idx[hk]])
        tm = [sum(v) / len(v) for v in by_table.values()]
        tlo, thi, _ = bootstrap_pct(lambda xs: sum(xs) / len(xs), tm,
                                    n_resample=BOOT_N, seed=BOOT_SEED)
        out_arms[key] = {
            "k": k, "eps": e, "primary": (k, e) == PRIMARY_ARM,
            "n_hands": len(deltas), "n_dp": results[key]["n_dp"],
            "delta_bb_hand": round(mean_bb_hand, 6),
            "delta_bb_100": round(mean_bb_hand * 100, 6),
            "ci95_hand_level": [round(lo, 6), round(hi, 6)],
            "ci95_bb100_hand_level": [round(lo * 100, 6), round(hi * 100, 6)],
            "ci95_table_block": [round(tlo, 6), round(thi, 6)],
            "statistical_significance": "pass (CI lower > 0)"
            if lo > 0 else "not significant",
            "practical_significance": (
                f"convention: |Δ| ≥ {PRACTICAL_BB100} bb/100（独立报告，"
                f"非统计判据）——本臂 Δbb100 = {round(mean_bb_hand * 100, 6)}"),
            "divergent_dp": divergence[key],
            "ev_label": EV_LABEL,
        }

    observed = []
    for hk, hand in hands.items():
        hs = hand["players"][0]
        if hs.get("stack_start") is not None and hs.get("stack_end") is not None:
            observed.append((hs["stack_end"] - hs["stack_start"])
                            / float(hand.get("bb") or 100.0))
    exp = {
        "phase": "8B",
        "status": ("replay/exploit mechanism validation + simulated EV "
                   "feasibility；NOT real EV / NOT P2 verdict"),
        "config": {
            "hero": hero,
            "hero_note": "固定玩家 Pluribus（预注册主座位；players 列表顺序按桌旋转，"
                        "故以固定玩家身份实现 8B Step 0 原意——已记录）",
            "n_rollouts": N_ROLLOUTS,
            "bootstrap": {"unit": "hand", "paired": True, "n": BOOT_N,
                          "seed": BOOT_SEED},
            "sensitivity": "table-block bootstrap（92 表）",
            "practical_bb100": PRACTICAL_BB100,
            "practical_note": "convention（预注册），统计显著与实践显著分开报告",
            "arms_grid": [[k, e] for k, e in arms_grid],
            "primary_arm": list(PRIMARY_ARM),
            "no_arm_selection": "禁止 replay 后选最优臂；主臂固定预注册",
            "split": "60/20/20 hand-level seed=42 per-table stratified（8B Step1）",
            "pam": "train-only fit（8B Step2）；同时充当 baseline 与 response model",
            "recommendation_provider": "synthetic（预注册参数；真实 7C 链禁止接入"
                                       "Pluribus replay——无时间顺序）",
            "board_continuation": ("真实公共牌优先；反事实路径超出真实发牌时，残余"
                                   "牌堆以 board_seed 确定性洗牌补足（两臂同 seed "
                                   "→ 环境齐性）；标注 synthesized board continuation"),
            "prohibited": ["exploit_utility_proven", "EV proven", "ROI proven",
                           "profitable", "real-world profitability",
                           "post-hoc arm selection", "post-hoc threshold tuning"],
        },
        "arms": out_arms,
        "observed_sanity": {
            "label": "observed real-path hero net（描述性，≠ 真实 EV）",
            "n_hands": len(observed),
            "mean_bb_hand": round(sum(observed) / len(observed), 6)
            if observed else None,
        },
        "divergence_note": ("两臂在同一 DP 同种子流下首行动不同的决策点数；"
                            "signal 关闭/中性时必为 0（测试断言）"),
        "n_eval_hands": len(hands), "n_total_dp": total_dp,
        "ev_label": EV_LABEL,
        "verdict": None,     # 8B 不产生 P2 verdict
    }
    (OUT_DIR / "experiment.json").write_text(
        json.dumps(exp, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: {kk: v[kk] for kk in ("delta_bb_100", "ci95_bb100_hand_level",
                                               "ci95_table_block", "divergent_dp")}
                      for k, v in out_arms.items()}, ensure_ascii=False, indent=2))
    print(f"eval_hands={len(hands)} total_dp={total_dp} observed_mean_bb_hand="
          f"{round(sum(observed)/len(observed),6) if observed else None}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
