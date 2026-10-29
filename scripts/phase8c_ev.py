"""Phase 8C Step 4：P2b —— 真实 7C 链 × 两臂 simulated ΔEV + adversarial controls。

- 8C 本地 self-play rollout（镜像 8B 冻结语义，唯一差异 = villain 决策按座位策略，
  使目标 villain 保持其预注册偏差——不改 8B 任何代码）
- 两臂：同 hand/board/villain/RNG/信息集；唯一差异 = 真实 7C recommendation
  驱动的 counter-action（冻结 ACTION_MAPPINGS 乘子）
- controls：C1 reverse direction / C3 random direction（C2 在 detection 阶段）
- EV 标注：simulated EV under frozen synthetic response model
- 输出 data/uoft/models/phase8/phase8c/ev.json
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
from phase8a_replay import (  # noqa: E402
    ACTION_MAPPINGS, FROZEN_EPS, FROZEN_K, PRIMARY_ARM, BaselineArm, ExploitArm)
from phase8b_arms import ACTIONS, PAMPolicy  # noqa: E402
from phase8b_experiment import _board_continuation, walk_dps  # noqa: E402
import phase8c_detect as detect  # noqa: E402
import phase8c_sim as sim  # noqa: E402
import phase8c_spec as spec  # noqa: E402

GEN_DIR = spec.OUT_DIR / "generated"


# ---------------------------------------------------------------- 8C rollout
def _sample(probs: dict, rng: random.Random) -> str:
    r = rng.random()
    acc = 0.0
    for a in sorted(probs):
        acc += probs[a]
        if r <= acc:
            return a
    return sorted(probs)[-1]


def rollout8c(hand: dict, hero: str, dp: dict, hero_probs, villain_probs,
              rng: random.Random, board_seed: str) -> float:
    """镜像 8B rollout 语义；villain_probs(pid, street, position, spot, bucket,
    legal) -> probs（按座位策略，保持目标 villain 偏差）。返回 hero 相对 DP 净筹码。"""
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
                    return False
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

    for _ in range(400):
        actors = active()
        if len(actors) <= 1:
            winner = actors[0] if actors else None
            return (st["pot"] if winner == hero else 0.0) - hero_contrib_post
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
                break
            if board_ptr >= 3:
                break
            st["street"] = ("flop", "turn", "river")[board_ptr]
            st["board"] = list(board_seq[board_ptr])
            board_ptr += 1
            st["committed"] = {}
            st["current_bet"] = 0.0
            st["acted"] = set()
            bbs = [p for p in orbit if hand["positions"].get(p) == "BB"]
            start = (orbit.index(bbs[0]) + 1) % len(orbit) if bbs else 0
            while orbit[start] not in actors:
                start = (start + 1) % len(orbit)
            pointer = actors.index(orbit[start])
            continue
        pid = actors[pointer % len(actors)]
        facing = st["current_bet"] - st["committed"].get(pid, 0.0)
        remaining = stacks[pid] - st["total_committed"].get(pid, 0.0)
        if remaining <= 1e-9:
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
            obs = _obs8c(st, hand, hero)
            action = _sample(hero_probs(obs), rng)
        else:
            legal = (["fold", "call"] + (["raise"] if remaining > facing else [])
                     if facing > 1e-9 else ["check", "bet"])
            action = _sample(villain_probs(pid, st["street"],
                                           hand["positions"].get(pid), spot,
                                           bucket, legal), rng)
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
            size = 0.5 * st["pot"]
            if action == "raise":
                target = st["committed"].get(pid, 0.0) + max(
                    size, max(st["current_bet"], bb))
            else:
                target = max(size, bb)
            target = min(target, st["committed"].get(pid, 0.0) + remaining)
            contrib = max(target - st["committed"].get(pid, 0.0), 0.0)
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
        if hero in winners:
            return st["pot"] / len(winners) - hero_contrib_post
        return 0.0 - hero_contrib_post
    winner = actors[0] if actors else st["last_aggressor"]
    return (st["pot"] if winner == hero else 0.0) - hero_contrib_post


def _obs8c(st: dict, hand: dict, hero: str) -> dict:
    facing = st["current_bet"] - st["committed"].get(hero, 0.0)
    hs = next(p["stack_start"] for p in hand["players"] if p["pid"] == hero)
    hero_remaining = hs - st["total_committed"].get(hero, 0.0)
    legal = (["fold", "call"] + (["raise"] if hero_remaining > facing else [])
             if facing > 1e-9 else ["check", "bet"] if hero_remaining > 0
             else ["check"])
    return {
        "hand_id": hand["hand_id"], "game_type": "cash",
        "street": st["street"], "board": list(st["board"]),
        "hero_hole_cards": next(p["hole_cards"] for p in hand["players"]
                                if p["pid"] == hero),
        "pot": round(st["pot"], 4), "current_bet": round(st["current_bet"], 4),
        "to_call": round(max(facing, 0.0), 4),
        "min_raise_to": round(max(st["current_bet"] * 2, hand.get("bb") or 0.0), 4),
        "blinds": {"sb": hand.get("sb") or 0.0, "bb": hand.get("bb") or 0.0,
                   "ante": 0.0},
        "hero_stack": round(hero_remaining, 4),
        "hero_position": hand["positions"].get(hero),
        "action_history": [dict(h) for h in st["history"]],
        "available_actions": legal,
        "players": [{"player_id": p["pid"],
                     "stack": round(stacks_view(st, hand, p["pid"]), 4),
                     "status": "folded" if p["pid"] in st["folded"] else "active",
                     "position": hand["positions"].get(p["pid"]),
                     "is_hero": p["pid"] == hero}
                    for p in hand["players"]],
    }


def stacks_view(st: dict, hand: dict, pid: str) -> float:
    hs = next(p["stack_start"] for p in hand["players"] if p["pid"] == pid)
    return hs - st["total_committed"].get(pid, 0.0)


# ---------------------------------------------------------------- 真实 7C provider
class Real7CProvider:
    """真实 7C 检测记录 → 冻结 ACTION_MAPPINGS 乘子信号（禁止合成 provider 冒充）。"""

    def __init__(self, recs: dict, villain_id: str, sign_mode: str = "normal",
                 seed: int = 0):
        assert sign_mode in ("normal", "reverse", "random")
        self.recs = recs
        self.villain_id = villain_id
        self.sign_mode = sign_mode
        self.rng = random.Random(f"8c-sign-{seed}")

    def _rec(self, feature: str, direction: str) -> dict | None:
        r = self.recs.get(feature)
        if not r or r.get("direction") != direction:
            return None
        return {"eligible": True, "label": "real-7c",
                "mapping": {"action_class": "raise",
                            "sign": ACTION_MAPPINGS[(feature, direction)]["sign"]},
                "confidence": r["confidence"],
                "deviation_magnitude": r["deviation_magnitude"] or 0.0}

    def __call__(self, obs: dict) -> dict | None:
        if self.villain_id not in [p["player_id"] for p in obs.get("players", [])]:
            return None
        active = any(p["player_id"] == self.villain_id
                     and p["status"] != "folded" for p in obs.get("players", []))
        if not active:
            return None
        street = obs.get("street")
        pos = obs.get("hero_position")
        to_call = obs.get("to_call") or 0.0
        history = obs.get("action_history") or []
        rec = None
        if street == "preflop":
            voluntary = any(h.get("action") in ("call", "bet", "raise")
                            for h in history if h.get("street") == "preflop")
            if not voluntary and pos in ("BTN", "LP"):
                # steal spot（无人入池 + hero BTN/LP，与冻结映射/pam 语义一致）
                rec = self._rec("vpip", "under")
            elif voluntary and to_call > 1e-9:
                # villain open：最后翻前加注者必须是目标 villain
                last = None
                for h in history:
                    if h.get("street") == "preflop" and h.get("action") in (
                            "bet", "raise"):
                        last = h.get("player_id")
                if last == self.villain_id:
                    rec = self._rec("pfr", "over") or \
                        self._rec("fold_to_3bet", "over") or \
                        self._rec("fold_to_3bet", "under")
        if rec is None:
            return None
        if self.sign_mode == "reverse":
            rec["mapping"]["sign"] = -rec["mapping"]["sign"]   # C1
        elif self.sign_mode == "random":                        # C3
            rec["mapping"]["sign"] = 1 if self.rng.random() < 0.5 else -1
        return rec


# ---------------------------------------------------------------- 主实验
def main() -> int:
    cal = json.load(open(spec.OUT_DIR / "villains_calibration.json",
                         encoding="utf-8"))
    latent = json.load(open(spec.OUT_DIR / "villains_latent.json",
                            encoding="utf-8"))["villains"]
    pam = json.load(open(f"{_ROOT}/data/uoft/models/phase8/phase8b/pam.json",
                         encoding="utf-8"))
    v0_policy = sim.PlayerPolicy(pam, cal["cells"]["v0"]["multipliers"], 1.0)
    hero = spec.HERO_SEAT
    villain = spec.VILLAIN_SEAT

    arms_grid = [(k, e) for k in FROZEN_K for e in FROZEN_EPS]
    cells = {}
    per_villain: dict[str, dict] = {}
    n_villains_processed = 0

    for vid in sorted(latent):
        info = latent[vid]
        payload = json.load(open(GEN_DIR / f"{vid.replace('|', '_')}.json",
                                 encoding="utf-8"))
        recs = {}
        for r in detect.detect_villain(payload, cal):
            if r["horizon"] == 200 and r["eligible"]:
                recs[r["feature"]] = r
        # 目标 villain 策略（保持偏差）
        if info["cell"] == "v3":
            vpol = sim.PlayerPolicy(pam, cal["cells"]["v0"]["multipliers"],
                                    spec.V3_DONK_MULT)
        else:
            vpol = sim.PlayerPolicy(pam, cal["cells"][info["cell"]]["multipliers"],
                                    1.0)

        def villain_probs(pid, street, position, spot, bucket, legal):
            pol = vpol if pid == villain else v0_policy
            return pol.probs(street, position, spot, bucket, legal, 0,
                             None, pid)

        per_hand = {f"k{k}_eps{e}": [] for k, e in arms_grid}
        per_hand["c1_reverse"] = []
        per_hand["c3_random"] = []
        divergence = {f"k{k}_eps{e}": 0 for k, e in arms_grid}

        for hand in payload["eval_hands"]:
            dps = walk_dps(hand, hero)
            hand_d = {f"k{k}_eps{e}": 0.0 for k, e in arms_grid}
            hand_d["c1_reverse"] = 0.0
            hand_d["c3_random"] = 0.0
            if not dps:
                continue
            for i, dp in enumerate(dps):
                obs0 = _obs8c(dp, hand, hero)
                board_seed = f"8c-board-{vid}-{hand['hand_id']}-{i}"
                # baseline rollouts（共享）
                base_list = []
                for r in range(spec.R_ROLLOUTS_PRIMARY):
                    seed = f"8c-{vid}-{hand['hand_id']}-{i}-{r}"
                    base_list.append(rollout8c(
                        hand, hero, dp, PAMPolicy(pam), villain_probs,
                        random.Random(seed + "-rng"),
                        board_seed=f"{board_seed}-{r}"))
                base_mean = sum(base_list) / len(base_list)
                for k, e in arms_grid:
                    key = f"k{k}_eps{e}"
                    R = spec.R_ROLLOUTS_PRIMARY if (k, e) == PRIMARY_ARM \
                        else spec.R_ROLLOUTS_OTHER
                    provider = Real7CProvider(recs, villain, "normal")
                    expl_list = []
                    for r in range(R):
                        seed = f"8c-{vid}-{hand['hand_id']}-{i}-{r}"
                        arm = ExploitArm(random.Random(seed), k=k, eps=e,
                                         recommendation_provider=provider,
                                         policy=PAMPolicy(pam))
                        expl_list.append(rollout8c(
                            hand, hero, dp, arm.action_probs, villain_probs,
                            random.Random(seed + "-rng"),
                            board_seed=f"{board_seed}-{r}"))
                    hand_d[key] += sum(expl_list) / len(expl_list) - base_mean
                # 主臂首行动分歧（同 seed 流）
                b = BaselineArm(random.Random(f"{vid}-{hand['hand_id']}-{i}-0"),
                                policy=PAMPolicy(pam))
                ex = ExploitArm(random.Random(f"{vid}-{hand['hand_id']}-{i}-0"),
                                k=PRIMARY_ARM[0], eps=PRIMARY_ARM[1],
                                recommendation_provider=Real7CProvider(
                                    recs, villain, "normal"),
                                policy=PAMPolicy(pam))
                if b.decide(obs0) != ex.decide(obs0):
                    divergence["k1.0_eps1.0"] += 1
                # C1 reverse / C3 random（主臂口径）
                for cname, mode in (("c1_reverse", "reverse"),
                                    ("c3_random", "random")):
                    provider = Real7CProvider(recs, villain, mode, seed=i)
                    expl_list = []
                    for r in range(spec.R_ROLLOUTS_PRIMARY):
                        seed = f"8c-{vid}-{hand['hand_id']}-{i}-{r}"
                        arm = ExploitArm(random.Random(seed), k=PRIMARY_ARM[0],
                                         eps=PRIMARY_ARM[1],
                                         recommendation_provider=provider,
                                         policy=PAMPolicy(pam))
                        expl_list.append(rollout8c(
                            hand, hero, dp, arm.action_probs, villain_probs,
                            random.Random(seed + "-rng"),
                            board_seed=f"{board_seed}-{r}"))
                    hand_d[cname] += sum(expl_list) / len(expl_list) - base_mean
            for key in hand_d:
                per_hand[key].append(round(hand_d[key] / float(hand.get("bb") or 1.0), 6))
        cells[info["cell"]] = {
            "cell": info["cell"], "feature": info["feature"],
            "direction": info["direction"], "tier": info["tier"],
            "per_hand": per_hand,
            "divergent_dp_primary": divergence["k1.0_eps1.0"],
            "n_hands": len(payload["eval_hands"]),
        }
        per_villain.setdefault(info["cell"], {})[vid] = per_hand
        n_villains_processed += 1
        print(f"ev done {vid} ({n_villains_processed})")

    # ---- 聚合（每 cell：主臂 + controls + hand-level bootstrap + villain-block）----
    def ci(xs):
        if len(xs) < 5:
            return None
        lo, hi, _ = bootstrap_pct(lambda v: sum(v) / len(v), xs,
                                  n_resample=spec.BOOT_N, seed=spec.BOOT_SEED)
        return [round(lo, 6), round(hi, 6)]

    per_cell = {}
    for cell, c in cells.items():
        primary = c["per_hand"]["k1.0_eps1.0"]
        mean = sum(primary) / len(primary) if primary else 0.0
        lo, hi, _ = bootstrap_pct(lambda v: sum(v) / len(v), primary,
                                  n_resample=spec.BOOT_N, seed=spec.BOOT_SEED)
        # villain-block sensitivity：cell 内 15 个 villain 的主臂均值 → 重采样 villain
        tm = []
        for vid, ph in per_villain.get(cell, {}).items():
            xs = ph["k1.0_eps1.0"]
            if xs:
                tm.append(sum(xs) / len(xs))
        tlo, thi = (None, None)
        if len(tm) >= 5:
            tlo, thi, _ = bootstrap_pct(lambda v: sum(v) / len(v), tm,
                                        n_resample=spec.BOOT_N,
                                        seed=spec.BOOT_SEED)
        per_cell[cell] = {
            "cell": cell, "feature": c["feature"], "direction": c["direction"],
            "tier": c["tier"], "n_villains": spec.N_VILLAINS_PER_CELL,
            "n_hands": c["n_hands"],
            "primary_delta_bb100": round(mean * 100, 6),
            "primary_delta_bb_hand": round(mean, 6),
            "primary_ci95_hand_level_bb100": [round(lo * 100, 6),
                                              round(hi * 100, 6)],
            "primary_ci95_villain_block_bb100": (
                [round(tlo * 100, 6), round(thi * 100, 6)]
                if tlo is not None else None),
            "primary_statistical": "pass (CI lower > 0)" if lo > 0
            else "not significant",
            "practical_convention": f"|Δ| ≥ {spec.PRACTICAL_BB100} bb/100"
                                    f"（独立报告）",
            "c1_reverse_delta_bb100": round(
                (sum(c["per_hand"]["c1_reverse"]) / len(c["per_hand"]["c1_reverse"]))
                * 100, 6) if c["per_hand"]["c1_reverse"] else None,
            "c3_random_delta_bb100": round(
                (sum(c["per_hand"]["c3_random"]) / len(c["per_hand"]["c3_random"]))
                * 100, 6) if c["per_hand"]["c3_random"] else None,
            "divergent_dp_primary": c["divergent_dp_primary"],
            "ev_label": spec.EV_LABEL,
        }
    out = {
        "phase": "8C", "proposition": "P2b（synthetic 条件版）：paired simulated ΔEV",
        "config": {
            "hero": spec.HERO_SEAT, "villain": spec.VILLAIN_SEAT,
            "rollouts": {"primary": spec.R_ROLLOUTS_PRIMARY,
                         "other_arms": spec.R_ROLLOUTS_OTHER},
            "bootstrap": {"unit": "hand", "paired": True, "n": spec.BOOT_N,
                          "seed": spec.BOOT_SEED},
            "sensitivity": "villain-block（cell 内 15 villain 分块重采样）",
            "arms_grid": [[k, e] for k, e in arms_grid],
            "primary_arm": list(PRIMARY_ARM),
            "controls": ["C1 reverse direction（主臂）",
                         "C2 orthogonal deviation（detection 阶段）",
                         "C3 random direction（主臂）"],
            "chain": "真实 7C 检测记录 → 冻结 ACTION_MAPPINGS 乘子",
            "no_arm_selection": True, "no_posthoc_tuning": True,
        },
        "per_cell": per_cell,
        "ev_label": spec.EV_LABEL,
        "verdict": None,
    }
    (spec.OUT_DIR / "ev.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out["per_cell"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
