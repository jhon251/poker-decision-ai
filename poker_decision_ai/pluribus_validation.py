"""P54 — CASH decision-level validation harness（只读；确定性；单遍重放）。

对照（eval 决策点，人类座位 only；Pluribus AI 座位 = solver_label 全程隔离）：
  - fold 校准：BASELINE（population 条件率）vs REAL（player 条件估计）
    vs observed_human_action → Brier / agreement；
  - 决策级 EV/Action：单遍重放重建 state → BASELINE（静态范围）vs
    REAL（P46 posterior + P43 条件 fold，证据仅来自 train 分割）→
    compute_ev → model_recommendation vs observed_human_action、
    regret proxy、action-change rate、legality、UNKNOWN 传播。

标签纪律（禁止混淆）：observed_human_action / pluribus_ai_action /
solver_label / model_recommendation。
"""
from __future__ import annotations

import statistics
from typing import Any

from .domain.poker.conditional_model import (
    ConditionalFoldIndex,
    size_bucket,
    texture_tag,
)
from .domain.poker.equity import EquityCalculator
from .domain.poker.range_posterior import (
    RangePosteriorIndex,
    build_range_posterior,
)
from .domain.poker.ranges import Range, parse_range
from .domain.poker.strategy_core import compute_ev
from .pluribus_decision_adapter import (
    AI_SEAT,
    PLURIBUS_DATASET_META,
    _positions_for_seats,
    legal_action_types,
    manifest,
)

EVIDENCE_FRAC = 0.7
EQUITY_ITERATIONS = 1500
NEUTRAL_FOLD = 0.5
MIN_SAMPLES = 30


# ---------------------------------------------------------------- 证据事件
def build_evidence_events(parsed: dict) -> list[dict]:
    """结构化手牌 → 证据事件（human seats only；确定性）。

    fold 事件（facing≠none）→ ConditionalFoldIndex；
    preflop 未开池 entry 事件（facing=none）→ RangePosteriorIndex。
    AI 座位事件不产出（solver_label 隔离）。
    """
    n = len(parsed["players"])
    pos_map = _positions_for_seats(n, n)
    seat_of = {p: i + 1 for i, p in enumerate(parsed["players"])}
    boards = parsed["board_by_street"]
    hid = parsed["hand_id"]
    events: list[dict] = []
    for st in parsed["streets"]:
        street = st["street"]
        if street == "preflop":
            board: list[str] = []
        else:
            board = list(boards.get("flop", []))
            if street in ("turn", "river"):
                board += list(boards.get("turn", []))
            if street == "river":
                board += list(boards.get("river", []))
        texture = "preflop" if street == "preflop" else texture_tag(board)
        for a in st["actions"]:
            pid = a["player"]
            if pid == AI_SEAT:
                continue
            if (a.get("to_call") or 0) > 1e-9:
                events.append({
                    "player": pid, "street": street,
                    "facing": "bet",
                    "size_bucket": size_bucket(a["to_call"],
                                               a["pot_before"]),
                    "position": str(pos_map.get(seat_of.get(pid)) or ""),
                    "texture": texture, "action": a["action"],
                    "fold": 1 if a["action"] == "fold" else 0,
                    "hand_id": hid,
                })
            elif street == "preflop":
                events.append({
                    "hand_id": hid, "player": pid, "street": "preflop",
                    "facing": "none",
                    "position": str(pos_map.get(seat_of.get(pid)) or ""),
                    "action": ("raise" if a["action"] in ("bet", "raise")
                               else "call" if a["action"] == "call"
                               else "fold"),
                })
    return events


def split_hands(hands: list[dict],
                evidence_frac: float = EVIDENCE_FRAC) -> tuple[list, list]:
    """hand_id 字典序确定性排序 → 前 evidence_frac=证据，其余=评估。"""
    ordered = sorted(hands, key=lambda h: str(h["hand_id"]))
    k = int(len(ordered) * evidence_frac)
    return ordered[:k], ordered[k:]


def build_indexes(train_hands: list[dict]) -> dict:
    """train 分割 → player/population 双索引（P43 + P46；human only）。"""
    player_cfi: dict[str, ConditionalFoldIndex] = {}
    player_rpi: dict[str, RangePosteriorIndex] = {}
    pop_cfi = ConditionalFoldIndex()
    pop_rpi = RangePosteriorIndex()
    for h in train_hands:
        for ev in build_evidence_events(h):
            pid = ev["player"]
            if ev.get("facing") == "none":
                pop_rpi.add_events([ev])
                player_rpi.setdefault(pid,
                                      RangePosteriorIndex()).add_events([ev])
            else:
                pop_cfi.add_events([ev])
                player_cfi.setdefault(pid,
                                      ConditionalFoldIndex()).add_events([ev])
    return {"player_cfi": player_cfi, "player_rpi": player_rpi,
            "pop_cfi": pop_cfi, "pop_rpi": pop_rpi}


# ---------------------------------------------------------------- 范围
def _range_from_distribution(dist: dict) -> Range:
    out = Range(name="from_dist")
    for name, w in (dist or {}).items():
        try:
            sub = parse_range(str(name))
        except ValueError:
            continue
        for combo in sub.weights:
            out.weights[combo] = out.weights.get(combo, 0.0) + w
    return out


def _villain_base_range(cfg: Any, villain_pos: str, route: str,
                        dead_cards: list[str]) -> Range | None:
    """BASELINE 静态范围（configs/ranges.yaml；复用 RangeAgent 纯函数）。"""
    from .agents.range.agent import _base_range_string
    defaults = cfg.default_ranges() if cfg is not None else None
    if not defaults:
        return None
    base = _base_range_string(defaults, route, villain_pos)
    if not base:
        base = (defaults.get("open") or {}).get("BB") or "random:1.0"
    try:
        return parse_range(base).remove_cards(_parse_cards(dead_cards))
    except ValueError:
        return None


def _parse_cards(cards: list[str]):
    from .domain.poker import parse_card
    return [parse_card(str(c)) for c in (cards or [])]


def _route_of(action_history: list[dict], villain: str, street: str) -> str:
    """复用 RangeAgent._detect_route 路线语义（只读纯函数）。"""
    from .agents.range.agent import _detect_route
    return _detect_route({"action_history": action_history,
                          "street": street}, villain)


def _villain_range_arm(cfg, *, villain, villain_pos, street, route,
                       dead_cards, idx, with_evidence, hole_cards, board):
    """villain 范围：BASELINE=静态；REAL=P46 posterior（train 证据）。"""
    prov = {"arm": "BASELINE" if not with_evidence else "REAL",
            "villain": villain}
    vr_full = _villain_base_range(cfg, villain_pos, route, dead_cards)
    if vr_full is None:
        return None, prov | {"status": "no_base_range"}
    vr_live = vr_full.remove_cards(_parse_cards(dead_cards))
    if not with_evidence:
        return vr_live, prov | {"status": "baseline_static"}
    post_kind = "starting" if street == "preflop" else "continuing"
    pc = idx["player_cfi"].get(villain)
    pr = idx["player_rpi"].get(villain)
    vr_post, res = build_range_posterior(
        vr_full, vr_live, kind=post_kind, position=villain_pos,
        street=street, route=route, hero_cards=hole_cards, board=board,
        player_index=pr, population_index=idx["pop_rpi"],
        cond_player=pc.to_dict() if pc else None,
        cond_population=idx["pop_cfi"].to_dict())
    prov |= {"status": res.status,
             "lambda": (res.tilt or {}).get("lambda"),
             "fallback_reason": res.fallback_reason,
             "data_kind": "REAL"}
    used = vr_post if (res.available and vr_post is not None) else vr_live
    return used, prov


def _profile(villain: str, idx: dict) -> dict:
    pc = idx["player_cfi"].get(villain)
    pr = idx["player_rpi"].get(villain)
    return {"player_id": villain, "name": villain,
            "range_posterior": pr.to_dict() if pr else {},
            "range_posterior_population": idx["pop_rpi"].to_dict(),
            "conditional_model": pc.to_dict() if pc else {},
            "conditional_population": idx["pop_cfi"].to_dict(),
            "data_kind": "REAL", "dataset": dict(PLURIBUS_DATASET_META),
            "behavior_statistics": {}}


def _load_config():
    from .core.config import load_config
    return load_config()


# ---------------------------------------------------------------- 主验证
def validate(hands: list[dict], *, eval_hand_limit: int = 0,
             decision_hand_limit: int = 0,
             equity_iterations: int = EQUITY_ITERATIONS,
             diagnose: bool = False) -> dict:
    """决策级验证主入口（单遍重放；确定性；AI 座位全程隔离）。

    fold 校准跑全部 eval 手（廉价，无 equity）；决策级 EV/Action 在
    eval 的确定性前缀 decision_hand_limit 手上运行（equity 成本有界，
    manifest 显式声明）。
    """
    cfg = _load_config()
    train, eval_hands = split_hands(hands)
    if eval_hand_limit:
        eval_hands = eval_hands[:eval_hand_limit]
    decision_hands = eval_hands[:decision_hand_limit]         if decision_hand_limit else eval_hands
    idx = build_indexes(train)
    leak = ({h["hand_id"] for h in train}
            & {h["hand_id"] for h in eval_hands})
    decision_hand_ids = {h["hand_id"] for h in decision_hands}
    calc = EquityCalculator(iterations=equity_iterations, seed=42)

    fold_rows = {"BASELINE": {"p": [], "y": []}, "REAL": {"p": [], "y": []}}
    decision_rows: list[dict] = []
    prov_sample: list[dict] = []
    unknown_ct = {"no_state": 0, "no_range": 0, "ev_unavailable": 0}
    legality_ct = {"legal": 0, "illegal": 0}
    labels = {"observed_human_action": 0,
              "pluribus_ai_action_excluded": 0,
              "model_recommendation": 0}

    for hand in eval_hands:
        bb = hand["bb"] or 1.0
        n = len(hand["players"])
        pos_map = _positions_for_seats(n, n)
        seat_of = {p: i + 1 for i, p in enumerate(hand["players"])}
        folded: set[str] = set()
        action_history: list[dict] = []
        boards = hand["board_by_street"]
        for st in hand["streets"]:
            street = st["street"]
            if street == "preflop":
                board: list[str] = []
            else:
                board = list(boards.get("flop", []))
                if street in ("turn", "river"):
                    board += list(boards.get("turn", []))
                if street == "river":
                    board += list(boards.get("river", []))
            for a in st["actions"]:
                pid = a["player"]
                if pid == AI_SEAT:
                    # AI 动作仅作对手历史；永不出现在评估侧
                    labels["pluribus_ai_action_excluded"] += 1
                    if a["action"] in ("bet", "raise", "call"):
                        action_history.append(
                            {"street": street, "player_id": pid,
                             "action": a["action"],
                             "amount": a.get("amount") or 0.0})
                    continue
                labels["observed_human_action"] += 1
                observed = a["action"] if not (
                    a["action"] == "call"
                    and (a.get("to_call") or 0) <= 1e-9) else "check"
                la = legal_action_types(a.get("to_call") or 0,
                                        a.get("remaining_stack") or 0)
                if la is None or observed not in la:
                    legality_ct["illegal" if la is not None else "unknown"] += 1
                else:
                    legality_ct["legal"] += 1
                # ---- fold 校准 ----
                if (a.get("to_call") or 0) > 1e-9 and observed in ("fold",
                                                                   "call"):
                    y = 1 if observed == "fold" else 0
                    facing = ("raise" if _agg_n_before(hand, street, a) >= 2
                              else "bet")
                    args = (street, facing,
                            size_bucket(a["to_call"], a["pot_before"]),
                            str(pos_map.get(seat_of.get(pid)) or ""),
                            texture_tag(board) if board else "preflop")
                    pb = idx["pop_cfi"].estimate(*args)
                    p_base = (float(pb["p_fold"]) if pb is not None
                              else NEUTRAL_FOLD)
                    pc = idx["player_cfi"].get(pid)
                    pr_est = (pc.estimate(*args, population=idx["pop_cfi"])
                              if pc is not None else None)
                    if pr_est is not None:
                        fold_rows["BASELINE"]["p"].append(p_base)
                        fold_rows["BASELINE"]["y"].append(y)
                        fold_rows["REAL"]["p"].append(float(pr_est["p_fold"]))
                        fold_rows["REAL"]["y"].append(y)
                # ---- 决策级 EV/action（有界子集） ----
                if hand["hand_id"] not in decision_hand_ids:
                    if a["action"] == "fold":
                        folded.add(pid)
                    if a["action"] in ("bet", "raise", "call"):
                        action_history.append(
                            {"street": street, "player_id": pid,
                             "action": a["action"],
                             "amount": a.get("amount") or 0.0})
                    continue
                row, prov = _decision_ev_row(
                    cfg, calc, idx, hand, a, street, board, pos_map,
                    seat_of, folded, action_history, equity_iterations)
                if prov and len(prov_sample) < 10:
                    prov_sample.append(prov)
                if row is None:
                    unknown_ct["no_range"] += 1
                else:
                    if row.get("error") == "ev_unavailable":
                        unknown_ct["ev_unavailable"] += 1
                    if row.get("real_best"):
                        labels["model_recommendation"] += 1
                    decision_rows.append(row)
                # ---- 状态推进（人类） ----
                if a["action"] == "fold":
                    folded.add(pid)
                if a["action"] in ("bet", "raise", "call"):
                    action_history.append({"street": street,
                                           "player_id": pid,
                                           "action": a["action"],
                                           "amount": a.get("amount") or 0.0})

    def _metrics(d):
        if not d["p"]:
            return None
        return {"n": len(d["p"]),
                "brier": round(sum((p - y) ** 2 for p, y in
                                   zip(d["p"], d["y"])) / len(d["p"]), 6),
                "agreement": round(sum(1 for p, y in zip(d["p"], d["y"])
                                       if (p >= 0.5) == bool(y))
                                   / len(d["p"]), 6)}

    changes = [d["action_change"] for d in decision_rows
               if d.get("action_change") is not None]
    regrets = [d["regret_proxy"] for d in decision_rows
               if d.get("regret_proxy") is not None]
    with_best = [d for d in decision_rows if d.get("real_best")]
    return {
        "manifest": manifest(),
        "split": {"evidence_hands": len(train),
                  "evaluation_hands": len(eval_hands),
                  "decision_ev_hands": len(decision_hands),
                  "leakage_free": not leak,
                  "leaked_hand_ids": sorted(leak)[:5]},
        "labels": labels,
        "unknown": unknown_ct,
        "legality": legality_ct,
        "fold_calibration": {"arms": {a: _metrics(d)
                                      for a, d in fold_rows.items()}},
        "decision_ev": {
            "n_decision_points": len(decision_rows),
            "observed_agreement_baseline": (
                round(sum(1 for d in with_best
                          if d.get("baseline_best")
                          == d["observed_human_action"]) / len(with_best), 6)
                if with_best else None),
            "observed_agreement_real": (
                round(sum(1 for d in with_best
                          if d.get("real_best")
                          == d["observed_human_action"]) / len(with_best), 6)
                if with_best else None),
            "action_change_rate": (round(sum(changes) / len(changes), 6)
                                   if changes else None),
            # regret 定义 = model_best_utility − observed_action_utility
            # （同模型族内；恒 ≤0 是构造性的）——zero_share = 观测动作即
            # 模型最优的份额；median = 观测动作相对模型最优的平均次优幅度
            "regret_zero_share": (
                round(sum(1 for x in regrets if x == 0) / len(regrets), 6)
                if regrets else None),
            "regret_proxy_median": (round(statistics.median(regrets), 6)
                                    if regrets else None),
        },
        "range_prov_sample": prov_sample,
        "decision_rows_sample": decision_rows[:50],
        **({"decision_rows_all": decision_rows} if diagnose else {}),
    }


def _decision_ev_row(cfg, calc, idx, hand, a, street, board, pos_map,
                     seat_of, folded, action_history, equity_iterations):
    """单决策点：BASELINE vs REAL → compute_ev vs observed（确定性）。"""
    pid = a["player"]
    n = len(hand["players"])
    bb = hand["bb"] or 1.0
    hero_hole = list(hand["hole_cards"].get(pid) or [])
    if len(hero_hole) != 2:
        return None, {"status": "no_hole_cards"}
    villain = _current_aggressor(hand, street, a)
    if villain is None or villain == pid:
        villain = next((p for p in hand["players"]
                        if p != pid and p not in folded), None)
    if villain is None:
        return None, {"status": "no_villain"}
    villain_pos = str(pos_map.get(seat_of.get(villain)) or "")
    route = _route_of(action_history, villain, street)
    dead_cards = hero_hole + list(board)
    arms: dict[str, dict] = {}
    prov_last: dict = {}
    for arm, with_evidence in (("BASELINE", False), ("REAL", True)):
        vr, prov = _villain_range_arm(
            cfg, villain=villain, villain_pos=villain_pos, street=street,
            route=route, dead_cards=dead_cards, idx=idx,
            with_evidence=with_evidence, hole_cards=hero_hole, board=board)
        prov_last = prov
        if vr is None:
            return None, prov | {"status": "no_range"}
        dist = vr.distribution()
        if not dist:
            return None, prov | {"status": "no_range"}
        er = calc.hand_vs_range(hero_hole, board, vr,
                                iterations=equity_iterations, seed=42)
        hero_eq, tie = er.hero_equity, er.tie
        eq_src = {"hero_equity": hero_eq,
                  "outcome": {"win": round(hero_eq - tie / 2, 6),
                              "tie": round(tie, 6),
                              "lose": round(1 - hero_eq - tie / 2, 6)}}
        state = _state_for(hand, a, street, board, pos_map, folded,
                           action_history, pid, bb)
        res = compute_ev(state, eq_src,
                         range_src={"villains": {villain:
                                                 {"distribution": dist}}},
                         opponent_src=({"players": {villain:
                                                    _profile(villain, idx)}}
                                       if with_evidence else None))
        if not res.available:
            return {"error": "ev_unavailable",
                    "reason": res.unavailable_reason,
                    "hand_id": hand["hand_id"]}, prov
        arms[arm] = {"best": {"action": res.best_action,
                              "amount_bb": res.best_amount_bb,
                              "ev_bb": round(res.best_ev_bb, 6)},
                     "objective": res.objective,
                     "hero_equity": round(hero_eq, 6),
                     "ranking": [{"action": c.action,
                                  "amount_bb": c.amount_bb,
                                  "utility_bb": round(c.utility_bb, 6)}
                                 for c in res.candidates]}
    b, r = arms["BASELINE"], arms["REAL"]
    observed = a["action"] if not (a["action"] == "call"
                                   and (a.get("to_call") or 0) <= 1e-9) \
        else "check"
    obs_u = [c["utility_bb"] for c in r["ranking"]
             if c["action"] == observed]
    # ---- P55 诊断字段（additive；不影响 P54 输出语义） ----
    pc = idx["player_cfi"].get(villain)
    facing = "raise" if (a.get("to_call") or 0) > 1e-9 and \
        _agg_n_before(hand, street, a) >= 2 else \
        ("bet" if (a.get("to_call") or 0) > 1e-9 else "check")
    fold_args = (street, facing,
                 size_bucket(a.get("to_call") or 0.0,
                             a.get("pot_before") or 1.0),
                 str(pos_map.get(seat_of.get(villain)) or ""),
                 texture_tag(board) if board else "preflop")
    pb = idx["pop_cfi"].estimate(*fold_args)
    pfold_base = (float(pb["p_fold"]) if pb is not None else None)
    pfold_real = None
    if facing in ("bet", "raise"):
        pr_est = (pc.estimate(*fold_args, population=idx["pop_cfi"])
                  if pc is not None else None)
        pfold_real = (float(pr_est["p_fold"]) if pr_est is not None else None)
    row = {
        "hand_id": hand["hand_id"], "player": pid, "street": street,
        "facing": "bet" if (a.get("to_call") or 0) > 1e-9 else "check",
        "observed_human_action": observed,
        "baseline_best": (b.get("best") or {}).get("action"),
        "real_best": (r.get("best") or {}).get("action"),
        "baseline_amount_bb": (b.get("best") or {}).get("amount_bb"),
        "real_amount_bb": (r.get("best") or {}).get("amount_bb"),
        "action_change": ((b.get("best") or {}).get("action")
                          != (r.get("best") or {}).get("action")),
        "regret_proxy": (round(max(obs_u)
                               - r["ranking"][0]["utility_bb"], 6)
                         if obs_u else None),
        "legal": observed in legal_action_types(a.get("to_call") or 0,
                                                a.get("remaining_stack") or 0),
        "real_objective": r.get("objective"),
        "villain": villain,
        "villain_position": str(pos_map.get(seat_of.get(villain)) or ""),
        "real_lambda": prov_last.get("lambda"),
        "real_status": prov_last.get("status"),
        # ---- P55 diagnostics ----
        "equity_base": b.get("hero_equity"),
        "equity_real": r.get("hero_equity"),
        "ev_best_base": (b.get("best") or {}).get("ev_bb"),
        "ev_best_real": (r.get("best") or {}).get("ev_bb"),
        "p_fold_baseline": pfold_base,
        "p_fold_real": pfold_real,
        "villain_cfi_events": (pc.total_events if pc is not None else 0),
        "villain_rpi_hands": ((idx["player_rpi"].get(villain).total_hands)
                              if idx["player_rpi"].get(villain) is not None
                              else 0),
        "to_call_bb": round((a.get("to_call") or 0.0), 6),
        "pot_bb": round((a.get("pot_before") or 0.0), 6),
    }
    return row, prov_last


def _state_for(hand, a, street, board, pos_map, folded, action_history, pid,
               bb) -> dict:
    n = len(hand["players"])
    players = []
    for i, p in enumerate(hand["players"]):
        players.append({
            "player_id": p, "seat": i + 1,
            "position": str(pos_map.get(i + 1) or ""),
            "stack": round(hand["stacks"][i] / bb, 6),
            "status": "folded" if p in folded else "active",
            "is_hero": p == pid,
            "hole_cards": (list(hand["hole_cards"].get(p) or [])
                           if p == pid else []),
        })
    return {
        "game_type": "cash", "street": street, "board": list(board),
        "pot": round((a.get("pot_before") or 0.0) / bb, 6),
        "to_call": round((a.get("to_call") or 0.0) / bb, 6),
        "bb": 1.0, "sb": 0.5, "ante": 0.0, "min_raise_to": None,
        "hero_id": pid, "button_seat": n,
        "players": players, "action_history": list(action_history),
        "available_actions": (["fold", "call", "raise"]
                              if (a.get("to_call") or 0) > 1e-9
                              else ["fold", "check", "bet"]),
        "cash": {"stakes": "pluribus_50_100"}, "tournament": None,
        "source": "pluribus_phh",
    }


def _agg_n_before(hand: dict, street: str, target: dict) -> int:
    agg = 0
    for st in hand["streets"]:
        if st["street"] != street:
            continue
        for a in st["actions"]:
            if a is target:
                return agg
            if a["action"] in ("bet", "raise"):
                agg += 1
    return agg


def _current_aggressor(hand: dict, street_target: dict, target: dict):
    last = None
    for st in hand["streets"]:
        for a in st["actions"]:
            if a is target:
                return last
            if a["action"] in ("bet", "raise"):
                last = a["player"]
    return last
