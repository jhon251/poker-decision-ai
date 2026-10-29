"""P57 — Strategy Core v3.1 Robustness Matrix（只读诊断；零策略修改）。

对 v3.1（SACA + A3-v2）做街/位置/牌力/SPR/facing 全场景稳定性、单调性、
边界与常数审计。策略数学只读：A8/A9/A10 归因用运行时常数覆写（诊断
上下文管理器，退出即恢复，不触碰冻结文件）；A3-v2 影响用外部解析式
ΔEV = p_c×(eq_cont−eq_full)×(P+s) 计算。

语义声明：regret/EV 均为模型内一致性度量；本模块不宣称盈利/GTO。
"""
from __future__ import annotations

import statistics
from contextlib import contextmanager
from typing import Any

from .domain.poker import strategy_core as sc
from .domain.poker.fold_equity import FoldEstimator
from .domain.poker.ranges import parse_range
from .domain.poker.strategy_core import compute_ev

# ---- 牌类（确定性手牌定义；board 固定 Ah Kd 7c 系） ----
HAND_CLASSES = {
    "made_top_pair": (["Ah", "Kd"], ["Ah", "Kd", "7c"]),      # 顶两对
    "made_second_pair": (["Kd", "Qc"], ["Ah", "Kd", "7c"]),   # 第二对
    "flush_draw": (["9c", "4c"], ["Ah", "7c", "2c"]),         # 同花听 9
    "oesd": (["Tc", "Jc"], ["Qd", "Kd", "2h"]),               # OESD 8
    "weak_showdown": (["7d", "3c"], ["Ah", "Kd", "7c"]),      # 第三对
    "air": (["6d", "2d"], ["Ah", "Kd", "7c"]),                # 纯空气
    "overcards": (["Qc", "Jc"], ["Ah", "Kd", "7c"]),          # 双高牌
}
POSITIONS = {"IP": ("BTN", 2, "BB", 3, "SB", 1),
             "OOP": ("SB", 1, "BB", 3, "BTN", 2)}
SPRS = (3.0, 10.0, 25.0)
FACINGS = {"check": 0.0, "bet_05": 0.5, "bet_10": 1.0, "bet_20": 2.0,
           "raise_30": 3.0}


@contextmanager
def cv_constants(**overrides: dict):
    """运行时覆写 SACA 常数（诊断用；退出恢复——冻结文件零修改）。"""
    saved = {k: getattr(sc, k) for k in overrides}
    for k, v in overrides.items():
        setattr(sc, k, v)
    try:
        yield
    finally:
        for k, v in saved.items():
            setattr(sc, k, v)


def _a8_off():
    return {k: 1.0 for k in sc.REALIZATION}


def _a10_off():
    return {k: 1.0 for k in sc.INITIATIVE}


def make_state(street: str, hero_hole: list[str], board: list[str],
               pos_key: str, spr: float, facing: str,
               game_type: str = "cash") -> dict:
    """确定性场景 state（6-max；pot=10bb；stack=spr×pot）。"""
    pos_hero, seat_hero, pos_vil, seat_vil, pos_third, seat_third = \
        POSITIONS[pos_key]
    pot = 10.0
    stack = round(spr * pot, 4)
    to_call = round(FACINGS[facing] * pot, 4)
    return {
        "game_type": game_type, "street": street, "board": list(board),
        "pot": pot, "to_call": to_call, "bb": 1.0, "sb": 0.5,
        "hero_id": "hero", "button_seat": seat_hero,
        "players": [
            {"player_id": "v1", "seat": seat_third, "position": pos_third,
             "status": "folded", "hole_cards": [], "stack": stack},
            {"player_id": "hero", "seat": seat_hero, "position": pos_hero,
             "is_hero": True, "status": "active",
             "hole_cards": list(hero_hole), "stack": stack},
            {"player_id": "v2", "seat": seat_vil, "position": pos_vil,
             "status": "active", "hole_cards": [], "stack": stack}],
        "action_history": [{"street": "preflop", "player_id": "hero",
                            "action": "raise", "amount": 2.5}],
        "available_actions": (["fold", "call", "raise"] if to_call > 0
                              else ["fold", "check", "bet"]),
        "cash": {"stakes": "robustness"},
        "tournament": None,
    }


def _cand_map(res) -> dict:
    return {(c.action, None if c.amount_bb is None else round(c.amount_bb, 4)):
            c for c in res.candidates}


def _best_utility(res) -> float | None:
    return res.best_utility_bb


# ---------------------------------------------------------------- 矩阵
def robustness_matrix() -> dict:
    """街 × 位置 × 牌类 × SPR × facing 全网格（确定性）。"""
    cells = []
    for street, board_fix in (("flop", None), ("turn", None), ("river", None)):
        for pos_key in ("IP", "OOP"):
            for hc_name, (hole, base_board) in HAND_CLASSES.items():
                board = list(base_board) if street != "preflop" else []
                if street == "turn":
                    board = board + ["2s"]
                elif street == "river":
                    board = board + ["2s", "9d"]
                for spr in SPRS:
                    for facing in FACINGS:
                        st = make_state(street, hole, board, pos_key, spr,
                                        facing)
                        res = compute_ev(st, {"hero_equity": 0.5})
                        cells.append({
                            "street": street, "position": pos_key,
                            "hand_class": hc_name, "spr": spr,
                            "facing": facing,
                            "available": res.available,
                            "cv_model": res.cv_model,
                            "best_action": res.best_action,
                            "best_utility_bb": (round(res.best_utility_bb, 4)
                                                if res.available else None),
                            "competitive": res.competitive_actions,
                            "cv_total": (round(sum(
                                c.cv_bb or 0.0 for c in res.candidates), 4)
                                if res.available else None),
                        })
    return {"n_cells": len(cells), "cells": cells}


# ---------------------------------------------------------------- attribution
def attribution() -> dict:
    """A8/A9/A10 逐项开关归因（代表性场景网格；确定性）。"""
    out = {"A8": [], "A9": [], "A10": []}
    scenarios = []
    for street in ("flop", "turn"):
        for pos_key in ("IP", "OOP"):
            for hc_name in ("made_top_pair", "flush_draw", "oesd",
                            "overcards", "air"):
                hole, board = HAND_CLASSES[hc_name]
                board = list(board) if street == "flop" \
                    else list(board) + ["2s"]
                for spr in (10.0,):
                    for facing in ("check", "bet_05"):
                        scenarios.append((street, pos_key, hc_name, hole,
                                          board, spr, facing))
    for (street, pos_key, hc_name, hole, board, spr, facing) in scenarios:
        st = make_state(street, hole, board, pos_key, spr, facing)
        eq = {"hero_equity": 0.5}
        base = compute_ev(copy_state(st), eq)
        with cv_constants(REALIZATION=_a8_off()):
            no_a8 = compute_ev(copy_state(st), eq)
        with cv_constants(KAPPA_IMPLIED=0.0):
            no_a9 = compute_ev(copy_state(st), eq)
        with cv_constants(INITIATIVE=_a10_off()):
            no_a10 = compute_ev(copy_state(st), eq)

        def _check_u(r):
            c = next((c for c in r.candidates if c.action == "check"), None)
            return c.ev_bb if c else None

        def _bet_u(r):
            c = next((c for c in r.candidates if c.action == "bet"), None)
            return c.ev_bb if c else None

        out["A8"].append({
            "street": street, "position": pos_key, "hand": hc_name,
            "facing": facing,
            "check_cv_A8": (round(_check_u(base) - _check_u(no_a8), 6)
                            if _check_u(base) is not None
                            and _check_u(no_a8) is not None else None),
        })
        out["A9"].append({
            "street": street, "position": pos_key, "hand": hc_name,
            "facing": facing,
            "check_cv_A9": (round(_check_u(base) - _check_u(no_a9), 6)
                            if _check_u(base) is not None
                            and _check_u(no_a9) is not None else None),
        })
        out["A10"].append({
            "street": street, "position": pos_key, "hand": hc_name,
            "facing": facing,
            "bet_cv_A10": (round(_bet_u(base) - _bet_u(no_a10), 6)
                           if _bet_u(base) is not None
                           and _bet_u(no_a10) is not None else None),
        })
    return out


def copy_state(st: dict) -> dict:
    import copy
    return copy.deepcopy(st)


# ---------------------------------------------------------------- A3-v2
def a3v2_shift() -> dict:
    """A3-v2：继续范围 equity 与全范围 equity 的差异（方向不变量 +
    幅度分布）。eq_cont ≤ eq_full 恒成立（继续范围更强）。"""
    rows = []
    for street in ("flop", "turn"):
        for pos_key in ("IP", "OOP"):
            for hc_name, (hole, board0) in HAND_CLASSES.items():
                board = list(board0) if street == "flop" \
                    else list(board0) + ["2s"]
                for vr_name, vr_text in (("random", "random:1.0"),
                                         ("wide", "22+,A2s+,K9s+,Q8s+,"
                                          "J8s+,T8s+,97s+,87s,76s,65s,"
                                          "A9o+,KTo+,QTo+,JTo"),
                                         ("tight", "JJ+,AJs+,A5s-A2s,"
                                          "KQs,AQo+")):
                    pot, s = 10.0, 5.0
                    fe = FoldEstimator(pot_bb=pot, facing="bet",
                                       street=street,
                                       villain_dist=dict(
                                           parse_range(vr_text)
                                           .distribution()),
                                       hero_cards=list(hole),
                                       board=list(board), seed=42)
                    req_v = s / (pot + 2.0 * s)
                    combos = fe._combo_equities()
                    if combos is None:
                        continue
                    total_w = sum(w for _, w in combos)
                    eq_full = sum((1.0 - eq) * w for eq, w in combos) \
                        / total_w
                    w_cont = sum(w for eq, w in combos if eq >= req_v)
                    eq_cont = (sum((1.0 - eq) * w for eq, w in combos
                                   if eq >= req_v) / w_cont
                               if w_cont > 0 else None)
                    rows.append({
                        "street": street, "position": pos_key,
                        "hand": hc_name, "villain_range": vr_name,
                        "eq_full": round(eq_full, 6),
                        "eq_continuing": (round(eq_cont, 6)
                                          if eq_cont is not None else None),
                        "shift": (round(eq_cont - eq_full, 6)
                                  if eq_cont is not None else None),
                    })
    shifts = [r["shift"] for r in rows if r["shift"] is not None]
    return {"n": len(rows),
            "shift_mean": (round(statistics.mean(shifts), 6)
                           if shifts else None),
            "shift_median": (round(statistics.median(shifts), 6)
                             if shifts else None),
            "shift_max": (round(max(shifts), 6) if shifts else None),
            "always_nonpositive": all(x <= 1e-9 for x in shifts),
            "rows": rows}


# ---------------------------------------------------------------- monotonicity
def monotonicity() -> dict:
    """equity 扫描（best utility 单调不降）+ fold 概率扫描（条件单调）。"""
    hole, board = HAND_CLASSES["made_top_pair"]
    eqs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    curve = []
    for street in ("flop", "turn", "river"):
        b = list(board) if street != "preflop" else []
        if street == "turn":
            b = b + ["2s"]
        elif street == "river":
            b = b + ["2s", "9d"]
        utils = []
        for eq in eqs:
            st = make_state(street, hole, b, "IP", 10.0, "check")
            r = compute_ev(st, {"hero_equity": eq})
            utils.append(round(r.best_utility_bb, 6) if r.available else None)
        bad = [i for i in range(1, len(utils))
               if utils[i] is not None and utils[i - 1] is not None
               and utils[i] < utils[i - 1] - 1e-9]
        curve.append({"street": street, "eqs": eqs, "best_utility": utils,
                      "monotonic_nondecreasing": not bad,
                      "violations": bad})
    # fold 概率扫描：villain 范围 random（高弃牌）→ tight（低弃牌）
    # 条件不变量：dEV/dp_fold = P − W_called；W_called > P 时（强手价值
    # 场景）EV 随 p_fold 下降是正确扑克数学（fold 掉更差手=损失价值）。
    hole, board = HAND_CLASSES["made_top_pair"]
    fold_sweep = []
    for eq_src, tag in ((0.30, "weak_bluff_region"),
                        (0.80, "strong_value_region")):
        evs = []
        pfolds = []
        for vr_name, vr_text in (("random", "random:1.0"),
                                 ("wide", "22+,A2s+,K9s+,Q8s+,J8s+,T8s+,"
                                  "97s+,87s,76s,65s,A9o+,KTo+,QTo+,JTo"),
                                 ("tight", "JJ+,AJs+,AQo+")):
            st = make_state("flop", hole, list(board), "IP", 10.0, "check")
            r = compute_ev(st, {"hero_equity": eq_src},
                           range_src={"villains": {"v2": {
                               "distribution": dict(
                                   parse_range(vr_text).distribution())}}})
            bet = next((c for c in r.candidates if c.action == "bet"
                        and c.amount_bb == 5.0), None)
            fe = FoldEstimator(pot_bb=10.0, facing="bet", street="flop",
                               villain_dist=dict(
                                   parse_range(vr_text).distribution()),
                               hero_cards=list(hole), board=list(board),
                               seed=42)
            fr = fe.for_size(5.0)
            evs.append(round(bet.ev_bb, 6) if bet else None)
            pfolds.append(round(fr.p_fold, 6) if fr.available else None)
        fold_sweep.append({"equity_region": tag, "ranges": ["random",
                                                            "wide", "tight"],
                           "p_fold": pfolds, "bet_ev": evs})
    return {"equity_sweep": curve, "fold_sweep": fold_sweep}


# ---------------------------------------------------------------- boundary
def boundaries() -> dict:
    """边界：allin 封顶 / 近零底池 / 缺信息中性 / UNKNOWN。"""
    hole, board = HAND_CLASSES["made_top_pair"]
    out = {}
    # all-in 封顶：stack = 12bb（bet 封顶）
    st = make_state("flop", hole, list(board), "IP", 1.2, "check")
    st["players"][1]["stack"] = 12.0
    st["players"][2]["stack"] = 12.0
    r = compute_ev(st, {"hero_equity": 0.5})
    out["allin_cap"] = {
        "available": r.available,
        "max_amount_bb": max((c.amount_bb or 0.0) for c in r.candidates),
        "eff_stack_bb": 12.0,
        "capped_ok": max((c.amount_bb or 0.0)
                         for c in r.candidates) <= 12.0 + 1e-9}
    # 近零底池
    st = make_state("flop", hole, list(board), "IP", 10.0, "check")
    st["pot"] = 0.5
    r = compute_ev(st, {"hero_equity": 0.5})
    out["near_zero_pot"] = {"available": r.available,
                            "best": r.best_action,
                            "best_utility": (round(r.best_utility_bb, 6)
                                             if r.available else None)}
    # pot=0 → UNKNOWN（输入门）
    st0 = make_state("flop", hole, list(board), "IP", 10.0, "check")
    st0["pot"] = 0.0
    r0 = compute_ev(st0, {"hero_equity": 0.5})
    out["zero_pot_unknown"] = {"available": r0.available,
                               "reason": r0.unavailable_reason}
    # 缺信息中性（位置/手牌全缺）
    st = {"game_type": "cash", "street": "flop", "board": list(board),
          "pot": 10.0, "to_call": 0.0, "bb": 1.0, "hero_id": "hero",
          "players": [{"player_id": "hero", "is_hero": True, "stack": 100}]}
    r = compute_ev(st, {"hero_equity": 0.6})
    check = next(c for c in r.candidates if c.action == "check")
    out["missing_info_neutral"] = {"cv_model": r.cv_model,
                                   "check_ev": round(check.ev_bb, 6),
                                   "bit_frozen": check.ev_bb
                                   == pytest_approx(6.0)}
    # 极端 draw（12-out 组合听）
    st = make_state("flop", ["9c", "4c"], ["Ah", "7c", "2c"], "IP", 10.0,
                    "check")
    r = compute_ev(st, {"hero_equity": 0.35})
    check = next(c for c in r.candidates if c.action == "check")
    out["extreme_draw"] = {"cv_check": round(check.cv_bb, 6),
                           "bounded": check.cv_bb < 12.0}
    return out


def pytest_approx(v: float, tol: float = 1e-9) -> float:
    return v


# ---------------------------------------------------------------- 主入口
def full_report() -> dict:
    return {
        "report_version": "p57-robustness-v1",
        "matrix": robustness_matrix(),
        "attribution": attribution(),
        "a3v2_shift": a3v2_shift(),
        "monotonicity": monotonicity(),
        "boundaries": boundaries(),
        "honesty": {"no_tuning": ("本阶段未修改任何常数/公式；诊断用常数"
                                  "覆写为运行时上下文，退出即恢复"),
                    "profitability": "不构成任何盈利性/GTO 证明"},
    }
