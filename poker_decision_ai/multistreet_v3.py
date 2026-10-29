"""R2.2 — 多街 v3：层内语义正确性修正（opt-in 实验层；R2/R2.1 只读复用）。

修正项
------
F1 节点方向位置感知：IP → 对手先行；**OOP → hero 先行**（此前恒让对手先行）
F2 hero 下注后对手响应三分支：fold（冻结通道）× call × **raise**（`facing_bet` 证据）
F3 all-in / 退化守卫：hero 投入后栈≈0 → 解析摊牌闭包（不进入 0 栈闭包）
F4 诊断常驻化：benchmark 输出 per-action 分位数 + 三消融（frozen / v3 / no-aggression）
F5 bounty 语义标注：MTT/PKO 沿用冻结核 bounty 候选值（**不改任何 bounty 公式**）

记账约定（与冻结核同口径；测试断言恒等式与零修正等价）
---------------------------------------------------
终端/节点价值均为「相对该节点的净值」（冻结闭包输出）：
    V_checked   = 冻结闭包(to_call=0)             （hero 可 check/bet）
    V_faced     = 冻结闭包(to_call=villain bet)   （hero 可 fold/call/raise）
    V_faced_raise = 冻结闭包(to_call=villain raise)
    V_terminal  = eq × pot                        （F3 allin / 退化）

候选：
    fold          : 0
    check / call  : V_node − invest
    bet / raise   : p_fold×P + p_call×(V_node_call − invest)
                              + p_raise×(V_faced_raise − invest)
其中 V_node：
    IP  : p_a×V_faced + (1−p_a)×V_checked
    OOP : max( p_a×V_faced + (1−p_a)×V_checked ,
               p_fv×pot + (1−p_fv)×[(1−r)×V_show + r×V_faced_raise] )
    V_show = eq×(pot + node_bet) − (1−eq)×node_bet（hero 在节点自行下注的净 EV）

恒等式：utility_chip_v3 = immediate_ev + future_realization − aggression_risk
（future_realization = 「对手在 checked_to 不自发下注」反事实；raise 分支两侧同值）
"""
from __future__ import annotations

from typing import Any

from .domain.poker.strategy_core import compute_ev
from .multistreet_v2 import (MS2_SEED, _child, _chip_values, _hero_of,
                             child_equity, mode_transform_available,
                             mode_utility_from_chip, remaining_deck,
                             sample_runouts)

MS3_VERSION = "r22-multistreet-v3"
MS3_DEPTH_CAP_STREETS = 1
MS3_RUNOUTS = 8
MS3_VILLAIN_BET_FRAC = 0.66        # 声明式常数（与 R2.1 同口径；非调参）
MS3_VILLAIN_RAISE_FRAC = 1.0       # 对手加注至 = pot-sized raise（声明式）
MS3_ALLIN_EPS_BB = 1e-6
MS3_SEED = MS2_SEED
_FUTURE = {"flop": "turn", "turn": "river"}


def hero_in_position(state: dict, villain_pid: str) -> bool | None:
    """IP/OOP（只读调用冻结核 `_hero_in_position`；不可判定 → None，不猜）。"""
    try:
        from .domain.poker.strategy_core import _hero_in_position
        return _hero_in_position(state, villain_pid)
    except Exception:  # noqa: BLE001
        return None


def _bet_size(pot: float, stack: float, frac: float) -> float:
    s = frac * pot
    return min(s, stack) if stack > 0 else s


def _raise_to(pot: float, to_call: float, stack: float) -> float:
    target = max(pot * MS3_VILLAIN_RAISE_FRAC, 2.0 * to_call)
    return min(target, stack) if stack > 0 else target


def evaluate_v3(state: dict, equity_src: dict, range_src: dict | None = None,
                opponent_src: dict | None = None, *,
                aggression: dict | None = None,
                villain_id: str | None = None,
                villain_position: str | None = None,
                runouts: int = MS3_RUNOUTS,
                equity_iterations: int = 300,
                seed: int = MS3_SEED) -> dict:
    """R2.2 层估值（位置感知 + 三分支 + allin 守卫；不覆盖冻结结果）。"""
    base = compute_ev(state, equity_src, range_src=range_src,
                      opponent_src=opponent_src, seed=seed)
    mode = str(state.get("game_type") or "")
    out: dict[str, Any] = {
        "ms3_version": MS3_VERSION,
        "depth_cap_streets": MS3_DEPTH_CAP_STREETS,
        "constants": {"villain_bet_frac": MS3_VILLAIN_BET_FRAC,
                      "villain_raise_frac": MS3_VILLAIN_RAISE_FRAC},
        "seed": seed, "runouts_per_branch": runouts,
        "bounty_semantics": ("n/a" if mode == "cash" else
                             "frozen_value_carried（不重算淘汰概率；"
                             "bounty 公式未改）"),
        "assumptions": [
            "R2.2-A1 节点方向位置感知：IP → 对手先行；OOP → hero 先行（F1）",
            "R2.2-A2 hero 下注后对手响应三分支 fold/call/raise（F2）；尺寸为"
            "声明式常数（非调参）",
            "R2.2-A3 all-in / 退化状态走解析摊牌闭包（F3）",
            "R2.2-A4 深度上限 1 条未来街；节点 hero 应对由冻结闭包给出",
            "R2.2-A5 仅建模上下文条件侵略性；**牌力条件侵略性未建模**（无证据）"
            "——不得据此声称真实 opponent model 已完成",
            "R2.2-A6 runout 仅用未见牌确定性采样（anti-lookahead）",
            "R2.2-A7 单人主对手（多人池交互不建模）",
            "R2.2-A8 no-aggression 反事实仅关闭 checked_to 自发下注；"
            "raise 分支两侧同值（声明）",
            "R2.2-A9 OOP 节点的 hero 先行仅建模 check（不自行下注）——"
            "节点下注估值不属冻结口径；allin 候选整支跳过节点（F3）",
        ],
    }
    if not base.available:
        out.update({"available": False,
                    "decline_reason": f"frozen core unavailable: "
                                      f"{base.unavailable_reason}",
                    "base": None, "candidates": []})
        return out

    mt = mode_transform_available(state)
    street = str(state.get("street") or "")
    next_street = _FUTURE.get(street)
    hero = _hero_of(state)
    hero_cards = list((hero or {}).get("hole_cards") or [])
    board = [str(c) for c in (state.get("board") or [])]
    bb = float(state.get("bb") or 0.0)
    vpid = villain_id or str(next(iter((range_src or {}).get("villains") or {}
                                       or {}), "") or "")
    vdist = dict(((range_src or {}).get("villains") or {})
                 .get(vpid, {}).get("distribution") or {}) if vpid else {}
    hero_ip = hero_in_position(state, vpid) if vpid else None

    reasons: list[str] = []
    if not mt.get("available"):
        reasons.append(str(mt.get("reason")))
    if next_street is None:
        reasons.append(f"no future street from {street!r}")
    if len(hero_cards) != 2:
        reasons.append("hero hole cards missing")
    if not vpid or not vdist:
        reasons.append("villain range/distribution missing")
    if bb <= 0:
        reasons.append("bb<=0")
    if aggression is None:
        reasons.append("aggression index missing（UNKNOWN：不伪造对手模型）")
    if hero_ip is None:
        reasons.append("hero position unverifiable（F1 需要 IP/OOP；不猜）")

    out["base"] = {
        "best_action": base.best_action, "best_amount_bb": base.best_amount_bb,
        "objective": base.objective, "mode": base.mode, "equity": base.equity,
        "candidates": [{"action": c.action, "amount_bb": c.amount_bb,
                        "ev_bb": round(float(c.ev_bb or 0.0), 6),
                        "utility_ss": round(float(
                            c.utility_bb if c.utility_bb is not None
                            else c.ev_bb or 0.0), 6),
                        "p_fold": c.p_fold,
                        "bounty_util_bb": c.bounty_util_bb}
                       for c in base.candidates],
    }
    if reasons:
        out.update({
            "available": False, "decline_reason": "; ".join(reasons),
            "candidates": [_declined(c) for c in base.candidates],
            "best_ss": base.best_action, "best_v3": None,
            "provenance": {"layer": MS3_VERSION, "declined": True,
                           "reason": "; ".join(reasons), "hero_ip": hero_ip},
        })
        return out

    pop_idx = (aggression or {}).get("population")
    pl_idx = ((aggression or {}).get("players") or {}).get(vpid)

    def _est(street_q: str, facing_q: str) -> dict | None:
        if pl_idx is not None:
            e = pl_idx.estimate(street_q, facing_q, None, villain_position,
                                None, population=pop_idx)
            if e:
                return e
        if pop_idx is not None:
            return pop_idx.estimate(street_q, facing_q, None, villain_position,
                                    None, population=None)
        return None

    est_bet = _est(next_street, "checked_to")
    if est_bet is None:
        out.update({
            "available": False,
            "decline_reason": (f"aggression evidence insufficient for "
                               f"{next_street}/checked_to（player<30 且 "
                               f"population<30）"),
            "candidates": [_declined(c) for c in base.candidates],
            "best_ss": base.best_action, "best_v3": None,
            "provenance": {"layer": MS3_VERSION, "declined": True,
                           "reason": "aggression UNKNOWN", "hero_ip": hero_ip},
        })
        return out
    p_a = float(est_bet["p_aggress"])
    src_bet = str(est_bet.get("source"))
    est_raise = _est(street, "facing_bet")
    r_rate = float(est_raise["p_aggress"]) if est_raise else 0.0
    src_raise = str(est_raise.get("source")) if est_raise else None

    pot_bb = float(state.get("pot") or 0.0) / bb
    hero_stack_bb = float((hero or {}).get("stack") or 0.0) / bb
    to_call_bb = float(state.get("to_call") or 0.0) / bb
    v_stack_bb = max((float(p.get("stack") or 0.0)
                      for p in state.get("players") or []
                      if isinstance(p, dict)
                      and str(p.get("player_id")) == vpid), default=0.0) / bb
    deck = remaining_deck(hero_cards, board)
    picked = sample_runouts(deck, runouts, seed)

    def _closure(board_plus: list, pot: float, hero_stack: float,
                 to_call: float, eq: float) -> float | None:
        st = _child(state, street=next_street, board=board_plus, pot=pot,
                    hero_stack=hero_stack, to_call=to_call, bb=bb)
        vals = _chip_values(st, eq, vdist, vpid, seed)
        if not vals:
            return None
        return round(float(max(v for v, _ in vals.values())), 6)

    cands: list[dict] = []
    for c in base.candidates:
        ev = float(c.ev_bb or 0.0)
        util_ss = float(c.utility_bb if c.utility_bb is not None else ev)
        amt = float(c.amount_bb or 0.0)
        entry: dict[str, Any] = {
            "action": c.action, "amount_bb": c.amount_bb,
            "utility_ss": round(util_ss, 6), "frozen_ev_bb": round(ev, 6),
            "immediate_ev": None, "future_realization": None,
            "aggression_risk": None, "utility_chip_v3": None,
            "utility_v3": None, "delta": None, "p_fold": c.p_fold,
            "opponent_response_distribution": None, "leaf": None,
            "allin_guard": False,
        }
        if c.action == "fold":
            entry.update({"immediate_ev": 0.0, "future_realization": 0.0,
                          "aggression_risk": 0.0, "utility_chip_v3": 0.0,
                          "opponent_response_distribution": {},
                          "leaf": {"terminal": "fold EV=0"}})
            _finish(entry, state, util_ss, c, seed)
            cands.append(entry)
            continue

        if c.action == "call":
            invest = min(amt, hero_stack_bb)
            pot_node = pot_bb + invest
            p_f = None
        elif c.action == "check":
            invest = 0.0
            pot_node = pot_bb
            p_f = None
        else:
            if c.p_fold is None:
                entry["leaf"] = {"declined": "fold probability unavailable"}
                cands.append(entry)
                continue
            p_f = float(c.p_fold)
            invest = amt
            pot_node = (pot_bb + 2.0 * amt if c.action == "bet"
                        else pot_bb + amt + max(0.0, amt - to_call_bb))
        stack_node = hero_stack_bb - invest
        if stack_node < -1e-9:
            entry["leaf"] = {"declined": "hero stack exceeded"}
            cands.append(entry)
            continue
        allin = stack_node <= MS3_ALLIN_EPS_BB
        entry["allin_guard"] = bool(allin)
        if allin:
            # F3：hero 投入后无后续筹码 → **不进入多街节点**，保留冻结核值
            #（含冻结核自身的 continuing-range equity；本层不得用全范围
            # equity 替代——那会系统性高估 allin 被跟分支）
            entry.update({
                "immediate_ev": round(ev, 6), "future_realization": 0.0,
                "aggression_risk": 0.0, "utility_chip_v3": round(ev, 6),
                "opponent_response_distribution": {
                    "fold": (round(float(p_f), 6) if p_f is not None else None),
                    "continuation": "frozen_continuing_equity（本层不重估）"},
                "leaf": {"allin_guard": True, "node_skipped": True,
                         "reason": "hero stack exhausted after invest"},
            })
            _finish(entry, state, util_ss, c, seed)
            cands.append(entry)
            continue

        if c.action in ("bet", "raise"):
            remaining = max(0.0, 1.0 - float(p_f))
            p_raise_v = remaining * r_rate
            p_call_v = remaining * (1.0 - r_rate)
        else:
            p_raise_v = p_call_v = None

        v_act: list[float] = []
        v_noagg: list[float] = []
        for card in picked:
            eq_c = child_equity(hero_cards, board + [card], vdist,
                                equity_iterations, seed)
            if eq_c is None:
                continue
            bplus = board + [card]
            bet_size = _bet_size(pot_node, v_stack_bb, MS3_VILLAIN_BET_FRAC)
            v_faced = _closure(bplus, pot_node + bet_size, stack_node * bb,
                               bet_size * bb, eq_c)
            v_checked = _closure(bplus, pot_node * bb, stack_node * bb, 0.0,
                                 eq_c)
            if v_faced is None or v_checked is None:
                continue
            if hero_ip:
                # F1：IP → 对手先行（hero 的 check 后仍可 bet：由冻结闭包给出）
                v_node = p_a * v_faced + (1.0 - p_a) * v_checked
                v_node_noagg = v_checked
            else:
                # F1：OOP → hero 先行，仅建模 check（不自行下注——节点下注
                # 估值不属冻结口径，避免非冻结近似污染；声明于 R2.2-A9）
                v_showdown = round(eq_c * pot_node, 6)
                v_node = p_a * v_faced + (1.0 - p_a) * v_showdown
                v_node_noagg = v_showdown
            if c.action in ("bet", "raise"):
                # call 支 → 节点；raise 支 → hero 面对对手加注
                if allin:
                    v_raise_faced = round(eq_c * pot_node, 6)
                else:
                    raise_to2 = _raise_to(pot_node, max(invest, 1e-9), v_stack_bb)
                    v_raise_faced = _closure(
                        bplus,
                        pot_node + invest + max(0.0, raise_to2 - invest),
                        stack_node * bb, max(0.0, raise_to2 - invest) * bb,
                        eq_c)
                    if v_raise_faced is None:
                        continue
                v_act.append(p_call_v * (v_node - invest)
                             + p_raise_v * (v_raise_faced - invest))
                v_noagg.append(p_call_v * (v_node_noagg - invest)
                               + p_raise_v * (v_raise_faced - invest))
            else:
                v_act.append(v_node - invest)
                v_noagg.append(v_node_noagg - invest)

        if not v_act:
            entry["leaf"] = {"declined": "no node evaluation available"}
            cands.append(entry)
            continue

        actual_future = round(sum(v_act) / len(v_act), 6)
        check_future = round(sum(v_noagg) / len(v_noagg), 6)
        imm = (0.0 if c.action in ("check", "call")
               else round(float(p_f) * pot_bb, 6))
        risk = round(check_future - actual_future, 6)
        chip_v3 = round(imm + actual_future, 6)
        opp_dist = ({"fold": round(float(p_f), 6),
                     "call": round(float(p_call_v or 0.0), 6),
                     "raise": round(float(p_raise_v or 0.0), 6)}
                    if c.action in ("bet", "raise")
                    else {"bet": round(p_a, 6), "check": round(1.0 - p_a, 6)})
        entry.update({
            "immediate_ev": imm, "future_realization": check_future,
            "aggression_risk": risk, "utility_chip_v3": chip_v3,
            "opponent_response_distribution": opp_dist,
            "leaf": {"n_runouts": len(v_act), "next_street": next_street,
                     "hero_ip": bool(hero_ip),
                     "node_order": "villain_first" if hero_ip else "hero_first",
                     "p_aggress": round(p_a, 6), "aggression_source": src_bet,
                     "raise_rate_facing_bet": round(r_rate, 6),
                     "raise_rate_source": src_raise,
                     "hero_investment_bb": round(invest, 6),
                     "pot_node_bb": round(pot_node, 6),
                     "villain_bet_size_bb": round(bet_size, 6),
                     "allin": bool(allin)},
        })
        _finish(entry, state, util_ss, c, seed)
        cands.append(entry)

    scored = [e for e in cands if e.get("utility_v3") is not None]
    best_v3 = max(scored, key=lambda e: e["utility_v3"])["action"] if scored \
        else None
    out.update({
        "available": True, "decline_reason": None, "candidates": cands,
        "best_ss": base.best_action, "best_v3": best_v3,
        "hero_ip": bool(hero_ip), "p_aggress": round(p_a, 6),
        "aggression_source": src_bet,
        "provenance": {
            "layer": MS3_VERSION, "seed": seed,
            "runouts_per_branch": len(picked), "hero_ip": bool(hero_ip),
            "node_order": "villain_first" if hero_ip else "hero_first",
            "aggression_source": src_bet, "p_aggress": round(p_a, 6),
            "raise_rate_facing_bet": round(r_rate, 6),
            "raise_rate_source": src_raise,
            "future_cards": "sampled_unseen_only（anti-lookahead）",
            "mode_space": ("chip_ev" if mode == "cash"
                           else "gain×k+ − loss×k-(+bounty)"),
            "frozen_core_untouched": True,
            "solver": "not used（declarative；非 GTO）",
            "strength_conditioned_aggression": "NOT MODELED（无证据）",
            "bounty_semantics": out["bounty_semantics"],
        },
    })
    return out


def _declined(c) -> dict:
    return {"action": c.action, "amount_bb": c.amount_bb,
            "utility_ss": round(float(c.utility_bb if c.utility_bb is not None
                                      else c.ev_bb or 0.0), 6),
            "immediate_ev": None, "future_realization": None,
            "aggression_risk": None, "utility_chip_v3": None,
            "utility_v3": None, "delta": None,
            "opponent_response_distribution": None, "leaf": None}


def _finish(entry: dict, state: dict, util_ss: float, cand, seed: int) -> None:
    if entry.get("utility_chip_v3") is None:
        return
    tf = mode_utility_from_chip(state, entry["utility_chip_v3"], cand, seed)
    entry["utility_v3"] = tf.get("utility")
    entry["mode_transform"] = {k: tf.get(k) for k in
                               ("k_p", "k_m", "bounty_util", "error")}
    if entry["utility_v3"] is not None:
        entry["delta"] = round(entry["utility_v3"] - util_ss, 6)
