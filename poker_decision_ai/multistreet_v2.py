"""R2.1 — 多街 v2：双方节点 + 分量化解剖（opt-in 层；零解冻）。

与 R2 的关键差别
----------------
R2 的叶子 = 冻结 `compute_ev` 的最优动作 → 继承 bet 偏置，且对手在下一街
不会行动（G1/G2）。R2.1 改为：

  候选动作
    ├─ immediate_ev      本街纯终局期望 = 冻结 `ev_bb` − 冻结 `cv_bb`
    │                    （即剔除 SACA 闭式延续价值后的单街值）
    └─ 节点 T+1（**双方**）：
         对手先行动：bet（P_aggress，来自 ConditionalAggressionIndex）
                     └→ hero 应对（冻结闭包：fold/call/raise）
                     check（1 − P_aggress）
                     └→ hero 行动（冻结闭包：check/bet）
         future_realization = 「对手恒 check」反事实下的节点价值（净值，已扣投入）
         aggression_risk    = future_realization − 实际节点价值（符号化：强牌可为负）

恒等式（构造保证）：
      utility_chip_v2 = immediate_ev + future_realization − aggression_risk

模式一致性（裁定 A）
--------------------
用**只读**导入的冻结私有助手重建 utility 空间：
  `strategy_core._icm_kappa`（k±）、`strategy_core._bounty_gain_bb`（bounty gain）
  + `bounty.knockout_probability`
并以候选自身 (ev_bb, utility_bb, bounty_util_bb) **精确反解** gain/loss，
从而 `Δ=0` 时 `utility_v2 == utility_bb` **逐位相等**（零修正等价测试）。

失败回退（UNKNOWN-first，绝不静默降级）
  * 缺 hero 手牌 / 对手范围 / bb / 无未来街（river）→ 拒绝前瞻
  * 对手侵略性证据不足（player<30 且 population<30）→ 拒绝前瞻
  * `_icm_kappa`/`_bounty_gain_bb` 缺失或反解不一致 → **MTT/PKO 整体拒绝**
  * 分量非有限 / 超栈 → 拒绝该候选或该点
"""
from __future__ import annotations

import random
from typing import Any

from .domain.poker.strategy_core import compute_ev
from .opponent_aggression import _texture_tag as _texture

MS2_VERSION = "r21-multistreet-v2"
MS2_DEPTH_CAP_STREETS = 1          # 前瞻深度上限（1 条未来街；声明式）
MS2_RUNOUTS = 8
MS2_VILLAIN_BET_FRAC = 0.66        # 节点 T+1 对手下注尺寸（声明式常数，非调参）
MS2_SEED = 42
_FUTURE = {"flop": "turn", "turn": "river"}
_RANKS = "23456789TJQKA"
_SUITS = "shdc"


def _norm(c: str) -> str:
    return str(c).strip().capitalize()


def remaining_deck(hero_cards: list, board: list) -> list[str]:
    known = {_norm(c) for c in list(hero_cards) + list(board)}
    return [r + s for r in _RANKS for s in _SUITS if (r + s) not in known]


def sample_runouts(deck: list[str], k: int, seed: int = MS2_SEED) -> list[str]:
    if k <= 0 or not deck:
        return []
    pool = sorted(deck)
    return sorted(random.Random(seed).sample(pool, min(k, len(pool))))


def _hero_of(state: dict) -> dict | None:
    return next((p for p in state.get("players") or []
                 if isinstance(p, dict) and p.get("is_hero")), None)


def _child(state: dict, *, street: str, board: list, pot: float,
           hero_stack: float, to_call: float, bb: float) -> dict:
    st = {k: v for k, v in state.items() if k != "tournament"}
    if isinstance(state.get("tournament"), dict):
        st["tournament"] = dict(state["tournament"])
    st.update({"street": street, "board": list(board), "to_call": to_call,
               "pot": round(pot, 6)})
    players = []
    for p in state.get("players") or []:
        if not isinstance(p, dict):
            continue
        q = dict(p)
        if p.get("is_hero"):
            q["stack"] = round(max(0.0, hero_stack), 6)
        players.append(q)
    st["players"] = players
    st.pop("min_raise_to", None)
    return st


def _range_of(villain_dist: dict) -> Any:
    from .domain.poker.ranges import Range, parse_range
    rng_obj = Range(name="ms2")
    for name, w in (villain_dist or {}).items():
        try:
            sub = parse_range(str(name))
        except ValueError:
            continue
        for combo in sub.weights:
            rng_obj.weights[combo] = rng_obj.weights.get(combo, 0.0) + w
    return rng_obj if rng_obj.weights else None


def child_equity(hero_cards: list, board: list, villain_dist: dict,
                 iterations: int = 300, seed: int = MS2_SEED) -> float | None:
    from .domain.poker.equity import EquityCalculator
    rng_obj = _range_of(villain_dist)
    if rng_obj is None:
        return None
    try:
        er = EquityCalculator(iterations=iterations, seed=seed).hand_vs_range(
            hero_cards, board, rng_obj, iterations=iterations, seed=seed)
    except Exception:
        return None
    eq = getattr(er, "hero_equity", None)
    return float(eq) if eq is not None else None


def _chip_values(state: dict, eq: float, villain_dist: dict,
                 villain_pid: str, seed: int) -> dict | None:
    """冻结闭包：返回 {action: (ev_chip_bb, cv_bb)}（剔除闭式 CV 后的即时值）。"""
    res = compute_ev(state, {"hero_equity": eq},
                     range_src={"villains": {villain_pid: {
                         "distribution": dict(villain_dist)}}}, seed=seed)
    if not res.available or not res.candidates:
        return None
    out = {}
    for c in res.candidates:
        cv = float(c.cv_bb or 0.0)
        if out.get(c.action, (None,))[0] is None or \
                float(c.ev_bb or 0.0) > out[c.action][0]:
            out[c.action] = (round(float(c.ev_bb or 0.0), 6), cv)
    return out


# ── 模式变换（裁定 A：只读私有助手 + 零修正等价） ────────────────────────
def mode_transform_available(state: dict) -> dict:
    """检查冻结私有助手可用性（缺失 → 调用方必须整体拒绝 MTT/PKO）。"""
    mode = str(state.get("game_type") or "")
    if mode == "cash":
        return {"available": True, "k_p": 1.0, "k_m": 1.0, "bounty_gain": None}
    try:
        from .domain.poker.strategy_core import (_bounty_gain_bb,  # noqa: F401
                                                 _icm_kappa)      # noqa: F401
    except Exception as e:  # noqa: BLE001
        return {"available": False,
                "reason": f"frozen mode helpers unavailable: {e}"}
    return {"available": True}


def mode_utility_from_chip(state: dict, chip_ev: float, cand, seed: int = 42
                           ) -> dict:
    """把 chip 域修正值映射回候选自身的 utility 空间（精确反解 + Δ 施加）。

    返回 {utility, k_p, k_m, bounty_util, recovered_gain, recovered_loss}；
    任何不一致 → {"error": …}（调用方必须拒绝该点，不得静默使用）。
    """
    ev = float(cand.ev_bb or 0.0)
    util = float(cand.utility_bb if cand.utility_bb is not None else ev)
    bu = float(cand.bounty_util_bb or 0.0)
    mode = str(state.get("game_type") or "")
    if mode == "cash":
        return {"utility": round(chip_ev, 6), "k_p": 1.0, "k_m": 1.0,
                "bounty_util": 0.0, "recovered_gain": None,
                "recovered_loss": None, "space": "chip_ev(=utility)"}
    try:
        from .domain.poker.strategy_core import _bounty_gain_bb, _icm_kappa
    except Exception as e:  # noqa: BLE001
        return {"error": f"mode helpers unavailable: {e}"}
    hero = _hero_of(state) or {}
    bb = float(state.get("bb") or 0.0)
    try:
        kappa = _icm_kappa(state, str(hero.get("player_id") or ""),
                           float(hero.get("stack") or 0.0), bb, seed)
    except Exception as e:  # noqa: BLE001
        return {"error": f"icm_kappa failed: {e}"}
    if kappa is None:
        k_p = k_m = 1.0
    else:
        k_p, k_m = float(kappa[0]), float(kappa[1])
    if k_p <= 0 or k_m <= 0:
        return {"error": f"invalid kappa k+= {k_p}, k-= {k_m}"}
    delta = chip_ev - ev
    if abs(k_m - k_p) < 1e-12:
        # utility = ev×k + bu → Δ 直接按同系数缩放
        return {"utility": round((chip_ev) * k_p + bu, 6), "k_p": k_p,
                "k_m": k_m, "bounty_util": bu, "recovered_gain": None,
                "recovered_loss": None, "space": "chip_ev×k(+bounty)"}
    loss = (ev * k_p - (util - bu)) / (k_m - k_p)
    gain = ev + loss
    if loss < -1e-6 or gain < -1e-6:
        return {"error": f"gain/loss recovery inconsistent (gain={gain}, "
                         f"loss={loss})"}
    if delta >= 0:
        gain += delta
    else:
        loss += -delta
    util2 = gain * k_p - loss * k_m + bu
    return {"utility": round(util2, 6), "k_p": k_p, "k_m": k_m,
            "bounty_util": bu, "recovered_gain": round(gain, 6),
            "recovered_loss": round(loss, 6), "space": "gain×k+ − loss×k-(+bounty)"}


# ── 主入口 ─────────────────────────────────────────────────────────────
def evaluate_v2(state: dict, equity_src: dict, range_src: dict | None = None,
                opponent_src: dict | None = None, *,
                aggression: dict | None = None,
                villain_id: str | None = None,
                villain_position: str | None = None,
                runouts: int = MS2_RUNOUTS,
                equity_iterations: int = 300,
                seed: int = MS2_SEED) -> dict:
    """R2.1 多街 v2 估值（分量化解剖 + 双方节点；不覆盖冻结结果）。"""
    base = compute_ev(state, equity_src, range_src=range_src,
                      opponent_src=opponent_src, seed=seed)
    out: dict[str, Any] = {
        "ms2_version": MS2_VERSION,
        "depth_cap_streets": MS2_DEPTH_CAP_STREETS,
        "villain_bet_frac": MS2_VILLAIN_BET_FRAC,
        "seed": seed, "runouts_per_branch": runouts,
        "assumptions": [
            "R2.1-A1 节点 T+1 的 hero 应对由**冻结闭包**给出（单街核，非 solver）",
            "R2.1-A2 对手 T+1 先行动：bet 概率来自真实观测的上下文条件索引；"
            "bet 尺寸为声明式常数（非调参）",
            "R2.1-A3 深度上限 1 条未来街（不递归到更远）",
            "R2.1-A4 仅建模**上下文条件**侵略性；**牌力条件侵略性未建模**"
            "（无证据）——不得据此声称真实 opponent model 已完成",
            "R2.1-A5 runout 仅用未见牌确定性采样（anti-lookahead）",
            "R2.1-A6 单人主对手（多人池交互不建模）",
        ],
        "components_defined_in": "chip_ev（bb）；模式变换见 mode_utility_from_chip",
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

    out["base"] = {
        "best_action": base.best_action, "best_amount_bb": base.best_amount_bb,
        "objective": base.objective, "mode": base.mode, "equity": base.equity,
        "candidates": [{"action": c.action, "amount_bb": c.amount_bb,
                        "ev_bb": round(float(c.ev_bb or 0.0), 6),
                        "cv_bb": (round(float(c.cv_bb), 6)
                                  if c.cv_bb is not None else None),
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
            "candidates": [{"action": c.action, "amount_bb": c.amount_bb,
                            "utility_ss": round(float(
                                c.utility_bb if c.utility_bb is not None
                                else c.ev_bb or 0.0), 6),
                            "immediate_ev": None, "future_realization": None,
                            "aggression_risk": None, "utility_chip_v2": None,
                            "utility_v2": None, "delta": None}
                           for c in base.candidates],
            "best_ss": base.best_action, "best_v2": None,
            "provenance": {"layer": MS2_VERSION, "declined": True,
                           "reason": "; ".join(reasons)},
        })
        return out

    pop_idx = (aggression or {}).get("population")
    pl_idx = ((aggression or {}).get("players") or {}).get(vpid)
    texture = _texture(state_context_board(board))
    p_a = None
    src = None
    if pl_idx is not None:
        est = pl_idx.estimate(next_street, "checked_to", None,
                              villain_position, texture, population=pop_idx)
        if est:
            p_a, src = est["p_aggress"], est["source"]
    if p_a is None and pop_idx is not None:
        est = pop_idx.estimate(next_street, "checked_to", None,
                               villain_position, texture, population=None)
        if est:
            p_a, src = est["p_aggress"], "population"
    if p_a is None:
        out.update({
            "available": False,
            "decline_reason": (f"aggression evidence insufficient for "
                               f"{next_street}/checked_to（player<{30} 且 "
                               f"population<{30}）"),
            "candidates": [{"action": c.action, "amount_bb": c.amount_bb,
                            "utility_ss": round(float(
                                c.utility_bb if c.utility_bb is not None
                                else c.ev_bb or 0.0), 6),
                            "immediate_ev": None, "future_realization": None,
                            "aggression_risk": None, "utility_chip_v2": None,
                            "utility_v2": None, "delta": None}
                           for c in base.candidates],
            "best_ss": base.best_action, "best_v2": None,
            "provenance": {"layer": MS2_VERSION, "declined": True,
                           "reason": "aggression UNKNOWN"},
        })
        return out

    pot_bb = float(state.get("pot") or 0.0) / bb
    hero_stack_bb = float((hero or {}).get("stack") or 0.0) / bb
    to_call_bb = float(state.get("to_call") or 0.0) / bb
    deck = remaining_deck(hero_cards, board)
    picked = sample_runouts(deck, runouts, seed)

    cands: list[dict] = []
    for c in base.candidates:
        ev = float(c.ev_bb or 0.0)
        cv = float(c.cv_bb or 0.0)
        immediate = round(ev - cv, 6)
        util_ss = float(c.utility_bb if c.utility_bb is not None else ev)
        amt = float(c.amount_bb or 0.0)
        entry: dict[str, Any] = {
            "action": c.action, "amount_bb": c.amount_bb,
            "utility_ss": round(util_ss, 6), "immediate_ev": None,
            "frozen_ev_minus_cv_bb": round(ev - cv, 6),
            "frozen_ev_bb": round(ev, 6),
            "cv_closed_form": (round(cv, 6) if c.cv_bb is not None else None),
            "future_realization": None, "aggression_risk": None,
            "utility_chip_v2": None, "utility_v2": None, "delta": None,
            "p_fold": c.p_fold, "leaf": None,
        }
        if c.action == "fold":
            entry.update({"immediate_ev": 0.0, "future_realization": 0.0,
                          "aggression_risk": 0.0,
                          "utility_chip_v2": 0.0,
                          "leaf": {"terminal": "fold EV=0"}})
            tf = mode_utility_from_chip(state, 0.0, c, seed)
            entry["utility_v2"] = tf.get("utility")
            entry["delta"] = (round(entry["utility_v2"] - util_ss, 6)
                              if entry["utility_v2"] is not None else None)
            entry["mode_transform"] = {k: tf.get(k) for k in
                                       ("k_p", "k_m", "bounty_util", "error")}
            cands.append(entry)
            continue

        # 到达节点 T+1 的投入（沉没成本桥）
        if c.action == "call":
            invest = min(amt, hero_stack_bb)
            pot_node = pot_bb + invest
        elif c.action == "check":
            invest = 0.0
            pot_node = pot_bb
        else:                                     # bet / raise：被跟分支
            p_f = c.p_fold
            if p_f is None:
                entry["leaf"] = {"declined": "fold probability unavailable"}
                cands.append(entry)
                continue
            invest = amt
            pot_node = (pot_bb + 2.0 * amt if c.action == "bet"
                        else pot_bb + amt + max(0.0, amt - to_call_bb))
        stack_node = hero_stack_bb - invest
        if stack_node < -1e-9:
            entry["leaf"] = {"declined": "hero stack exceeded"}
            cands.append(entry)
            continue

        v_actual: list[float] = []
        v_check_only: list[float] = []
        for card in picked:
            eq_c = child_equity(hero_cards, board + [card], vdist,
                                equity_iterations, seed)
            if eq_c is None:
                continue
            # 分支 B：对手 check → hero 行动（to_call=0）
            st_chk = _child(state, street=next_street, board=board + [card],
                            pot=pot_node * bb, hero_stack=stack_node * bb,
                            to_call=0.0, bb=bb)
            chk_vals = _chip_values(st_chk, eq_c, vdist, vpid, seed)
            if not chk_vals:
                continue
            v_chk = max(v for v, _ in chk_vals.values())
            # 分支 A：对手 bet（声明尺寸，封顶对手栈）→ hero 应对
            villain_stack_bb = max((float(p.get("stack") or 0.0)
                                    for p in state.get("players") or []
                                    if isinstance(p, dict)
                                    and str(p.get("player_id")) == vpid),
                                   default=0.0) / bb
            bet_size = min(MS2_VILLAIN_BET_FRAC * pot_node,
                           villain_stack_bb if villain_stack_bb > 0 else
                           MS2_VILLAIN_BET_FRAC * pot_node)
            st_bet = _child(state, street=next_street, board=board + [card],
                            pot=(pot_node + bet_size) * bb,
                            hero_stack=stack_node * bb, to_call=bet_size * bb,
                            bb=bb)
            bet_vals = _chip_values(st_bet, eq_c, vdist, vpid, seed)
            if not bet_vals:
                continue
            v_faced = max(v for v, _ in bet_vals.values())
            v_actual.append(p_a * v_faced + (1.0 - p_a) * v_chk - invest)
            v_check_only.append(v_chk - invest)
        if not v_actual:
            entry["leaf"] = {"declined": "no node evaluation available"}
            cands.append(entry)
            continue

        v_act = sum(v_actual) / len(v_actual)
        v_chk_only = sum(v_check_only) / len(v_check_only)
        # ── 互斥分解（避免双重计数）──────────────────────────────────
        #   immediate_terminal：现在终结的价值
        #     fold=0；call/check=0（跟注/过牌不终结，价值全在节点）
        #     bet/raise = p_f×P（对手弃牌分支终局）
        #   actual_future：真实节点价值（含对手侵略性）
        #   check_future ：对手恒 check 反事实
        if c.action in ("bet", "raise"):
            p_c = (float(c.p_continue) if getattr(c, "p_continue", None)
                   is not None else None)
            w = p_c if p_c is not None else 1.0 - float(c.p_fold or 0.0)
            immediate_terminal = round(float(c.p_fold or 0.0) * pot_bb, 6)
            actual_future = v_act * w
            check_future = v_chk_only * w
        else:
            immediate_terminal = 0.0
            actual_future = v_act
            check_future = v_chk_only
        fut = round(check_future, 6)
        risk = round(check_future - actual_future, 6)
        chip_v2 = round(immediate_terminal + actual_future, 6)
        tf = mode_utility_from_chip(state, chip_v2, c, seed)
        err = tf.get("error")
        entry["immediate_ev"] = immediate_terminal
        entry.update({
            "future_realization": fut, "aggression_risk": risk,
            "utility_chip_v2": chip_v2,
            "utility_v2": tf.get("utility"), "mode_transform_error": err,
            "leaf": {"n_runouts": len(v_actual), "next_street": next_street,
                     "p_aggress": round(p_a, 6), "aggression_source": src,
                     "villain_bet_size_bb": round(bet_size, 6),
                     "hero_investment_bb": round(invest, 6),
                     "actual_future_bb": round(actual_future, 6),
                     "check_future_bb": round(check_future, 6)},
        })
        if entry["utility_v2"] is not None:
            entry["delta"] = round(entry["utility_v2"] - util_ss, 6)
        cands.append(entry)

    scored = [e for e in cands if e.get("utility_v2") is not None]
    best_v2 = max(scored, key=lambda e: e["utility_v2"])["action"] if scored \
        else None
    out.update({
        "available": True, "decline_reason": None,
        "candidates": cands,
        "best_ss": base.best_action, "best_v2": best_v2,
        "p_aggress": round(p_a, 6), "aggression_source": src,
        "provenance": {
            "layer": MS2_VERSION, "seed": seed,
            "runouts_per_branch": len(picked),
            "villain_position": villain_position,
            "aggression_source": src, "p_aggress": round(p_a, 6),
            "future_cards": "sampled_unseen_only（anti-lookahead）",
            "mode_space": ("chip_ev" if str(state.get("game_type")) == "cash"
                           else "gain×k+ − loss×k-(+bounty)"),
            "frozen_core_untouched": True,
            "solver": "not used（declarative；非 GTO）",
            "strength_conditioned_aggression": "NOT MODELED（无证据）",
        },
    })
    return out


def state_context_board(board: list) -> list:
    return list(board)
