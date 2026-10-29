"""R2 — 多街前瞻估值层（opt-in；冻结核心零修改）。

问题（R1/P55 实证）：
  冻结单街核 `compute_ev`（A1）对 call/check 只给「本街终结 + 闭式延续价值
  （A8 realization / A9 implied odds / A10 initiative）」，**没有未来街节点**：
  set mining、implied odds、多街价值无法被表达 → `call` 一致率 ≈0.7%、
  REAL 臂 96–97% 偏向 bet（结构性天花板）。

R2 方案（最小、可回归）：
  在**不修改** `compute_ev` 的前提下，对冻结结果的每个候选做**下一街前瞻**：

      候选动作 → 对手响应（fold 概率取自冻结 fold 通道；被跟则进入下一街）
        → 街推进：未见牌 → 确定性采样 runout（**禁用真实未来牌**）
        → 叶子估值 = 冻结 `compute_ev`(子状态) 的最优 utility
      聚合 → `utility_ms`；与冻结 `utility_ss` 并列输出（旧指标不覆盖）

硬约束（与验收条款一一对应）：
  1. **anti-lookahead**：只用未见牌采样；实际 runout 置换不改变输出（测试覆盖）。
  2. **UNKNOWN-first**：缺 hero 手牌 / board / 对手范围 / equity，或已到 river
     （无未来街）→ 不进前瞻，返回 `available=False` + 原因；调用方回退冻结单街结果。
     任何情况下**不猜测**。
  3. **三模式独立**：叶子一律经冻结 `compute_ev` → objective 自动继承
     （CASH `chip_ev` / MTT `chip_ev+icm` / PKO `chip_ev+icm+bounty`）；
     本层**不注入** ICM/bounty/exploit 任何项。
  4. **确定性**：固定 seed、固定 runout 采样顺序、无 wall-clock。
  5. **不接 Solver / 不伪造 GTO**：声明式近似（declarative），叶子即冻结单街核。
  6. **合法性继承**：候选动作与尺寸全部来自冻结核输出（本层只重估 utility，
     不新增动作、不放大尺寸），并做栈/合法性复核。
"""
from __future__ import annotations

import random
from typing import Any

from .domain.poker.strategy_core import compute_ev

MS_VERSION = "r2-multistreet-v1"
MS_LEAF_MODEL = "frozen-compute_ev（单街闭包）"
MS_MAX_LOOKAHEAD_STREETS = 1        # 最小实现：向前看 1 条街（flop→turn / turn→river）
MS_RUNOUT_SAMPLES = 12              # 每个分支确定性采样 runout 数
MS_CHILD_EQUITY_ITERATIONS = 400    # 子街 equity（vs 对手范围）迭代数
MS_SEED = 42
_FUTURE_STREETS = {"flop": "turn", "turn": "river"}
_RANKS = "23456789TJQKA"
_SUITS = "shdc"


def _norm(card: str) -> str:
    return str(card).strip().capitalize()


def remaining_deck(hero_cards: list, board: list) -> list[str]:
    """未见牌（排除 hero 手牌与公共牌；对手手牌未知 → 保留在牌堆中）。"""
    known = {_norm(c) for c in list(hero_cards) + list(board)}
    return [r + s for r in _RANKS for s in _SUITS if (r + s) not in known]


def sample_runouts(deck: list[str], k: int, seed: int = MS_SEED) -> list[str]:
    """确定性 runout 采样（排序后随机选取 → 与输入顺序无关）。"""
    if k <= 0 or not deck:
        return []
    pool = sorted(deck)
    rng = random.Random(seed)
    k = min(k, len(pool))
    return sorted(rng.sample(pool, k))


def _hero_of(state: dict) -> dict | None:
    return next((p for p in state.get("players") or []
                 if isinstance(p, dict) and p.get("is_hero")), None)


def _child_state(state: dict, *, street: str, board: list, pot_bb_new: float,
                 hero_stack_bb_new: float, bb: float) -> dict:
    """构造下一街子状态（to_call=0；hero 视角；对手栈同步扣除）。"""
    st = {k: v for k, v in state.items() if k != "tournament"}
    if isinstance(state.get("tournament"), dict):
        st["tournament"] = dict(state["tournament"])
    st["street"] = street
    st["board"] = list(board)
    st["to_call"] = 0.0
    st["pot"] = round(pot_bb_new * bb, 6)
    players = []
    for p in state.get("players") or []:
        if not isinstance(p, dict):
            continue
        q = dict(p)
        if p.get("is_hero"):
            q["stack"] = round(max(0.0, hero_stack_bb_new) * bb, 6)
        players.append(q)
    st["players"] = players
    st.pop("min_raise_to", None)
    return st


def _child_equity(hero_cards: list, board: list, villain_dist: dict,
                  iterations: int, seed: int) -> float | None:
    """子街 hero equity（vs 对手范围；缺失 → None，不猜测）。"""
    if not villain_dist:
        return None
    from .domain.poker.equity import EquityCalculator
    from .domain.poker.ranges import Range, parse_range
    try:
        rng_obj = Range(name="ms_from_dist")
        for name, w in dict(villain_dist).items():
            try:
                sub = parse_range(str(name))
            except ValueError:
                continue
            for combo in sub.weights:
                rng_obj.weights[combo] = rng_obj.weights.get(combo, 0.0) + w
        if not rng_obj.weights:
            return None
    except Exception:
        return None
    calc = EquityCalculator(iterations=iterations, seed=seed)
    try:
        er = calc.hand_vs_range(hero_cards, board, rng_obj,
                                iterations=iterations, seed=seed)
    except Exception:
        return None
    eq = getattr(er, "hero_equity", None)
    return float(eq) if eq is not None else None


def _leaf_value(state_leaf: dict, eq: float, villain_dist: dict,
                villain_pid: str | None, seed: int,
                leaf_mode: str = "best_action") -> dict | None:
    """叶子估值。

    leaf_mode:
      "best_action"（默认）——叶子 = 冻结 compute_ev 的最优动作 utility。
        注意：该口径**继承冻结核的既有偏置**（P55 实证 REAL 臂 ~96% 偏 bet），
        前瞻一步会放大之——报告须明示。
      "checkdown"——叶子 = 冻结核 check 候选 utility（= eq×pot，摊牌实现值），
        仅隔离「未来街实现权益」而不假设 hero 能在下一街盈利下注。用于
        区分「隐含赔率实现值」与「下一街再下注的期权价值」。
    """
    eq_src = {"hero_equity": eq,
              "outcome": {"win": eq, "tie": 0.0, "lose": round(1 - eq, 6)}}
    res = compute_ev(state_leaf, eq_src,
                     range_src=({"villains": {villain_pid: {
                         "distribution": dict(villain_dist)}}}
                         if villain_pid else None),
                     seed=seed)
    if not res.available or not res.candidates:
        return None
    if leaf_mode == "checkdown":
        chk = next((c for c in res.candidates if c.action == "check"), None)
        if chk is None:
            return None
        u = float(chk.utility_bb if chk.utility_bb is not None else chk.ev_bb)
        return {"action": "check(checkdown)", "utility_bb": round(u, 6),
                "check_utility_bb": round(u, 6),
                "objective": res.objective, "mode": res.mode,
                "ranking": [{"action": "check", "utility_bb": round(u, 6)}]}
    top = res.candidates[0]
    chk = next((c for c in res.candidates if c.action == "check"), None)
    chk_u = (float(chk.utility_bb if chk.utility_bb is not None else chk.ev_bb)
             if chk is not None else None)
    return {"action": top.action, "utility_bb": round(float(
        top.utility_bb if top.utility_bb is not None else top.ev_bb), 6),
        "check_utility_bb": (round(chk_u, 6) if chk_u is not None else None),
        "objective": res.objective, "mode": res.mode,
        "ranking": [{"action": c.action,
                     "utility_bb": round(float(
                         c.utility_bb if c.utility_bb is not None else c.ev_bb), 6)}
                    for c in res.candidates]}


def _leaf_best_utility(state_leaf: dict, eq: float, villain_dist: dict,
                       villain_pid: str | None, seed: int) -> dict | None:
    """向后兼容包装：叶子 = 冻结 compute_ev 最优动作（leaf_mode=best_action）。"""
    return _leaf_value(state_leaf, eq, villain_dist, villain_pid, seed,
                       leaf_mode="best_action")


def multistreet_evaluate(state: dict, equity_src: dict,
                         range_src: dict | None = None,
                         opponent_src: dict | None = None, *,
                         runouts: int = MS_RUNOUT_SAMPLES,
                         child_equity_iterations: int = MS_CHILD_EQUITY_ITERATIONS,
                         seed: int = MS_SEED) -> dict:
    """冻结单街结果 + 下一街前瞻估值（并列；不覆盖旧指标）。

    返回 dict：
      available / decline_reason：UNKNOWN-first 门
      base：冻结 compute_ev 结果摘要（utility_ss 来源）
      candidates：每候选 {action, amount_bb, utility_ss, utility_ms, delta, leaf…}
      best_ss / best_ms：两套口径的最优动作
      comparison：{changed, deltas, call/check 专项}
      provenance / assumptions / mode_isolation
    """
    base = compute_ev(state, equity_src, range_src=range_src,
                      opponent_src=opponent_src, seed=seed)
    out: dict[str, Any] = {
        "ms_version": MS_VERSION,
        "leaf_model": MS_LEAF_MODEL,
        "seed": seed, "runouts_per_branch": runouts,
        "assumptions": [
            "R2-A1 叶子估值 = 冻结单街核（declarative approximation，非 solver、非 GTO）",
            "R2-A2 对手在下一街的主动下注未建模（仅本街响应 fold/call 分支）",
            "R2-A3 runout 为未见牌确定性采样（禁用真实未来牌，anti-lookahead）",
            "R2-A4 单人主对手（多人池交互仍不建模，与 A4 一致）",
        ],
        "objective_inherited": True,
    }
    if not base.available:
        out.update({"available": False,
                    "decline_reason": f"frozen core unavailable: "
                                      f"{base.unavailable_reason}",
                    "base": None, "candidates": [],
                    "best_ss": None, "best_ms": None,
                    "best_ms_checkdown": None,
                    "comparison": {"changed": False,
                                   "changed_checkdown": False,
                                   "reason": "frozen core unavailable"}})
        return out

    street = str(state.get("street") or "")
    next_street = _FUTURE_STREETS.get(street)
    hero = _hero_of(state)
    hero_cards = list((hero or {}).get("hole_cards") or [])
    board = [str(c) for c in (state.get("board") or [])]
    bb = float(state.get("bb") or 0.0)
    villain_pid = str(range_src and next(iter((range_src.get("villains") or {})
                                              or {}), "") or "")
    vdist = dict(((range_src or {}).get("villains") or {})
                 .get(villain_pid, {}).get("distribution") or {}) \
        if villain_pid else {}

    out["base"] = {
        "best_action": base.best_action,
        "best_amount_bb": base.best_amount_bb,
        "objective": base.objective, "mode": base.mode,
        "equity": base.equity,
        "candidates": [{"action": c.action, "amount_bb": c.amount_bb,
                        "utility_ss": round(float(
                            c.utility_bb if c.utility_bb is not None else c.ev_bb), 6),
                        "p_fold": c.p_fold}
                       for c in base.candidates],
    }

    # ---- UNKNOWN-first 门（任何一项缺失 → 不进前瞻，不猜测） ----
    reasons: list[str] = []
    if next_street is None:
        reasons.append(f"no future street from {street!r}（river/未知街）")
    if len(hero_cards) != 2:
        reasons.append("hero hole cards missing")
    if not villain_pid or not vdist:
        reasons.append("villain range/distribution missing")
    if bb <= 0:
        reasons.append("bb<=0")
    if reasons:
        out.update({"available": False,
                    "decline_reason": "; ".join(reasons),
                    "candidates": [
                        {"action": c.action, "amount_bb": c.amount_bb,
                         "utility_ss": round(float(
                             c.utility_bb if c.utility_bb is not None else c.ev_bb), 6),
                         "utility_ms": None, "delta": None,
                         "leaf": None}
                        for c in base.candidates],
                    "best_ss": base.best_action, "best_ms": None,
                    "best_ms_checkdown": None,
                    "comparison": {"changed": False, "changed_checkdown": False,
                                   "reason": "; ".join(reasons)}})
        return out

    pot_bb = float(state.get("pot") or 0.0) / bb
    hero_stack_bb = float((hero or {}).get("stack") or 0.0) / bb
    to_call_bb = float(state.get("to_call") or 0.0) / bb
    deck = remaining_deck(hero_cards, board)
    picked = sample_runouts(deck, runouts, seed)

    cand_out: list[dict] = []
    for c in base.candidates:
        u_ss = float(c.utility_bb if c.utility_bb is not None else c.ev_bb)
        amt = float(c.amount_bb or 0.0)
        entry: dict[str, Any] = {
            "action": c.action, "amount_bb": c.amount_bb,
            "utility_ss": round(u_ss, 6), "p_fold": c.p_fold,
            "utility_ms": None, "delta": None, "leaf": None,
            "branches": None,
        }
        if c.action == "fold":
            entry.update({"utility_ms": 0.0, "delta": round(0.0 - u_ss, 6),
                          "leaf": {"terminal": "fold EV=0"}})
            cand_out.append(entry)
            continue
        if c.action in ("check", "call"):
            pot_new = pot_bb + (amt if c.action == "call" else 0.0)
            stack_new = hero_stack_bb - (amt if c.action == "call" else 0.0)
            if stack_new < -1e-9:
                entry["leaf"] = {"declined": "hero stack exceeded"}
                cand_out.append(entry)
                continue
            vals: list[float] = []
            vals_chk: list[float] = []
            for card in picked:
                eq_c = _child_equity(hero_cards, board + [card], vdist,
                                     child_equity_iterations, seed)
                if eq_c is None:
                    continue
                st_leaf = _child_state(state, street=next_street,
                                       board=board + [card],
                                       pot_bb_new=pot_new,
                                       hero_stack_bb_new=stack_new, bb=bb)
                leaf = _leaf_best_utility(st_leaf, eq_c, vdist, villain_pid,
                                          seed)
                if leaf is not None:
                    vals.append(leaf["utility_bb"])
                    if leaf.get("check_utility_bb") is not None:
                        vals_chk.append(leaf["check_utility_bb"])
            if not vals:
                entry["leaf"] = {"declined": "no leaf evaluation available"}
                cand_out.append(entry)
                continue
            # 记账桥：叶子 utility 相对「叶子节点」计数（含已投入），
            # 本决策点净 EV = 叶子 utility − 到达叶子的投入 x（沉没成本）
            invest = amt if c.action == "call" else 0.0
            u_ms = sum(vals) / len(vals) - invest
            u_ms_chk = (sum(vals_chk) / len(vals_chk) - invest
                        if vals_chk else None)
            entry.update({
                "utility_ms": round(u_ms, 6), "delta": round(u_ms - u_ss, 6),
                "utility_ms_checkdown": (round(u_ms_chk, 6)
                                         if u_ms_chk is not None else None),
                "delta_checkdown": (round(u_ms_chk - u_ss, 6)
                                    if u_ms_chk is not None else None),
                "leaf": {"n_runouts": len(vals),
                         "mean_leaf_utility_bb": round(sum(vals) / len(vals), 6),
                         "hero_investment_bb": round(invest, 6),
                         "min_utility_bb": round(min(vals) - invest, 6),
                         "max_utility_bb": round(max(vals) - invest, 6),
                         "next_street": next_street},
            })
            cand_out.append(entry)
            continue
        if c.action in ("bet", "raise"):
            # 本街 fold 分支（终局：赢得当前 pot）与被跟分支（进入下一街）
            p_f = c.p_fold
            if p_f is None:
                entry["leaf"] = {"declined": "fold probability unavailable "
                                           "(always-called fallback)"}
                cand_out.append(entry)
                continue
            if c.action == "bet":
                hero_cost = amt                      # 下注额 = hero 投入
                pot_after_call = pot_bb + 2.0 * amt  # hero + 对手跟注
            else:                                    # raise：amt = raise-to 总额
                hero_cost = amt
                pot_after_call = pot_bb + amt + max(0.0, amt - to_call_bb)
            stack_new = hero_stack_bb - hero_cost
            if stack_new < -1e-9:
                entry["leaf"] = {"declined": "hero stack exceeded"}
                cand_out.append(entry)
                continue
            vals = []
            vals_chk = []
            for card in picked:
                eq_c = _child_equity(hero_cards, board + [card], vdist,
                                     child_equity_iterations, seed)
                if eq_c is None:
                    continue
                st_leaf = _child_state(state, street=next_street,
                                       board=board + [card],
                                       pot_bb_new=pot_after_call,
                                       hero_stack_bb_new=stack_new, bb=bb)
                leaf = _leaf_best_utility(st_leaf, eq_c, vdist, villain_pid,
                                          seed)
                if leaf is not None:
                    vals.append(leaf["utility_bb"])
                    if leaf.get("check_utility_bb") is not None:
                        vals_chk.append(leaf["check_utility_bb"])
            if not vals:
                entry["leaf"] = {"declined": "no leaf evaluation available"}
                cand_out.append(entry)
                continue
            u_leaf = sum(vals) / len(vals) - hero_cost   # 记账桥（扣除投入）
            u_ms = float(p_f) * pot_bb + (1.0 - float(p_f)) * u_leaf
            if vals_chk:
                u_leaf_chk = sum(vals_chk) / len(vals_chk) - hero_cost
                u_ms_chk = float(p_f) * pot_bb + (1.0 - float(p_f)) * u_leaf_chk
            else:
                u_ms_chk = None
            entry.update({
                "utility_ms": round(u_ms, 6), "delta": round(u_ms - u_ss, 6),
                "utility_ms_checkdown": (round(u_ms_chk, 6)
                                         if u_ms_chk is not None else None),
                "delta_checkdown": (round(u_ms_chk - u_ss, 6)
                                    if u_ms_chk is not None else None),
                "leaf": {"n_runouts": len(vals),
                         "fold_branch_utility_bb": round(pot_bb, 6),
                         "called_leaf_mean_bb": round(sum(vals) / len(vals), 6),
                         "hero_investment_bb": round(hero_cost, 6),
                         "net_called_leaf_bb": round(u_leaf, 6),
                         "called_leaf_min_bb": round(min(vals) - hero_cost, 6),
                         "called_leaf_max_bb": round(max(vals) - hero_cost, 6),
                         "next_street": next_street},
                "branches": {"fold_prob": float(p_f),
                             "call_prob": round(1.0 - float(p_f), 6)},
            })
            cand_out.append(entry)
            continue
        entry["leaf"] = {"declined": f"unknown action {c.action!r}"}
        cand_out.append(entry)

    scored = [e for e in cand_out if e["utility_ms"] is not None]
    best_ms = max(scored, key=lambda e: e["utility_ms"])["action"] if scored \
        else None
    scored_chk = [e for e in cand_out
                  if e.get("utility_ms_checkdown") is not None]
    best_ms_chk = (max(scored_chk,
                       key=lambda e: e["utility_ms_checkdown"])["action"]
                   if scored_chk else None)
    best_ss = base.best_action
    out.update({
        "available": True, "decline_reason": None,
        "candidates": cand_out,
        "best_ss": best_ss, "best_ms": best_ms,
        "best_ms_checkdown": best_ms_chk,
        "comparison": {
            "changed": bool(best_ms) and best_ms != best_ss,
            "changed_checkdown": (bool(best_ms_chk)
                                  and best_ms_chk != best_ss),
            "best_ss": best_ss, "best_ms": best_ms,
            "best_ms_checkdown": best_ms_chk,
            "deltas": {e["action"]: e["delta"] for e in cand_out
                       if e["delta"] is not None},
            "deltas_checkdown": {e["action"]: e["delta_checkdown"]
                                 for e in cand_out
                                 if e.get("delta_checkdown") is not None},
            "call_delta": next((e["delta"] for e in cand_out
                                if e["action"] == "call"), None),
            "check_delta": next((e["delta"] for e in cand_out
                                 if e["action"] == "check"), None),
            "call_delta_checkdown": next(
                (e.get("delta_checkdown") for e in cand_out
                 if e["action"] == "call"), None),
            "check_delta_checkdown": next(
                (e.get("delta_checkdown") for e in cand_out
                 if e["action"] == "check"), None),
        },
        "provenance": {
            "layer": MS_VERSION, "leaf_model": MS_LEAF_MODEL,
            "leaf_variants": ["best_action（继承冻结核偏置）",
                              "checkdown（仅实现权益，隔离下注期权价值）"],
            "runouts_per_branch": len(picked), "seed": seed,
            "future_cards": "sampled_unseen_only（anti-lookahead）",
            "frozen_core_untouched": True,
            "solver": "not used（declarative approximation）",
        },
    })
    return out
