"""PI-EV 性能 profiling + 并行模型测量（**只读性能审计**；无验收语义）。

自我约束（代码内可验证）：
  * 不写任何封存路径（`docs/V3F_S3_*`）：计时/per-point 输出只落 `.tmp/`；
  * 不调用 `run_s3(...)`；不改 seed / budget / decision set / 契约 / roster；
  * 不减少枚举精度（PI 参照仍是精确全枚举）；
  * instrumented 副本与 sealed 原实现**逐字段等值断言**（独立等价 pass）；
  * profiling 切片 = **手序 index < 1500**（S/S′/S″/S‴ 全部使用 ≥1500），
    故不触碰任何阶段的验收样本，也不读取任何阶段的结果文件。

并行模型的合法性前提（本模块用 `prove_order_independence()` 实测）：
  每个 decision point 的全部随机通道（`hand_vs_range` / `child_equity` /
  `sample_runouts` / `compute_ev` 折叠通道）都用**固定常量种子**在调用内部
  新建 `random.Random(seed)`，不共享全局 RNG → 逐点结果与执行顺序/进程无关。
"""
from __future__ import annotations

import json
import math
import os
import time
from collections import Counter
from contextlib import contextmanager
from itertools import combinations

from ..domain.poker.cards import parse_card
from ..domain.poker.equity import EquityCalculator
from ..domain.poker.evaluator import evaluate7
from ..domain.poker.strategy_core import _hero_in_position, compute_ev
from ..multistreet_v3 import evaluate_v3
from ..pluribus_decision_adapter import _positions_for_seats
from ..pluribus_validation import (AI_SEAT, _current_aggressor, _load_config,
                                   _profile, _route_of, _state_for,
                                   _villain_range_arm, legal_action_types)
from .pi_ev_contract import UnitContractError, pi_ev_inputs, sanity_check
from .unfreeze_core import PI_EV_MAX_RUNOUTS, PiEvReference, compute_ev_variant

PERF_VERSION = "pi-ev-perf-audit-v2"
PROFILE_HAND_MAX_INDEX = 1500          # profiling 切片（任何阶段均未使用）
_RANKS = "23456789TJQKA"
_SUITS = "shdc"


class Timers:
    def __init__(self) -> None:
        self.wall: Counter = Counter()
        self.calls: Counter = Counter()
        self.counts: Counter = Counter()

    @contextmanager
    def t(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.wall[name] += time.perf_counter() - t0
            self.calls[name] += 1

    def bump(self, name: str, n: int = 1) -> None:
        self.counts[name] += n

    def as_dict(self) -> dict:
        return {"wall_s": {k: round(v, 4) for k, v in self.wall.items()},
                "calls": dict(self.calls), "counts": dict(self.counts)}

    def merge(self, other: dict) -> None:
        for k, v in (other.get("wall_s") or {}).items():
            self.wall[k] += v
        for k, v in (other.get("calls") or {}).items():
            self.calls[k] += v
        for k, v in (other.get("counts") or {}).items():
            self.counts[k] += v


# ── 忠实副本：`_pi_equity`（拆分 deck/枚举 与 showdown） ───────────────
def _deck_excluding(known: set) -> list:
    return [r + s for r in _RANKS for s in _SUITS if (r + s) not in known]


def pi_equity_profiled(hero: list, others: list, board: list,
                       timers: Timers | None = None):
    """逐字复制 sealed `unfreeze_core._pi_equity`，仅插入分段计时。"""
    known = {c for c in hero + board}
    for o in others:
        known |= set(o)
    t0 = time.perf_counter()
    deck = [c for c in _deck_excluding(known) if c not in board]
    need = 5 - len(board)
    if need < 0 or not others:
        if timers:
            timers.wall["pi_deck_enum"] += time.perf_counter() - t0
        return None, 0
    runouts = list(combinations(deck, need)) if need else [()]
    n_run = len(runouts)
    if timers:
        timers.wall["pi_deck_enum"] += time.perf_counter() - t0
        timers.bump("pi_runouts", n_run)
        timers.bump("pi_showdown_units", n_run * (1 + len(others)))
    if n_run > PI_EV_MAX_RUNOUTS:
        return None, n_run
    t1 = time.perf_counter()
    tot = 0.0
    ev7 = evaluate7
    card = parse_card
    for ro in runouts:
        full = board + list(ro)
        hs = ev7([card(c) for c in hero + full])
        best_other = max(ev7([card(c) for c in o + full]) for o in others)
        if hs > best_other:
            tot += 1.0
        elif hs == best_other:
            tot += 0.5
    if timers:
        timers.wall["pi_showdown"] += time.perf_counter() - t1
    return tot / len(runouts), len(runouts)


def pi_reference_profiled(*, hero_cards, villain_hands, board, pot_bb,
                          to_call_bb, hero_stack_bb, candidates,
                          rule: str = "villain_calls",
                          timers: Timers | None = None) -> dict:
    """逐字复制 sealed `PiEvReference.reference` 的输出构造（含 provenance）。"""
    eq, n_run = pi_equity_profiled(list(hero_cards),
                                   [list(v) for v in villain_hands],
                                   [str(c) for c in board], timers)
    prov = {"kind": "pi_ev", "evidence_level": "L2",
            "reference": "perfect-information", "equilibrium": False,
            "rule": f"checkdown_{rule}", "runouts_enumerated": n_run,
            "NOT_exact_gto": True}
    if eq is None:
        return {"available": False,
                "unavailable_reason": ("runout budget exceeded or no "
                                       "active opponent"),
                "provenance": prov, "values": {}, "ordering": []}
    t0 = time.perf_counter()
    vals: dict = {}
    for c in candidates:
        act, amt = str(c["action"]), float(c.get("amount_bb") or 0.0)
        if act == "fold":
            v = 0.0
        elif act == "check":
            v = eq * pot_bb
        elif act == "call":
            s = min(to_call_bb, hero_stack_bb)
            v = eq * pot_bb - (1.0 - eq) * s
        elif act == "bet":
            v = pot_bb if rule == "villain_folds" else (
                eq * (pot_bb + amt) - (1.0 - eq) * amt)
        elif act == "raise":
            v = pot_bb if rule == "villain_folds" else (
                eq * (pot_bb + max(0.0, amt - to_call_bb)) - (1.0 - eq) * amt)
        else:
            continue
        vals[act] = {"value_bb": round(v, 6), "pi_equity": round(eq, 6),
                     "n_runouts": n_run, "ci95": None}
    ordering = [k for k, _ in sorted(vals.items(),
                                     key=lambda kv: -kv[1]["value_bb"])]
    if timers:
        timers.wall["pi_candidates"] += time.perf_counter() - t0
    return {"available": True, "unavailable_reason": None, "values": vals,
            "ordering": ordering, "provenance": prov,
            "pi_equity": round(eq, 6), "n_runouts": n_run}


# ── 等价性（instrumented / optimized vs sealed） ──────────────────────
class EquivalenceError(AssertionError):
    pass


def equivalence_check(fixtures: list) -> dict:
    bad = 0
    detail = []
    for f in fixtures:
        sealed = PiEvReference().reference(
            hero_cards=f["hole"], villain_hands=f["vhands"], board=f["board"],
            pot_bb=f["inputs"]["pot_bb"], to_call_bb=f["inputs"]["to_call_bb"],
            hero_stack_bb=f["inputs"]["hero_stack_bb"], candidates=f["cands"])
        prof = pi_reference_profiled(
            hero_cards=f["hole"], villain_hands=f["vhands"], board=f["board"],
            pot_bb=f["inputs"]["pot_bb"], to_call_bb=f["inputs"]["to_call_bb"],
            hero_stack_bb=f["inputs"]["hero_stack_bb"], candidates=f["cands"])
        if sealed != prof:
            bad += 1
            if len(detail) < 5:
                detail.append({"hand_id": f.get("hand_id"),
                               "street": f.get("street")})
    return {"checked": len(fixtures), "mismatches": bad, "detail": detail,
            "basis": "PiEvReference.reference == instrumented 副本（逐字段）"}


# ── 核心：逐点循环（runner 调用序；无判定、不落封存路径） ──────────────
def run_points(scope: list, arm: str, max_points: int, *, idx, agg, cfg=None,
               timers: Timers | None = None, fixture_cap: int = 0,
               optimized: bool = False, case_cap: int = 0,
               full_rows: bool = True, attempt_offset: int = 0,
               attempt_window: tuple | None = None) -> dict:
    """逐点循环。

    `attempt_window=(start, end)`：只处理**全局 attempt 序号** ∈ [start,end)
    的决策点（`attempt_offset` = 本 scope 首个 attempt 的全局序号）。用于让
    进程池与单进程处理**完全相同的 attempt 前缀**，从而使
    attempted/coverage/legality/valid 等门语义逐位可比。
    """
    from .s3_validation import _adv
    if optimized:
        # v2 接入集：**仅 S1（combo_equities 不可变记忆化）+ S2（child_equity）**
        # —— 不替换 `compute_ev`（S4：实测 0 命中且返回可变对象 → 禁止共享）。
        from .perf_optimize import evaluate_v3_s1s2 as _eval_v3

        _compute_ev = compute_ev

        def _pi_call(**kw):
            return pi_reference_profiled(timers=tm, **kw)
    else:
        _compute_ev = compute_ev
        _eval_v3 = evaluate_v3

        def _pi_call(**kw):
            return pi_reference_profiled(timers=tm, **kw)

    if cfg is None:
        cfg = _load_config()
    tm = timers if timers is not None else Timers()
    calc = EquityCalculator(iterations=1500, seed=42)
    counters = {"attempted": 0, "unavailable": 0, "instrument_blocked": 0,
                "skipped_hero_or_villain": 0, "skipped_range": 0,
                "skipped_frozen": 0}
    rows: list = []
    fixtures: list = []
    cases: list = []
    point_times: list = []
    runout_per_point: list = []
    n_vill: list = []
    done = 0
    stop_scope = False
    scanned = 0
    for h in scope:
        if done >= max_points or stop_scope:
            break
        bb = h["bb"] or 1.0
        n = len(h["players"])
        pos_map = _positions_for_seats(n, n)
        seat_of = {p: i + 1 for i, p in enumerate(h["players"])}
        folded: set = set()
        hist: list = []
        boards = h["board_by_street"]
        for st in h["streets"]:
            if done >= max_points or stop_scope:
                break
            street = st["street"]
            if street == "preflop":
                board: list = []
            elif street == "flop":
                board = list(boards.get("flop", []))
            elif street == "turn":
                board = list(boards.get("flop", [])) + list(
                    boards.get("turn", []))
            else:
                board = (list(boards.get("flop", []))
                         + list(boards.get("turn", []))
                         + list(boards.get("river", [])))
            record = street in ("flop", "turn")
            for a in st["actions"]:
                if done >= max_points or stop_scope:
                    break
                pid = a["player"]
                if pid == AI_SEAT:
                    if a["action"] in ("bet", "raise", "call"):
                        hist.append({"street": street, "player_id": pid,
                                     "action": a["action"],
                                     "amount": a.get("amount") or 0.0})
                    continue
                if not record:
                    _adv(a, pid, street, folded, hist)
                    continue
                tp0 = time.perf_counter()
                # --- decision point：守卫 / 状态构造 / 单位契约 ---
                t0 = time.perf_counter()
                gidx = attempt_offset + scanned
                scanned += 1
                if attempt_window is not None:
                    if gidx >= attempt_window[1]:
                        stop_scope = True
                        break
                    if gidx < attempt_window[0]:
                        _adv(a, pid, street, folded, hist)
                        continue
                counters["attempted"] += 1
                observed = a["action"] if not (
                    a["action"] == "call"
                    and (a.get("to_call") or 0) <= 1e-9) else "check"
                la = legal_action_types(a.get("to_call") or 0,
                                        a.get("remaining_stack") or 0)
                hole = list(h["hole_cards"].get(pid) or [])
                villain = _current_aggressor(h, street, a)
                if villain is None or villain == pid:
                    villain = next((p for p in h["players"]
                                    if p != pid and p not in folded), None)
                if len(hole) != 2 or villain is None:
                    counters["skipped_hero_or_villain"] += 1
                    _adv(a, pid, street, folded, hist)
                    continue
                vpos = str(pos_map.get(seat_of.get(villain)) or "")
                state = _state_for(h, a, street, board, pos_map, folded, hist,
                                   pid, bb)
                try:
                    inputs = pi_ev_inputs(state, raw=a, hand_bb=bb)
                    sanity_check(inputs, raw=a, hand_bb=bb)
                except UnitContractError:
                    counters["instrument_blocked"] += 1
                    _adv(a, pid, street, folded, hist)
                    continue
                hero = next(p for p in state["players"]
                            if isinstance(p, dict) and p.get("is_hero"))
                hero_ip = _hero_in_position(state, villain)
                tm.wall["decision"] += time.perf_counter() - t0

                # --- villain range（对手手牌范围枚举） ---
                t0 = time.perf_counter()
                vr, _p = _villain_range_arm(
                    cfg, villain=villain, villain_pos=vpos, street=street,
                    route=_route_of(hist, villain, street),
                    dead_cards=hole + list(board), idx=idx, with_evidence=True,
                    hole_cards=hole, board=board)
                if vr is None or not vr.distribution():
                    counters["skipped_range"] += 1
                    _adv(a, pid, street, folded, hist)
                    continue
                dist = vr.distribution()
                tm.wall["villain_range"] += time.perf_counter() - t0

                # --- 范围权益（Monte Carlo 1500 iters） ---
                t0 = time.perf_counter()
                er = calc.hand_vs_range(hole, board, vr, iterations=1500,
                                        seed=42)
                eq, tie = er.hero_equity, er.tie
                eq_src = {"hero_equity": eq,
                          "outcome": {"win": round(eq - tie / 2, 6),
                                      "tie": round(tie, 6),
                                      "lose": round(1 - eq - tie / 2, 6)}}
                tm.wall["range_equity_mc"] += time.perf_counter() - t0

                rs = {"villains": {villain: {"distribution": dist}}}
                prof = _profile(villain, idx)
                osrc = {"players": {villain: prof}} if prof else None

                # --- frozen baseline ---
                t0 = time.perf_counter()
                frozen = _compute_ev(state, eq_src, range_src=rs,
                                     opponent_src=osrc, seed=42)
                tm.wall["frozen"] += time.perf_counter() - t0
                if not frozen.available or not frozen.candidates:
                    counters["skipped_frozen"] += 1
                    _adv(a, pid, street, folded, hist)
                    continue

                # --- PI 参照 ---
                cands = [{"action": c.action, "amount_bb": c.amount_bb}
                         for c in frozen.candidates]
                vhands = []
                for p in h["players"]:
                    if p == AI_SEAT or p == pid or p in folded:
                        continue
                    cards = list(h["hole_cards"].get(p) or [])
                    if len(cards) == 2:
                        vhands.append(cards)
                kw = dict(hero_cards=hole, villain_hands=vhands, board=board,
                          pot_bb=inputs["pot_bb"],
                          to_call_bb=inputs["to_call_bb"],
                          hero_stack_bb=inputs["hero_stack_bb"],
                          candidates=cands)
                t0 = time.perf_counter()
                ref = _pi_call(**kw)
                tm.wall["pi_total"] += time.perf_counter() - t0
                if not ref["available"]:
                    counters["unavailable"] += 1
                    _adv(a, pid, street, folded, hist)
                    continue

                # --- 臂 ---
                t0 = time.perf_counter()
                cand_best = None
                applicable = False
                if arm == "U1":
                    res, meta = compute_ev_variant(state, eq_src, rs, osrc,
                                                   variant="U1", seed=42,
                                                   base=frozen)
                    cand_best = res.best_action if res.available else None
                    applicable = bool(meta["u1_applied"])
                    declined = (not res.available) or not applicable
                    tm.wall["u1"] += time.perf_counter() - t0
                elif arm == "R2.2":
                    r22 = _eval_v3(state, eq_src, range_src=rs,
                                   aggression=agg, villain_id=villain,
                                   villain_position=vpos, runouts=8,
                                   equity_iterations=300, seed=42)
                    cand_best = r22.get("best_v3")
                    applicable = bool(r22.get("available"))
                    declined = not applicable
                    tm.wall["r22"] += time.perf_counter() - t0
                else:
                    declined = False
                    tm.wall["frozen_only"] += time.perf_counter() - t0

                n_run = int(ref.get("n_runouts") or 0)
                runout_per_point.append(n_run)
                n_vill.append(len(vhands))
                if len(cases) < case_cap:
                    cases.append({"state": state, "eq": eq_src, "rs": rs,
                                  "os": osrc, "agg": agg, "vid": villain,
                                  "vpos": vpos})
                if len(fixtures) < fixture_cap:
                    fixtures.append({"hand_id": str(h["hand_id"]),
                                     "street": street, "hole": hole,
                                     "vhands": vhands, "board": list(board),
                                     "inputs": inputs, "cands": cands})
                if full_rows:
                    ratios = inputs.get("ratios_vs_raw") or {}
                    rows.append({
                        "arm": arm, "hand_id": str(h["hand_id"]),
                        "street": street, "observed": observed,
                        "legal": bool(la is not None and observed in la),
                        "hero_ip": bool(hero_ip) if hero_ip is not None
                        else None,
                        "spr": round(inputs["hero_stack_bb"] / inputs["pot_bb"],
                                     6) if inputs["pot_bb"] > 0 else None,
                        "pot_bb": inputs["pot_bb"],
                        "to_call_bb": inputs["to_call_bb"],
                        "hero_stack_bb": inputs["hero_stack_bb"],
                        "raw_pot_before_chips": float(a.get("pot_before")
                                                      or 0.0),
                        "raw_to_call_chips": float(a.get("to_call") or 0.0),
                        "hand_bb_chips": float(bb),
                        "state_pot_bb": round(float(state.get("pot") or 0.0), 9),
                        "state_to_call_bb": round(
                            float(state.get("to_call") or 0.0), 9),
                        "ratio_pot_input_over_raw": ratios.get("pot_bb"),
                        "ratio_stack_input_over_state": round(
                            inputs["hero_stack_bb"]
                            / float(hero.get("stack") or 1.0), 9),
                        "frozen_best": frozen.best_action,
                        "cand_best": cand_best,
                        "cand_applicable": bool(applicable),
                        "cand_declined": bool(declined),
                        "pi_best": ref["ordering"][0],
                        "pi_ordering": ref["ordering"],
                        "pi_equity": round(float(ref["pi_equity"]), 6),
                        "n_runouts": n_run,
                        "position": vpos,
                    })
                else:
                    rows.append({"arm": arm, "hand_id": str(h["hand_id"]),
                                 "street": street, "observed": observed,
                                 "n_runouts": n_run,
                                 "pi_equity": ref.get("pi_equity"),
                                 "pi_best": (ref.get("ordering") or [None])[0],
                                 "frozen_best": frozen.best_action,
                                 "cand_best": cand_best})
                done += 1
                point_times.append(time.perf_counter() - tp0)
                _adv(a, pid, street, folded, hist)
    return {"rows": rows, "fixtures": fixtures, "cases": cases,
            "point_times": point_times, "counters": counters,
            "runout_per_point": runout_per_point, "n_vill": n_vill,
            "timers": tm.as_dict(), "points": done}


def _slice_hands(hands: list) -> list:
    return [h for i, h in enumerate(hands) if i < PROFILE_HAND_MAX_INDEX]


def _runout_stats(runout_per_point: list, n_vill: list) -> dict:
    if not runout_per_point:
        return {}
    return {"points": len(runout_per_point),
            "mean_runouts_per_point": round(sum(runout_per_point)
                                            / len(runout_per_point), 3),
            "max_runouts": max(runout_per_point),
            "min_runouts": min(runout_per_point),
            "mean_villains_per_point": round(sum(n_vill) / len(n_vill), 3),
            "total_runouts": sum(runout_per_point),
            "total_showdown_units": sum(r * v for r, v in
                                        zip(runout_per_point, n_vill))}


# ── 单进程 profiling ─────────────────────────────────────────────────
def profile_arm(hands: list, arm: str, *, max_points: int,
                persist_path=None, fixture_cap: int = 200,
                optimized: bool = False, idx=None, agg=None,
                cfg=None, attempt_window: tuple | None = None) -> dict:
    from .s3_validation import (build_aggression_index_excluding,
                                build_indexes)
    scope = _slice_hands(hands) if idx is None or agg is None else hands
    t_idx = time.perf_counter()
    if idx is None:
        idx = build_indexes(scope, exclude_players=set())
    if agg is None:
        agg = build_aggression_index_excluding(scope, self_exclude=set())
    index_build_s = time.perf_counter() - t_idx
    if cfg is None:
        cfg = _load_config()
    timers = Timers()
    wall0 = time.perf_counter()
    cpu0 = time.process_time()
    out = run_points(scope, arm, max_points, idx=idx, agg=agg, cfg=cfg,
                     timers=timers, fixture_cap=fixture_cap,
                     optimized=optimized, attempt_window=attempt_window)
    wall = time.perf_counter() - wall0
    cpu = time.process_time() - cpu0
    persist_s = 0.0
    if persist_path is not None:
        t0 = time.perf_counter()
        blob = "".join(json.dumps(r, ensure_ascii=False) + "\n"
                       for r in out["rows"])
        with open(persist_path, "w", encoding="utf-8") as fh:
            fh.write(blob)
        persist_s = time.perf_counter() - t0
    return {"perf_version": PERF_VERSION, "arm": arm, "mode": "single-process",
            "optimized": bool(optimized),
            "profile_slice": "hand_index < %d" % PROFILE_HAND_MAX_INDEX,
            "points": out["points"], "wall_s": round(wall, 4),
            "cpu_s": round(cpu, 4), "index_build_s": round(index_build_s, 4),
            "persist_s": round(persist_s, 4), "timers": out["timers"],
            "point_times_s": out["point_times"], "rows": out["rows"],
            "fixtures": out["fixtures"], "counters": out["counters"],
            "cache_stats": _cache_stats(optimized),
            "runout_stats": _runout_stats(out["runout_per_point"],
                                          out["n_vill"])}


def _cache_stats(optimized: bool) -> dict | None:
    if not optimized:
        return None
    try:
        from .perf_optimize import STATS
        return {k: dict(v) for k, v in STATS.items()}
    except Exception:  # noqa: BLE001
        return None


# ── 多进程 profiling（worker 并行；索引/配置由父进程统一构造后下传） ──
def _pool_worker(payload: dict) -> dict:
    hands, arm, cap, idx, agg, cfg = (payload["hands"], payload["arm"],
                                      payload["cap"], payload["idx"],
                                      payload["agg"], payload["cfg"])
    timers = Timers()
    w0 = time.perf_counter()
    c0 = time.process_time()
    out = run_points(hands, arm, cap, idx=idx, agg=agg, cfg=cfg, timers=timers,
                     optimized=bool(payload.get("optimized")),
                     attempt_offset=int(payload.get("attempt_offset") or 0),
                     attempt_window=payload.get("attempt_window"))
    return {"rows": out["rows"], "point_times": out["point_times"],
            "runout_per_point": out["runout_per_point"], "n_vill": out["n_vill"],
            "timers": out["timers"], "points": out["points"],
            "counters": out["counters"],
            "worker_wall_s": time.perf_counter() - w0,
            "worker_cpu_s": time.process_time() - c0}


def _candidate_counts(scope: list) -> list:
    """轻量预扫：每手「attempt 数」= flop/turn 上非 AI 座位的动作数。

    必须与 `run_points` 中 `counters["attempted"]` 的口径**完全一致**
    （attempted 在 hero/villain 守卫**之前**自增），否则进程池的
    attempt 窗口会与真实序号错位。
    """
    from .s3_validation import _adv
    out = []
    for h in scope:
        c = 0
        folded: set = set()
        hist: list = []
        for st in h["streets"]:
            street = st["street"]
            record = street in ("flop", "turn")
            for a in st["actions"]:
                pid = a["player"]
                if record and pid != AI_SEAT:
                    c += 1
                _adv(a, pid, street, folded, hist)
        out.append(c)
    return out


def profile_arm_pool(hands: list, arm: str, *, max_points: int,
                     workers: int, fixture_cap: int = 0,
                     optimized: bool = False, idx=None, agg=None,
                     cfg=None, attempt_window: tuple | None = None) -> dict:
    """手级分片的进程池并行；每点结果必须与单进程逐字段相同。

    分片策略：按「attempt 序号」把 `attempt_window` 切成 W 个**连续**子窗口，
    每个 worker 只处理自己窗口内的 attempt → 所有 worker 合起来恰好是单进程
    的同一前缀（attempted/coverage/legality/valid 完全可比）。
    结果由 `executor.map` **按提交序**收集（per-point 行序确定）。
    """
    import concurrent.futures as cf
    from .s3_validation import (build_aggression_index_excluding,
                                build_indexes)
    scope = hands if (idx is not None and agg is not None) else _slice_hands(hands)
    t_idx = time.perf_counter()
    if idx is None:
        idx = build_indexes(scope, exclude_players=set())
    if agg is None:
        agg = build_aggression_index_excluding(scope, self_exclude=set())
    index_build_s = time.perf_counter() - t_idx
    if cfg is None:
        cfg = _load_config()
    counts = _candidate_counts(scope)
    cum = []
    acc = 0
    for c in counts:
        acc += c
        cum.append(acc)
    total = acc
    if attempt_window is None:
        lo, hi = 0, max(1, min(total, max_points))
    else:
        lo = max(0, attempt_window[0])
        hi = min(total, attempt_window[1])
    # 目标记录点不足时，向后扩展窗口（保持各模式同一规则）
    target_hi = hi
    span = max(1, hi - lo)
    per = max(1, math.ceil(span / max(1, workers)))
    chunks = []
    start = lo
    for w in range(workers):
        if start >= target_hi:
            break
        end = min(target_hi, start + per)
        # 找到覆盖 [start, end) 的手区间
        h_start = next((j for j, c in enumerate(cum) if c > start), len(cum))
        h_end = next((j for j, c in enumerate(cum) if c >= end), len(cum))
        base = cum[h_start - 1] if h_start > 0 else 0
        chunks.append({"hands": scope[h_start:h_end + 1], "window": (start, end),
                       "offset": base})
        start = end
    payloads = [{"hands": c["hands"], "arm": arm, "cap": 10 ** 9,
                 "idx": idx, "agg": agg, "cfg": cfg,
                 "optimized": bool(optimized),
                 "attempt_offset": c["offset"], "attempt_window": c["window"]}
                for c in chunks]
    timers = Timers()
    w0 = time.perf_counter()
    c0 = time.process_time()
    results = []
    with cf.ProcessPoolExecutor(max_workers=workers) as ex:
        for r in ex.map(_pool_worker, payloads):
            results.append(r)
    wall = time.perf_counter() - w0
    cpu = time.process_time() - c0
    rows: list = []
    ptimes: list = []
    runouts: list = []
    nvill: list = []
    counters: Counter = Counter()
    wcpu = 0.0
    wwall = 0.0
    for r in results:
        rows.extend(r["rows"])
        ptimes.extend(r["point_times"])
        runouts.extend(r["runout_per_point"])
        nvill.extend(r["n_vill"])
        timers.merge(r["timers"])
        for k, v in (r.get("counters") or {}).items():
            counters[k] += v
        wcpu += r["worker_cpu_s"]
        wwall += r["worker_wall_s"]
    wcpu_total = wcpu
    eff = (wcpu_total / (wall * len(chunks))) if wall > 0 and chunks else None
    return {"perf_version": PERF_VERSION, "arm": arm,
            "mode": "process-pool(%d workers)" % workers,
            "optimized": bool(optimized),
            "profile_slice": "hand_index < %d" % PROFILE_HAND_MAX_INDEX,
            "points": len(rows), "wall_s": round(wall, 4),
            "cpu_s": round(cpu, 4), "index_build_s": round(index_build_s, 4),
            "worker_cpu_s_sum": round(wcpu, 4),
            "worker_wall_s_sum": round(wwall, 4),
            "worker_efficiency": (round(eff, 4) if eff is not None else None),
            "chunks": len(chunks),
            "attempt_window": [lo, target_hi],
            "timers": timers.as_dict(), "point_times_s": ptimes,
            "rows": rows, "fixtures": [],
            "counters": counters,
            "cache_stats": _cache_stats(optimized),
            "runout_stats": _runout_stats(runouts, nvill)}


# ── 顺序无关性证明（并行合法性的前提） ───────────────────────────────
def prove_order_independence(hands: list, arm: str, n_hands: int) -> dict:
    """同一批点在正序 / 逆序 / 重复执行下**逐点**一致（并行合法性的前提）。

    取前 `n_hands` 手并跑完其中**全部**决策点，因此正序与逆序覆盖同一集合、
    只有执行顺序不同。
    """
    from .s3_validation import (build_aggression_index_excluding,
                                build_indexes)
    scope = _slice_hands(hands)[:n_hands]
    idx = build_indexes(scope, exclude_players=set())
    agg = build_aggression_index_excluding(scope, self_exclude=set())
    cfg = _load_config()
    cap = 10 ** 6

    def _run(sc):
        out = run_points(sc, arm, cap, idx=idx, agg=agg, cfg=cfg)
        d: dict = {}
        for r in out["rows"]:
            d.setdefault((r["hand_id"], r["street"]), []).append(
                (r["n_runouts"], r["pi_equity"], r["pi_best"],
                 r["frozen_best"], r["cand_best"]))
        for k in d:
            d[k] = sorted(d[k], key=repr)
        return d

    a = _run(scope)
    b = _run(list(reversed(scope)))
    c = _run(scope)
    common = set(a) & set(b)
    mism = [k for k in common if a[k] != b[k]]
    return {"points_forward": sum(len(v) for v in a.values()),
            "points_reverse": sum(len(v) for v in b.values()),
            "common_hand_street_keys": len(common),
            "order_mismatches": len(mism),
            "repeat_run_identical": a == c,
            "sample_mismatch": mism[:3],
            "basis": "全部随机通道为调用内固定种子 → 逐点结果与执行顺序无关"}


def percentiles(values: list, qs=(50, 90, 95, 99)) -> dict:
    if not values:
        return {}
    s = sorted(values)
    out = {}
    for q in qs:
        k = min(len(s) - 1, int(math.ceil(q / 100.0 * len(s))) - 1)
        out["p%d" % q] = round(s[max(0, k)], 6)
    out["mean"] = round(sum(s) / len(s), 6)
    out["max"] = round(s[-1], 6)
    out["min"] = round(s[0], 6)
    return out


def cpu_report() -> dict:
    return {"cpu_count": os.cpu_count(), "workers_single_process": 1,
            "note": "runner 单进程单线程 → worker 利用率 = 1/cpu_count"}
