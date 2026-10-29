"""PI-EV 性能优化的**候选实现 + 等价性证明工具**（不接入生产、不改封存文件）。

设计原则
--------
1. 只做**可证明语义等价**的优化：确定性缓存的都是纯函数
   （`parse_card` / `_pi_equity` / `compute_ev` / `child_equity`），
   相同输入 → 逐位相同输出；缓存不改变调用序之外的任何东西。
2. 不改 seed / budget / decision set / 契约 / roster / 阈值；
   **不减少枚举精度**（PI 仍是精确全枚举，缓存只复用相同结果）。
3. 本模块**不写入**任何封存路径，也不被生产代码导入。

monkey-patch 用法说明：`optimized_evaluate_v3()` 在调用 sealed `evaluate_v3`
期间把 `multistreet_v3.child_equity` / `multistreet_v2.compute_ev` 等纯函数
替换为缓存版本，调用后立即还原（try/finally）。这是**测量候选**，属于
高风险项（依赖模块内名字绑定），默认不启用。
"""
from __future__ import annotations

import dataclasses
import json
from contextlib import contextmanager
from functools import lru_cache

from ..domain.poker.cards import parse_card
from ..domain.poker.strategy_core import compute_ev
from ..multistreet_v2 import child_equity
from ..multistreet_v3 import evaluate_v3
from .. import multistreet_v2 as _ms2_mod
from .. import multistreet_v3 as _ms3_mod
from .unfreeze_core import PI_EV_MAX_RUNOUTS

OPT_VERSION = "pi-ev-perf-optimize-v1"
_RANKS = "23456789TJQKA"
_SUITS = "shdc"

STATS: dict = {}


def reset_stats() -> None:
    STATS.clear()
    for name in ("pi_equity", "compute_ev", "child_equity"):
        STATS[name] = {"calls": 0, "hits": 0, "misses": 0}


reset_stats()


def _bump(name: str, hit: bool) -> None:
    s = STATS.setdefault(name, {"calls": 0, "hits": 0, "misses": 0})
    s["calls"] += 1
    s["hits" if hit else "misses"] += 1


# ── 1) 牌面解析缓存（parse_card 为纯函数） ────────────────────────────
@lru_cache(maxsize=1 << 16)
def _card_cached(code: str):
    return parse_card(code)


# ── 2) _pi_equity 结果缓存（精确全枚举；只复用相同输入的相同结果） ────
def _deck_excluding(known: set) -> list:
    return [r + s for r in _RANKS for s in _SUITS if (r + s) not in known]


def _pi_equity_impl(hero: tuple, others: tuple, board: tuple):
    known = set(hero) | set(board)
    for o in others:
        known |= set(o)
    deck = [c for c in _deck_excluding(known) if c not in board]
    need = 5 - len(board)
    if need < 0 or not others:
        return None, 0
    from itertools import combinations
    runouts = list(combinations(deck, need)) if need else [()]
    n_run = len(runouts)
    if n_run > PI_EV_MAX_RUNOUTS:
        return None, n_run
    tot = 0.0
    ev7 = _ev7_cached
    card = _card_cached
    for ro in runouts:
        full = board + tuple(ro)
        hs = ev7(tuple(sorted(c.code for c in
                              [card(c) for c in hero + full])))
        best_other = max(ev7(tuple(sorted(c.code for c in
                                          [card(c) for c in o + full])))
                         for o in others)
        if hs > best_other:
            tot += 1.0
        elif hs == best_other:
            tot += 0.5
    return tot / len(runouts), len(runouts)


@lru_cache(maxsize=1 << 15)
def _ev7_cached(sorted_codes: tuple) -> int:
    """等价于 `evaluate7(list[Card])`：入参为已排序 code 元组。"""
    from ..domain.poker.evaluator import evaluate7_cached
    return evaluate7_cached(sorted_codes)


@lru_cache(maxsize=1 << 17)
def _pi_equity_cached(hero: tuple, others: tuple, board: tuple):
    return _pi_equity_impl(hero, others, board)


def pi_equity_cached(hero, others, board):
    key = (tuple(hero), tuple(sorted(tuple(o) for o in others)), tuple(board))
    info_before = _pi_equity_cached.cache_info()
    out = _pi_equity_cached(*key)
    _bump("pi_equity", _pi_equity_cached.cache_info().hits > info_before.hits)
    return out


# ── 3) compute_ev 结果缓存（冻结核为纯函数：相同输入 → 相同输出） ──────
def _state_key(state: dict) -> str:
    return json.dumps(state, sort_keys=True, default=str, ensure_ascii=False)


def _src_key(src) -> str:
    return json.dumps(src, sort_keys=True, default=str, ensure_ascii=False)


@lru_cache(maxsize=1 << 14)
def _compute_ev_cached(state_key: str, eq_key: str, rs_key: str, os_key: str,
                       seed: int):
    return compute_ev(json.loads(state_key), json.loads(eq_key),
                      range_src=(json.loads(rs_key) if rs_key != "" else None),
                      opponent_src=(json.loads(os_key) if os_key != ""
                                    else None), seed=seed)


def compute_ev_cached(state, equity_src, range_src=None, opponent_src=None,
                      seed: int = 42):
    sk = _state_key(state)
    ek = _src_key(equity_src)
    rk = _src_key(range_src) if range_src is not None else ""
    ok = _src_key(opponent_src) if opponent_src is not None else ""
    before = _compute_ev_cached.cache_info()
    out = _compute_ev_cached(sk, ek, rk, ok, seed)
    _bump("compute_ev", _compute_ev_cached.cache_info().hits > before.hits)
    return out


# ── 4) child_equity 结果缓存（R2.2 每点每候选重复同一 8 张 runout） ────
def _vdist_key(villain_dist: dict) -> str:
    return json.dumps(villain_dist or {}, sort_keys=True, default=str)


@lru_cache(maxsize=1 << 16)
def _child_equity_cached(hero: tuple, board: tuple, vkey: str, iters: int,
                         seed: int):
    return child_equity(list(hero), list(board), json.loads(vkey), iters,
                        seed)


def child_equity_cached(hero_cards, board, villain_dist,
                        iterations: int = 300, seed: int = 42):
    key = (tuple(hero_cards), tuple(board), _vdist_key(villain_dist),
           iterations, seed)
    before = _child_equity_cached.cache_info()
    out = _child_equity_cached(*key)
    _bump("child_equity", _child_equity_cached.cache_info().hits > before.hits)
    return out


# ── 5) evaluate_v3 候选（内部纯函数缓存；调用后还原） ─────────────────
_COMBO_MEMO: dict = {}
_ORIG_COMBO_EQ = None


def _fe_key(fe) -> tuple:
    return (tuple(fe.hero_cards or ()), tuple(fe.board or ()),
            json.dumps(fe.villain_dist or {}, sort_keys=True, default=str),
            int(fe.seed), int(getattr(fe, "runouts", 0) or 0))


def combo_equities_cached(self):
    """`FoldEstimator._combo_equities` 的确定性记忆化（**不可变存储，防污染**）。

    该方法只依赖 (hero_cards, board, villain_dist, seed, runouts)，**与
    底池/下注尺寸无关**，且内部用 `random.Random(self.seed)` → 纯函数；
    同一实例内原本已有 `self._combo_eq` 缓存，但 `compute_ev` 每次调用都
    新建实例，故跨实例的同一 (board, range) 会被反复重算。

    防可变对象污染：缓存中以 **tuple 元组**（不可变）保存，每次返回**新的
    list 副本**；调用方持有/修改返回值不会影响缓存或其它实例。
    """
    key = _fe_key(self)
    hit = _COMBO_MEMO.get(key, _MISS)
    if hit is not _MISS:
        _bump("combo_equities", True)
        fresh = [tuple(p) for p in hit]
        self._combo_eq = fresh
        return fresh
    _bump("combo_equities", False)
    out = _ORIG_COMBO_EQ(self)
    if out is not None and len(_COMBO_MEMO) < (1 << 14):
        _COMBO_MEMO[key] = tuple(tuple(p) for p in out)
    return out


class _Miss:
    pass


_MISS = _Miss()


@contextmanager
def _patched(with_combo_memo: bool = True):
    global _ORIG_COMBO_EQ
    from ..domain.poker.fold_equity import FoldEstimator as FoldEquity
    orig = (_ms3_mod.child_equity, _ms3_mod.compute_ev, _ms2_mod.compute_ev)
    _ms3_mod.child_equity = child_equity_cached
    _ms3_mod.compute_ev = compute_ev_cached
    _ms2_mod.compute_ev = compute_ev_cached
    if with_combo_memo:
        if _ORIG_COMBO_EQ is None:
            _ORIG_COMBO_EQ = FoldEquity._combo_equities
        FoldEquity._combo_equities = combo_equities_cached
    try:
        yield
    finally:
        (_ms3_mod.child_equity, _ms3_mod.compute_ev,
         _ms2_mod.compute_ev) = orig
        if with_combo_memo and _ORIG_COMBO_EQ is not None:
            FoldEquity._combo_equities = _ORIG_COMBO_EQ


def reset_memos() -> None:
    _COMBO_MEMO.clear()
    _child_equity_cached.cache_clear()
    _compute_ev_cached.cache_clear()
    _pi_equity_cached.cache_clear()


# ── 6) `evaluate7_cached` 容量（仅淘汰策略；不可能改变返回值） ────────
def install_evaluate7_cache(maxsize: int):
    """把 `evaluator.evaluate7_cached` 换成同实现、更大容量的 LRU。

    返回 `(restore_callable, new_func)`。**只改淘汰策略**：被缓存的计算体
    （`__wrapped__`）为同一函数对象，因此任何输入的返回值逐位不变。
    """
    from ..domain.poker import evaluator as ev
    orig = ev.evaluate7_cached
    new = lru_cache(maxsize=maxsize)(orig.__wrapped__)
    ev.evaluate7_cached = new

    def restore():
        ev.evaluate7_cached = orig

    return restore, new


def evaluate7_cache_stats() -> dict:
    from ..domain.poker import evaluator as ev
    info = ev.evaluate7_cached.cache_info()
    return {"hits": info.hits, "misses": info.misses,
            "maxsize": info.maxsize, "currsize": info.currsize}


def optimized_evaluate_v3(*args, **kwargs):
    with _patched():
        return evaluate_v3(*args, **kwargs)


def optimized_evaluate_v3_plus(*args, **kwargs):
    """额外启用 `FoldEquity._combo_equities` 跨实例记忆化。"""
    with _patched(with_combo_memo=True):
        return evaluate_v3(*args, **kwargs)


# ── v2 runner 专用补丁集：**只含已证明且不可变/值类型**的 S1+S2 ───────
@contextmanager
def patched_s1_s2():
    """S1（combo_equities，不可变存储）+ S2（child_equity，float 值）补丁。

    与 `_patched()` 的区别：**不替换 `compute_ev`**（S4 实测 0 命中且返回
    可变 `EvResult` 对象，存在共享可变对象风险 → 不进入接入路径）。
    """
    global _ORIG_COMBO_EQ
    from ..domain.poker.fold_equity import FoldEstimator
    orig_child = _ms3_mod.child_equity
    _ms3_mod.child_equity = child_equity_cached
    if _ORIG_COMBO_EQ is None:
        _ORIG_COMBO_EQ = FoldEstimator._combo_equities
    FoldEstimator._combo_equities = combo_equities_cached
    try:
        yield
    finally:
        _ms3_mod.child_equity = orig_child
        FoldEstimator._combo_equities = _ORIG_COMBO_EQ


def evaluate_v3_s1s2(*args, **kwargs):
    """S1+S2 加速版 `evaluate_v3`（输出必须与 sealed 逐字段相同）。"""
    with patched_s1_s2():
        return evaluate_v3(*args, **kwargs)


# ── 6) PI 参照候选（缓存版；输出必须与 sealed 逐字段相同） ─────────────
def pi_reference_optimized(*, hero_cards, villain_hands, board, pot_bb,
                           to_call_bb, hero_stack_bb, candidates,
                           rule: str = "villain_calls") -> dict:
    eq, n_run = pi_equity_cached(list(hero_cards),
                                 [list(v) for v in villain_hands],
                                 [str(c) for c in board])
    prov = {"kind": "pi_ev", "evidence_level": "L2",
            "reference": "perfect-information", "equilibrium": False,
            "rule": f"checkdown_{rule}", "runouts_enumerated": n_run,
            "NOT_exact_gto": True}
    if eq is None:
        return {"available": False,
                "unavailable_reason": ("runout budget exceeded or no "
                                       "active opponent"),
                "provenance": prov, "values": {}, "ordering": []}
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
    return {"available": True, "unavailable_reason": None, "values": vals,
            "ordering": ordering, "provenance": prov,
            "pi_equity": round(eq, 6), "n_runouts": n_run}


# ── 7) 等值证明helpers ───────────────────────────────────────────────
def prove_pi_equivalence(fixtures: list) -> dict:
    from .unfreeze_core import PiEvReference
    bad = 0
    checked = 0
    for f in fixtures:
        sealed = PiEvReference().reference(
            hero_cards=f["hole"], villain_hands=f["vhands"], board=f["board"],
            pot_bb=f["inputs"]["pot_bb"], to_call_bb=f["inputs"]["to_call_bb"],
            hero_stack_bb=f["inputs"]["hero_stack_bb"], candidates=f["cands"])
        opt = pi_reference_optimized(
            hero_cards=f["hole"], villain_hands=f["vhands"], board=f["board"],
            pot_bb=f["inputs"]["pot_bb"], to_call_bb=f["inputs"]["to_call_bb"],
            hero_stack_bb=f["inputs"]["hero_stack_bb"], candidates=f["cands"])
        checked += 1
        if sealed != opt:
            bad += 1
    return {"checked": checked, "mismatches": bad,
            "basis": "PiEvReference.reference == pi_reference_optimized（逐字段）"}


def prove_compute_ev_equivalence(cases: list) -> dict:
    bad = 0
    for c in cases:
        a = compute_ev(c["state"], c["eq"], range_src=c.get("rs"),
                       opponent_src=c.get("os"), seed=c.get("seed", 42))
        b = compute_ev_cached(c["state"], c["eq"], range_src=c.get("rs"),
                              opponent_src=c.get("os"), seed=c.get("seed", 42))
        if dataclasses.asdict(a) != dataclasses.asdict(b):
            bad += 1
    return {"checked": len(cases), "mismatches": bad,
            "basis": "dataclasses.asdict(compute_ev) 深比较"}


def prove_evaluate_v3_equivalence(cases: list) -> dict:
    bad = 0
    for c in cases:
        a = evaluate_v3(c["state"], c["eq"], range_src=c.get("rs"),
                        aggression=c.get("agg"), villain_id=c.get("vid"),
                        villain_position=c.get("vpos"), runouts=c.get("runouts", 8),
                        equity_iterations=c.get("iters", 300),
                        seed=c.get("seed", 42))
        b = optimized_evaluate_v3(c["state"], c["eq"], range_src=c.get("rs"),
                                  aggression=c.get("agg"), villain_id=c.get("vid"),
                                  villain_position=c.get("vpos"),
                                  runouts=c.get("runouts", 8),
                                  equity_iterations=c.get("iters", 300),
                                  seed=c.get("seed", 42))
        if a != b:
            bad += 1
    return {"checked": len(cases), "mismatches": bad,
            "basis": "evaluate_v3(...) == optimized_evaluate_v3(...)（整字典深比较）"}
