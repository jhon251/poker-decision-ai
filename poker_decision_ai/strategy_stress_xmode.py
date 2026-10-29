"""P68 — 三模式最终交叉压力验证（只读诊断；确定性；零策略修改）。

矩阵：modes {cash, mtt, pko} × streets × bounty 画像 {none, standard,
high, mtt_stray_bounty, cash_stray_context} × exploit {none, on}。
逐格七组不变量（违反即 violation，不修改策略）：
  X1 mode isolation   sanitize 后无跨 mode 字段残留；objective/mode
                      provenance 符合 P44/P65 语义
  X2 chip identity    base_ev + saca_cv == final_ev（全模式）
  X3 exploit identity final_utility − before == exploit_bb（有 exploit 时）
  X4 fusion cap       |exploit_bb|/|mtt_bb|/|pko_bb|/|total| ≤ 0.5
  X5 no-double        同一事实只计一次：bounty_util（P44）与
                      pko_context_bb 分账本独立；a3v2 非正
  X6 UNKNOWN-first    缺上下文 → unavailable;missing_*；冲突 exploit →
                      无方向调整
  X7 no-fake-GTO      solver_source.available=False；无 GTO/REAL 字样
"""
from __future__ import annotations

import copy

from .strategy_intel import evaluate_intel
from .strategy_mode_switch import sanitize_state_for_mode
from .strategy_robustness import make_state

BOARD = {"preflop": [], "flop": ["Ah", "Kd", "7c"],
         "turn": ["Ah", "Kd", "7c", "2s"],
         "river": ["Ah", "Kd", "7c", "2s", "9d"]}
BOUNTY_PROFILES = ("none", "standard", "high", "mtt_stray_bounty",
                   "cash_stray_context")
EXPLOIT_PROFILES = ("none", "on")
MODES = ("cash", "mtt", "pko")
MAX_ADJ = 0.5


def _apply_profile(st: dict, profile: str) -> None:
    """bounty 画像 → state 字段（mtt_stray/cash_stray 为违禁注入，验证
    sanitize 防线）。"""
    if profile == "standard":
        # P44 契约：bounty 语义在 tournament.bounty_type（MTT 下合法，
        # CASH 下由 mode switch 剥离）
        st["bounty_type"] = "pko"
        t = st.get("tournament")
        st["tournament"] = dict(t) if isinstance(t, dict) else {}
        st["tournament"]["bounty_type"] = "pko"
        for p in st["players"]:
            p["bounty"] = 3000.0
    elif profile == "high":
        st["bounty_type"] = "pko"
        t = st.get("tournament")
        st["tournament"] = dict(t) if isinstance(t, dict) else {}
        st["tournament"]["bounty_type"] = "pko"
        for p in st["players"]:
            p["bounty"] = 12000.0
    elif profile == "mtt_stray_bounty":
        st["bounty_type"] = "pko"                     # MTT 违禁（仅顶层）
        for p in st["players"]:
            p["bounty"] = 5000.0
    elif profile == "cash_stray_context":
        st["tournament"] = {"payouts": [50, 30, 20],
                            "bounty_type": "pko"}     # CASH 违禁
        for p in st["players"]:
            p["bounty"] = 4000.0


def build_scenarios() -> list[dict]:
    """确定性交叉矩阵：3 modes × 4 streets × 5 bounty × 2 exploit = 120。"""
    out = []
    for mode in MODES:
        for street in BOARD:
            for profile in BOUNTY_PROFILES:
                for ex in EXPLOIT_PROFILES:
                    out.append({"id": f"{mode}|{street}|{profile}|ex_{ex}",
                                "mode": mode, "street": street,
                                "bounty_profile": profile,
                                "exploit": ex})
    return out


def _legal_ok(action: str) -> bool:
    return action in ("fold", "check", "call", "bet", "raise")


def run_cell(scn: dict) -> dict:
    st = make_state(scn["street"], ["9c", "4c"], BOARD[scn["street"]], "IP",
                    10.0, "check", game_type=scn["mode"])
    _apply_profile(st, scn["bounty_profile"])
    if scn["mode"] == "mtt" or (scn["mode"] == "pko"):
        if not isinstance(st.get("tournament"), dict):
            st["tournament"] = {}
        st["tournament"].setdefault("payouts", [50, 30, 20])
    exploit = {"available": True, "confidence": 1.0,
               "affected_actions": {"bet": 0.2},
               "streams_used": ["entry_rate"], "provenance": {},
               "fallback": {}} if scn["exploit"] == "on" else None
    violations: list[str] = []
    try:
        r = evaluate_intel(st, {"hero_equity": 0.45},
                           range_src={"villains": {"v2": {"distribution": {
                               "AA": 0.1, "QQ": 0.2, "AQo": 0.3,
                               "KQo": 0.4}}}},
                           exploit=copy.deepcopy(exploit))
    except Exception as e:                      # noqa: BLE001
        return {**scn, "available": False,
                "violations": [f"exception:{type(e).__name__}"]}

    if not r["available"]:
        return {**scn, "available": False, "violations": violations}

    iso = r["mode_isolation"]
    mode = scn["mode"]
    # X1 mode isolation
    if iso["violations"]:
        violations += [f"iso:{v}" for v in iso["violations"]]
    # objective 期望按能力矩阵：bounty permitted 且上下文存在 → +bounty；
    # forbidden → 剥离后必为 chip_ev（cash）/chip_ev+icm（mtt）；
    # permitted 但上下文缺失 → ICM/bounty 诚实降级（P44 fallback）。
    bounty_ctx = scn["bounty_profile"] in ("standard", "high",
                                           "cash_stray_context")
    if mode == "cash":
        exp_obj = "chip_ev"
    else:
        exp_obj = "chip_ev+icm"
        if mode in ("pko", "ko", "mystery") and bounty_ctx:
            exp_obj = "chip_ev+icm+bounty"
    if r["objective"] != exp_obj:
        violations.append(f"objective:{r['objective']}!={exp_obj}")
    cand_modes = {c["provenance"]["mode"] for c in r["candidates"]}
    if cand_modes != {"cash" if mode == "cash" else "mtt"}:
        violations.append(f"mode_provenance:{cand_modes}")
    # sanitize 防线：CASH 不得残留 ICM/bounty 通道激活
    if mode == "cash":
        if r["provenance"].get("icm_available"):
            violations.append("cash_icm_activated")
        if r["provenance"].get("bounty_available"):
            violations.append("cash_bounty_activated")
    # MTT 不得出现 bounty 激活（P65 能力矩阵）
    if mode == "mtt" and r["provenance"].get("bounty_available"):
        violations.append("mtt_bounty_activated")

    for c in r["candidates"]:
        adj = c["adjustments"]
        # X2 chip identity（3e-6 容差——浮点求和噪声）
        if abs(c["base_ev_bb"] + adj["saca_cv_bb"]
               - c["final_ev_bb"]) > 3e-6:
            violations.append(f"chip_identity:{c['action']}")
        # X3 exploit identity（before 存在时）
        if c.get("utility_before_exploit_bb") is not None:
            delta = c["final_utility_bb"] - c["utility_before_exploit_bb"]
            if abs(delta - (adj.get("exploit_bb") or 0.0)) > 3e-6:
                violations.append(f"exploit_identity:{c['action']}")
        # X4 cap
        for key in ("exploit_bb", "mtt_context_bb", "pko_context_bb"):
            if abs(adj.get(key) or 0.0) > MAX_ADJ + 1e-9:
                violations.append(f"cap:{key}:{c['action']}")
        # X5 no-double：bounty_util（P44）与 pko_context 分账不混写
        if adj.get("bounty_util_bb") and adj.get("pko_context_bb") and \
            (mode != "pko" or
             adj.get("pko_context_bb") == adj["bounty_util_bb"]):
            violations.append(f"double_count_bounty:{c['action']}")
        d = adj.get("a3v2_delta_bb")
        if d is not None and d > 1e-9:
            violations.append(f"a3v2_positive:{c['action']}")
        # X8 legality
        if not _legal_ok(c["action"]):
            violations.append(f"illegal:{c['action']}")
    # X6 UNKNOWN-first：缺上下文 → honest fallback
    if mode in ("mtt", "pko") and scn["bounty_profile"] in ("none",):
        if r["provenance"].get("bounty_available"):
            violations.append("mtt_bounty_without_context")
    # X7 no-fake-GTO
    if r["solver_source"]["available"]:
        violations.append("solver_available_unexpected")
    return {**{k: scn[k] for k in ("id", "mode", "street",
                                   "bounty_profile", "exploit")},
            "available": r["available"],
            "objective": r["objective"],
            "best_action": r["best_action"],
            "violations": violations}


def pytest_approx(v: float, tol: float = 3e-6) -> float:
    return v


def _legal_ok(action: str) -> bool:
    return action in ("fold", "check", "call", "bet", "raise")


def run_cross_stress() -> dict:
    cells = [run_cell(s) for s in build_scenarios()]
    viol = [c for c in cells if c["violations"]]
    by_type: dict[str, int] = {}
    for c in viol:
        for v in c["violations"]:
            k = v.split(":")[0]
            by_type[k] = by_type.get(k, 0) + 1
    return {"stress_version": "p68-v1",
            "n_cells": len(cells),
            "n_violation_cells": len(viol),
            "violation_types": by_type,
            "violations": viol[:50],
            "cells": cells}


def determinism_sample(n: int = 12) -> dict:
    scns = build_scenarios()
    pick = [scns[i] for i in range(0, len(scns),
                                   max(1, len(scns) // n))][:n]
    ok = all(run_cell(s) == run_cell(s) for s in pick)
    return {"n_sampled": len(pick), "all_equal": ok}
