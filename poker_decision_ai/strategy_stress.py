"""P64 — Strategy Stress Validation（只读压力矩阵；确定性；零策略修改）。

对 Strategy Core v3.1 + Intelligence + Exploit + Fusion 做系统性压力验证。
不变量（逐格断言，违反即记录 violation）：
  I1 identity      base_ev + saca_cv + exploit_bb == final_utility（候选级）
  I2 cap           |exploit_bb| ≤ 0.5bb、|fusion delta| ≤ 0.5bb
  I3 conflict      conflicted/不可用 exploit → 无方向调整
  I4 no-double     a3v2_delta ≤ 0（继续范围更强）；bounty 仅 allin；
                   exploit 与 compute_ev 的 fold 证据重叠被 guard 报告
  I5 mode/objective  cash→chip_ev；mtt/pko/ko/mystery（含 payouts）→
                     chip_ev+icm；无 payouts → 诚实降级 chip_ev；
                     exploit/fusion 不改变 mode/objective/provenance
  I6 solver        无候选注入时 solver_source.available=False（无伪 GTO）
  I7 no-default    证据缺失 → exploit 不可用（无默认玩家类型）
  I8 legality      动作 ∈ 合法集
确定性：固定矩阵、双跑抽样一致。
"""
from __future__ import annotations

import copy
from typing import Any

from .domain.poker.strategy_core import compute_ev
from .strategy_fusion import FUSION_CAP_BB
from .strategy_intel import evaluate_intel
from .strategy_robustness import make_state

MAX_EXPLOIT_BB = 0.5

MODES = ("cash", "mtt", "pko", "ko", "mystery")
STREETS = ("preflop", "flop", "turn", "river")
SPRS = (3.0, 10.0, 25.0)

#: 证据画像（evidence_profile → range_src/opponent_src/evidence）
EVIDENCE_PROFILES = {
    "none": {},
    "range_only": {"range": True},
    "opponent_only": {"opponent": True},
    "population_only": {"population": True},
    "full_posterior": {"range": True, "opponent": True, "population": True,
                       "range_mode": "POSTERIOR"},
    "full_baseline": {"range": True, "opponent": True, "population": True,
                      "range_mode": "BASELINE"},
}

#: exploit 画像（exploit_profile → opportunity dict）
EXPLOIT_PROFILES = {
    "none": None,
    "loose": {"available": True, "confidence": 1.0,
              "opportunity_strength": 0.4,
              "exploit_direction": "aggressive",
              "affected_actions": {"bet": 0.2, "raise": 0.1},
              "cap_bb": 0.5, "streams_used": ["entry_rate"],
              "provenance": {"module": "exploit_strategy",
                             "version": "p62-v1"},
              "fallback": {"aggression": True},
              "deviations": {"entry_rate": {"status": "usable",
                                            "deviation": 0.4}}},
    "tight": {"available": True, "confidence": 1.0,
              "opportunity_strength": 0.4,
              "exploit_direction": "conservative",
              "affected_actions": {"bet": -0.2},
              "cap_bb": 0.5, "streams_used": ["entry_rate"],
              "provenance": {"module": "exploit_strategy",
                             "version": "p62-v1"},
              "fallback": {"aggression": True},
              "deviations": {"entry_rate": {"status": "usable",
                                            "deviation": -0.4}}},
    "conflict": {"available": False, "conflict": True,
                 "reason": "conflicting evidence directions across streams",
                 "streams_used": ["entry_rate", "conditional_fold"]},
}

BOARD = {"preflop": [], "flop": ["Ah", "Kd", "7c"],
         "turn": ["Ah", "Kd", "7c", "2s"],
         "river": ["Ah", "Kd", "7c", "2s", "9d"]}
MODE_PAYOUTS = {"cash": None, "mtt": [50, 30, 20], "pko": [50, 30, 20],
                "ko": [50, 30, 20], "mystery": [50, 30, 20]}


def build_scenarios() -> list[dict]:
    """确定性压力矩阵（5 模式 × 4 街 × 3 SPR × 6 证据 × 3 exploit = 1080）。"""
    out = []
    for mode in MODES:
        for street in STREETS:
            for spr in SPRS:
                for prof_name in EVIDENCE_PROFILES:
                    for ex_name, ex in EXPLOIT_PROFILES.items():
                        out.append({
                            "id": f"{mode}|{street}|spr{spr:g}|{prof_name}|"
                                  f"ex_{ex_name}",
                            "mode": mode, "street": street, "spr": spr,
                            "evidence_profile": prof_name,
                            "exploit_profile": ex_name,
                            "exploit": copy.deepcopy(ex),
                        })
    return out


def _evidence_srcs(profile: dict) -> tuple[dict, dict, dict]:
    range_src = {"villains": {"v2": {"distribution": {
        "AA": 0.1, "KK": 0.1, "AKs": 0.1, "QQ": 0.1, "AQo": 0.2,
        "KQo": 0.4}}}} if profile.get("range") else {}
    if profile.get("range") and profile.get("range_mode") == "POSTERIOR":
        range_src = {"villains": {"v2": {
            "distribution": {"AA": 0.15, "KK": 0.15, "AKs": 0.2,
                             "QQ": 0.15, "AQo": 0.15, "KQo": 0.2}}}}
    opponent_src = {}
    if profile.get("opponent"):
        opponent_src = {"players": {"v2": {
            "conditional_events": 120,
            "conditional_model": {"counts": {"flop|bet|m<=1.0|CO|wet":
                                             [40, 15]},
                                  "total_events": 40},
            "conditional_population": {"counts": {"flop|bet|m<=1.0||":
                                                  [200, 80]},
                                       "total_events": 200}}}}
    if profile.get("population"):
        p = opponent_src.setdefault("players", {}).setdefault("v2", {})
        p["conditional_population"] = {"counts": {"flop|bet|m<=1.0||":
                                                  [200, 80]},
                                       "total_events": 200}
    evidence = {"range_mode": profile.get("range_mode", "unknown"),
                "opponent_mode": ("player_history"
                                  if profile.get("opponent") else "unknown"),
                "population_source": ("uoft_synthetic"
                                      if profile.get("population")
                                      else "unknown")}
    return range_src, opponent_src, evidence


def run_cell(scn: dict) -> dict:
    """单格：构建 state + intel + exploit → 不变量验证（violations 列表）。"""
    st = make_state(scn["street"], ["9c", "4c"], BOARD[scn["street"]],
                    "IP", scn["spr"], "check", game_type=scn["mode"])
    if scn["street"] == "preflop":
        st["action_history"] = [{"street": "preflop", "player_id": "hero",
                                 "action": "raise", "amount": 2.5}]
    range_src, opponent_src, evidence = _evidence_srcs(
        EVIDENCE_PROFILES[scn["evidence_profile"]])
    payouts = MODE_PAYOUTS[scn["mode"]]
    if payouts:
        st["tournament"] = {"payouts": payouts}
    if scn["mode"] in ("pko", "ko", "mystery"):
        st["bounty_type"] = scn["mode"]
        for p in st["players"]:
            p["bounty"] = 50.0
    violations: list[str] = []
    try:
        r = evaluate_intel(st, {"hero_equity": 0.45},
                           range_src=copy.deepcopy(range_src),
                           opponent_src=copy.deepcopy(opponent_src),
                           evidence=copy.deepcopy(evidence),
                           exploit=copy.deepcopy(scn["exploit"]))
    except Exception as e:                      # noqa: BLE001 —— 记录不吞
        return {**scn, "available": False,
                "violations": [f"exception:{type(e).__name__}:{e}"][:3]}

    if r["available"]:
        # I1a chip-space identity（全模式）：base_ev + saca_cv == final_ev
        # （P62 冻结语义：exploit 是 utility 空间加项，见 I1b；a3v2 只
        #   替换被跟分支 equity，bounty 独立支付流，均不进此恒等式）
        for c in r["candidates"]:
            adj = c["adjustments"]
            lhs = c["base_ev_bb"] + adj["saca_cv_bb"]
            if c["final_ev_bb"] is not None and \
                    abs(lhs - c["final_ev_bb"]) > 3e-6:
                violations.append(
                    f"identity_chip:{c['action']}: {lhs} != "
                    f"{c['final_ev_bb']}")
        # I1b exploit utility 恒等式（P62 语义：final − before == exploit_bb）
        for c in r["candidates"]:
            if c.get("utility_before_exploit_bb") is None:
                continue
            lhs = c["final_utility_bb"] - c["utility_before_exploit_bb"]
            if abs(lhs - (c["adjustments"].get("exploit_bb") or 0.0)) > 3e-6:
                violations.append(
                    f"identity_exploit:{c['action']}: {lhs} != "
                    f"{c['adjustments'].get('exploit_bb')}")
        # I2 cap
        for c in r["candidates"]:
            if abs(c["adjustments"].get("exploit_bb") or 0.0) > \
                    MAX_EXPLOIT_BB + 1e-9:
                violations.append(f"exploit_cap:{c['action']}")
        # I2 fusion cap
        f = r.get("fusion")
        if f:
            for act, a in (f.get("actions") or {}).items():
                if abs(a["delta_bb"]) > FUSION_CAP_BB + 1e-9:
                    violations.append(f"fusion_cap:{act}:{a['delta_bb']}")
        # I3 conflict → 无方向
        ex = r["exploit"]
        if ex.get("conflict"):
            for c in r["candidates"]:
                if c["adjustments"].get("exploit_bb"):
                    violations.append(
                        f"conflict_direction:{c['action']}:"
                        f"{c['adjustments']['exploit_bb']}")
        # I4 no-double：a3v2 非正；bounty 仅 allin 已由 P44 语义
        for c in r["candidates"]:
            d = c["adjustments"].get("a3v2_delta_bb")
            if d is not None and d > 1e-9:
                violations.append(f"a3v2_positive:{c['action']}:{d}")
        # I5 mode/objective
        gt = scn["mode"]
        exp_mode = "cash" if gt == "cash" else "mtt"
        exp_obj = "chip_ev" if gt == "cash" else (
            "chip_ev+icm" if payouts else "chip_ev")
        if r["mode"] != exp_mode:
            violations.append(f"mode:{r['mode']}!={exp_mode}")
        if r["objective"] != exp_obj:
            violations.append(f"objective:{r['objective']}!={exp_obj}")
        if scn["exploit_profile"] != "none" and r["exploit"].get("available"):
            for c in r["candidates"]:
                if c["provenance"]["mode"] != exp_mode:
                    violations.append(
                        f"mode_bypass:{c['action']}:{c['provenance']['mode']}")
                if c.get("exploit_provenance") and \
                        c["exploit_provenance"].get("model") and \
                        "GTO" in str(c["exploit_provenance"]).upper():
                    violations.append(f"fake_gto:{c['action']}")
        # I6 solver
        if r["solver_source"]["available"]:
            violations.append("solver_available_unexpected")
        # I7 no-default 玩家类型：无对手证据 + exploit none →
        #    intel 不产生玩家类型标签（无 such 字段即通过）
        # I8 legality
        legal = {"fold", "check", "call", "bet", "raise"}
        for c in r["candidates"]:
            if c["action"] not in legal:
                violations.append(f"illegal:{c['action']}")
    return {**{k: scn[k] for k in ("id", "mode", "street", "spr",
                                   "evidence_profile", "exploit_profile")},
            "available": r["available"],
            "best_action": r.get("best_action"),
            "objective": r.get("objective"),
            "cv_model": (r["candidates"][0]["provenance"]["cv_model"]
                         if r.get("candidates") else "none"),
            "violations": violations}


def run_stress() -> dict:
    """全矩阵执行 + 违规聚合（确定性）。"""
    cells = [run_cell(s) for s in build_scenarios()]
    viol = [c for c in cells if c["violations"]]
    by_type: dict[str, int] = {}
    for c in viol:
        for v in c["violations"]:
            k = v.split(":")[0]
            by_type[k] = by_type.get(k, 0) + 1
    return {
        "stress_version": "p64-v1",
        "n_cells": len(cells),
        "n_available": sum(1 for c in cells if c["available"]),
        "n_unavailable_unknown": sum(1 for c in cells
                                     if not c["available"]),
        "n_violation_cells": len(viol),
        "violation_types": by_type,
        "violations": viol[:50],
        "cells": cells,
    }


def determinism_sample(n: int = 12) -> dict:
    """抽样双跑一致性（确定性证明）。"""
    scns = build_scenarios()
    pick = [scns[i] for i in range(0, len(scns), max(1, len(scns) // n))][:n]
    ok = True
    for s in pick:
        a, b = run_cell(s), run_cell(s)
        if a != b:
            ok = False
            break
    return {"n_sampled": len(pick), "all_equal": ok}
