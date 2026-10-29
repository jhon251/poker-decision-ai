"""P60 — Strategy Intelligence / Candidate Evaluation 层（只读聚合；确定性）。

定位：P41–P57 语义之上的**统一候选评估与归因层**——不新增 Agent、不改
策略数学：内部调用冻结的 `strategy_core.compute_ev`（其内部已整合
range posterior / fold evidence / SACA / A3-v2 / mode utility），把结果
与证据清单装配为每个候选可审计的 intelligence 块。

每个候选回答（P60 授权要求）：
  1. evidence_used      哪些证据参与（range/opponent/population/equity/fold/solver）
  2. math_sources       数学来源（terminal EV / fold 通道 / SACA-A8/A9 /
                         A3-v2 / ICM / bounty）
  3. base_ev            基础 EV（= ev_bb − cv − a3v2 贡献，即 P41-P55 终端语义）
  4. adjustments        各调整项贡献（cv_bb / a3v2_delta_bb / bounty_util_bb）
  5. final_utility      utility_bb（含 mode k± 缩放与 bounty）
  6. confidence         声明式合成（ev_mode 等级 × fold 通道质量 × 样本量）
  7. provenance         ev_mode/objective/mode/assumptions/cv_model/fold_channel
  8. fallback/UNKNOWN   always_called/point_fallback/icm/bounty 缺失显式标注

多来源竞争：ranked candidates（utility 降序）+ best/competitive/utility gap +
recommendation confidence——无单一模块接管；各来源贡献独立可追溯。

Double-counting 审计（结构保证，测试钉扎）：
  - population 仅作 player 条件估计的先验（P43 阶梯，非加项）；
  - fold 证据从 villain 范围推导（range posterior 是输入不是叠加项）；
  - A3-v2 只替换被跟分支 equity（eq_full 不重复计入）；
  - A8 与 A9 分属当前权益实现与听牌改进（不同权益来源）；
  - bounty 只在淘汰型候选且独立于 chip EV（不同支付流）。

Solver：P35 BLOCKED——`solver_source` 恒 BLOCKED/available=False；提供
`solver_candidate` 接口（未来真实 solver 以 solver_label 身份作为独立
候选来源接入），绝不把 heuristic 标成 GTO/REAL。
"""
from __future__ import annotations

import copy
from typing import Any

from .domain.poker.strategy_core import compute_ev

#: P60 声明式 confidence 权重（非调参对象；常数 contract）
EV_MODE_WEIGHT = {"distribution+fold": 1.0, "distribution": 0.9,
                  "point+fold": 0.85, "point_fallback": 0.6}
FOLD_CHANNEL_WEIGHT = {"conditional_player": 1.0,
                       "conditional_population": 0.8,
                       "stat+range": 0.8,
                       "range": 0.6,
                       "stat": 0.7,
                       "always_called_fallback": 0.3,
                       "none": 0.5}
SAMPLE_FULL = 100                    # fold 证据样本量饱和点（与 P43 一致）
MARGIN_DENOM_FRAC = 0.10             # recommendation confidence 的边际归一


def evaluate_intel(state: dict, equity_src: dict,
                   range_src: dict | None = None,
                   opponent_src: dict | None = None,
                   *,
                   evidence: dict | None = None,
                   solver_candidate: dict | None = None,
                   solver_source_override: dict | None = None,
                   exploit: dict | None = None,
                   fusion: dict | None = None,
                   mode_isolation: bool = True,
                   mtt_context: dict | None = None,
                   pko_context: dict | None = None,
                   seed: int = 42) -> dict:
    """候选评估层主入口（确定性；UNKNOWN-first）。

    evidence（可选，调用方从 Opponent/Range/Population 输出装配）：
      {"range_mode": str, "range_villains": [pid...],
       "opponent_mode": str, "opponent_cfi_events": int,
       "population_source": str, "population_cfi_events": int}
    solver_candidate（可选）：未来真实 solver 的候选
      {"action", "utility_bb", "provenance": "solver_label", ...}——
      本阶段恒不可用（P35 BLOCKED），接口占位。
    """
    # ---- P65 mode switch：语义模式解析 + 跨 mode 上下文剥离 ----
    from .strategy_mode_switch import (
        mode_capability_report, resolve_mode, sanitize_state_for_mode,
        verify_mode_isolation)
    resolved_mode, mode_unknown_reason = resolve_mode(state)
    mode_cap_report = mode_capability_report(state, resolved_mode)         if resolved_mode != "unknown" else {
            "mode": "unknown", "status": "unknown",
            "unknown_reason": mode_unknown_reason}
    run_state, forbidden_stripped = sanitize_state_for_mode(
        state, resolved_mode)

    res = compute_ev(run_state, equity_src, range_src=range_src,
                     opponent_src=opponent_src, seed=seed)

    ev_man = _evidence_manifest(run_state, equity_src, range_src,
                                opponent_src, evidence)

    if not res.available:
        return {
            "intel_version": "p60-v1",
            "available": False,
            "unavailable_reason": res.unavailable_reason,
            "evidence": ev_man,
            "solver_source": (solver_source_override
                              or _solver_source(solver_candidate)),
            "unknown_first": True,
            "mode_isolation": {
                "resolved_mode": resolved_mode,
                "capabilities": mode_cap_report.get("capabilities"),
                "unknown_reason": mode_unknown_reason,
                "forbidden_context_stripped": forbidden_stripped,
                "violations": []},
            "provenance": {"module": "strategy_intel",
                           "version": "p60-v1"},
        }

    # ---- P62 双计守卫：exploit 消费流 vs 喂给 compute_ev 的证据 ----
    guard = _double_counting_guard(exploit, opponent_src)
    # ---- P63：range/posterior 与 exploit 流的重叠审计扩展 ----
    if exploit and exploit.get("available"):
        streams = exploit.get("streams_used") or []
        if any(x in streams for x in ("entry_rate", "pfr_raise"))                 and (ev_man["range_source"]["present"]
                     and ev_man["range_source"]["mode"] == "POSTERIOR"):
            guard["range_exploit_overlap"] = {
                "detected": True,
                "note": ("exploit entry/pfr streams derive from the same "
                         "history as the active range posterior — caller "
                         "must route once")}
    cands = []
    for c in res.candidates:
        base_ev = c.ev_bb - (c.cv_bb or 0.0)
        adjustments = {
            "saca_cv_bb": round(c.cv_bb, 6) if c.cv_bb is not None else 0.0,
            "a3v2_delta_bb": (round(c.a3v2_delta_bb, 6)
                              if c.a3v2_delta_bb is not None else None),
            "bounty_util_bb": (round(c.bounty_util_bb, 6)
                               if c.bounty_util_bb is not None else None),
        }
        fallback = {
            "fold_channel": (c.fold_channel
                             if c.fold_channel else "none"),
            "is_always_called_fallback": (
                c.fold_channel == "always_called_fallback"),
            "point_fallback": res.ev_mode == "point_fallback",
            "icm_unavailable": (res.mode == "mtt"
                                and not res.icm_available),
            "bounty_unavailable_for_mode": (
                _is_bounty_mode(state) and not res.bounty_available),
        }
        cands.append({
            "action": c.action,
            "amount_bb": c.amount_bb,
            "base_ev_bb": round(base_ev, 6),
            "adjustments": adjustments,
            "final_ev_bb": round(c.ev_bb, 6),
            "final_utility_bb": (round(c.utility_bb, 6)
                                 if c.utility_bb is not None else None),
            "confidence": _candidate_confidence(res, c, ev_man),
            "p_fold": c.p_fold,
            "p_continue": c.p_continue,
            "fold_channel": c.fold_channel or "none",
            "fallback": fallback,
            "provenance": {"rule": c.rule,
                           "cv_model": res.cv_model,
                           "objective": res.objective,
                           "mode": res.mode},
        })

    # ---- P62：exploit 作为独立调整层应用（base/既有调整不变；重排序） ----
    # ---- P63：fusion——exploit/P62 路径经 ledger 融合后应用 ----
    # ---- P66：mtt_context 作为受 mode 门控的独立融合来源 ----
    from .strategy_fusion import (
        exploit_to_entries, fused_delta_for_action, fuse_entries)
    from .mtt_context import mtt_context_to_entries
    fused = None
    mtt_gate = {"applied": False,
                "reason": "no mtt_context provided"}
    if mtt_context is not None:
        if resolved_mode == "mtt":
            mtt_gate = {"applied": True, "reason": ""}
        else:
            mtt_gate = {"applied": False,
                        "reason": (f"mtt_context requires mode=mtt, got "
                                   f"{resolved_mode!r} "
                                   "(mode switch isolation)")}
    exploit_entries = (exploit_to_entries(exploit)
                       if exploit and exploit.get("available") else [])
    mtt_entries = (mtt_context_to_entries(mtt_context)
                   if mtt_gate.get("applied") else [])
    # ---- P67 mode 门控：pko_context 仅 bounty 模式应用 ----
    pko_gate = {"applied": False, "reason": "no pko_context provided"}
    pko_entries = []
    if pko_context is not None:
        if resolved_mode in ("pko", "ko", "mystery"):
            pko_gate = {"applied": True, "reason": ""}
            from .pko_context import pko_context_to_entries
            pko_entries = pko_context_to_entries(pko_context)
        else:
            pko_gate = {"applied": False,
                        "reason": (f"pko_context requires bounty mode "
                                   f"(pko/ko/mystery), got {resolved_mode!r} "
                                   "(mode switch isolation)")}
    all_entries = exploit_entries + mtt_entries + pko_entries
    if all_entries:
        fused = fuse_entries(all_entries)
        # 分账本（来源归因）：exploit-only / mtt-only 的每动作 delta
        fused_exploit = (fuse_entries(exploit_entries)
                         if exploit_entries else None)
        fused_mtt = (fuse_entries(mtt_entries)
                     if mtt_entries else None)
        fused_pko = (fuse_entries(pko_entries)
                     if pko_entries else None)
        for c in cands:
            act = c.get("action")
            bb_total, conflicted = fused_delta_for_action(fused, act)
            bb_exp = (fused_delta_for_action(fused_exploit, act)[0]
                      if fused_exploit else 0.0)
            bb_mtt = (fused_delta_for_action(fused_mtt, act)[0]
                      if fused_mtt else 0.0)
            bb_pko = (fused_delta_for_action(fused_pko, act)[0]
                      if fused_pko else 0.0)
            if c.get("adjustments") is not None:
                if exploit_entries:
                    c["adjustments"]["exploit_bb"] = bb_exp
                if mtt_entries:
                    c["adjustments"]["mtt_context_bb"] = bb_mtt
                if pko_entries:
                    c["adjustments"]["pko_context_bb"] = bb_pko
            c["utility_before_exploit_bb"] = c.get("final_utility_bb")
            if c.get("final_utility_bb") is not None:
                c["final_utility_bb"] = round(
                    c["final_utility_bb"] + bb_total, 6)
            c["exploit_confidence"] = (exploit or {}).get("confidence")
            c["exploit_provenance"] = (exploit or {}).get("provenance")
            c["fusion"] = {"delta_bb": bb_total,
                           "delta_exploit_bb": bb_exp,
                           "delta_mtt_context_bb": bb_mtt,
                           "delta_pko_context_bb": bb_pko,
                           "conflicted": conflicted,
                           "resolution": (fused.get("actions", {}).get(
                               act) or {}).get("resolution", "stack"),
                           "sources": list(
                               (fused.get("actions", {}).get(
                                   act) or {}).get("sources", []))}
        cands.sort(key=lambda c: -(c.get("final_utility_bb")
                                   if c.get("final_utility_bb")
                                   is not None else -1e18))
    else:
        cands.sort(key=lambda c: 0)          # 无调整——保持原序（空操作）
    best_u = res.best_utility_bb
    margin = 0.10 * max(res.pot_bb, 1e-9)
    gap = (best_u - res.candidates[1].utility_bb
           if len(res.candidates) >= 2
           and res.candidates[1].utility_bb is not None else None)
    rec_conf = None
    if gap is not None and cands and cands[0].get('confidence') is not None:
        rec_conf = round(cands[0]['confidence']
                         * (0.5 + 0.5 * min(1.0, max(0.0, gap)
                                            / (MARGIN_DENOM_FRAC
                                               * max(res.pot_bb, 1e-9)))), 6)

    result = {
        "intel_version": "p60-v1",
        "available": True,
        "mode": res.mode,
        "objective": res.objective,
        "ev_mode": res.ev_mode,
        "evidence": ev_man,
        "candidates": cands,
        "best_action": res.best_action,
        "best_amount_bb": res.best_amount_bb,
        "best_action_exploited": (cands[0].get("action")
                                  if cands and exploit
                                  and exploit.get("available") else None),
        "competitive_actions": res.competitive_actions,
        "utility_gap_bb": (round(gap, 6) if gap is not None else None),
        "exploit": (exploit or {"available": False}),
        "fusion": ({"actions": fused.get("actions"),
                    "conflicted_actions": fused.get("conflicted_actions"),
                    "capped_actions": fused.get("capped_actions")}
                   if fused else None),
        "double_counting_guard": guard,
        "mtt_context_gate": mtt_gate,
        "pko_context_gate": pko_gate,
        "recommendation_confidence": rec_conf,
        "solver_source": (solver_source_override
                          or _solver_source(solver_candidate)),
        "unknown_first": True,
        "mode_isolation": {
            "resolved_mode": resolved_mode,
            "capabilities": mode_cap_report.get("capabilities"),
            "forbidden_context_stripped": forbidden_stripped,
            "violations": []},
        "provenance": {"module": "strategy_intel", "version": "p60-v1",
                       "assumptions": list(res.assumptions),
                       "icm_available": res.icm_available,
                       "bounty_available": res.bounty_available},
    }
    result["mode_isolation"]["violations"] = verify_mode_isolation(
        result, resolved_mode)
    return result


def _double_counting_guard(exploit: dict | None,
                           opponent_src: dict | None) -> dict:
    """P62 双计守卫（报告制）：exploit 消费的 evidence 流 vs 调用方喂给
    compute_ev 的 opponent_src（conditional fold 证据）。重叠 = 同一
    fold 证据既进 compute_ev 又进 exploit 层——显式报告，由调用方路由
    证据（喂一次）。"""
    streams = [s for s in ((exploit or {}).get("streams_used") or [])]
    opp_has_fold = any(
        isinstance(p, dict) and p.get("conditional_model")
        for p in ((opponent_src or {}).get("players") or {}).values())
    overlap = "conditional_fold" in streams and opp_has_fold
    return {"exploit_streams": streams,
            "opponent_src_fed_to_compute_ev": opp_has_fold,
            "overlap_detected": overlap,
            "policy": ("route each evidence stream once: either into "
                       "compute_ev (fold/posterior) or into the exploit "
                       "layer — overlap is reported, not silently added")}


def _is_bounty_mode(state: dict) -> bool:
    gt = str(state.get("game_type") or "").lower()
    return any(k in gt for k in ("pko", "ko", "mystery", "bounty"))


def _evidence_manifest(state: dict, equity_src: dict | None,
                       range_src: dict | None, opponent_src: dict | None,
                       evidence: dict | None) -> dict:
    """证据清单（presence + 质量；确定性；缺失=unknown，不伪造）。"""
    range_src = range_src or {}
    opponent_src = opponent_src or {}
    villains = range_src.get("villains") or {}
    players = opponent_src.get("players") or {}
    has_range = any((v or {}).get("distribution") for v in villains.values()
                    if isinstance(v, dict))
    opp_cfi = sum(int((p or {}).get("conditional_events") or 0)
                  for p in players.values() if isinstance(p, dict))
    has_opp = any(isinstance(p, dict) and p.get("conditional_model")
                  for p in players.values())
    has_pop = any(isinstance(p, dict) and p.get("conditional_population")
                  and p["conditional_population"].get("counts")
                  for p in players.values())
    equity_present = bool((equity_src or {}).get("hero_equity") is not None)
    outcome_dist = (equity_src or {}).get("outcome") is not None
    man = {
        "range_source": {"present": has_range,
                         "mode": (evidence or {}).get("range_mode",
                                                      "unknown"),
                         "villains": [str(k) for k in villains]},
        "opponent_source": {"present": has_opp,
                            "mode": (evidence or {}).get("opponent_mode",
                                                         "unknown"),
                            "cfi_events": opp_cfi},
        "population_source": {"present": has_pop,
                              "mode": (evidence or {}).get(
                                  "population_source", "unknown")},
        "equity_source": {"present": equity_present,
                          "outcome_distribution": bool(outcome_dist)},
        "saca": {"present": None,          # 由候选层填充（cv_model）
                 "model": "saca-1.0|none"},
        "solver": _solver_source(None),
        "unknown_fields": sorted(
            k for k, v in (("range", has_range), ("opponent", has_opp),
                           ("population", has_pop),
                           ("equity", equity_present)) if not v),
    }
    if evidence:
        man["caller_evidence"] = dict(evidence)
    return man


def _solver_source(solver_candidate: dict | None) -> dict:
    """solver 来源块：P35 BLOCKED；未来真实 solver 以 solver_label 接入。"""
    if solver_candidate:
        # 接口占位：调用方显式提供才出现，且必须带 solver_label provenance
        sc = dict(solver_candidate)
        sc.setdefault("provenance", "solver_label")
        sc["is_human_ground_truth"] = False
        return {"available": True, "candidate": sc,
                "note": "future real solver — never heuristic-as-GTO"}
    return {"available": False,
            "status": "BLOCKED (P35: no solver environment)",
            "note": "heuristic 不会被标注为 GTO/REAL"}


def _candidate_confidence(res, cand, ev_man: dict) -> float | None:
    """声明式候选置信：ev_mode 等级 × fold 通道质量 × 样本饱和。"""
    w_mode = EV_MODE_WEIGHT.get(res.ev_mode, 0.5)
    w_fold = FOLD_CHANNEL_WEIGHT.get(cand.fold_channel or "none", 0.5)
    n = (ev_man.get("opponent_source") or {}).get("cfi_events") or 0
    sample = min(1.0, n / SAMPLE_FULL) if n else 0.5
    conf = w_mode * w_fold * max(sample, 0.25)     # 无样本不低于 0.25 底
    return round(min(max(conf, 0.0), 1.0), 6)
