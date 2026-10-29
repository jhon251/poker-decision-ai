"""Judge Agent (最终决策融合): deterministic weighted fusion of every source.

The Judge never lets the LLM decide numbers: equity / pot-odds / ICM premium are
computed deterministically (or read from upstream agents), each source casts a
weighted score on the action space, and the argmax becomes the final action.
The LLM is used only to optionally polish the natural-language `reasoning`.

Input : {state, board, range, equity, solver, exploit, opponent, population,
         human_style, confidence, bluff, image, emotion, strategy_evolution,
         risk, icm, tournament_stage, satellite, icm_exploit, blind_pressure,
         debate}  (all dict|None)
Output: JudgeOutput (below)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar

from poker_decision_ai.core.errors import ProviderError
from poker_decision_ai.core.orchestration.base_agent import BaseAgent
from poker_decision_ai.core.orchestration.context import AgentContext
from poker_decision_ai.domain.poker import pot_odds
from poker_decision_ai.domain.poker.strategy_anchor import (
    ANCHOR_EXPLOIT_DEVIATION,
    ANCHOR_MISS_FALLBACK,
    ANCHOR_PREFLOP_CHART,
    ANCHOR_SOLVER_HIT,
    anchor_config,
    anchor_from_chart,
    anchor_from_solver,
    bounded_deviation,
)
from poker_decision_ai.domain.poker.strategy_core import compute_ev


@dataclass
class JudgeOutput:
    action: str = "fold"                       # fold|check|call|bet|raise|allin
    recommended_size: float | None = None      # bb (bet/raise/allin only)
    confidence: float = 0.5
    reasoning: str = ""
    key_evidence: list = field(default_factory=list)
    uncertainty: list = field(default_factory=list)
    alternative_action: str | None = None
    weights_used: dict = field(default_factory=dict)
    final_inputs_summary: dict = field(default_factory=dict)
    evidence: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    # P41：EVCore v1 结果块（可解释 EV 表 + 假设声明；CASH 可用）
    ev_core: dict = field(default_factory=dict)
    # MC1-R5：影子对比（互不覆盖、字段分开——频率基线 vs 旧对手建模链）
    shadow_comparison: dict = field(default_factory=dict)
    # SOLVER-2：决策来源溯源（四类可区分：
    #   solver_hit / preflop_chart / exploit_deviation / solver_miss_fallback）
    provenance: dict = field(default_factory=dict)


@dataclass
class _ReasoningPolish:
    reasoning: str


ACTIONS: list[str] = ["fold", "check", "call", "bet", "raise", "allin"]


def _num(v: Any) -> float | None:
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    return None


def _clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return float(max(lo, min(hi, x)))


def _action_from_bounty_type(rtype: str, to_call: float) -> str | None:
    """bounty_strategy 建议类型 -> 动作（确定性映射）。"""
    if rtype in ("hunt_big_bounty", "exploit_bounty_ignorer"):
        return "raise" if to_call > 0 else "bet"
    if rtype in ("exploit_bounty_hunter", "use_bounty_pressure"):
        return "bet" if to_call == 0 else "raise"
    if rtype in ("avoid_big_bounty", "protect_stack"):
        return "fold" if to_call > 0 else "check"
    return None


def _hero_stack_bb(state: dict, bb: float) -> float | None:
    for p in state.get("players") or []:
        if isinstance(p, dict) and p.get("is_hero"):
            st = _num(p.get("stack"))
            if st is not None:
                return st / bb
    return None


class JudgeAgent(BaseAgent):
    name: ClassVar[str] = "judge"
    version: ClassVar[str] = "1.0.0"
    zh_name: ClassVar[str] = "最终决策融合 Agent"
    layer: ClassVar[str] = "decision"
    depends: ClassVar[list[str]] = ["equity", "solver", "exploit", "bluff", "risk", "icm", "debate"]
    Input: ClassVar[type] = dict
    Output: ClassVar[type] = JudgeOutput

    async def applicable(self, raw_input: dict, ctx: AgentContext) -> tuple[bool, str]:
        # Judge always runs: with zero sources it degrades to a conservative action.
        return True, ""

    async def handle(self, inp: dict, ctx: AgentContext) -> JudgeOutput:
        state = inp.get("state") or {}
        equity_src = inp.get("equity") or {}
        solver_src = inp.get("solver") or {}
        exploit_src = inp.get("exploit") or {}
        bluff_src = inp.get("bluff") or {}
        risk_src = inp.get("risk") or {}
        icm_src = inp.get("icm") or {}
        out = JudgeOutput()
        warnings: list[dict] = []

        # ------------------------------------------------------------------
        # Step 1) state basics
        # ------------------------------------------------------------------
        to_call = float(state.get("to_call") or 0.0)
        pot = float(state.get("pot") or 0.0)
        bb = float(state.get("bb") or 0.0)
        if bb <= 0:
            bb = 1.0
        to_call_bb = to_call / bb
        pot_bb_val = pot / bb

        # equity + required equity
        hero_equity = _num(equity_src.get("hero_equity"))
        required = _num(equity_src.get("pot_odds", {}).get("required_equity"))
        if required is None:
            # pot_odds(pot_before_call, to_call).required_equity (0 when no bet)
            required = pot_odds(pot, to_call).required_equity

        # ------------------------------------------------------------------
        # Step 2) ICM premium: linear approximation -> raise required equity.
        #   required_equity += max(risk_premium)  (survival pressure)
        # ------------------------------------------------------------------
        icm_premium = 0.0
        rp = icm_src.get("risk_premium") or {}
        if rp:
            try:
                icm_premium = max(float(v) for v in rp.values())
            except (TypeError, ValueError):
                icm_premium = 0.0
        required += icm_premium

        # ------------------------------------------------------------------
        # Step 2b) V1.1 模式修饰器：赏金调整所需胜率 / 卫星赛优先级
        #   use_bounty_adjusted_equity: 用 Bounty Equity Agent 的确定性
        #     有效所需胜率替换普通所需胜率（赏金把所需胜率向下拉）
        #   satellite_priority: 席位概率极高/极低时直接调整动作分
        # ------------------------------------------------------------------
        bounty_src = inp.get("bounty") or {}
        bounty_eq = inp.get("bounty_equity") or {}
        bounty_strat = inp.get("bounty_strategy") or {}
        bounty_icm_src = inp.get("bounty_icm") or {}
        satellite_src = inp.get("satellite") or {}
        modifiers = inp.get("mode_modifiers") or {}
        if not isinstance(modifiers, dict):
            modifiers = {}
        if modifiers.get("use_bounty_adjusted_equity") and bounty_eq.get("effective_decision_equity") is not None:
            adjusted = _num(bounty_eq.get("effective_decision_equity"))
            if adjusted is not None:
                required = adjusted

        # ------------------------------------------------------------------
        # Step 2c) SOLVER-2 求解器锚定基座（基座反转）────────────────────
        #   spot 库命中（HU postflop to_call=0 首入节点）或 preflop 图表
        #   命中 → base_decision = solver/chart 策略（argmax 频率；尺寸用
        #   solver sizings，替换 BET_FRACTIONS 常数）。exploit/频率先验/
        #   bluff 只能在锚点附近做**有预算偏离**（strategy_core 配置）。
        #   未命中 → 继续走旧启发式加权路径（诚实降级，provenance=miss）。
        # ------------------------------------------------------------------
        try:
            anchor_cfg = anchor_config(ctx.config.strategy_core_config())
        except AttributeError:
            anchor_cfg = anchor_config(None)
        if str(anchor_cfg.get("anchor") or "solver") == "solver":
            anchor = (anchor_from_solver(state, solver_src, anchor_cfg)
                      or anchor_from_chart(state, solver_src, anchor_cfg))
            if anchor is not None:
                return await self._anchored_output(
                    out, inp, ctx, anchor, anchor_cfg,
                    {"to_call": to_call, "pot": pot, "bb": bb,
                     "hero_equity": hero_equity, "required": required,
                     "icm_premium": icm_premium})
        # 未命中（或锚定关闭）→ 旧启发式加权路径，provenance 如实标注
        out.provenance = {
            "source": ANCHOR_MISS_FALLBACK, "anchored": False,
            "reason": ("spot/chart 未命中 → 启发式加权融合（诚实降级）"
                       if str(anchor_cfg.get("anchor") or "solver") == "solver"
                       else "strategy_core.anchor=off → 启发式加权融合"),
        }

        # ------------------------------------------------------------------
        # Step 3) action scoring table (all weights documented inline)
        # ------------------------------------------------------------------
        scores: dict[str, float] = {a: 0.0 for a in ACTIONS}
        weights_used: dict[str, float] = {}
        summary: dict[str, str] = {}
        source_confs: list[tuple[str, float]] = []

        # ---- P41 EVCore primary source (CASH chip_ev; weight 0.5) ----
        # 确定性 EV 排序作为主策略来源；各 (action,size) 的 EV/底池 归一后
        # 计入动作分。旧来源（equity/solver/exploit/...）全部保留为佐证
        # prior；RULES/外部 Solver 语义不变；可用性缺失 → 诚实回退旧记分板。
        # EVCore 永不绕过安全门（本输出仍走 judge 下游 validate_action）。
        ev_res = compute_ev(state, equity_src,
                            range_src=inp.get("range") or {},
                            opponent_src=inp.get("opponent") or {})
        out.ev_core = ev_res.to_summary()
        if ev_res.available:
            weights_used["ev_core"] = 0.5
            summary["ev_core"] = "ok"
            source_confs.append(("ev_core", 1.0))   # 确定性数学——无置信衰减
            for cand in ev_res.candidates:
                if cand.action in ACTIONS:
                    # P44：mode-aware utility 记分（CASH 下 utility==ev，
                    # 与 P41-P43 冻结语义逐位一致）
                    u = (cand.utility_bb if cand.utility_bb is not None
                         else cand.ev_bb)
                    scores[cand.action] += 0.5 * _clip(
                        u / max(ev_res.pot_bb, 1e-9), -1.0, 1.0)
        else:
            summary["ev_core"] = "unavailable"

        # ---- equity source (weight 0.3) ----
        # to_call>0 : call  += (equity-required)/0.2 clipped [-1,1]; fold += -call
        # to_call==0: check += 0.2 ; bet += (equity-0.5)*0.6
        if hero_equity is not None:
            summary["equity"] = "ok"
            weights_used["equity"] = 0.3
            ec = _num(equity_src.get("confidence"))
            source_confs.append(("equity", ec if ec is not None else 0.5))
            if to_call > 0:
                call_raw = _clip((hero_equity - required) / 0.2, -1.0, 1.0)
                scores["call"] += 0.3 * call_raw
                scores["fold"] += 0.3 * (-call_raw)
            else:
                scores["check"] += 0.3 * 0.2
                scores["bet"] += 0.3 * (hero_equity - 0.5) * 0.6
        else:
            summary["equity"] = "missing"

        # ---- solver source (weight 0.2 if approximate else 0.35) ----
        # gto_action += 0.5*w ; second-highest-frequency action += 0.2*w
        if solver_src.get("gto_action"):
            w = 0.2 if solver_src.get("is_approximate") else 0.35
            weights_used["solver"] = w
            summary["solver"] = "degraded" if solver_src.get("is_approximate") else "ok"
            sc = _num(solver_src.get("confidence"))
            source_confs.append(("solver", sc if sc is not None else 0.5))
            gto = str(solver_src["gto_action"])
            if gto in ACTIONS:
                scores[gto] += 0.5 * w
            freqs: list[tuple[str, float]] = []
            for a in solver_src.get("actions") or []:
                if isinstance(a, dict) and a.get("name") in ACTIONS:
                    f = _num(a.get("frequency"))
                    if f is not None:
                        freqs.append((str(a["name"]), f))
            freqs.sort(key=lambda x: -x[1])
            if len(freqs) >= 2:
                scores[freqs[1][0]] += 0.2 * w
        else:
            summary["solver"] = "missing"

        # ---- exploit source (weight 0.25) ----
        # best_exploit.suggested_action += 0.4 * confidence (+0.15 with icm_exploit)
        best = exploit_src.get("best_exploit") or {}
        if best.get("suggested_action"):
            act = str(best["suggested_action"])
            weights_used["exploit"] = 0.25
            summary["exploit"] = "ok"
            conf = _num(best.get("confidence"))
            if conf is None:
                conf = _num(exploit_src.get("confidence"))
            conf = conf if conf is not None else 0.5
            source_confs.append(("exploit", conf))
            if act in ACTIONS:
                exploit_score = 0.4 * _clip(conf)
                if inp.get("icm_exploit"):
                    exploit_score += 0.15
                scores[act] += 0.25 * exploit_score
        else:
            summary["exploit"] = "missing"

        # ---- V1.1 bounty_strategy source (weight 0.15) ----
        # 建议映射：hunt->raise, avoid/protect->保守, exploit_hunter->bet,
        #           exploit_ignorer->raise, use_pressure->bet
        if isinstance(bounty_strat.get("recommendations"), list) and bounty_strat["recommendations"]:
            weights_used["bounty_strategy"] = 0.15
            summary["bounty_strategy"] = "ok"
            bsc = _num(bounty_strat.get("confidence"))
            source_confs.append(("bounty_strategy", bsc if bsc is not None else 0.5))
            for r in bounty_strat["recommendations"][:2]:
                if not isinstance(r, dict):
                    continue
                act = _action_from_bounty_type(str(r.get("type") or ""), to_call)
                if act in ACTIONS:
                    conf = _num(r.get("confidence"))
                    scores[act] += 0.15 * 0.3 * _clip(conf if conf is not None else 0.5)
        else:
            summary["bounty_strategy"] = "missing"
        if bounty_icm_src.get("combined_ev"):
            summary["bounty_icm"] = "ok"
            weights_used["bounty_icm"] = 0.05      # 仅记录，动作分由 bounty_strategy 表达
        elif bounty_icm_src:
            summary["bounty_icm"] = "degraded"

        # ---- V1.1 satellite priority (weight 0.1) ----
        if modifiers.get("satellite_priority"):
            sev = _num(satellite_src.get("hero_seat_ev"))
            if sev is not None:
                weights_used["satellite"] = 0.1
                summary["satellite"] = "ok"
                scc = _num(satellite_src.get("confidence"))
                source_confs.append(("satellite", scc if scc is not None else 0.5))
                if sev > 0.9:
                    # 晋级几乎锁定：极端保守
                    scores["fold" if to_call > 0 else "check"] += 0.1 * 0.4
                elif sev < 0.15:
                    # 晋级希望渺茫：可激进
                    scores["raise" if to_call > 0 else "bet"] += 0.1 * 0.3
        elif satellite_src.get("hero_seat_ev") is not None:
            summary["satellite"] = "ok"

        # ---- bluff source (weight 0.15) ----
        # bluff_probability>0.5 -> call +0.2, raise +0.1 ; value_probability>0.6 -> fold +0.2
        bp = _num(bluff_src.get("bluff_probability"))
        vp = _num(bluff_src.get("value_probability"))
        if bp is not None or vp is not None:
            weights_used["bluff"] = 0.15
            summary["bluff"] = "ok"
            bc = _num(bluff_src.get("confidence"))
            source_confs.append(("bluff", bc if bc is not None else 0.5))
            if bp is not None and bp > 0.5:
                scores["call"] += 0.15 * 0.2
                scores["raise"] += 0.15 * 0.1
            if vp is not None and vp > 0.6:
                scores["fold"] += 0.15 * 0.2
        else:
            summary["bluff"] = "missing"

        # ---- risk / icm source (weight 0.1) ----
        # high-risk -> conservative leader among check/call/fold += 0.15
        high_risk = bool(risk_src.get("high_risk_decision")) or icm_premium > 0.2
        summary["risk"] = "ok" if risk_src else "missing"
        summary["icm"] = "ok" if icm_src else "missing"
        if high_risk:
            weights_used["risk"] = 0.1
            rc = _num(risk_src.get("confidence")) if risk_src else None
            source_confs.append(("risk", rc if rc is not None else 0.5))
            conservative = max(["check", "call", "fold"], key=lambda a: scores[a])
            scores[conservative] += 0.1 * 0.15

        # ---- MC1-R5 villain_frequency_prior source (weight from config) ----
        # 确定性对手频率先验：全部数值来自 domain/poker/frequency_prior.py
        # （仓库手牌统计 + Wilson 区间 + 时间衰减），LLM 不参与任何数值。
        # 通道（语义见 docs/MC1_R5_FREQUENCY_WIRING_REPORT.md）：
        #   fold-equity 通道：villain 面对下注弃牌率 f_fold → hero 下注/加注
        #     的弃牌权益 ↑（bet/raise += w·gain_fe·f_fold）；
        #   call 通道（仅 hero 面对下注）：villain 面对进攻的继续率 p_cont
        #     （松→其下注更薄→call += w·gain_cc·p_cont；紧→fold 加权）。
        # 权重全部来自 configs/config.yaml::frequency_prior（代码不写死）。
        shadow_comparison: dict = {}
        freq_channels_applied = False
        freq_evidence_items: list = []
        freq_src = inp.get("villain_frequency_prior") or {}
        if not isinstance(freq_src, dict):
            freq_src = {}
        fp_cfg = {}
        try:
            fp_cfg = dict(ctx.config.frequency_prior_config() or {})
        except AttributeError:
            fp_cfg = {}
        freq_enabled = bool(fp_cfg.get("enabled", True))
        sel_pid = str((freq_src.get("selection") or {}).get("player_id") or "")
        sel_prior = (freq_src.get("villains") or {}).get(sel_pid) or {}
        freq_available = bool(freq_enabled and freq_src.get("available")
                              and sel_prior.get("frequencies"))
        if freq_src:
            summary["villain_frequency_prior"] = (
                "ok" if freq_available and
                str(sel_prior.get("source", "")).startswith("player")
                else ("degraded" if freq_available else
                      ("disabled" if not freq_enabled else "unavailable")))
        if freq_available:
            w_fp = _num(fp_cfg.get("weight"))
            w_fp = 0.15 if w_fp is None else w_fp
            gain_fe = _num(fp_cfg.get("fold_equity_gain"))
            gain_fe = 0.30 if gain_fe is None else gain_fe
            gain_cc = _num(fp_cfg.get("call_channel_gain"))
            gain_cc = 0.20 if gain_cc is None else gain_cc
            f_fold = _num((sel_prior.get("frequencies") or {}).get(
                "fold", {}).get("rate"))
            p_cont = _num(sel_prior.get("p_continue"))
            channels: list[str] = []
            if f_fold is not None and f_fold > 0:
                scores["bet"] += w_fp * gain_fe * f_fold
                scores["raise"] += w_fp * gain_fe * f_fold
                channels.append(f"fold_equity(f_fold={f_fold:.2f})")
            if to_call > 0 and p_cont is not None:
                scores["call"] += w_fp * gain_cc * p_cont
                scores["fold"] += w_fp * gain_cc * (1.0 - p_cont)
                channels.append(f"call_channel(p_cont={p_cont:.2f})")
            if channels:
                freq_channels_applied = True
                weights_used["villain_frequency_prior"] = w_fp
                source_confs.append((
                    "villain_frequency_prior",
                    _num(freq_src.get("confidence")) or 0.5))
                freq_evidence_items.append({
                    "kind": "frequency_prior",
                    "text": (f"villain={sel_pid} "
                             f"{sel_prior.get('street')}/"
                             f"{sel_prior.get('position')} "
                             f"facing_bet={sel_prior.get('facing_bet')}: "
                             + "；".join(channels)
                             + f"（source={sel_prior.get('source')}, "
                               f"n={sel_prior.get('n_total')}, "
                               f"wilson_fold="
                               f"{(sel_prior.get('frequencies') or {}).get('fold', {}).get('wilson_low')}"
                               f"~{(sel_prior.get('frequencies') or {}).get('fold', {}).get('wilson_high')}）"),
                    "source": "frequency_baseline",
                    "weight": w_fp,
                    "provenance": {
                        "source": "frequency_baseline",
                        "n": sel_prior.get("n_total"),
                        "interval": {
                            "fold_wilson_low":
                                (sel_prior.get("frequencies") or {}).get(
                                    "fold", {}).get("wilson_low"),
                            "fold_wilson_high":
                                (sel_prior.get("frequencies") or {}).get(
                                    "fold", {}).get("wilson_high")},
                            "prior_source": sel_prior.get("source"),
                            "p_continue": p_cont,
                            "warnings": sel_prior.get("warnings") or []},
                })
            # ---- 影子对比（互不覆盖；字段分开）----
            ctx_actions = sel_prior.get("context_actions") or []
            freqs = sel_prior.get("frequencies") or {}
            fb_best = None
            if freqs:
                ranked = sorted((a for a in ctx_actions if a in freqs),
                                key=lambda a: -(_num(freqs[a].get("rate")) or 0.0))
                fb_best = ranked[0] if ranked else None
            legacy_raw = best.get("suggested_action")
            legacy_best = str(legacy_raw) if legacy_raw else None
            shadow_comparison = {
                "frequency_baseline": {
                    "suggested_action": fb_best,
                    "p_continue": p_cont,
                    "p_fold": _num(sel_prior.get("p_fold")),
                    "source": sel_prior.get("source"),
                    "n": sel_prior.get("n_total"),
                    "wilson_fold_low":
                        (freqs.get("fold") or {}).get("wilson_low"),
                    "wilson_fold_high":
                        (freqs.get("fold") or {}).get("wilson_high"),
                    "provenance": "frequency_baseline"},
                "legacy_response_model": {
                    "suggested_action": legacy_best,
                    "source": ("legacy_opponent_exploit_chain"
                               if legacy_best else "missing"),
                    "provenance": "opponent_conditioned_chain"},
            }
        out.shadow_comparison = shadow_comparison

        # debate present?
        summary["debate"] = "ok" if inp.get("debate") else "missing"

        # ------------------------------------------------------------------
        # all sources missing -> conservative fallback (confidence 0.3)
        # ------------------------------------------------------------------
        any_source = (ev_res.available
                      or hero_equity is not None
                      or bool(solver_src.get("gto_action"))
                      or bool(best.get("suggested_action"))
                      or (bp is not None or vp is not None)
                      or bool(bounty_strat.get("recommendations"))
                      or _num(satellite_src.get("hero_seat_ev")) is not None
                      or high_risk
                      or freq_channels_applied)

        if not any_source:
            out.action = "fold" if to_call > 0 else "check"
            out.confidence = 0.3
            out.alternative_action = None
            out.recommended_size = None
            out.weights_used = {}
            out.final_inputs_summary = {k: "missing" for k in
                                        ("equity", "solver", "exploit", "bluff", "risk", "icm")}
            out.final_inputs_summary["ev_core"] = summary.get(
                "ev_core", "unavailable")
            out.uncertainty = [
                "缺少 equity 来源", "缺少 solver 来源", "缺少 exploit 来源",
                "缺少 bluff 来源", "缺少 risk/icm 来源",
            ]
            out.reasoning = ("所有决策来源缺失，采取保守动作 "
                             + ("fold（面对下注且胜率未知）。" if to_call > 0 else "check（无人下注）。"))
            out.key_evidence = []
            out.evidence = [{"kind": "judge", "text": "no sources; conservative fallback",
                             "source": "judge", "weight": 1.0}]
            out.warnings = [{"code": "NO_DECISION_SOURCES", "message": "all decision sources missing"}]
            return out

        # ------------------------------------------------------------------
        # Step 4) choose action + alternative + recommended size
        # ------------------------------------------------------------------
        best_score = max(scores.values())
        if best_score <= 0:
            # neutral: no source advocated a positive action -> call/check
            action = "call" if to_call > 0 else "check"
        else:
            action = max(ACTIONS, key=lambda a: scores[a])
        ranked = sorted(ACTIONS, key=lambda a: (-scores[a], ACTIONS.index(a)))
        alternative = next((a for a in ranked if a != action), None)

        recommended_size: float | None = None
        if action in ("bet", "raise"):
            sugg: float | None = None
            # P41：EVCore 最优尺寸优先（primary source）
            if (ev_res.available and action == ev_res.best_action
                    and ev_res.best_amount_bb):
                sugg = ev_res.best_amount_bb
            if sugg is None:
                sugg = _num(best.get("suggested_size_bb"))
            if sugg is None:
                sugg = _num(exploit_src.get("suggested_size_bb"))
            if sugg is not None and sugg > 0:
                recommended_size = sugg
            else:
                sizings = [f for f in (_num(s) for s in (solver_src.get("bet_sizings_bb") or []))
                           if f is not None and f > 0]
                if sizings:
                    # pick the solver sizing closest to 0.66 * pot
                    recommended_size = min(sizings, key=lambda s: abs(s - 0.66))
                else:
                    recommended_size = 0.66 * pot_bb_val
            # all-in condition: effective stack <= size -> just go all-in
            stack_bb = _hero_stack_bb(state, bb)
            if stack_bb is not None and recommended_size >= stack_bb:
                action = "allin"
                recommended_size = stack_bb

        # ------------------------------------------------------------------
        # Step 5) confidence + uncertainty + key evidence
        # ------------------------------------------------------------------
        margin = scores[action] - (scores[alternative] if alternative else 0.0)
        avg_conf = (sum(c for _, c in source_confs) / len(source_confs)) if source_confs else 0.5
        confidence = _clip(0.5 + margin * 0.8 + avg_conf * 0.2, 0.2, 0.95)
        out.confidence = round(confidence, 3)

        uncertainty: list[str] = []
        for k, v in summary.items():
            if v == "missing":
                uncertainty.append(f"缺少 {k} 来源")
            elif v == "degraded":
                uncertainty.append(f"{k} 来源降级（近似/低置信）")
        for src, c in source_confs:
            if c < 0.5:
                uncertainty.append(f"低置信来源 {src}（confidence={c:.2f}）")

        key_evidence: list[str] = []
        for src_key in ("equity", "solver", "exploit", "bluff", "risk", "icm", "debate"):
            s = inp.get(src_key) or {}
            for ev in s.get("evidence") or []:
                if isinstance(ev, dict) and ev.get("text"):
                    key_evidence.append(str(ev["text"]))
                elif isinstance(ev, str):
                    key_evidence.append(ev)
        # dedupe, keep top 5
        seen: set[str] = set()
        deduped: list[str] = []
        for e in key_evidence:
            if e not in seen:
                seen.add(e)
                deduped.append(e)
        if not deduped:
            if hero_equity is not None:
                deduped.append(f"胜率 {hero_equity:.2f} vs 所需 {required:.2f}")
        out.key_evidence = deduped[:5]

        # ------------------------------------------------------------------
        # Step 6) reasoning (deterministic template, then optional LLM polish)
        # ------------------------------------------------------------------
        parts = [f"底池 {pot_bb_val:.1f}bb，需跟注 {to_call_bb:.2f}bb"]
        if ev_res.available and ev_res.best_action:
            second = (ev_res.candidates[1] if len(ev_res.candidates) > 1
                      else None)
            obj_tag = (f"[{ev_res.objective}]"
                       if getattr(ev_res, "objective", "chip_ev") != "chip_ev"
                       else "")
            parts.append(
                f"EVCore{obj_tag} 最优 {ev_res.best_action}"
                + (f" {ev_res.best_amount_bb:.1f}bb" if ev_res.best_amount_bb else "")
                + f" EV≈{ev_res.best_ev_bb:.1f}bb"
                + (f"（次优 {second.action} {second.ev_bb:.1f}bb）" if second else "")
                + "；单街近似/无弃牌权益假设")
        if hero_equity is not None:
            parts.append(f"胜率 {hero_equity:.2f}，所需胜率 {required:.2f}"
                         + (f"（含 ICM 溢价 {icm_premium:.3f}）" if icm_premium > 0 else ""))
        else:
            parts.append("胜率未知")
        if solver_src.get("gto_action"):
            parts.append(f"solver 建议 {solver_src['gto_action']}")
        if best.get("suggested_action"):
            parts.append(f"exploit 建议 {best['suggested_action']}")
        if freq_channels_applied:
            parts.append(f"对手频率先验（source={sel_prior.get('source')}, "
                         f"n={sel_prior.get('n_total')}）")
        if bp is not None or vp is not None:
            parts.append(f"bluff 信号（bluff={bp}, value={vp}）")
        if high_risk:
            parts.append("高风险决策，倾向保守")
        if bounty_src.get("is_bounty") and bounty_src.get("bounty_chips"):
            parts.append(f"赏金 EV 上下文（最大赏金≈{bounty_src.get('bounty_chips', 0):.0f} 筹码）")
        if modifiers.get("satellite_priority") and _num(satellite_src.get("hero_seat_ev")) is not None:
            parts.append(f"卫星赛席位概率 {satellite_src.get('hero_seat_ev'):.2f}")
        parts.append(f"最终动作 {action}"
                     + (f"，建议尺寸 {recommended_size:.2f}bb" if recommended_size else ""))
        reasoning = "；".join(parts) + "。"

        if not getattr(ctx.llm(), "name", "") == "mock":
            try:
                result = await ctx.llm_json(
                    system="你是扑克决策助理，把下列确定性分析改写成一段清晰的中文决策说明，不改动动作/数字。",
                    user=reasoning,
                    schema_cls=_ReasoningPolish,
                )
                # ctx.llm_json 返回 (data_dict, usage) 元组
                polished = result[0] if result else None
                if polished and polished.get("reasoning"):
                    reasoning = str(polished["reasoning"])
            except ProviderError as e:
                warnings.append({"code": "LLM_POLISH_FAILED", "message": str(e)})

        out.action = action
        out.alternative_action = alternative
        out.recommended_size = round(recommended_size, 3) if recommended_size is not None else None
        out.weights_used = weights_used
        out.final_inputs_summary = summary
        out.uncertainty = uncertainty
        out.reasoning = reasoning
        out.evidence = [{
            "kind": "judge",
            "text": f"action={action}, margin={margin:.3f}, weights={weights_used}",
            "source": "judge",
            "weight": 1.0,
        }] + freq_evidence_items
        # ── V3G-2：CASH 状态完整性 **证据旁路**（非评分、不改判定） ──────
        # 只追加一条 evidence 记录（在 action/amount/ev/scores 全部确定之后），
        # 不参与任何 EV / 候选 / 尺寸 / 评分计算；非 CASH 模式返回 present=False。
        try:
            from ...cash_integrity_context import cash_integrity_context
            ctx_ev = cash_integrity_context(state)
            if ctx_ev.get("present"):
                out.evidence.append({
                    "kind": ctx_ev["kind"],
                    "text": (f"state_integrity={ctx_ev['level']} "
                             f"completeness={ctx_ev['completeness']} "
                             f"evidence_grade={ctx_ev['evidence_grade']} "
                             f"missing={ctx_ev['missing']} "
                             f"conflicts={ctx_ev['conflicts']}"),
                    "source": "cash_state_integrity",
                    "weight": 0.0,          # 明确零权重：不参与任何评分
                    "state_integrity": ctx_ev["state_integrity"],
                })
        except Exception:                   # 旁路失败绝不阻断决策
            pass
        out.warnings = warnings
        return out

    # ------------------------------------------------------------------
    # SOLVER-2：锚定基座输出（base_decision = solver/chart 策略 + 有界偏离）
    # 评分板被基座反转取代：solver 动作即 base_decision；exploit/频率先验/
    # bluff 只能通过 bounded_deviation（预算内）改变动作，并记录 justification。
    # EVCore 仅作零权重影子（可解释性 + 下游契约形状不变）。
    # ------------------------------------------------------------------
    async def _anchored_output(self, out: JudgeOutput, inp: dict,
                               ctx: AgentContext, anchor,
                               anchor_cfg: dict, basics: dict) -> JudgeOutput:
        state = inp.get("state") or {}
        bb = float(basics.get("bb") or 0.0) or 1.0
        to_call = float(basics.get("to_call") or 0.0)
        pot_bb_val = float(basics.get("pot") or 0.0) / bb
        hero_equity = basics.get("hero_equity")
        required = basics.get("required")
        icm_premium = basics.get("icm_premium") or 0.0

        dev = bounded_deviation(state, anchor, inp, anchor_cfg)
        action = anchor.action
        recommended_size = anchor.recommended_size_bb
        if dev is not None:
            action = str(dev["action"])
            recommended_size = dev["size_bb"]
        # all-in 折算（与旧路径同一规则：尺寸 ≥ 有效栈 → 全下）
        stack_bb = _hero_stack_bb(state, bb)
        if (action in ("bet", "raise") and stack_bb is not None
                and recommended_size is not None
                and recommended_size >= stack_bb):
            action = "allin"
            recommended_size = stack_bb

        # EVCore 影子（零权重——不参与锚定动作选择；保留下游契约形状）
        ev_res = compute_ev(state, inp.get("equity") or {},
                            range_src=inp.get("range") or {},
                            opponent_src=inp.get("opponent") or {})
        out.ev_core = ev_res.to_summary()

        # 次选：solver 支持内频率次高动作（≠ 最终动作）
        ranked = sorted(anchor.support,
                        key=lambda r: (-r["frequency"],
                                       ACTIONS.index(r["action"]),
                                       r["size_bb"] or 0.0))
        alternative = next((r["action"] for r in ranked
                            if r["action"] != action), None)

        # provenance（四类可区分；偏离附 justification）
        provenance = {"source": (ANCHOR_EXPLOIT_DEVIATION if dev is not None
                                 else anchor.source),
                      "anchored": True,
                      "deviated": dev is not None}
        provenance.update(anchor.provenance)
        provenance["deviation"] = (dict(dev["justification"])
                                   if dev is not None else None)
        out.provenance = provenance

        # 影子对比（MC1-R5 契约形状；锚定模式下频率基线仅作影子）
        shadow_comparison: dict = {}
        freq_src = inp.get("villain_frequency_prior") or {}
        if isinstance(freq_src, dict) and freq_src.get("available"):
            sel_pid = str((freq_src.get("selection") or {}).get("player_id") or "")
            sel_prior = (freq_src.get("villains") or {}).get(sel_pid) or {}
            freqs = sel_prior.get("frequencies") or {}
            ctx_actions = sel_prior.get("context_actions") or []
            ranked_fb = sorted((a for a in ctx_actions if a in freqs),
                               key=lambda a: -(_num(freqs[a].get("rate")) or 0.0))
            best_raw = (inp.get("exploit") or {}).get("best_exploit") or {}
            shadow_comparison = {
                "frequency_baseline": {
                    "suggested_action": ranked_fb[0] if ranked_fb else None,
                    "p_continue": sel_prior.get("p_continue"),
                    "p_fold": sel_prior.get("p_fold"),
                    "source": sel_prior.get("source"),
                    "n": sel_prior.get("n_total"),
                    "wilson_fold_low": (freqs.get("fold") or {}).get("wilson_low"),
                    "wilson_fold_high": (freqs.get("fold") or {}).get("wilson_high"),
                    "provenance": "frequency_baseline"},
                "legacy_response_model": {
                    "suggested_action": (str(best_raw["suggested_action"])
                                         if best_raw.get("suggested_action")
                                         else None),
                    "source": ("legacy_opponent_exploit_chain"
                               if best_raw.get("suggested_action") else "missing"),
                    "provenance": "opponent_conditioned_chain"},
            }
        out.shadow_comparison = shadow_comparison

        out.action = action
        out.alternative_action = alternative
        out.recommended_size = (round(recommended_size, 3)
                                if recommended_size is not None else None)
        out.confidence = anchor.confidence
        out.weights_used = {"solver_anchor": 1.0}
        out.final_inputs_summary = {
            "mode": "solver_anchor",
            "anchor": anchor.source,
            "solver": "ok" if anchor.source == ANCHOR_SOLVER_HIT else "chart",
            "ev_core": "ok" if ev_res.available else "unavailable",
            "deviation": ("bounded" if dev is not None else "none"),
        }
        out.uncertainty = (["锚定为近似套用（nearest/chart），已降级标注"]
                           if anchor.provenance.get("is_approximate") else [])

        key_evidence: list[str] = []
        for src_key in ("solver", "exploit", "range"):
            s = inp.get(src_key) or {}
            for ev in s.get("evidence") or []:
                if isinstance(ev, dict) and ev.get("text"):
                    key_evidence.append(str(ev["text"]))
                elif isinstance(ev, str):
                    key_evidence.append(ev)
        if dev is not None:
            j = dev["justification"]
            key_evidence.append(
                f"有界偏离: {j['anchor_action']} → {action}"
                f"（signal={j['signal']}, conf={j['signal_confidence']}, "
                f"预测EV增量={j['predicted_ev_gain_bb']}, "
                f"solver预测EV损失={j['solver_predicted_ev_loss_bb']}bb, "
                f"预算={j['budget']['dev_max_ev_loss_bb']}bb）")
        if hero_equity is not None:
            key_evidence.append(f"胜率 {hero_equity:.2f} vs 所需 {required:.2f}")
        seen: set[str] = set()
        out.key_evidence = [e for e in key_evidence
                            if not (e in seen or seen.add(e))][:6]

        parts = [f"底池 {pot_bb_val:.1f}bb，需跟注 {to_call / bb:.2f}bb"]
        if anchor.source == ANCHOR_SOLVER_HIT:
            size_txt = (f" {anchor.recommended_size_bb:.1f}bb"
                        if anchor.recommended_size_bb else "")
            parts.append(f"求解器锚定（spot={anchor.provenance.get('spot_id')}"
                         f"，mode={anchor.provenance.get('lookup_mode')}）："
                         f"{anchor.action}"
                         f"@{anchor.provenance.get('anchor_action_freq')}"
                         f"{size_txt}（尺寸=solver sizing）")
        else:
            parts.append(f"preflop 图表基线（scenario="
                         f"{anchor.provenance.get('scenario')}，source=chart，"
                         f"非均衡解）：{anchor.action}")
        if dev is not None:
            j = dev["justification"]
            parts.append(f"有界剥削偏离（{j['signal']}，置信 "
                         f"{j['signal_confidence']:.2f}，预测 EV 增量 "
                         f"{j['predicted_ev_gain_bb']}，solver 预测 EV 损失 "
                         f"{j['solver_predicted_ev_loss_bb']}bb ≤ 预算 "
                         f"{j['budget']['dev_max_ev_loss_bb']}bb）")
        if ev_res.available and ev_res.best_action:
            parts.append(f"EVCore 影子最优 {ev_res.best_action}（零权重佐证）")
        if hero_equity is not None:
            parts.append(f"胜率 {hero_equity:.2f}，所需胜率 {required:.2f}"
                         + (f"（含 ICM 溢价 {icm_premium:.3f}）"
                            if icm_premium > 0 else ""))
        parts.append(f"最终动作 {action}"
                     + (f"，建议尺寸 {recommended_size:.2f}bb"
                        if recommended_size else ""))
        reasoning = "；".join(parts) + "。"

        warnings: list[dict] = []
        if not getattr(ctx.llm(), "name", "") == "mock":
            try:
                result = await ctx.llm_json(
                    system="你是扑克决策助理，把下列确定性分析改写成一段清晰的中文决策说明，不改动动作/数字。",
                    user=reasoning,
                    schema_cls=_ReasoningPolish,
                )
                polished = result[0] if result else None
                if polished and polished.get("reasoning"):
                    reasoning = str(polished["reasoning"])
            except ProviderError as e:
                warnings.append({"code": "LLM_POLISH_FAILED", "message": str(e)})

        out.reasoning = reasoning
        out.evidence = [{
            "kind": "judge",
            "text": (f"solver_anchored: anchor={anchor.source}, "
                     f"action={action}, deviated={dev is not None}"),
            "source": "judge", "weight": 1.0,
        }]
        # ── V3G-2：CASH 状态完整性 **证据旁路**（与旧路径同一行为）──────
        try:
            from ...cash_integrity_context import cash_integrity_context
            ctx_ev = cash_integrity_context(state)
            if ctx_ev.get("present"):
                out.evidence.append({
                    "kind": ctx_ev["kind"],
                    "text": (f"state_integrity={ctx_ev['level']} "
                             f"completeness={ctx_ev['completeness']} "
                             f"evidence_grade={ctx_ev['evidence_grade']} "
                             f"missing={ctx_ev['missing']} "
                             f"conflicts={ctx_ev['conflicts']}"),
                    "source": "cash_state_integrity",
                    "weight": 0.0,
                    "state_integrity": ctx_ev["state_integrity"],
                })
        except Exception:                   # 旁路失败绝不阻断决策
            pass
        out.warnings = warnings
        return out
