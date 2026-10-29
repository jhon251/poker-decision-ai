"""Debate Agent (辩论/反方): deterministic conflict detection + pro/con framing.

Each upstream source (solver / exploit / bluff / risk / icm / equity) implies
a preferred action.  When two sources imply mutually-exclusive actions we flag
a conflict and construct a pro/con argument pair.  All numeric facts come from
the sources themselves (never LLM-computed); the LLM only optionally polishes
the natural-language argument text.

Input : {state, equity, solver, exploit, bluff, risk, icm, confidence} (all dict|None)
Output: DebateOutput (below)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar

from poker_decision_ai.core.errors import ProviderError
from poker_decision_ai.core.orchestration.base_agent import BaseAgent
from poker_decision_ai.core.orchestration.context import AgentContext
from poker_decision_ai.domain.poker import pot_odds


@dataclass
class DebateOutput:
    conflict_detected: bool = False
    positions: list = field(default_factory=list)        # [{source, action, argument}]
    strongest_argument: str = ""
    counterargument: str = ""
    unresolved_uncertainty: list = field(default_factory=list)
    recommendation: str = ""
    confidence: float = 0.5
    evidence: list = field(default_factory=list)
    warnings: list = field(default_factory=list)


@dataclass
class _DebatePolish:
    strongest_argument: str
    counterargument: str
    recommendation: str


def _num(v: Any) -> float | None:
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    return None


def _clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return float(max(lo, min(hi, x)))


class DebateAgent(BaseAgent):
    name: ClassVar[str] = "debate"
    version: ClassVar[str] = "1.0.0"
    zh_name: ClassVar[str] = "辩论 Agent"
    layer: ClassVar[str] = "decision"
    depends: ClassVar[list[str]] = ["state", "equity", "solver", "exploit", "bluff", "risk", "icm"]
    Input: ClassVar[type] = dict
    Output: ClassVar[type] = DebateOutput

    async def applicable(self, raw_input: dict, ctx: AgentContext) -> tuple[bool, str]:
        state = raw_input.get("state") or {}
        sources = ("equity", "solver", "exploit", "bluff", "risk", "icm")
        has_source = any(raw_input.get(s) for s in sources)
        if not state and not has_source:
            return False, "no debate sources (state and all sources missing)"
        return True, ""

    async def handle(self, inp: dict, ctx: AgentContext) -> DebateOutput:
        state = inp.get("state") or {}
        equity = inp.get("equity") or {}
        solver = inp.get("solver") or {}
        exploit = inp.get("exploit") or {}
        bluff = inp.get("bluff") or {}
        risk = inp.get("risk") or {}
        icm = inp.get("icm") or {}
        out = DebateOutput()
        warnings: list[dict] = []

        to_call = float(state.get("to_call") or 0.0)
        pot = float(state.get("pot") or 0.0)
        required = equity.get("pot_odds", {}).get("required_equity")
        if required is None and to_call > 0:
            required = pot_odds(pot, to_call).required_equity

        # ---- collect implied actions + arguments (deterministic) ----
        positions: list[dict] = []
        implied: dict[str, str] = {}
        conf_by_source: dict[str, float] = {}

        if solver.get("gto_action"):
            act = str(solver["gto_action"])
            conf = _num(solver.get("confidence"))
            conf_by_source["solver"] = conf if conf is not None else 0.5
            positions.append({
                "source": "solver", "action": act,
                "argument": (f"solver 基线建议 {act}（is_approximate={bool(solver.get('is_approximate'))},"
                             f" confidence={conf_by_source['solver']:.2f}）"),
            })
            implied["solver"] = act

        best = exploit.get("best_exploit") or {}
        if best.get("suggested_action"):
            act = str(best["suggested_action"])
            conf = _num(best.get("confidence")) or _num(exploit.get("confidence")) or 0.5
            conf_by_source["exploit"] = conf
            positions.append({
                "source": "exploit", "action": act,
                "argument": f"剥削策略建议 {act}（confidence={conf:.2f}）",
            })
            implied["exploit"] = act

        bp = _num(bluff.get("bluff_probability"))
        vp = _num(bluff.get("value_probability"))
        bluff_act = bluff.get("implied_action")
        if not bluff_act and bp is not None and bp > 0.5:
            bluff_act = "call"      # opponent likely bluffing -> catch
        elif not bluff_act and vp is not None and vp > 0.6:
            bluff_act = "fold"      # opponent likely value-heavy -> fold
        if bluff_act:
            act = str(bluff_act)
            conf = _num(bluff.get("confidence")) or 0.5
            conf_by_source["bluff"] = conf
            positions.append({
                "source": "bluff", "action": act,
                "argument": (f"诈唬/价值信号：bluff_probability={bp}, value_probability={vp} → 隐含 {act}"),
            })
            implied["bluff"] = act

        if risk.get("high_risk_decision"):
            act = "fold" if to_call > 0 else "check"
            conf = _num(risk.get("confidence")) or 0.5
            conf_by_source["risk"] = conf
            positions.append({
                "source": "risk", "action": act,
                "argument": (f"高风险决策（decision_risk={risk.get('decision_risk')}）→ 倾向保守 {act}"),
            })
            implied["risk"] = act

        rp = icm.get("risk_premium") or {}
        if rp:
            try:
                premium = max(float(v) for v in rp.values())
            except (TypeError, ValueError):
                premium = 0.0
            if premium > 0.05:
                act = "fold" if to_call > 0 else "check"
                conf = _num(icm.get("confidence")) or 0.5
                conf_by_source["icm"] = conf
                positions.append({
                    "source": "icm", "action": act,
                    "argument": f"ICM 风险溢价 {premium:.3f} 偏高 → 倾向保守 {act}",
                })
                implied["icm"] = act

        out.positions = positions

        # ---- conflict detection: >1 distinct implied action -> conflict ----
        distinct = set(implied.values())
        out.conflict_detected = len(distinct) > 1

        # ---- strongest / counter argument (confidence-weighted ordering) ----
        ordered = sorted(positions, key=lambda p: conf_by_source.get(p["source"], 0.5), reverse=True)
        if ordered:
            top = ordered[0]
            out.strongest_argument = f"{top['source']} 认为 {top['action']}：{top['argument']}"
            # pick the highest-confidence source that DISAGREES with the top action
            opposing = next((p for p in ordered if p["action"] != top["action"]), None)
            if opposing:
                out.counterargument = f"{opposing['source']} 反驳（{opposing['action']}）：{opposing['argument']}"
            else:
                out.counterargument = "无对立来源，缺少反方论据。"

        # ---- unresolved uncertainty ----
        for src in ("equity", "solver", "exploit", "bluff", "risk", "icm"):
            if not inp.get(src):
                out.unresolved_uncertainty.append(f"缺少 {src} 来源")
        for src, conf in conf_by_source.items():
            if conf < 0.5:
                out.unresolved_uncertainty.append(f"低置信来源 {src}（confidence={conf:.2f}）")

        # ---- recommendation (deterministic template) ----
        if not ordered:
            out.recommendation = "无任何证据来源，无法形成辩论或建议。"
        elif out.conflict_detected:
            top = ordered[0]
            opp = next((p for p in ordered if p["action"] != top["action"]), None)
            if opp:
                out.recommendation = (
                    f"存在冲突：倾向 {top['source']} 的 {top['action']}，但保留对 "
                    f"{opp['source']}（{opp['action']}）的尊重，建议执行 {top['action']} 时缩小尺寸并谨慎。"
                )
            else:
                out.recommendation = f"各来源存在分歧，但无明确对立证据，倾向 {top['source']} 的 {top['action']}。"
        else:
            out.recommendation = f"各来源一致，建议 {ordered[0]['action']}（无冲突）。"

        # ---- confidence: average of available source confidences ----
        if conf_by_source:
            out.confidence = round(_clip(sum(conf_by_source.values()) / len(conf_by_source)), 3)
        else:
            out.confidence = 0.0

        # ---- optional LLM polish (fall back to template) ----
        if ordered and not getattr(ctx.llm(), "name", "") == "mock":
            try:
                result = await ctx.llm_json(
                    system="你是扑克决策辩论助理，只润色语言，不改动事实与倾向。",
                    user=(
                        f"冲突检测={out.conflict_detected}；各方立场："
                        + "；".join(f"{p['source']}:{p['action']} ({p['argument']})" for p in positions)
                        + f"。当前最强论点：{out.strongest_argument}；反方：{out.counterargument}；"
                        + f"建议：{out.recommendation}"
                    ),
                    schema_cls=_DebatePolish,
                )
                # ctx.llm_json 返回 (data_dict, usage) 元组
                polished = result[0] if result else None
                if polished and polished.get("strongest_argument"):
                    out.strongest_argument = str(polished["strongest_argument"])
                    out.counterargument = str(polished.get("counterargument") or out.counterargument)
                    out.recommendation = str(polished.get("recommendation") or out.recommendation)
            except ProviderError as e:
                warnings.append({"code": "LLM_POLISH_FAILED", "message": str(e)})

        if not positions:
            warnings.append({"code": "NO_DEBATE_SOURCES", "message": "no evidence sources provided"})

        out.evidence = [{
            "kind": "debate",
            "text": f"conflict={out.conflict_detected}, positions={sorted(implied)}",
            "source": "debate",
            "weight": 1.0,
        }]
        out.warnings = warnings
        return out
