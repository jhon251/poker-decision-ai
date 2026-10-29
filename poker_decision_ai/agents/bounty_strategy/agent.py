"""Bounty Strategy Agent (赏金策略): 确定性规则引擎 + 可选 LLM summary。

V1.1 §4b 契约。规则阈值（注释基准）：

  hunt_big_bounty       : hero covered 且 bounty_pressure > 0.4
  avoid_big_bounty      : hero 短码(<20bb) 且对手大赏金 且 bubble
  exploit_bounty_hunter : 对手 bounty_hunting_tendency > 0.6（其范围偏宽，可 value 加注）
  exploit_bounty_ignorer: 对手 bounty_hunting_tendency < 0.4 且其手上赏金 > 0
                          （忽视赏金 → 对其施加更多压力）
  bounty_hunting_tendency 来源：opponent 档案（feature_vector.bounty_hunting_tendency /
                          bounty_features.hunting_tendency / 顶层字段）→ population 段位
                          stats["bounty_hunting_tendency"] → 缺省 0.5 中性。

Input : {state, bounty, bounty_equity, bounty_icm, opponent, population,
         tournament_stage, icm}（dict|None）
Output: BountyStrategyOutput
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, ClassVar

from poker_decision_ai.core.errors import ProviderError
from poker_decision_ai.core.orchestration.base_agent import BaseAgent
from poker_decision_ai.core.orchestration.context import AgentContext

# ---- 规则阈值基准（见模块 docstring） ----
_PRESSURE_HUNT = 0.4        # bounty_pressure 高于此且 covered -> hunt
_SHORT_STACK_BB = 20.0      # 短码阈值
_HUNTER_TENDENCY = 0.6      # 高于此 -> bounty hunter
_IGNORER_TENDENCY = 0.4     # 低于此 -> bounty ignorer


@dataclass
class _BountyStrategySummary:
    summary: str = ""


@dataclass
class BountyStrategyOutput:
    recommendations: list = field(default_factory=list)  # [{type,target,description,confidence,suggested_adjustment}]
    hunt_big_bounty: bool = False
    avoid_big_bounty: bool = False
    bounty_hunting_detected: dict = field(default_factory=dict)   # {pid: bool}
    bubble_bounty_note: str = ""
    final_table_bounty_note: str = ""
    summary: str = ""
    confidence: float = 0.0
    evidence: list = field(default_factory=list)
    warnings: list = field(default_factory=list)


def _num(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _bb(state: dict) -> float:
    return float(state.get("bb") or 0.0) or 1.0


def _active_villains(players: list) -> list:
    return [p for p in players
            if isinstance(p, dict) and not p.get("is_hero")
            and p.get("status") in ("active", "allin")]


def _bounty_hunting_tendency(opponent: dict | None, population: dict | None,
                             pid: str) -> float:
    """对手赏金猎取倾向：opponent 档案优先，population 段位兜底，缺省 0.5 中性。"""
    opp = ((opponent or {}).get("players") or {}).get(pid)
    if isinstance(opp, dict):
        fv = opp.get("feature_vector") or {}
        if "bounty_hunting_tendency" in fv:
            return _num(fv["bounty_hunting_tendency"], 0.5)
        bf = opp.get("bounty_features") or {}
        if isinstance(bf, dict):
            for k in ("hunting_tendency", "bounty_hunting_tendency"):
                if k in bf:
                    return _num(bf[k], 0.5)
        if "bounty_hunting_tendency" in opp:
            return _num(opp["bounty_hunting_tendency"], 0.5)
    # population：stats 或 primary_segment 里的 bounty_hunting_tendency
    stats = (population or {}).get("stats") or {}
    m = stats.get("bounty_hunting_tendency")
    if isinstance(m, dict):
        return _num(m.get("value"), 0.5)
    if m is not None:
        return _num(m, 0.5)
    segments = (population or {}).get("segments") or {}
    primary = (population or {}).get("primary_segment") or ""
    seg = segments.get(primary) if primary else None
    if isinstance(seg, dict):
        m = seg.get("bounty_hunting_tendency")
        if isinstance(m, dict):
            return _num(m.get("value"), 0.5)
        if m is not None:
            return _num(m, 0.5)
    return 0.5


def _deterministic_summary(hunt: bool, avoid: bool, recs: list,
                           hunters: list, ignorers: list) -> str:
    if not recs:
        return "无显著赏金策略信号（数据中性或缺失）。"
    parts = [r.get("type") for r in recs[:5]]
    base = f"赏金策略建议: {', '.join(parts)}"
    if hunters:
        base += f"; 赏金猎手: {', '.join(hunters)}"
    if ignorers:
        base += f"; 忽视赏金: {', '.join(ignorers)}"
    return base


class BountyStrategyAgent(BaseAgent):
    name: ClassVar[str] = "bounty_strategy"
    version: ClassVar[str] = "1.1.0"
    zh_name: ClassVar[str] = "赏金策略 Agent"
    layer: ClassVar[str] = "analysis"
    depends: ClassVar[list[str]] = ["state", "bounty", "bounty_equity", "bounty_icm",
                                    "opponent", "population", "tournament_stage", "icm"]
    supported_modes: ClassVar[list[str]] = ["bounty"]
    Input: ClassVar[type] = dict
    Output: ClassVar[type] = BountyStrategyOutput

    async def applicable(self, raw_input: dict, ctx: AgentContext) -> tuple[bool, str]:
        if not raw_input.get("state"):
            return False, "state input is required for bounty strategy analysis"
        return True, ""

    async def handle(self, inp: dict, ctx: AgentContext) -> BountyStrategyOutput:
        state = inp.get("state") or {}
        bounty = inp.get("bounty") or {}
        opponent = inp.get("opponent")
        population = inp.get("population")
        tournament_stage = inp.get("tournament_stage") or {}
        out = BountyStrategyOutput()
        warnings: list[dict] = []
        recs: list[dict] = []

        bb = _bb(state)
        players = state.get("players") or []
        hero_p = next((p for p in players if isinstance(p, dict) and p.get("is_hero")), None)
        hero_bb = _num(hero_p.get("stack")) / bb if hero_p else 0.0

        stage = tournament_stage.get("stage") or (state.get("tournament") or {}).get("stage")
        stage = str(stage or "")
        remaining = int((state.get("tournament") or {}).get("players_remaining") or 0)
        is_bubble = stage == "bubble"
        is_ft = stage == "final_table" or (0 < remaining <= 9)

        # ---- bounty 上下文（缺失 → 无策略 + warning） ----
        if not isinstance(bounty, dict) or not bounty.get("is_bounty"):
            warnings.append({"code": "NO_BOUNTY",
                             "message": "bounty 输入缺失/无赏金，输出空 recommendations"})
            out.summary = _deterministic_summary(False, False, [], [], [])
            out.warnings = warnings
            return out

        covered = bool(bounty.get("covered"))
        pressure = _num(bounty.get("bounty_pressure"))
        big_bounty = _num(bounty.get("villain_bounty")) > 0

        # ---- 规则 1：hunt（covered 且压力高） ----
        if covered and pressure > _PRESSURE_HUNT and not (hero_bb < _SHORT_STACK_BB and is_bubble and big_bounty):
            out.hunt_big_bounty = True
            recs.append({
                "type": "hunt_big_bounty", "target": "population",
                "description": "已覆盖最大赏金对手且赏金压力高（>0.4），积极追淘汰以兑现赏金",
                "confidence": 0.7, "suggested_adjustment": "放宽跟注/加注范围，优先与可淘汰的大赏金对手摊牌",
            })

        # ---- 规则 2：avoid / protect（短码 + bubble + 大赏金） ----
        if hero_bb < _SHORT_STACK_BB and big_bounty and is_bubble:
            out.avoid_big_bounty = True
            recs.append({
                "type": "avoid_big_bounty", "target": "population",
                "description": "短码(<20bb)且 bubble 阶段面对大赏金对手，淘汰收益需与生存压力权衡，避免无谓冒险",
                "confidence": 0.8, "suggested_adjustment": "收紧范围，保护生存筹码；除非 EV 极佳否则不追边缘 KO",
            })
            recs.append({
                "type": "protect_stack", "target": "population",
                "description": "bubble 阶段保护码量优先，赏金 EV 低于生存压力",
                "confidence": 0.8, "suggested_adjustment": "减少大额波动决策，优先保底晋级",
            })

        # ---- 规则 3/4：对手 bounty 倾向 ----
        hunters: list[str] = []
        ignorers: list[str] = []
        for v in _active_villains(players):
            pid = str(v.get("player_id") or "")
            if not pid:
                continue
            tendency = _bounty_hunting_tendency(opponent, population, pid)
            v_bounty = _num(v.get("bounty"))
            if tendency > _HUNTER_TENDENCY:
                hunters.append(pid)
                out.bounty_hunting_detected[pid] = True
                recs.append({
                    "type": "exploit_bounty_hunter", "target": pid,
                    "description": f"对手 {pid} 赏金猎取倾向高（>{_HUNTER_TENDENCY}），其范围偏宽、过度追 KO",
                    "confidence": 0.6,
                    "suggested_adjustment": "对其价值加注，用强牌扩大下注尺寸诱其跟注",
                })
            elif tendency < _IGNORER_TENDENCY and v_bounty > 0:
                ignorers.append(pid)
                recs.append({
                    "type": "exploit_bounty_ignorer", "target": pid,
                    "description": f"对手 {pid} 忽视赏金（倾向<{_IGNORER_TENDENCY}）且持有赏金，可对其施加更多压力",
                    "confidence": 0.6,
                    "suggested_adjustment": "提高对其的加注/全下频率，利用其不愿为保护赏金而战的弱点",
                })
            elif tendency < _IGNORER_TENDENCY and big_bounty:
                recs.append({
                    "type": "use_bounty_pressure", "target": pid,
                    "description": "对手忽视赏金，可主动施加赏金压力迫其弃牌",
                    "confidence": 0.5,
                    "suggested_adjustment": "增加诈唬与半诈唬频率，迫使持有赏金的对手弃牌",
                })

        # ---- 规则 5：bubble + 大赏金提示 ----
        if is_bubble and big_bounty:
            out.bubble_bounty_note = ("bubble 阶段 + 大赏金：淘汰收益与生存压力并存，"
                                      "需比较 bounty_ev 与 icm 风险溢价后再决定追 KO。")
        # ---- 规则 6：FT 提示 ----
        if is_ft:
            out.final_table_bounty_note = ("最终桌阶段：赏金与 pay jump 并存，"
                                           "决策需同时考虑淘汰赏金与名次奖金跳跃。")

        out.recommendations = recs
        confs = [float(r.get("confidence", 0.0) or 0.0) for r in recs]
        out.confidence = round(max(confs) if confs else 0.0, 4)
        if not recs:
            warnings.append({"code": "NO_SIGNAL", "message": "无显著赏金策略信号（数据中性或缺失）"})

        # ---- summary：LLM 润色，失败/None 回退确定性模板 ----
        summary = _deterministic_summary(out.hunt_big_bounty, out.avoid_big_bounty,
                                         recs, hunters, ignorers)
        if recs:
            try:
                llm_out = await ctx.llm_json(
                    system="你是扑克赏金赛策略助手，用一句话总结赏金策略建议（中文）。",
                    user=json.dumps({
                        "hunt_big_bounty": out.hunt_big_bounty,
                        "avoid_big_bounty": out.avoid_big_bounty,
                        "recommendations": recs,
                    }, ensure_ascii=False, default=str),
                    schema_cls=_BountyStrategySummary,
                )
            except ProviderError as e:
                llm_out = None
                warnings.append({"code": "LLM_FAILED",
                                 "message": f"summary LLM failed ({e}); using deterministic template"})
            if llm_out:
                data, _ = llm_out
                s = (data or {}).get("summary")
                if isinstance(s, str) and s:
                    summary = s
        out.summary = summary

        out.evidence = [{
            "kind": "bounty_strategy", "source": "bounty_strategy",
            "text": (f"hunt={out.hunt_big_bounty} avoid={out.avoid_big_bounty} "
                     f"recs={len(recs)} hunters={len(hunters)} ignorers={len(ignorers)}"),
            "weight": 1.0,
        }]
        out.warnings = warnings
        return out
