"""Hypothesis Agent (假设生成): propose testable hypotheses, never facts.

Deterministic template pool selects context-relevant hypotheses (each carries
required_data / variables / control_group / test_method / expected_result so
it is falsifiable).  An optional LLM call (`ctx.llm_json`) can add a
personalized hypothesis; on None / ProviderError we fall back to templates.
Every hypothesis is stored via ctx.repos.research.add("hypothesis", ...) with
status="proposed" (spec: never treated as a confirmed fact).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, ClassVar

from poker_decision_ai.core.errors import ProviderError
from poker_decision_ai.core.orchestration.base_agent import BaseAgent
from poker_decision_ai.core.orchestration.context import AgentContext
from poker_decision_ai.core.schemas import utc_now_iso


@dataclass
class _LlmHypothesis:
    """Small schema for the optional LLM-generated hypothesis."""
    hypothesis: str = ""
    reason: str = ""
    required_data: list = field(default_factory=list)
    variables: list = field(default_factory=list)
    control_group: str = ""
    test_method: str = ""
    expected_result: str = ""


@dataclass
class HypothesisOutput:
    hypotheses: list = field(default_factory=list)
    confidence: float = 0.8
    evidence: list = field(default_factory=list)
    warnings: list = field(default_factory=list)


# 确定性模板池。requires=None 表示无条件候选（保证空输入也产出假设）；
# 否则仅当对应输入存在且非空时加入。
_TEMPLATES: list[dict] = [
    {
        "key": "river_bluff_after_losing", "requires": None,
        "hypothesis": "玩家连败后河牌诈唬率会上升",
        "reason": "情绪/损失厌恶可能让连败玩家在河牌更频繁诈唬",
        "required_data": ["连败窗口手牌", "河牌下注与诈唬标注", "胜负结果"],
        "variables": ["连败长度", "河牌诈唬率"],
        "control_group": "同段位非连败玩家",
        "test_method": "对连败前后河牌诈唬率做 Wilson 区间差异检验",
        "expected_result": "连败后河牌诈唬率显著高于连败前",
    },
    {
        "key": "three_bet_steal", "requires": None,
        "hypothesis": "3bet 频率异常高的玩家偷盲过多",
        "reason": "高 3bet 频率可能来自盲注偷取而非强牌",
        "required_data": ["3bet 手牌样本", "摊牌底牌标注", "位置信息"],
        "variables": ["3bet 频率", "3bet 摊牌价值占比"],
        "control_group": "同位置正常 3bet 频率玩家",
        "test_method": "比较高 3bet 玩家的摊牌强度分布与群体均值",
        "expected_result": "高 3bet 玩家的摊牌强度显著低于群体",
    },
    {
        "key": "overbet_value_tendency", "requires": "population",
        "hypothesis": "{stakes} 玩家 overbet 更偏向价值而非诈唬",
        "reason": "低级别玩家的大额下注往往反映强牌",
        "required_data": ["overbet 手牌样本", "摊牌底牌标注"],
        "variables": ["overbet 频率", "overbet 摊牌价值占比"],
        "control_group": "同段位普通下注样本",
        "test_method": "比较 overbet 与普通下注的价值/诈唬比例",
        "expected_result": "overbet 价值占比显著更高",
    },
    {
        "key": "bubble_overfold", "requires": "population",
        "hypothesis": "bubble 阶段 {pos} 玩家过度弃牌",
        "reason": "ICM 生存压力让 bubble 阶段玩家对加注/全下过度弃牌",
        "required_data": ["bubble 阶段弃牌样本", "fold_to_cbet/fold_to_3bet 指标"],
        "variables": ["fold_to_cbet", "fold_to_3bet"],
        "control_group": "非 bubble 阶段同段位玩家",
        "test_method": "比较 bubble 与非 bubble 的 fold_to_* 相对中性基线偏移",
        "expected_result": "bubble 阶段 fold_to_* 显著高于中性基线",
    },
    {
        "key": "low_confidence_shrink", "requires": "learning",
        "hypothesis": "低样本玩家统计被过度向群体均值收缩",
        "reason": "bayes_shrink 在小样本时权重偏向先验，可能抹掉真实个体偏差",
        "required_data": ["个体样本量与收缩前后指标", "群体均值"],
        "variables": ["个体样本量", "收缩强度"],
        "control_group": "高样本玩家的个体统计",
        "test_method": "检验收缩后个体指标对真实结果的预测力",
        "expected_result": "低样本玩家收缩后指标预测力偏低",
    },
]


def _data_context(statistics: dict | None, population: dict | None,
                  opponent: dict | None, learning: dict | None) -> dict:
    """Extract substitution values / references from available inputs."""
    subs: dict[str, str] = {}
    refs: list[str] = []

    if population and isinstance(population, dict):
        seg = population.get("primary_segment")
        segs = population.get("segments")
        if not seg and isinstance(segs, dict) and segs:
            seg = next(iter(segs))
        if seg:
            subs["stakes"] = str(seg).split(":")[0]
        stats = population.get("stats")
        if isinstance(stats, dict):
            f2c = stats.get("fold_to_cbet")
            if isinstance(f2c, dict) and f2c.get("value") is not None:
                refs.append(f"fold_to_cbet={f2c['value']:.2f}(n={f2c.get('samples')})")

    if opponent and isinstance(opponent, dict):
        players = opponent.get("players")
        if isinstance(players, dict) and players:
            first_pid = next(iter(players))
            subs["pos"] = first_pid
            p = players[first_pid]
            if isinstance(p, dict):
                bs = p.get("behavior_statistics")
                if isinstance(bs, dict):
                    t3 = bs.get("three_bet")
                    if isinstance(t3, dict) and t3.get("value") is not None:
                        refs.append(f"three_bet={t3['value']:.2f}(n={t3.get('samples')})")

    if statistics and isinstance(statistics, dict):
        hs = statistics.get("hero_stats")
        if isinstance(hs, dict):
            for name in ("vpip", "pfr"):
                m = hs.get(name)
                if isinstance(m, dict) and m.get("value") is not None:
                    refs.append(f"{name}={m['value']:.2f}(n={m.get('samples')})")

    if learning and isinstance(learning, dict):
        ups = learning.get("updates")
        if isinstance(ups, list) and ups:
            refs.append(f"{len(ups)} 条学习更新")

    return {"subs": subs, "ref": "；".join(refs)}


def _fill(tpl: dict, ctx_data: dict) -> dict:
    hypothesis = tpl["hypothesis"]
    reason = tpl["reason"]
    for k, v in ctx_data.get("subs", {}).items():
        hypothesis = hypothesis.replace("{" + k + "}", str(v))
        reason = reason.replace("{" + k + "}", str(v))
    if ctx_data.get("ref"):
        reason = f"{reason}（数据引用：{ctx_data['ref']}）"
    return {
        "hypothesis": hypothesis,
        "reason": reason,
        "required_data": list(tpl["required_data"]),
        "variables": list(tpl["variables"]),
        "control_group": tpl["control_group"],
        "test_method": tpl["test_method"],
        "expected_result": tpl["expected_result"],
    }


class HypothesisAgent(BaseAgent):
    name: ClassVar[str] = "hypothesis"
    version: ClassVar[str] = "1.0.0"
    zh_name: ClassVar[str] = "假设生成 Agent"
    layer: ClassVar[str] = "learning"
    depends: ClassVar[list[str]] = ["statistics", "population", "opponent",
                                    "learning", "research"]
    Input: ClassVar[type] = dict
    Output: ClassVar[type] = HypothesisOutput

    async def handle(self, inp: dict, ctx: AgentContext) -> HypothesisOutput:
        statistics = inp.get("statistics")
        population = inp.get("population")
        opponent = inp.get("opponent")
        learning = inp.get("learning")

        out = HypothesisOutput()
        ctx_data = _data_context(statistics, population, opponent, learning)
        ts = int(time.time() * 1000)
        created_at = utc_now_iso()

        # ---- 选择模板：无条件模板 + 有数据时叠加相关模板 ----
        selected: list[dict] = []
        for tpl in _TEMPLATES:
            req = tpl["requires"]
            if req is None:
                selected.append(tpl)
            elif req == "population" and population:
                selected.append(tpl)
            elif req == "learning" and learning:
                selected.append(tpl)

        for i, tpl in enumerate(selected):
            filled = _fill(tpl, ctx_data)
            record = {
                "id": f"hyp-{ts}-{i}",
                "hypothesis": filled["hypothesis"],
                "reason": filled["reason"],
                "required_data": filled["required_data"],
                "variables": filled["variables"],
                "control_group": filled["control_group"],
                "test_method": filled["test_method"],
                "expected_result": filled["expected_result"],
                "status": "proposed",
                "created_at": created_at,
            }
            out.hypotheses.append(record)
            ctx.repos.research.add("hypothesis", record)

        # ---- 可选 LLM 生成更个性化假设（失败/None -> 保留模板） ----
        try:
            res = await ctx.llm_json(
                system="你是扑克研究助理，提出可证伪的玩家行为假设。",
                user=f"基于输入 {json_fragment(inp)} 提出一条新的、可检验的扑克假设。",
                schema_cls=_LlmHypothesis,
            )
            if res:
                data, _usage = res
                if data and data.get("hypothesis"):
                    record = {
                        "id": f"hyp-{ts}-llm",
                        "hypothesis": str(data.get("hypothesis")),
                        "reason": str(data.get("reason") or "LLM 生成假设"),
                        "required_data": data.get("required_data") or ["历史手牌样本"],
                        "variables": data.get("variables") or ["待定变量"],
                        "control_group": str(data.get("control_group") or "同段位群体"),
                        "test_method": str(data.get("test_method") or "对照检验"),
                        "expected_result": str(data.get("expected_result") or "存在显著差异"),
                        "status": "proposed",
                        "created_at": created_at,
                    }
                    out.hypotheses.append(record)
                    ctx.repos.research.add("hypothesis", record)
        except ProviderError:
            out.warnings.append({"code": "LLM_FALLBACK",
                                 "message": "LLM 假设生成失败，回退确定性模板"})

        out.evidence = [{
            "kind": "hypothesis",
            "text": f"生成 {len(out.hypotheses)} 条假设（status=proposed）",
            "source": "hypothesis",
            "weight": 1.0,
        }]
        return out


def json_fragment(inp: dict) -> str:
    """Compact, safe JSON-ish summary of the input for the LLM prompt."""
    try:
        import json
        return json.dumps(inp, ensure_ascii=False, default=str)[:2000]
    except Exception:
        return str(inp)[:2000]
