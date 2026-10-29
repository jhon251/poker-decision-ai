"""Population Agent (群体分析): 段位统计查找 + 逐级 fallback + 手牌聚合兜底。

Input : {state: dict, statistics: dict|None, tournament_stage: dict|None}
Output: PopulationOutput {segment_keys, primary_segment, segments, stats,
         fallback_used, missing, aggregated_from_hands, confidence, evidence, warnings}

段位 key 构造（契约 §4 population）：
  现金桌   f"{stakes}:cash:{position}:{n}max" 逐级 fallback：
           "NL10:cash:CO:6max" -> "NL10:cash:6max" -> "NL10:cash" -> "cash:CO:6max" -> "cash"
  锦标赛   f"mtt:{stage}:{position}:{n}max" -> "mtt:{stage}:{n}max" -> "mtt:{stage}" -> "mtt"
  satellite 同理 f"satellite:..."
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar

from poker_decision_ai.core.orchestration.base_agent import BaseAgent
from poker_decision_ai.core.orchestration.context import AgentContext
from poker_decision_ai.domain.poker import evidence_gate as eg
from poker_decision_ai.domain.poker.stats import compute_all

# 段位记录里被当作"元字段"而非指标的 key
_META_KEYS = {"segment", "updated_at", "sample_size", "aggregated_from_hands"}

# V1.1 赏金行为指标（Population 段位可携带；对手模型三级收缩可用）
BOUNTY_BEHAVIOR_KEYS = (
    "bounty_hunting_tendency", "bounty_overcall", "bounty_overfold",
    "large_bounty_aggression", "small_bounty_aggression", "bubble_bounty_behavior",
)


@dataclass
class PopulationOutput:
    segment_keys: list = field(default_factory=list)
    primary_segment: str = ""
    segments: dict = field(default_factory=dict)       # key -> {metrics..., sample_size}
    stats: dict = field(default_factory=dict)          # 主段位指标 {<metric>: {value,samples}}
    fallback_used: bool = False
    missing: bool = True
    aggregated_from_hands: bool = False
    bounty_behavior: dict | None = None                # V1.1 赏金行为（无则 None）
    # P47：数据来源与证据门控（additive——真实/合成/混合 + gate 汇总）
    data_kind: str | None = None                       # REAL | SYNTHETIC | MIXED | None(legacy)
    evidence_gate: dict | None = None                  # eg.provenance_block 形态
    confidence: float = 1.0
    evidence: list = field(default_factory=list)
    warnings: list = field(default_factory=list)


class PopulationAgent(BaseAgent):
    name: ClassVar[str] = "population"
    version: ClassVar[str] = "1.0.0"
    zh_name: ClassVar[str] = "群体分析 Agent"
    layer: ClassVar[str] = "analysis"
    depends: ClassVar[list[str]] = ["state", "statistics"]
    Input: ClassVar[type] = dict
    Output: ClassVar[type] = PopulationOutput

    async def applicable(self, raw_input: dict, ctx: AgentContext) -> tuple[bool, str]:
        state = raw_input.get("state") or {}
        if not state.get("game_type"):
            return False, "state has no game_type for segment key construction"
        return True, ""

    async def handle(self, inp: dict, ctx: AgentContext) -> PopulationOutput:
        state = inp.get("state") or {}
        tournament_stage = inp.get("tournament_stage") or {}
        warnings: list[dict] = []

        keys = _segment_keys(state, tournament_stage)
        primary: str | None = None
        segments: dict[str, dict] = {}
        stats: dict = {}

        # ---- 逐级尝试 get_segment（具体 -> 粗略）----
        for key in keys:
            try:
                seg = ctx.repos.populations.get_segment(key)
            except Exception as e:  # 仓库故障不抛
                warnings.append({"code": "REPO_ERROR", "message": f"populations.get_segment({key}) failed: {e}"})
                seg = None
            if seg is None:
                continue
            sample_size = int(seg.get("sample_size") or 0)
            metrics = {k: v for k, v in seg.items() if k not in _META_KEYS and isinstance(v, dict)}
            segments[key] = {**metrics, "sample_size": sample_size}
            if primary is None:
                primary = key
                stats = metrics
            # 命中即用（首个命中的为 primary_segment）；继续记录后续命中的作为展示

        if primary is None:
            # ---- 全部未命中：尝试从 repos.hands 聚合同段位记录 ----
            agg, data_kind, gate_block = _aggregate_from_hands(ctx, state)
            if agg:
                stats = agg
                primary = keys[0] if keys else ""
                segments = {primary: {**agg, "sample_size": _sample_size_from(agg)}}
                evidence = [{
                    "kind": "population", "source": "repos.hands",
                    "text": f"段位 {keys} 未预存，使用 repos.hands 聚合 (aggregated_from_hands)",
                    "weight": 0.7,
                }]
                warnings.append({"code": "AGGREGATED_FROM_HANDS",
                                 "message": "population segments 未预存，回退到 repos.hands 聚合"})
                return PopulationOutput(
                    segment_keys=keys, primary_segment=primary, segments=segments,
                    stats=stats, fallback_used=False, missing=True, aggregated_from_hands=True,
                    bounty_behavior=_extract_bounty_behavior(stats),
                    data_kind=data_kind, evidence_gate=gate_block,
                    confidence=0.6, evidence=evidence, warnings=warnings,
                )
            # 仍无 -> stats 空
            warnings.append({"code": "SEGMENT_MISSING", "message": f"段位 {keys} 全部未命中"})
            return PopulationOutput(
                segment_keys=keys, primary_segment="", segments={}, stats={},
                fallback_used=False, missing=True, aggregated_from_hands=False,
                bounty_behavior=None,
                data_kind=None, evidence_gate=None,
                confidence=0.0,
                evidence=[{"kind": "population", "source": "populations",
                           "text": f"段位 {keys} 全部未命中", "weight": 1.0}],
                warnings=warnings,
            )

        fallback_used = (primary != keys[0]) if keys else False
        evidence = [{
            "kind": "population", "source": f"populations:{primary}",
            "text": f"命中段位 {primary}（fallback={fallback_used}）",
            "weight": 1.0,
        }]
        if fallback_used:
            warnings.append({"code": "FALLBACK_USED",
                             "message": f"最具体段位 {keys[0]} 未命中，回退到 {primary}"})
        return PopulationOutput(
            segment_keys=keys, primary_segment=primary, segments=segments,
            stats=stats, fallback_used=fallback_used, missing=False,
            aggregated_from_hands=False, bounty_behavior=_extract_bounty_behavior(stats),
            confidence=1.0,
            evidence=evidence, warnings=warnings,
        )


# --------------------------------------------------------------------------- #
def _hero_position(state: dict) -> str | None:
    hero_id = str(state.get("hero_id") or "")
    for p in state.get("players") or []:
        if str(p.get("player_id")) == hero_id or p.get("is_hero"):
            return p.get("position") or None
    return None


def _segment_keys(state: dict, tournament_stage: dict) -> list[str]:
    """构造段位 key 的 fallback 链（顺序 = 契约 §4 population 的顺序）。"""
    game = str(state.get("game_type") or "cash")
    n = len(state.get("players") or [])
    nmax = f"{n}max" if n > 0 else ""
    pos = _hero_position(state)
    stage = (tournament_stage or {}).get("stage") or ((state.get("tournament") or {}).get("stage") or "")
    stakes = ((state.get("cash") or {}).get("stakes") or "") if state.get("cash") else ""

    keys: list[str] = []
    if game == "cash":
        if stakes and pos and nmax:
            keys.append(f"{stakes}:cash:{pos}:{nmax}")
        if stakes and nmax:
            keys.append(f"{stakes}:cash:{nmax}")
        if stakes:
            keys.append(f"{stakes}:cash")
        if pos and nmax:
            keys.append(f"cash:{pos}:{nmax}")
        keys.append("cash")
    elif game == "mtt":
        if stage and pos and nmax:
            keys.append(f"mtt:{stage}:{pos}:{nmax}")
        if stage and nmax:
            keys.append(f"mtt:{stage}:{nmax}")
        if stage:
            keys.append(f"mtt:{stage}")
        keys.append("mtt")
    else:  # satellite
        if stage and pos and nmax:
            keys.append(f"satellite:{stage}:{pos}:{nmax}")
        if stage and nmax:
            keys.append(f"satellite:{stage}:{nmax}")
        if stage:
            keys.append(f"satellite:{stage}")
        keys.append("satellite")
    return keys


def _aggregate_from_hands(ctx: AgentContext, state: dict) -> tuple[dict | None, str | None, dict | None]:
    """从 repos.hands 聚合同段位（stakes/game 匹配）记录。

    P47 数据纪律：存在 REAL 标记记录时**只用 REAL 记录聚合**（合成语料
    不得混入真实统计——MIXED 聚合被禁止）；无 REAL 记录 → 全部记录
    （legacy 语义不变）。返回 (stats, data_kind, evidence_gate_block)。
    """
    game = state.get("game_type")
    stakes = ((state.get("cash") or {}).get("stakes") or "") if state.get("cash") else ""
    try:
        records = ctx.repos.hands.list(limit=100000)
    except Exception:
        return None, None, None
    matched = []
    for h in records:
        if game and h.get("game") and str(h.get("game")) != str(game):
            continue
        if stakes and h.get("stakes") and str(h.get("stakes")) != str(stakes):
            continue
        matched.append(h)
    if not matched:
        return None, None, None
    # P47：REAL 优先——有真实记录时合成/legacy 记录不参与群体统计；
    # hand-level 容器记录（无 player_id，仅携带 parsed）不是 player 证据，
    # 不参与 per-player 统计聚合（legacy 容器行为不变，容器本就无 player_id
    # 匹配 profile，此处仅约束 REAL 聚合口径）。
    real_all = [h for h in matched
                if str(h.get("data_kind") or "") == "REAL"]
    real = [h for h in real_all if h.get("player_id")]
    if real:
        matched = real
    elif real_all:
        # 只有容器记录（无 player 级证据）→ 诚实缺失，不冒充群体统计
        return None, None, None
    data_kind = _data_kind_of(matched)
    gate_block = _gate_block_of(matched)
    stats = compute_all(matched).to_dict()
    # V1.1：手牌记录带 bounty 行为可选字段（chased_bounty / bounty_overcall /
    # bounty_overfold）时，聚合出 bounty_hunting_tendency 等群体赏金行为。
    chased = [h for h in matched if isinstance(h.get("chased_bounty"), bool)]
    if chased:
        rate = sum(1 for h in chased if h["chased_bounty"]) / len(chased)
        stats["bounty_hunting_tendency"] = {"value": round(rate, 4), "samples": len(chased)}
    overcall = [h for h in matched if isinstance(h.get("bounty_overcall"), bool)]
    if overcall:
        rate = sum(1 for h in overcall if h["bounty_overcall"]) / len(overcall)
        stats["bounty_overcall"] = {"value": round(rate, 4), "samples": len(overcall)}
    overfold = [h for h in matched if isinstance(h.get("bounty_overfold"), bool)]
    if overfold:
        rate = sum(1 for h in overfold if h["bounty_overfold"]) / len(overfold)
        stats["bounty_overfold"] = {"value": round(rate, 4), "samples": len(overfold)}
    return stats, data_kind, gate_block


def _data_kind_of(records: list) -> str | None:
    """记录集合的 data_kind 归并（单一值 / MIXED / None=legacy）。"""
    kinds = {str(h["data_kind"]) for h in records
             if isinstance(h, dict) and h.get("data_kind")}
    if not kinds:
        return None
    if len(kinds) == 1:
        return kinds.pop()
    return "MIXED"


def _gate_block_of(records: list) -> dict | None:
    """记录集合的 evidence gate 聚合块（无 gate 标记记录 → None）。"""
    tagged = [h for h in records
              if isinstance(h, dict) and isinstance(h.get("evidence_gate"), dict)]
    if not tagged:
        return None
    rep = eg.EvidenceReport()
    for h in tagged:
        rep.add(h["evidence_gate"])
    dataset = None
    for h in tagged:
        if isinstance(h.get("dataset"), dict):
            dataset = h["dataset"]
            break
    return eg.provenance_block(dataset, rep)


def _extract_bounty_behavior(stats: dict) -> dict | None:
    """从主段位 stats 提取 V1.1 赏金行为指标（无则 None）。"""
    out = {k: v for k, v in stats.items() if k in BOUNTY_BEHAVIOR_KEYS and isinstance(v, dict)}
    return out or None


def _sample_size_from(stats: dict) -> int:
    """从指标 dict 取最大 samples 作为聚合样本量。"""
    best = 0
    for v in stats.values():
        if isinstance(v, dict):
            best = max(best, int(v.get("samples") or 0))
    return best
