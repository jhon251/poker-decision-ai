"""Statistics Agent (学习层统计): 从 repos.hands 的归一化记录计算统计指标。

Input : {state: dict}
Output: StatisticsOutput {hero_stats, villains, hand_records_count,
         normalized_hands, confidence, evidence, warnings}

逻辑（契约 §4 statistics）：
  - 从 ctx.repos.hands 读归一化手牌记录（HAND_RECORD_FIELDS 形状，每条含 player_id）
  - 对 hero 与每个 active villain 用 compute_all() 计算
  - cash 时按 game 过滤（可选；记录缺失 game 字段则不过滤）
  - 没数据 -> samples=0 的 Metric（不编造）+ warning
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar

from poker_decision_ai.core.orchestration.base_agent import BaseAgent
from poker_decision_ai.core.orchestration.context import AgentContext
from poker_decision_ai.domain.poker.stats import compute_all


@dataclass
class StatisticsOutput:
    hero_stats: dict = field(default_factory=dict)       # {"<metric>": {"value","samples"}}
    villains: dict = field(default_factory=dict)          # {"<pid>": {"<metric>": {...}}}
    hand_records_count: int = 0
    normalized_hands: int = 0
    confidence: float = 1.0
    evidence: list = field(default_factory=list)
    warnings: list = field(default_factory=list)


class StatisticsAgent(BaseAgent):
    name: ClassVar[str] = "statistics"
    version: ClassVar[str] = "1.0.0"
    zh_name: ClassVar[str] = "统计 Agent"
    layer: ClassVar[str] = "learning"
    depends: ClassVar[list[str]] = ["state"]
    Input: ClassVar[type] = dict
    Output: ClassVar[type] = StatisticsOutput

    async def applicable(self, raw_input: dict, ctx: AgentContext) -> tuple[bool, str]:
        state = raw_input.get("state") or {}
        if not state.get("hero_id") and not state.get("players"):
            return False, "state has no players to aggregate"
        return True, ""

    async def handle(self, inp: dict, ctx: AgentContext) -> StatisticsOutput:
        state = inp.get("state") or {}
        warnings: list[dict] = []

        # ---- 读取全部归一化手牌记录 ----
        try:
            all_records = ctx.repos.hands.list(limit=100000)
        except Exception as e:  # 仓库故障不抛，降级为空
            all_records = []
            warnings.append({"code": "REPO_ERROR", "message": f"hands repo read failed: {e}"})

        hand_records_count = len(all_records)

        # ---- cash 时按 game 过滤（记录缺失 game 字段则保留）----
        game_type = state.get("game_type")
        records = _filter_by_game(all_records, game_type)

        hero_id = str(state.get("hero_id") or "")
        players = state.get("players") or []

        hero_hands = [h for h in records if str(h.get("player_id")) == hero_id]
        hero_bundle = compute_all(hero_hands)
        hero_stats = hero_bundle.to_dict()

        villains: dict[str, dict] = {}
        active_ids = [
            str(p.get("player_id"))
            for p in players
            if (p.get("status") if isinstance(p, dict) else getattr(p, "status", "active")) in ("active", "allin")
            and str(p.get("player_id")) != hero_id
        ]
        for vid in active_ids:
            vhands = [h for h in records if str(h.get("player_id")) == vid]
            villains[vid] = compute_all(vhands).to_dict()
            if not vhands:
                warnings.append({"code": "NO_DATA", "message": f"villain {vid}: 无归一化手牌记录"})

        normalized_hands = len(hero_hands)
        if normalized_hands == 0:
            warnings.append({"code": "NO_DATA", "message": "hero 无归一化手牌记录，统计为 samples=0"})

        evidence = [{
            "kind": "statistics", "source": "repos.hands",
            "text": f"hero 归一化手牌 {normalized_hands} 条 / 仓库共 {hand_records_count} 条",
            "weight": 1.0,
        }]

        return StatisticsOutput(
            hero_stats=hero_stats, villains=villains,
            hand_records_count=hand_records_count, normalized_hands=normalized_hands,
            confidence=1.0, evidence=evidence, warnings=warnings,
        )


def _filter_by_game(records: list, game_type: Any) -> list:
    """cash 时按 game 过滤（可选）：记录带 game 字段且不等于当前 game 则剔除。"""
    if not game_type:
        return records
    return [h for h in records
            if not h.get("game") or str(h.get("game")) == str(game_type)]
