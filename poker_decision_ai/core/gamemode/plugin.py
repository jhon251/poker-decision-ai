"""GameModePlugin: 可插拔游戏模式声明（V1.1 架构核心）。

一个 Plugin 声明：
  - 模式与子类型（mode_id / type_id / supported_modes 能力规格）
  - 参与该模式的 Agent 集合（plan_stages()，含输入路由模板）
  - 必需的 / 可选的 State 字段
  - Judge 决策修饰器（decision_modifiers，注入 Judge 输入）
  - 评估规则（evaluation_rules，供 Evaluation Agent 使用）

新增玩法（Spin&Go / Short Deck / PLO ...）= 新增一个 Plugin，
Manager / Judge / Learning / Evaluation / Memory / Research 全部零改动。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:  # 类型检查专用：避免 gamemode -> orchestration 循环导入
    from poker_decision_ai.core.orchestration.plan import Stage


class GameModePlugin(ABC):
    """Base class for every game mode plugin."""

    #: 顶层模式：cash | tournament | bounty
    mode_id: ClassVar[str] = ""
    #: 子类型：mtt|satellite|freezeout|pko|ko|mystery（None = 该模式默认类型）
    type_id: ClassVar[str | None] = None
    #: 展示名
    display_name: ClassVar[str] = ""
    #: 本插件对应的模式规格（用于能力注册表与文档）
    supported_modes: ClassVar[list[str]] = []
    #: 必需的 State 字段（缺失时该模式降级）
    required_state_fields: ClassVar[list[str]] = []
    #: 可选的 State 字段
    optional_state_fields: ClassVar[list[str]] = []
    #: Judge 决策修饰器（如 satellite: {satellite_priority: true}）
    decision_modifiers: ClassVar[dict[str, Any]] = {}
    #: 评估规则（如 satellite: {objective: "seat_probability"}）
    evaluation_rules: ClassVar[dict[str, Any]] = {}

    # ---------------------------------------------------------------- 身份
    def spec(self) -> str:
        return f"{self.mode_id}:{self.type_id}" if self.type_id else self.mode_id

    def matches(self, game_context) -> bool:
        from .context import GameContext

        gc: GameContext = game_context
        if gc.game_mode != self.mode_id:
            return False
        if self.type_id is None:
            return True
        if gc.game_mode == "tournament":
            return (gc.tournament_type or "mtt") == self.type_id
        if gc.game_mode == "bounty":
            return (gc.bounty_type or "pko") == self.type_id
        return False

    # ---------------------------------------------------------------- pipeline
    @abstractmethod
    def plan_stages(self) -> list[Stage]:
        """该模式的执行计划（stage 内并行、stage 间顺序；含 $引用 路由模板）。"""

    def plan_kind(self) -> str:
        """该模式默认的 request.kind 值（CLI / 请求兼容）。"""
        return f"{self.mode_id}_{self.type_id or 'default'}_hand"

    # ---------------------------------------------------------------- 校验
    def validate_state(self, state: dict | None) -> tuple[bool, list[str]]:
        """校验 State 是否满足模式要求；返回 (ok, missing_fields)。"""
        if not self.required_state_fields:
            return True, []
        state = state or {}
        missing = [f for f in self.required_state_fields if f not in state]
        return not missing, missing

    # ---------------------------------------------------------------- Judge 输入
    def extra_judge_inputs(self) -> dict:
        """额外注入 Judge 的 $引用 模板（如 bounty 输出）。"""
        return {}

    # ---------------------------------------------------------------- 元信息
    def meta(self) -> dict:
        return {
            "mode": self.mode_id, "type": self.type_id, "spec": self.spec(),
            "display_name": self.display_name,
            "supported_modes": list(self.supported_modes),
            "required_state_fields": list(self.required_state_fields),
            "optional_state_fields": list(self.optional_state_fields),
            "decision_modifiers": dict(self.decision_modifiers),
            "evaluation_rules": dict(self.evaluation_rules),
            "agents": [t.agent for s in self.plan_stages() for t in s.tasks],
        }
