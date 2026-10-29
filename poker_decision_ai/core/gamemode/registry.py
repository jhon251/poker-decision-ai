"""ModeRegistry: 游戏模式注册表 + Agent 能力查询。

Manager 通过 ModeRegistry 解析 GameContext → Plugin → Plan，
通过 AgentRegistry.capabilities()/supports() 做声明式 Agent 激活。
"""
from __future__ import annotations

from typing import Any

from poker_decision_ai.core.errors import OrchestrationError

from .context import GameContext, match_mode_spec
from .plugin import GameModePlugin


class ModeRegistry:
    def __init__(self) -> None:
        self._plugins: list[GameModePlugin] = []

    def register(self, plugin: GameModePlugin) -> None:
        for existing in self._plugins:
            if existing.mode_id == plugin.mode_id and existing.type_id == plugin.type_id:
                raise OrchestrationError(f"duplicate mode plugin: {plugin.spec()}")
        self._plugins.append(plugin)

    def plugins(self) -> list[GameModePlugin]:
        return list(self._plugins)

    def resolve(self, game_context: GameContext) -> GameModePlugin:
        """按模式+子类型精确匹配；子类型无专属插件时回退该模式的默认插件。"""
        exact = [p for p in self._plugins if p.matches(game_context) and p.type_id is not None]
        if exact:
            return exact[0]
        default = [p for p in self._plugins
                   if p.matches(game_context) and p.type_id is None]
        if default:
            return default[0]
        for p in self._plugins:
            if p.matches(game_context):
                return p
        raise OrchestrationError(f"no GameModePlugin for {game_context.spec()}")

    def meta_all(self) -> list[dict]:
        return [p.meta() for p in sorted(self._plugins, key=lambda p: p.spec())]


def register_default_modes(registry: ModeRegistry) -> ModeRegistry:
    """注册 V1.1 全部内置模式插件（导入放局部避免循环依赖）。"""
    from poker_decision_ai.domain.poker.gamemodes import register_all

    register_all(registry)
    return registry


def agent_supports_mode(supported_modes: list[str], game_context: GameContext | None) -> bool:
    """声明式能力判定（引擎用它决定调用/跳过 Agent）。"""
    return match_mode_spec(supported_modes, game_context)


def capability_summary(registry: Any, mode_registry: ModeRegistry) -> dict:
    """{mode_spec: [可用 Agent 名]} —— 文档与校验用。"""
    modes = [p.spec() for p in mode_registry.plugins()]
    out: dict[str, list[str]] = {}
    for spec in modes:
        part = spec.split(":")
        gc = GameContext(game_mode=part[0], mode_type=part[1] if len(part) > 1 else None)
        out[spec] = [
            name for name in registry.names()
            if agent_supports_mode(registry.get(name).supported_modes, gc)
        ]
    return out
