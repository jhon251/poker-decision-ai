"""Poker 内置游戏模式插件注册（V1.1）。

新增玩法 = 新增一个 Plugin 文件并在这里注册一行；Manager/Judge/Learning/
Evaluation/Memory/Research 零改动。
"""
from __future__ import annotations

from poker_decision_ai.core.gamemode.registry import ModeRegistry

from .bounty import BountyPkoPlugin
from .cash import CashGamePlugin
from .freezeout import FreezeoutPlugin
from .ko import BountyKoPlugin
from .mtt import MttPlugin
from .mystery import MysteryBountyPlugin
from .satellite import SatellitePlugin

PLUGIN_CLASSES = [
    CashGamePlugin,
    MttPlugin,
    SatellitePlugin,
    FreezeoutPlugin,
    BountyPkoPlugin,
    BountyKoPlugin,
    MysteryBountyPlugin,
]


def register_all(registry: ModeRegistry) -> ModeRegistry:
    for cls in PLUGIN_CLASSES:
        registry.register(cls())
    return registry


def build_mode_registry() -> ModeRegistry:
    return register_all(ModeRegistry())
