"""可插拔游戏模式架构（V1.1）：GameContext / GameModePlugin / ModeRegistry。

领域无关：未来的 Football 等领域也可以复用这套"模式声明 + 能力注册"机制。
"""
from .context import (
    BOUNTY_TYPES,
    MODE_BOUNTY,
    MODE_CASH,
    MODE_TOURNAMENT,
    TOURNAMENT_TYPES,
    GameContext,
    enrich_from_state,
    match_mode_spec,
    normalize_type,
    resolve_game_context,
)
from .plugin import GameModePlugin
from .registry import (
    ModeRegistry,
    agent_supports_mode,
    capability_summary,
    register_default_modes,
)

__all__ = [
    "BOUNTY_TYPES", "MODE_BOUNTY", "MODE_CASH", "MODE_TOURNAMENT",
    "TOURNAMENT_TYPES", "GameContext", "enrich_from_state", "match_mode_spec",
    "normalize_type", "resolve_game_context", "GameModePlugin", "ModeRegistry",
    "agent_supports_mode", "capability_summary", "register_default_modes",
]
