"""V1.1 游戏模式架构单元测试：GameContext 解析 / 模式匹配 / 插件注册 / 能力过滤。"""
from __future__ import annotations

import pytest

from poker_decision_ai.core.gamemode import (
    GameContext,
    ModeRegistry,
    match_mode_spec,
    register_default_modes,
    resolve_game_context,
)
from poker_decision_ai.core.orchestration.registry import register_all_agents


# ---------------------------------------------------------------- GameContext
def test_game_context_flags():
    gc = GameContext(game_mode="tournament", mode_type="satellite")
    assert gc.is_tournament and gc.is_satellite and not gc.is_cash and not gc.is_bounty
    assert gc.spec() == "tournament:satellite"

    gc2 = GameContext(game_mode="bounty", mode_type="pko")
    assert gc2.is_bounty and gc2.bounty_type == "pko" and gc2.spec() == "bounty:pko"

    gc3 = GameContext(game_mode="cash")
    assert gc3.is_cash and gc3.spec() == "cash"


def test_game_context_roundtrip():
    gc = GameContext(game_mode="bounty", mode_type="ko", hero_bounty=120.0,
                     villain_bounty=800.0, players_remaining=32, blinds={"sb": 100, "bb": 200})
    back = GameContext.from_dict(gc.to_dict())
    assert back.spec() == "bounty:ko"
    assert back.hero_bounty == 120.0 and back.villain_bounty == 800.0
    assert back.blinds == {"sb": 100, "bb": 200}


# ---------------------------------------------------------------- spec 匹配
def test_match_mode_spec():
    ctx_t = GameContext(game_mode="tournament", mode_type="mtt")
    ctx_s = GameContext(game_mode="tournament", mode_type="satellite")
    ctx_b = GameContext(game_mode="bounty", mode_type="pko")
    ctx_c = GameContext(game_mode="cash")

    assert match_mode_spec(["*"], ctx_t)
    assert match_mode_spec(["tournament"], ctx_t) and match_mode_spec(["tournament"], ctx_s)
    assert match_mode_spec(["tournament:satellite"], ctx_s)
    assert not match_mode_spec(["tournament:satellite"], ctx_t)
    assert match_mode_spec(["bounty"], ctx_b)
    assert match_mode_spec(["bounty:pko"], ctx_b)
    assert not match_mode_spec(["bounty"], ctx_c)
    assert not match_mode_spec(["tournament"], ctx_c)
    assert match_mode_spec([], ctx_c)          # 空声明 = 默认放行（V1 兼容）
    assert match_mode_spec(["*"], None)        # 无 GameContext = 放行


# ---------------------------------------------------------------- 解析优先级
def test_resolve_priority_request_over_config():
    cfg = type("Cfg", (), {"root": type("R", (), {"get": lambda self, k, d=None: "bounty" if k == "game_mode" else None})()})()
    # request 显式覆盖 config
    gc = resolve_game_context({"game_mode": "cash"}, cfg)
    assert gc.game_mode == "cash" and gc.source == "request"
    # config 生效
    gc = resolve_game_context({}, cfg)
    assert gc.game_mode == "bounty" and gc.source == "config"


def test_resolve_infer_from_hand_history():
    mtt_text = "PokerStars Hand #1: Tournament #9, $10+$1 USD Hold'em No Limit - Level VI (50/100) - 2025/01/01 12:00:00 ET\n"
    gc = resolve_game_context({"hand_history_text": mtt_text})
    assert gc.spec() == "tournament:mtt" and gc.source == "hand"

    pko_text = ("PokerStars Hand #1: Tournament #9, $10+$1 USD Hold'em No Limit "
                "Progressive Knockout - Level VI (50/100) - 2025/01/01 12:00:00 ET\n")
    gc = resolve_game_context({"hand_history_text": pko_text})
    assert gc.spec() == "bounty:pko" and gc.source == "hand"

    cash_text = "PokerStars Hand #1: No Limit Hold'em ($0.10/$0.25 USD) - 2025/01/01 12:00:00 ET\n"
    gc = resolve_game_context({"hand_history_text": cash_text})
    assert gc.spec() == "cash"


def test_resolve_kind_fallback_and_default():
    assert resolve_game_context({"kind": "tournament_hand"}).spec() == "tournament:mtt"
    assert resolve_game_context({"kind": "cash_hand"}).spec() == "cash"
    assert resolve_game_context({"kind": "whatever"}).spec() == "cash"   # 默认兜底
    assert resolve_game_context({}).spec() == "cash"


# ---------------------------------------------------------------- 插件注册
def test_mode_registry_plugins():
    reg = register_default_modes(ModeRegistry())
    specs = {p.spec() for p in reg.plugins()}
    assert specs == {"cash", "tournament:mtt", "tournament:satellite",
                     "tournament:freezeout", "bounty:pko", "bounty:ko", "bounty:mystery"}

    plugin = reg.resolve(GameContext(game_mode="bounty", mode_type="pko"))
    assert plugin.spec() == "bounty:pko"
    agents = [t.agent for s in plugin.plan_stages() for t in s.tasks]
    for name in ("bounty", "bounty_equity", "bounty_icm", "bounty_strategy"):
        assert name in agents, f"{name} 缺失于 bounty:pko pipeline"
    assert "icm" in agents and "blind_pressure" in agents
    assert "satellite" not in agents and "icm_exploit" not in agents

    sat = reg.resolve(GameContext(game_mode="tournament", mode_type="satellite"))
    sat_agents = [t.agent for s in sat.plan_stages() for t in s.tasks]
    assert "satellite" in sat_agents
    assert sat.decision_modifiers.get("satellite_priority") is True

    cash = reg.resolve(GameContext(game_mode="cash"))
    cash_agents = [t.agent for s in cash.plan_stages() for t in s.tasks]
    for banned in ("icm", "satellite", "bounty", "bounty_equity", "tournament_stage"):
        assert banned not in cash_agents, f"现金桌不应包含 {banned}"


# ---------------------------------------------------------------- 能力过滤
def test_agent_capability_filter():
    registry = register_all_agents()
    gc_cash = GameContext(game_mode="cash")
    gc_mtt = GameContext(game_mode="tournament", mode_type="mtt")
    gc_pko = GameContext(game_mode="bounty", mode_type="pko")

    assert registry.supports("equity", gc_cash) and registry.supports("equity", gc_pko)
    assert registry.supports("icm", gc_mtt) and registry.supports("icm", gc_pko)
    assert not registry.supports("icm", gc_cash)
    assert not registry.supports("satellite", gc_mtt)
    assert registry.supports("satellite", GameContext(game_mode="tournament", mode_type="satellite"))
    assert "bounty" not in registry.supported_by_mode(gc_cash)
    # 现金桌下 tournament_stage 被过滤
    assert not registry.supports("tournament_stage", gc_cash)


def test_mystery_plugin_includes_mystery_agent():
    reg = register_default_modes(ModeRegistry())
    plugin = reg.resolve(GameContext(game_mode="bounty", mode_type="mystery"))
    agents = [t.agent for s in plugin.plan_stages() for t in s.tasks]
    assert "mystery_bounty" in agents
    assert "bounty" in agents
