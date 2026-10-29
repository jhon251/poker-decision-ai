"""GameContext: 全系统统一的游戏模式上下文（V1.1 可插拔游戏模式架构）。

任何需要做决策的 Agent 都能从 ctx.game_context 读到当前模式，而不是自己去猜。
模式规格字符串（mode spec）：

    cash                     现金桌
    tournament               任意锦标赛类型
    tournament:mtt           普通 MTT / Freezeout
    tournament:satellite     卫星赛
    bounty                   任意赏金赛
    bounty:pko               Progressive Knockout
    bounty:ko                固定赏金 Knockout
    bounty:mystery           神秘赏金

Agent 通过 supported_modes 声明能力，用 match_mode_spec 判定激活。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

#: 顶层模式
MODE_CASH = "cash"
MODE_TOURNAMENT = "tournament"
MODE_BOUNTY = "bounty"

#: 锦标赛子类型
TOURNAMENT_TYPES = ("mtt", "satellite", "freezeout")
#: 赏金子类型
BOUNTY_TYPES = ("pko", "ko", "mystery")


def normalize_type(mode: str, mode_type: str | None) -> str | None:
    if mode_type is None:
        return None
    t = str(mode_type).lower()
    if mode == MODE_TOURNAMENT and t in TOURNAMENT_TYPES:
        return t
    if mode == MODE_BOUNTY and t in BOUNTY_TYPES:
        return t
    return t


@dataclass
class GameContext:
    """统一模式上下文。字段按模式动态启用，未启用的保持 None / 默认值。"""

    game_mode: str = MODE_CASH                  # cash | tournament | bounty
    mode_type: str | None = None                # mtt|satellite|freezeout|pko|ko|mystery
    is_cash: bool = True
    is_tournament: bool = False
    is_bounty: bool = False
    is_satellite: bool = False
    bounty_type: str | None = None              # 仅 bounty 模式（pko|ko|mystery）
    tournament_type: str | None = None          # 仅 tournament 模式
    stakes: str | None = None                   # 现金桌级别（NL10 ...）
    blinds: dict = field(default_factory=dict)  # {sb, bb}
    ante: float = 0.0
    players_remaining: int | None = None
    players_paid: int | None = None             # 钱圈人数
    hero_stack: float | None = None
    villain_stack: float | None = None
    hero_bounty: float = 0.0                    # 货币单位
    villain_bounty: float = 0.0
    tournament_stage: str | None = None
    source: str = "default"                     # request|config|hand|state|default
    extra: dict = field(default_factory=dict)

    # ------------------------------------------------------------ mode spec
    def spec(self) -> str:
        """规范模式字符串：cash | tournament:<type> | bounty:<type>"""
        if self.game_mode == MODE_CASH:
            return MODE_CASH
        if self.game_mode == MODE_BOUNTY:
            t = self.bounty_type or self.mode_type or "pko"
            return f"{MODE_BOUNTY}:{t}"
        t = self.tournament_type or self.mode_type or "mtt"
        return f"{MODE_TOURNAMENT}:{t}"

    # ------------------------------------------------------------ serialization
    def to_dict(self) -> dict:
        return {
            "game_mode": self.game_mode,
            "mode_type": self.mode_type,
            "is_cash": self.is_cash,
            "is_tournament": self.is_tournament,
            "is_bounty": self.is_bounty,
            "is_satellite": self.is_satellite,
            "bounty_type": self.bounty_type,
            "tournament_type": self.tournament_type,
            "stakes": self.stakes,
            "blinds": dict(self.blinds),
            "ante": self.ante,
            "players_remaining": self.players_remaining,
            "players_paid": self.players_paid,
            "hero_stack": self.hero_stack,
            "villain_stack": self.villain_stack,
            "hero_bounty": self.hero_bounty,
            "villain_bounty": self.villain_bounty,
            "tournament_stage": self.tournament_stage,
            "source": self.source,
            "extra": dict(self.extra),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "GameContext":
        if not isinstance(d, dict):
            raise ValueError("GameContext.from_dict expects a dict")
        mode = str(d.get("game_mode") or MODE_CASH).lower()
        if mode not in (MODE_CASH, MODE_TOURNAMENT, MODE_BOUNTY):
            raise ValueError(f"unknown game_mode '{mode}'")
        gc = cls(
            game_mode=mode,
            mode_type=normalize_type(mode, d.get("mode_type") or d.get("tournament_type") or d.get("bounty_type")),
            stakes=d.get("stakes"),
            blinds=dict(d.get("blinds") or {}),
            ante=float(d.get("ante") or 0.0),
            players_remaining=d.get("players_remaining"),
            players_paid=d.get("players_paid"),
            hero_stack=d.get("hero_stack"),
            villain_stack=d.get("villain_stack"),
            hero_bounty=float(d.get("hero_bounty") or 0.0),
            villain_bounty=float(d.get("villain_bounty") or 0.0),
            tournament_stage=d.get("tournament_stage"),
            source=str(d.get("source") or "dict"),
            extra=dict(d.get("extra") or {}),
        )
        gc._refresh_flags()
        return gc

    def _refresh_flags(self) -> None:
        self.is_cash = self.game_mode == MODE_CASH
        self.is_tournament = self.game_mode == MODE_TOURNAMENT
        self.is_bounty = self.game_mode == MODE_BOUNTY
        if self.game_mode == MODE_TOURNAMENT:
            self.tournament_type = normalize_type(MODE_TOURNAMENT, self.mode_type)
            self.bounty_type = None
            self.is_satellite = self.tournament_type == "satellite"
        elif self.game_mode == MODE_BOUNTY:
            self.bounty_type = normalize_type(MODE_BOUNTY, self.mode_type)
            self.tournament_type = None
            self.is_satellite = False
        else:
            self.bounty_type = None
            self.tournament_type = None
            self.is_satellite = False

    def __post_init__(self) -> None:
        self._refresh_flags()


# ---------------------------------------------------------------- spec matching
def _spec_matches(spec: str, ctx: GameContext) -> bool:
    spec = spec.strip().lower()
    if spec in ("*", "all"):
        return True
    parts = spec.split(":")
    mode = parts[0]
    if mode not in (MODE_CASH, MODE_TOURNAMENT, MODE_BOUNTY):
        return False
    if ctx.game_mode != mode:
        return False
    if len(parts) == 1:
        return True                      # "tournament" / "bounty" 匹配该模式的任意子类型
    sub = parts[1]
    if ctx.game_mode == MODE_TOURNAMENT:
        return (ctx.tournament_type or "mtt") == sub
    return (ctx.bounty_type or "pko") == sub


def match_mode_spec(supported: list[str], ctx: GameContext | None) -> bool:
    """Agent 是否支持当前模式。ctx=None（旧路径）时默认放行。"""
    if ctx is None:
        return True
    if not supported:
        return True
    return any(_spec_matches(s, ctx) for s in supported)


# ---------------------------------------------------------------- resolution
def resolve_game_context(request: dict | None, config=None,
                         state: dict | None = None) -> GameContext:
    """解析 GameContext。优先级：

    1. request.game_context（dict，显式）
    2. request.game_mode / tournament_type / bounty_type（CLI 传入）
    3. config.yaml 的 game_mode（用户手动配置，不改代码）
    4. request.hand_history_text 预解析（cash/mtt/satellite/bounty 推断）
    5. state.game_type / tournament 块（已有状态时）
    6. request.kind 提示（cash_hand→cash；tournament_hand→tournament:mtt）
    7. 默认 cash
    """
    request = request or {}
    cfg_root = getattr(config, "root", None)
    cfg_mode = cfg_root.get("game_mode") if cfg_root is not None else None

    # 1) 显式 game_context
    gc_in = request.get("game_context")
    if isinstance(gc_in, dict) and gc_in.get("game_mode"):
        return GameContext.from_dict(gc_in)

    # 2) request 显式字段
    mode = str(request.get("game_mode") or "").lower()
    if mode in (MODE_CASH, MODE_TOURNAMENT, MODE_BOUNTY):
        gc = GameContext(game_mode=mode,
                         mode_type=request.get("tournament_type") or request.get("bounty_type"),
                         source="request")
        return gc

    # 3) config.yaml 手动模式
    if cfg_mode in (MODE_CASH, MODE_TOURNAMENT, MODE_BOUNTY):
        gc = GameContext(game_mode=cfg_mode,
                         mode_type=(cfg_root.get("tournament_type") if cfg_root is not None else None)
                                   or (cfg_root.get("bounty_type") if cfg_root is not None else None),
                         source="config")
        return gc

    # 4) 手牌历史预解析推断
    hh = request.get("hand_history_text")
    if hh:
        inferred = _infer_from_hand_history(hh)
        if inferred is not None:
            gc = GameContext(game_mode=inferred[0], mode_type=inferred[1], source="hand")
            return gc

    # 5) 已有 state
    if isinstance(state, dict) and state.get("game_type"):
        gt = str(state["game_type"])
        if gt == "cash":
            gc = GameContext(game_mode=MODE_CASH, source="state")
            return gc
        if gt == "satellite":
            gc = GameContext(game_mode=MODE_TOURNAMENT, mode_type="satellite", source="state")
            return gc
        if gt == "mtt":
            t = state.get("tournament") or {}
            bt = t.get("bounty_type") if isinstance(t, dict) else None
            if bt in BOUNTY_TYPES or t.get("is_bounty"):
                gc = GameContext(game_mode=MODE_BOUNTY, mode_type=bt or "pko", source="state")
                return gc
            gc = GameContext(game_mode=MODE_TOURNAMENT, mode_type="mtt", source="state")
            return gc

    # 6) kind 提示（V1 兼容）
    kind = str(request.get("kind") or "")
    if kind == "tournament_hand":
        return GameContext(game_mode=MODE_TOURNAMENT, mode_type="mtt", source="kind")
    if kind in ("cash_hand", "replay_hand"):
        return GameContext(game_mode=MODE_CASH, source="kind")

    # 7) 默认
    return GameContext(game_mode=MODE_CASH, source="default")


def _infer_from_hand_history(text: str) -> tuple[str, str | None] | None:
    """低成本预解析：仅判断模式，完整解析交给 State Agent。"""
    try:
        from poker_decision_ai.domain.poker.hand_history import parse_hand_history

        hand = parse_hand_history(text)
    except Exception:
        return None
    if hand.game == "satellite":
        return MODE_TOURNAMENT, "satellite"
    if hand.game == "mtt":
        t = hand.tournament or {}
        bt = t.get("bounty_type")
        if bt in BOUNTY_TYPES or t.get("is_bounty"):
            return MODE_BOUNTY, bt or "pko"
        return MODE_TOURNAMENT, "mtt"
    if hand.game == "cash":
        return MODE_CASH, None
    return None


def enrich_from_state(gc: GameContext, state: dict | None) -> GameContext:
    """用 State Agent 输出充实上下文（盲注/筹码/赏金/阶段），不改变模式。"""
    if not isinstance(state, dict):
        return gc
    if not gc.blinds:
        bb = float(state.get("bb") or 0.0)
        sb = float(state.get("sb") or 0.0)
        if bb or sb:
            gc.blinds = {"sb": sb, "bb": bb}
    gc.ante = float(state.get("ante") or gc.ante or 0.0)
    t = state.get("tournament") if isinstance(state.get("tournament"), dict) else {}
    if t:
        gc.players_remaining = t.get("players_remaining") or gc.players_remaining
        gc.players_paid = t.get("itm_players") or gc.players_paid
        gc.tournament_stage = t.get("stage") or gc.tournament_stage
    players = state.get("players") or []
    for p in players:
        if not isinstance(p, dict):
            continue
        bounty = p.get("bounty")
        if p.get("is_hero"):
            gc.hero_stack = float(p.get("stack") or 0.0) or gc.hero_stack
            if bounty:
                gc.hero_bounty = float(bounty)
        else:
            if bounty and float(bounty) > gc.villain_bounty:
                gc.villain_bounty = float(bounty)
                gc.villain_stack = float(p.get("stack") or 0.0) or gc.villain_stack
    if gc.hero_bounty > 0 and gc.game_mode == MODE_TOURNAMENT:
        # 状态带赏金但模式未声明：升级为 bounty 模式（来自手牌数据的事实）
        gc.game_mode = MODE_BOUNTY
        gc.mode_type = t.get("bounty_type") or "pko"
        gc._refresh_flags()
        gc.source = "state"
    return gc
