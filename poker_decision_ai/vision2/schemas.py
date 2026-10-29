"""Vision Agent V2 数据契约（Raw / Normalized / 校验结果）。

硬规则：Raw 字段默认值一律 None（null），**不设** 0/[]/"" 兜底——"缺失"与
"空值"必须区分。uncertainty 显式记录每个未知字段。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

#: 行动枚举（视觉可确认的行动，含 allin；未知 → None）
ACTIONS = ("fold", "check", "call", "bet", "raise", "allin")
#: 街道顺序（用于"street 不得倒退"校验）
STREET_RANK = {"preflop": 0, "flop": 1, "turn": 2, "river": 3}
#: 有效街道
STREETS = tuple(STREET_RANK)


@dataclass
class UncertaintyEntry:
    field: str
    status: str                       # "uncertain" | "missing"
    confidence: float | None = None

    def to_dict(self) -> dict:
        return {"field": self.field, "status": self.status,
                "confidence": self.confidence}


@dataclass
class RawSeat:
    """单座位视觉事实（只观察，不推导 position）。"""
    seat: int | None = None
    occupied: bool | None = None
    anonymous_player_id: str | None = None     # player_n 或稳定匿名 id；禁止真实名
    stack: float | None = None
    current_bet: float | None = None
    folded: bool | None = None
    all_in: bool | None = None
    in_hand: bool | None = None
    visible_action: str | None = None          # ACTIONS 之一或 None


@dataclass
class RawAction:
    """视觉可确认的单条行动；无法确认的字段必须 None。"""
    player: str | None = None                  # anonymous_player_id
    action: str | None = None                  # ACTIONS 之一或 None
    amount: float | None = None
    street: str | None = None
    order: int | None = None
    confidence: float | None = None


@dataclass
class RawVisionState:
    """Raw Vision Observation（只描述视觉所见；无推导、无决策）。"""
    source: str = "vision"                     # vision | mock | passthrough
    synthetic_test_only: bool = True           # V0：synthetic fixture 恒 True
    captured_at: str | None = None

    # Game
    poker_variant: str | None = None           # "nlhe"
    game_type: str | None = None               # "cash"
    table_size: int | None = None
    blinds: dict | None = None                 # {sb, bb, ante}，各值可 null
    street: str | None = None

    # Hero
    hero_seat: int | None = None
    hero_hole_cards: list | None = None        # 长度 2；元素可为 None（不确定）
    hero_stack: float | None = None
    hero_current_bet: float | None = None
    hero_amount_to_call: float | None = None
    hero_all_in: bool | None = None
    hero_acted: bool | None = None

    # Board
    board: dict | None = None                  # {flop:[3], turn:[1], river:[1]}；元素可 None
    button_seat: int | None = None

    # Pot
    pot: float | None = None
    side_pots: list | None = None              # 元素可 None

    # Players / Action
    seats: list | None = None                  # list[RawSeat]
    action_history: list | None = None         # list[RawAction]

    # 不确定性 / 置信度
    confidence: float | None = None
    field_confidences: dict | None = None
    uncertainty: list | None = None            # list[UncertaintyEntry]


@dataclass
class ValidationResult:
    valid: bool = True
    errors: list = field(default_factory=list)       # [{code, field, message}]
    warnings: list = field(default_factory=list)     # [{code, field, message}]

    def to_dict(self) -> dict:
        return {"valid": self.valid, "errors": list(self.errors),
                "warnings": list(self.warnings)}


@dataclass
class NormalizedVisionState:
    """经 schema + 确定性 validator 后的标准化状态（下游唯一消费对象）。"""
    source: str = "vision"
    synthetic_test_only: bool = True
    validation: ValidationResult = field(default_factory=ValidationResult)
    uncertainty: list = field(default_factory=list)   # list[UncertaintyEntry dict]
    game: dict = field(default_factory=dict)
    hero: dict = field(default_factory=dict)
    players: list = field(default_factory=list)
    board: dict = field(default_factory=dict)
    pot: dict = field(default_factory=dict)
    action_history: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "synthetic_test_only": self.synthetic_test_only,
            "validation": self.validation.to_dict(),
            "uncertainty": [u if isinstance(u, dict) else u.to_dict()
                            for u in self.uncertainty],
            "game": self.game, "hero": self.hero, "players": self.players,
            "board": self.board, "pot": self.pot,
            "action_history": self.action_history,
        }


def to_dict(obj: Any) -> Any:
    """dataclass/嵌套 dataclass → 纯 dict（None 保留为 null，不填默认值）。"""
    if obj is None:
        return None
    if isinstance(obj, (int, float, str, bool)):
        return obj
    if isinstance(obj, dict):
        return {k: to_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_dict(v) for v in obj]
    if hasattr(obj, "__dataclass_fields__"):
        return {f: to_dict(getattr(obj, f))
                for f in obj.__dataclass_fields__}
    return str(obj)
