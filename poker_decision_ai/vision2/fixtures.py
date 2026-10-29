"""Vision Agent V2 — synthetic_test_only fixtures。

所有 fixture 均为**合成测试数据**（synthetic_test_only=True），只用于验证
schema/validator/assembly 契约，**不代表任何真实视觉能力**。禁止冒充真实数据。
"""
from __future__ import annotations

from .schemas import RawAction, RawSeat, RawVisionState, UncertaintyEntry


def six_seats(hero_seat: int = 3, button_seat: int = 2) -> list[RawSeat]:
    seats = []
    for n in range(1, 7):
        hero = (n == hero_seat)
        seats.append(RawSeat(
            seat=n, occupied=True,
            anonymous_player_id=f"player_{n}",
            stack=100.0, current_bet=0.0,
            folded=False, all_in=False,
            in_hand=hero or n in (1, 2),
            visible_action=None,
        ))
    return seats


def good_6max_flop() -> RawVisionState:
    """全字段合法 6-max flop（synthetic）。"""
    seats = six_seats(hero_seat=3, button_seat=2)
    seats[0].stack = 98.5          # seat1 已投 1.5
    seats[0].current_bet = 1.5
    seats[1].stack = 97.0
    seats[1].current_bet = 1.5
    seats[1].in_hand = False
    seats[2].stack = 96.0
    seats[2].current_bet = 4.0
    seats[3].stack = 100.0
    seats[3].in_hand = False
    seats[4].stack = 100.0
    seats[4].in_hand = False
    return RawVisionState(
        source="mock", synthetic_test_only=True,
        poker_variant="nlhe", game_type="cash", table_size=6,
        blinds={"sb": 0.5, "bb": 1.0}, street="flop",
        hero_seat=3, hero_hole_cards=["As", "Kd"],
        hero_stack=96.0, hero_current_bet=0.0, hero_amount_to_call=4.0,
        hero_all_in=False, hero_acted=False,
        board={"flop": ["Qh", "7d", "2c"], "turn": None, "river": None},
        button_seat=2, pot=14.0, side_pots=None,
        seats=seats,
        action_history=[
            RawAction(player="player_1", action="fold", amount=None,
                      street="preflop", order=1, confidence=0.95),
            RawAction(player="player_2", action="raise", amount=4.0,
                      street="preflop", order=2, confidence=0.9),
            RawAction(player="player_3", action="call", amount=4.0,
                      street="preflop", order=3, confidence=0.9),
        ],
        confidence=0.9, field_confidences={"hole_cards": 0.95, "board": 0.95,
                                           "pot": 0.9, "to_call": 0.9},
        uncertainty=[],
    )


def hero_card_uncertain() -> RawVisionState:
    """Hero 第二张牌看不清 → null + uncertain（禁止猜测）。"""
    r = good_6max_flop()
    r.hero_hole_cards = ["As", None]
    r.uncertainty = [UncertaintyEntry(field="hero.hole_cards[1]",
                                      status="uncertain", confidence=0.3)]
    return r


def duplicate_card() -> RawVisionState:
    """Hero 与 Board 出现重复牌 → invalid。"""
    r = good_6max_flop()
    r.hero_hole_cards = ["As", "Kd"]
    r.board = {"flop": ["As", "7d", "2c"], "turn": None, "river": None}
    return r


def negative_pot() -> RawVisionState:
    r = good_6max_flop()
    r.pot = -5.0
    return r


def bet_exceeds_stack() -> RawVisionState:
    r = good_6max_flop()
    r.hero_current_bet = 200.0        # > stack 96，非 all-in
    r.hero_all_in = False
    return r


def street_regression() -> RawVisionState:
    r = good_6max_flop()
    r.action_history = [
        RawAction(player="player_1", action="call", amount=4.0,
                  street="flop", order=1, confidence=0.9),
        RawAction(player="player_2", action="call", amount=4.0,
                  street="preflop", order=2, confidence=0.9),   # 倒退
    ]
    return r


def folded_then_action() -> RawVisionState:
    r = good_6max_flop()
    r.action_history = [
        RawAction(player="player_1", action="fold", amount=None,
                  street="preflop", order=1, confidence=0.9),
        RawAction(player="player_1", action="bet", amount=2.0,
                  street="flop", order=2, confidence=0.9),       # folded 后再行动
    ]
    return r


def no_hero() -> RawVisionState:
    r = good_6max_flop()
    r.hero_seat = None
    return r


def multiple_hero() -> RawVisionState:
    r = good_6max_flop()
    r.seats = six_seats(hero_seat=3, button_seat=2)
    r.seats[0].in_hand = True       # 双 hero 语义由重复座位模拟：另造 seat 重复
    r.seats = [RawSeat(seat=3, occupied=True, anonymous_player_id="player_3a",
                       stack=96.0, in_hand=True)] + r.seats
    return r


def preflop_minimal() -> RawVisionState:
    return RawVisionState(
        source="mock", synthetic_test_only=True,
        poker_variant="nlhe", game_type="cash", table_size=6,
        blinds={"sb": 0.5, "bb": 1.0}, street="preflop",
        hero_seat=3, hero_hole_cards=["Ts", "Ts"],
        hero_stack=100.0, hero_current_bet=0.0, hero_amount_to_call=1.0,
        hero_all_in=False, hero_acted=False,
        board={"flop": None, "turn": None, "river": None},
        button_seat=2, pot=1.5,
        seats=six_seats(hero_seat=3, button_seat=2),
        action_history=[],
        confidence=0.9, uncertainty=[],
    )


def button_missing() -> RawVisionState:
    """button 不可见 → position 一律 None（不猜）。"""
    r = good_6max_flop()
    r.button_seat = None
    return r


ALL_FIXTURES = {
    "good_6max_flop": good_6max_flop,
    "hero_card_uncertain": hero_card_uncertain,
    "duplicate_card": duplicate_card,
    "negative_pot": negative_pot,
    "bet_exceeds_stack": bet_exceeds_stack,
    "street_regression": street_regression,
    "folded_then_action": folded_then_action,
    "no_hero": no_hero,
    "multiple_hero": multiple_hero,
    "preflop_minimal": preflop_minimal,
    "button_missing": button_missing,
}
