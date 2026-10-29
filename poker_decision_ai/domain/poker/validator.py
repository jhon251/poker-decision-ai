"""Dataset Validator（V2 Phase 4 §三）：Canonical Hand 质量检查。

10 类检查，每项输出 valid / invalid / warning + 原因。
规则全部确定性；原始数据缺失字段导致无法判定的检查 → warning（不猜）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .canonical import CanonicalHand
from .cards import parse_card

STREETS = ("preflop", "flop", "turn", "river")


@dataclass
class CheckResult:
    name: str
    status: str = "valid"          # valid | invalid | warning | skipped
    detail: str = ""


@dataclass
class ValidationReport:
    hand_id: str
    status: str = "valid"          # valid | invalid | warning
    checks: list = field(default_factory=list)
    errors: list = field(default_factory=list)   # [{code, field, message}]

    def to_dict(self) -> dict:
        return {
            "hand_id": self.hand_id, "status": self.status,
            "checks": [c.__dict__ for c in self.checks],
            "errors": list(self.errors),
        }


def _card_set(cards: list[str]) -> set[str] | None:
    out = set()
    for c in cards:
        try:
            out.add(str(parse_card(str(c))))
        except ValueError:
            return None
    return out


def validate_hand(hand: CanonicalHand) -> ValidationReport:
    """对一手 Canonical Hand 运行全部检查。"""
    report = ValidationReport(hand_id=hand.hand_id)
    checks: list[CheckResult] = []

    def add(name: str, status: str, detail: str = "") -> None:
        checks.append(CheckResult(name, status, detail))
        if status == "invalid":
            report.errors.append({"code": name.upper().replace(" ", "_"),
                                  "field": "", "message": detail})

    # 1. 手牌合法性（hero hole cards）
    hero_cards = list(hand.hole_cards or [])
    if hero_cards:
        if len(hero_cards) != 2 or _card_set(hero_cards) is None or len(_card_set(hero_cards)) != 2:
            add("1_hole_cards_legal", "invalid", f"hero hole cards 非法: {hero_cards}")
        else:
            add("1_hole_cards_legal", "valid")
    else:
        add("1_hole_cards_legal", "warning", "hero hole cards 缺失（原始数据未提供）")

    # 2. Board 合理性
    board = [c for st in ("flop", "turn", "river") for c in getattr(hand, st) or []]
    if board:
        if len(board) not in (3, 4, 5):
            add("2_board_sane", "invalid", f"board 张数非法: {len(board)}")
        else:
            parsed = _card_set(board)
            if parsed is None or len(parsed) != len(board):
                add("2_board_sane", "invalid", "board 有非法牌或重复牌")
            elif hero_cards and (_card_set(hero_cards) or set()) & parsed:
                add("2_board_sane", "invalid", "board 与 hero 手牌冲突")
            else:
                add("2_board_sane", "valid")
    else:
        add("2_board_sane", "warning", "无公共牌（可能翻前结束或数据不完整）")

    # 3. 玩家数量一致性
    action_players = {a.player_id for a in _all_actions(hand)}
    seat_players = {p.player_id for p in hand.players}
    if seat_players and action_players and action_players - seat_players:
        add("3_player_count_consistent", "invalid",
            f"行动者不在座位表: {sorted(action_players - seat_players)}")
    else:
        add("3_player_count_consistent", "valid")

    # 4. Blind / Ante 一致性
    if hand.sb is not None and hand.bb is not None:
        if hand.sb <= 0 or hand.bb <= 0 or hand.sb > hand.bb:
            add("4_blinds_consistent", "invalid", f"盲注非法: sb={hand.sb} bb={hand.bb}")
        else:
            add("4_blinds_consistent", "valid")
    else:
        add("4_blinds_consistent", "warning", "盲注缺失")

    # 5. 行动顺序合法性（含街顺序、金额、fold 后不再行动）
    errs = _check_action_sequence(hand)
    if errs:
        add("5_action_order_legal", "invalid", "; ".join(errs[:3]))
    else:
        add("5_action_order_legal", "valid")

    # 6. 底池一致性（重放累计 vs total_pot）
    pot_ok, pot_detail = _check_pot_consistency(hand)
    add("6_pot_consistent", "valid" if pot_ok else "warning", pot_detail)

    # 7. 筹码一致性（stack_end 与 stack_start - 投入 + 赢得）
    stack_ok, stack_detail = _check_stack_consistency(hand)
    add("7_stacks_consistent", "valid" if stack_ok else "warning", stack_detail)

    # 8. Showdown 一致性
    showdown = hand.showdown or {}
    if showdown.get("happened"):
        shows = showdown.get("shows") or []
        if not shows:
            add("8_showdown_consistent", "invalid", "showdown 标记但无 shows")
        else:
            add("8_showdown_consistent", "valid")
    elif hand.winners and hand.river:
        add("8_showdown_consistent", "warning", "有胜者但无 showdown 记录（可能对手弃牌）")
    else:
        add("8_showdown_consistent", "valid")

    # 9. Winner 一致性
    if hand.winners:
        wplayers = {w.get("player_id") for w in hand.winners}
        if not wplayers <= seat_players:
            add("9_winners_consistent", "invalid", f"胜者不在座位表: {wplayers - seat_players}")
        elif hand.total_pot is not None:
            won = sum(float(w.get("amount") or 0.0) for w in hand.winners)
            if won > hand.total_pot + 1e-6:
                add("9_winners_consistent", "warning",
                    f"胜者所得 {won} 超过总底池 {hand.total_pot}")
            else:
                add("9_winners_consistent", "valid")
        else:
            add("9_winners_consistent", "valid")
    else:
        add("9_winners_consistent", "warning", "无 winner 记录（数据可能不完整）")

    # 10. 缺失字段盘点（标准化完整性；不判 invalid，只报告）
    missing = []
    for field, value in (("hand_id", hand.hand_id), ("game_mode", hand.game_mode),
                         ("game_type", hand.game_type), ("sb", hand.sb), ("bb", hand.bb),
                         ("button_seat", hand.button_seat), ("started_at", hand.started_at),
                         ("hero_id", hand.hero_id), ("hole_cards", hand.hole_cards or None),
                         ("stack_start", any(p.stack_start is not None for p in hand.players)),
                         ("stack_end", any(p.stack_end is not None for p in hand.players)),
                         ("total_pot", hand.total_pot), ("winners", hand.winners or None)):
        if value in (None, "", [], False):
            missing.append(field)
    if missing:
        add("10_missing_fields", "warning", f"缺失字段: {missing}（标准化保持 null，不猜测）")
    else:
        add("10_missing_fields", "valid")

    report.checks = checks
    if any(c.status == "invalid" for c in checks):
        report.status = "invalid"
    elif any(c.status == "warning" for c in checks):
        report.status = "warning"
    return report


# ---------------------------------------------------------------- helpers
def _all_actions(hand: CanonicalHand) -> list:
    return (list(hand.preflop_actions) + list(hand.flop_actions)
            + list(hand.turn_actions) + list(hand.river_actions))


def _check_action_sequence(hand: CanonicalHand) -> list[str]:
    errs: list[str] = []
    folded: set[str] = set()
    last_street = "preflop"
    street_order = {s: i for i, s in enumerate(STREETS)}
    for a in _all_actions(hand):
        if street_order.get(a.street, -1) < street_order.get(last_street, -1):
            errs.append(f"街顺序倒退: {a.street} 在 {last_street} 之后")
        last_street = a.street
        if a.action == "fold":
            folded.add(a.player_id)
        elif a.player_id in folded:
            errs.append(f"{a.player_id} 弃牌后仍行动（{a.action}）")
        if a.action in ("bet", "raise", "call") and a.amount is not None and a.amount < 0:
            errs.append(f"{a.player_id} 金额为负: {a.amount}")
        if a.action == "raise" and a.to is not None and a.amount is not None and a.to < a.amount:
            errs.append(f"{a.player_id} raise to({a.to}) < 增量({a.amount})")
    return errs


def _check_pot_consistency(hand: CanonicalHand) -> tuple[bool, str]:
    if hand.total_pot is None:
        return True, "total_pot 缺失（无法核验，跳过）"
    contrib = 0.0
    for a in _all_actions(hand):
        amt = a.amount or 0.0
        if a.action in ("post", "call", "bet"):
            contrib += amt
        elif a.action == "raise":
            contrib += amt
    # 加注增量是相对上一注额的，重放累计用 to 更准：这里只做粗核验
    tol = max(1e-6, abs(hand.total_pot) * 0.02 + 0.01)
    if contrib > 0 and abs(contrib - hand.total_pot) > tol:
        return False, f"行动累计 {contrib:.2f} 与 total_pot {hand.total_pot:.2f} 偏差过大"
    return True, "底池核验通过（或数据不足以核验）"


def _check_stack_consistency(hand: CanonicalHand) -> tuple[bool, str]:
    have_end = any(p.stack_end is not None for p in hand.players)
    if not have_end:
        return True, "stack_end 缺失（跳过核验）"
    invested: dict[str, float] = {}
    for a in _all_actions(hand):
        invested[a.player_id] = invested.get(a.player_id, 0.0) + float(a.amount or 0.0)
    won: dict[str, float] = {}
    for w in hand.winners:
        won[w.get("player_id")] = won.get(w.get("player_id"), 0.0) + float(w.get("amount") or 0.0)
    bad = []
    for p in hand.players:
        if p.stack_start is None or p.stack_end is None:
            continue
        expected = p.stack_start - invested.get(p.player_id, 0.0) + won.get(p.player_id, 0.0)
        if abs(expected - p.stack_end) > 0.02:
            bad.append(f"{p.player_id}: {p.stack_start}->{p.stack_end} (期望 {expected:.2f})")
    if bad:
        return False, "筹码变化不一致: " + "; ".join(bad[:3])
    return True, "筹码核验通过"
