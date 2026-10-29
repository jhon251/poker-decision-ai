"""Deterministic hand-history parser for the documented V1 text format.

Supported format (PokerStars-like, simplified - see docs/HAND_HISTORY.md):

    PokerStars Hand #123456789: No Limit Hold'em ($0.05/$0.10) - 2025/01/01 12:00:00 ET
    Table 'Test' 6-max Seat #2 is the button
    Seat 1: Alice ($10.00)
    Seat 2: Bob ($12.50)
    Seat 3: Hero ($9.75)
    ...
    Alice: posts small blind $0.05
    Bob: posts big blind $0.10
    *** HOLE CARDS ***
    Dealt to Hero [As Kd]
    Carol: folds
    Dave: calls $0.10
    ...
    Bob: raises $0.20 to $0.30
    *** FLOP *** [Qh 7d 2c]
    Bob: checks
    Hero: bets $1.50
    Bob: calls $1.50
    *** TURN *** [Qh 7d 2c] [3s]
    ...
    Hero collected $5.95 from pot
    *** SUMMARY ***
    Total pot $5.95 | Rake $0.30

All-in actions: "Hero: raises $9.65 to $9.75 and is all-in"
Shows:         "Bob: shows [Qd Qs] (three of a kind, Queens)"
Tournament:    "PokerStars Hand #1: Tournament #123, $1+$0.1 USD Hold'em No Limit - Level I (10/20) - ..."
               "Seat 1: Alice (1500 in chips)"
               "*** SUMMARY *** Total pot ... | Tournament players remain: 45 (of 100) | Next payout $2.00"

The parser is deliberately line-based and strict: unknown lines produce
warnings (never silent mis-parse).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ParsedHand:
    """Raw normalized output of the parser (State Agent maps this to state)."""

    hand_id: str = ""
    game: str = "cash"                      # cash | mtt | satellite
    stakes: str = ""                        # NL10 (from blinds) - resolved by State Agent
    sb: float = 0.0
    bb: float = 0.0
    currency: str = "USD"
    button_seat: int | None = None
    max_players: int | None = None
    seats: list = field(default_factory=list)          # [{seat,name,stack,is_hero}]
    hero_id: str = ""
    hole_cards: list = field(default_factory=list)
    actions: list = field(default_factory=list)        # [{street,player,action,amount,all_in}]
    board: list = field(default_factory=list)          # flop+turn+river cards in order
    street_boards: dict = field(default_factory=dict)  # {flop: [...], turn: [...], river: [...]}
    winners: list = field(default_factory=list)        # [{player, amount}]
    shows: list = field(default_factory=list)          # [{player, cards, hand_desc}]
    pot: float = 0.0
    rake: float = 0.0
    tournament: dict | None = None                     # tournament info when mtt/satellite
    satellite_seats: int | None = None
    started_at: str | None = None                      # ISO8601（头部时间；缺失 → None）
    warnings: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "hand_id": self.hand_id, "game": self.game, "stakes": self.stakes,
            "sb": self.sb, "bb": self.bb, "currency": self.currency,
            "button_seat": self.button_seat, "max_players": self.max_players,
            "seats": self.seats, "hero_id": self.hero_id, "hole_cards": self.hole_cards,
            "actions": self.actions, "board": self.board, "street_boards": self.street_boards,
            "winners": self.winners, "shows": self.shows, "pot": self.pot, "rake": self.rake,
            "tournament": self.tournament, "satellite_seats": self.satellite_seats,
            "warnings": self.warnings,
        }


_HEADER = re.compile(r"^PokerStars Hand #(?P<id>[0-9A-Za-z]+):\s*(?P<kind>.+?)\s*\(\$?(?P<sb>\d+\.?\d*)/\$?(?P<bb>\d+\.?\d*)\s*(?P<ccy>[A-Z]*)\s*\)(?:\s*-\s*(?P<ts>\d{4}/\d{2}/\d{2}\s+\d{2}:\d{2}:\d{2})(?:\s+\w+)?)?")
_TOURNAMENT_HEADER = re.compile(r"^PokerStars Hand #(?P<id>[0-9A-Za-z]+):\s*Tournament\s*#(?P<tid>\d+),\s*(?P<desc>.+?)\s*-\s*Level\s*\S+\s*\((?P<sb>\d+)/(?P<bb>\d+)\)(?:\s*-\s*(?P<ts>\d{4}/\d{2}/\d{2}\s+\d{2}:\d{2}:\d{2})(?:\s+\w+)?)?")
_TABLE = re.compile(r"^Table\s+'(?P<name>[^']+)'\s+(?P<max>\d+)-max\s+Seat #(?P<btn>\d+)\s+is the button")
_SEAT = re.compile(r"^Seat (?P<seat>\d+):\s*(?P<name>.+?)\s+\(\$?(?P<stack>\d+\.?\d*)(?:\s+in chips)?(?:,\s+\$?(?P<bounty>\d+\.?\d*)\s+bounty)?\)")
_DEALT = re.compile(r"^Dealt to (?P<name>.+?)\s*\[(?P<cards>[^\]]+)\]")
_POST = re.compile(r"^(?P<player>.+?):\s+posts\s+(?:small blind|big blind|the ante)\s+\$?(?P<amount>\d+\.?\d*)")
_ACTION = re.compile(
    r"^(?P<player>.+?):\s+(?P<action>folds|checks|calls|bets|raises)"
    r"(?:\s+\$?(?P<amount>\d+\.?\d*)(?:\s+to\s+\$?(?P<to>\d+\.?\d*))?\s*(?P<allin>and is all-in)?)?\s*$"
)
_SHOW = re.compile(r"^(?P<player>.+?):\s+shows\s*\[(?P<cards>[^\]]+)\](?:\s*\((?P<desc>.*?)\))?")
_COLLECT = re.compile(r"^(?P<player>.+?):?\s+collected\s+\$?(?P<amount>\d+\.?\d*)\s+from pot")
_SUMMARY_POT = re.compile(r"Total pot\s+\$?(?P<pot>\d+\.?\d*)\s*\|\s*Rake\s+\$?(?P<rake>\d+\.?\d*)")
_SUMMARY_REMAIN = re.compile(r"Tournament players remain:\s*(?P<rem>\d+)\s*\(of\s*(?P<total>\d+)\)")
_SUMMARY_NEXT = re.compile(r"Next payout\s+\$?(?P<next>\d+\.?\d*)")
_SUMMARY_PAYOUTS = re.compile(r"^Payouts:\s*(?P<pay>[\d\.\s]+)\s*$")
_SUMMARY_ITM = re.compile(r"(?P<n>\d+)\s+players?\s+(?:paid|in the money)")
_SUMMARY_BOUNTY_TYPE = re.compile(r"^Bounty Type:\s*(?P<bt>pko|ko|mystery)", re.I)
_SUMMARY_BOUNTY_POOL = re.compile(r"Bounty Pool:\s+\$?(?P<pool>\d+\.?\d*)")
_SUMMARY_PRIZE_POOL = re.compile(r"Prize Pool:\s+\$?(?P<pool>\d+\.?\d*)")
_SUMMARY_MYSTERY_PHASE = re.compile(r"Mystery Bounty phase:\s*(?P<phase>active|pending)", re.I)
_SUMMARY_TOTAL_CHIPS = re.compile(r"Total chips:\s*(?P<chips>\d+\.?\d*)")
_STREET_RE = {
    "HOLE CARDS": "preflop",
    "FLOP": "flop",
    "TURN": "turn",
    "RIVER": "river",
    "SHOW DOWN": "showdown",
    "SUMMARY": "summary",
}
_BOARD_CARDS = re.compile(r"\[([^\]]+)\]")


def _header_ts_to_iso(ts: str) -> str:
    """头部时间 "2025/01/12 12:00:00" → ISO8601。"""
    from datetime import datetime

    try:
        return datetime.strptime(ts.strip(), "%Y/%m/%d %H:%M:%S").isoformat()
    except ValueError:
        return ts.strip()

STREET_MARKERS = ("*** HOLE CARDS ***", "*** FLOP ***", "*** TURN ***", "*** RIVER ***",
                  "*** SHOW DOWN ***", "*** SUMMARY ***")


def parse_hand_history(text: str) -> ParsedHand:
    hand = ParsedHand()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    street = "preflop"

    for ln in lines:
        if ln.startswith("*** "):
            for marker, st in _STREET_RE.items():
                if marker in ln.upper():
                    street = st
                    break
            if street in ("flop", "turn", "river"):
                for m in _BOARD_CARDS.finditer(ln):
                    cards = [c for c in m.group(1).split() if c]
                    if street == "flop":
                        hand.street_boards["flop"] = cards
                    elif street == "turn":
                        hand.street_boards["turn"] = cards[3:] if len(cards) >= 4 else cards
                    elif street == "river":
                        hand.street_boards["river"] = cards[4:] if len(cards) >= 5 else cards
            continue

        m = _TOURNAMENT_HEADER.match(ln) or _HEADER.match(ln)
        if m and street == "preflop" and not hand.hand_id:
            hand.hand_id = m.group("id")
            hand.sb = float(m.group("sb"))
            hand.bb = float(m.group("bb"))
            if m.groupdict().get("ts"):
                hand.started_at = _header_ts_to_iso(m.group("ts"))
            if "Tournament" in ln:
                hand.game = "mtt"
                desc = m.groupdict().get("desc", "Tournament")
                tinfo: dict = {
                    "tournament_id": m.groupdict().get("tid", ""),
                    "name": desc,
                    "sb": hand.sb, "bb": hand.bb,
                }
                # V1.1 赏金模式：从锦标赛描述识别 PKO/KO/Mystery
                dl = desc.lower()
                if "progressive knockout" in dl or "progressive bounty" in dl:
                    tinfo["bounty_type"] = "pko"
                    tinfo["is_bounty"] = True
                    tinfo["progressive"] = True
                elif "mystery bounty" in dl:
                    tinfo["bounty_type"] = "mystery"
                    tinfo["is_bounty"] = True
                elif "knockout" in dl or "bounty" in dl:
                    tinfo["bounty_type"] = "ko"
                    tinfo["is_bounty"] = True
                hand.tournament = tinfo
            continue

        m = _TABLE.match(ln)
        if m:
            hand.button_seat = int(m.group("btn"))
            hand.max_players = int(m.group("max"))
            continue

        m = _SEAT.match(ln)
        if m:
            d = m.groupdict()
            hand.seats.append({"seat": int(d["seat"]), "name": d["name"].strip(),
                               "stack": float(d["stack"]), "is_hero": False,
                               "bounty": float(d["bounty"]) if d.get("bounty") else 0.0})
            continue

        m = _DEALT.match(ln)
        if m:
            hand.hero_id = m.group("name").strip()
            hand.hole_cards = [c for c in m.group("cards").split() if c]
            for s in hand.seats:
                if s["name"] == hand.hero_id:
                    s["is_hero"] = True
            continue

        m = _POST.match(ln)
        if m:
            hand.actions.append({"street": street, "player": m.group("player").strip(),
                                 "action": "post", "amount": float(m.group("amount")), "all_in": False})
            continue

        m = _SHOW.match(ln)
        if m:
            hand.shows.append({"player": m.group("player").strip(),
                               "cards": [c for c in m.group("cards").split() if c],
                               "desc": (m.group("desc") or "").strip()})
            continue

        m = _COLLECT.match(ln)
        if m:
            hand.winners.append({"player": m.group("player").strip().rstrip(":"),
                                 "amount": float(m.group("amount"))})
            continue

        m = _ACTION.match(ln)
        if m and street != "summary":
            d = m.groupdict()
            amount = float(d["amount"]) if d["amount"] else 0.0
            # `to` 是该玩家本街下注后的总注额（raises $0.90 to $1.20 -> to=1.20）。
            # 保留它才能正确重放底池：加注者的实际投入 = to - 其本街已投入。
            to_amount = float(d["to"]) if d["to"] else None
            hand.actions.append({"street": street, "player": d["player"].strip(),
                                 "action": d["action"].rstrip("s"), "amount": amount,
                                 "to": to_amount, "all_in": bool(d["allin"])})
            continue

        if street == "summary":
            m = _SUMMARY_POT.search(ln)
            if m:
                hand.pot = float(m.group("pot"))
                hand.rake = float(m.group("rake"))
                continue
            m = _SUMMARY_REMAIN.search(ln)
            if m:
                hand.tournament = hand.tournament or {}
                hand.tournament["players_remaining"] = int(m.group("rem"))
                hand.tournament["total_players"] = int(m.group("total"))
                # 同一行可能携带卫星赛席位（"| 10 seats paid"）
                if "satellite" in ln.lower() or "seats paid" in ln.lower():
                    m2 = re.search(r"(\d+)\s+seats?", ln, re.I)
                    if m2:
                        hand.satellite_seats = int(m2.group(1))
                        hand.game = "satellite"
                continue
            m = _SUMMARY_NEXT.search(ln)
            if m:
                hand.tournament = hand.tournament or {}
                hand.tournament["next_payout"] = float(m.group("next"))
                continue
            m = _SUMMARY_PAYOUTS.match(ln)
            if m:
                hand.tournament = hand.tournament or {}
                hand.tournament["payouts"] = [float(x) for x in m.group("pay").split() if x]
                continue
            m = _SUMMARY_ITM.search(ln)
            if m:
                hand.tournament = hand.tournament or {}
                hand.tournament["itm_players"] = int(m.group("n"))
                continue
            m = _SUMMARY_BOUNTY_TYPE.match(ln)
            if m:
                hand.tournament = hand.tournament or {}
                hand.tournament["bounty_type"] = m.group("bt").lower()
                hand.tournament["is_bounty"] = True
                if m.group("bt").lower() == "pko":
                    hand.tournament["progressive"] = True
                continue
            m = _SUMMARY_BOUNTY_POOL.search(ln)
            if m:
                hand.tournament = hand.tournament or {}
                hand.tournament["total_bounty_pool"] = float(m.group("pool"))
                continue
            m = _SUMMARY_PRIZE_POOL.search(ln)
            if m:
                hand.tournament = hand.tournament or {}
                hand.tournament["prize_pool"] = float(m.group("pool"))
                continue
            m = _SUMMARY_MYSTERY_PHASE.search(ln)
            if m:
                hand.tournament = hand.tournament or {}
                hand.tournament["mystery_phase_active"] = m.group("phase").lower() == "active"
                continue
            m = _SUMMARY_TOTAL_CHIPS.search(ln)
            if m:
                hand.tournament = hand.tournament or {}
                hand.tournament["total_chips"] = float(m.group("chips"))
                continue
            if "satellite" in ln.lower() or "seats paid" in ln.lower():
                m2 = re.search(r"(\d+)\s+seats?", ln, re.I)
                if m2:
                    hand.satellite_seats = int(m2.group(1))
                    hand.game = "satellite"
            continue

        hand.warnings.append(f"unparsed line: {ln}")

    hand.board = []
    for st in ("flop", "turn", "river"):
        hand.board.extend(hand.street_boards.get(st, []))
    return hand


def state_ctx_from_parsed(hand: ParsedHand) -> dict:
    """Turn a parsed hand into the dict the State Agent consumes."""
    return {"parsed_hand": hand.to_dict()}
