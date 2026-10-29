"""合成牌局生成器（V2 Phase 4 Smoke Dataset：100 手现金桌）。

确定性（seed 固定）；行动顺序、底池、抽水、摊牌胜者严格一致
（摊牌胜者由确定性牌型评估器裁决）。生成结果直接用于 Dataset Qualification。

用法: .\\.venv\\Scripts\\python.exe scripts/generate_synthetic_hands.py
输出: data/hand_history/synthetic_cash_100/*.txt + data/hand_history/source.json
"""
from __future__ import annotations

import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from poker_decision_ai.domain.poker.cards import full_deck, parse_cards  # noqa: E402
from poker_decision_ai.domain.poker.evaluator import evaluate7, score_name  # noqa: E402

RNG = random.Random(42)
NAMES = ["Alice", "Bob", "Carol", "Dave", "Eve", "Frank", "Grace", "Heidi",
         "Ivan", "Judy", "Ken", "Lucy"]
STAKES = [("NL2", 0.02, 0.05), ("NL10", 0.10, 0.25), ("NL20", 0.25, 0.50)]
BASE_DATE = datetime(2025, 1, 1, 12, 0, 0)


def deal(deck: list, n: int) -> list:
    return [deck.pop() for _ in range(n)]


def make_hand(i: int) -> str:
    stake_name, sb, bb = STAKES[i % len(STAKES)]
    date = BASE_DATE + timedelta(days=i * 2, hours=RNG.randint(0, 23))
    ts = date.strftime("%Y/%m/%d %H:%M:%S")
    button = RNG.randint(1, 6)
    pool = list(NAMES)
    RNG.shuffle(pool)
    names = pool[:6]
    hero_seat = 3
    names[2] = "Hero"
    stacks = {s: round(RNG.uniform(20, 50) * bb * 4, 2) for s in range(1, 7)}
    # 座位顺时针：SB = 按钮+1（回绕），BB = SB+1
    sb_seat = button % 6 + 1
    bb_seat = sb_seat % 6 + 1

    deck = full_deck()
    RNG.shuffle(deck)
    hero_hole = [str(c) for c in deal(deck, 2)]
    villain_hole = {s: [str(c) for c in deal(deck, 2)] for s in range(1, 7)
                    if s != hero_seat}
    board = [str(c) for c in deal(deck, 5)]

    lines = [f"PokerStars Hand #{70000 + i}: No Limit Hold'em (${sb:.2f}/${bb:.2f} USD) - {ts} ET"]
    lines.append(f"Table 'Syn' 6-max Seat #{button} is the button")
    for s in range(1, 7):
        lines.append(f"Seat {s}: {names[s-1]} (${stacks[s]:.2f})")
    lines.append(f"{names[sb_seat-1]}: posts small blind ${sb:.2f}")
    lines.append(f"{names[bb_seat-1]}: posts big blind ${bb:.2f}")
    lines.append("*** HOLE CARDS ***")
    lines.append(f"Dealt to Hero [{hero_hole[0]} {hero_hole[1]}]")

    pot = round(sb + bb, 2)
    folded: set[int] = set()
    posted = {sb_seat: sb, bb_seat: bb}

    def act(seat: int, kind: str, amt: float = 0.0, to: float | None = None) -> str:
        nonlocal pot
        name = names[seat - 1]
        if kind == "fold":
            folded.add(seat)
            return f"{name}: folds"
        if kind == "check":
            return f"{name}: checks"
        if kind == "call":
            pot = round(pot + amt, 2)
            return f"{name}: calls ${amt:.2f}"
        if kind == "raise":
            inc = round(amt, 2)
            pot = round(pot + inc, 2)
            return f"{name}: raises ${inc:.2f} to ${to:.2f}"
        if kind == "bet":
            pot = round(pot + amt, 2)
            return f"{name}: bets ${amt:.2f}"
        return ""

    def active_seats() -> list[int]:
        return [s for s in range(1, 7) if s not in folded]

    # ---- 翻前：UTG(BB+1) 起顺时针，BB 最后 ----
    order = [(bb_seat + k - 1) % 6 + 1 for k in range(1, 7)]
    current_bet = bb
    last_aggressor = None
    hero_acted_preflop = False
    for s in order:
        if s in folded:
            continue
        if s == hero_seat:
            if RNG.random() < 0.6:
                to = round(max(3 * bb, current_bet * 3), 2)
                lines.append(act(s, "raise", round(to - current_bet, 2), to))
                last_aggressor = s
                current_bet = to
            else:
                lines.append(act(s, "call", round(current_bet - posted.get(s, 0.0), 2)))
            hero_acted_preflop = True
            continue
        r = RNG.random()
        if r < 0.55:
            lines.append(act(s, "fold"))
        elif r < 0.85:
            diff = round(current_bet - posted.get(s, 0.0), 2)
            if diff <= 0:
                lines.append(act(s, "check"))
            else:
                lines.append(act(s, "call", diff))
        else:
            to = round(current_bet * 3, 2)
            lines.append(act(s, "raise", round(to - current_bet, 2), to))
            last_aggressor = s
            current_bet = to

    active = active_seats()
    winner: int | None = None
    if len(active) <= 1:
        winner = active[0]          # 全员弃牌 → 无需公共牌
    else:
        # ---- 翻牌 ----
        lines.append(f"*** FLOP *** [{' '.join(board[:3])}]")
        flop_better = active[0]
        if RNG.random() < 0.55:
            bet = round(0.66 * pot, 2)
            lines.append(act(flop_better, "bet", bet))
            last_aggressor = flop_better
            for t in active:
                if t == flop_better:
                    continue
                if RNG.random() < 0.5:
                    lines.append(act(t, "call", bet))
                else:
                    lines.append(act(t, "fold"))
        else:
            for t in active:
                lines.append(act(t, "check"))
        active = active_seats()

    if winner is None and len(active) >= 2:
        # ---- 转牌/河牌：check/check 或 河牌下注 ----
        lines.append(f"*** TURN *** [{' '.join(board[:4])}]")
        for t in active:
            lines.append(act(t, "check"))
        lines.append(f"*** RIVER *** [{' '.join(board)}]")
        showdown = RNG.random() < 0.35 and hero_seat in active
        if showdown:
            for t in active:
                lines.append(act(t, "check"))
            hero_score = evaluate7(parse_cards(hero_hole + board))
            opp = next(s for s in active if s != hero_seat)
            opp_score = evaluate7(parse_cards(villain_hole[opp] + board))
            hero_win = hero_score >= opp_score
            lines.append("*** SHOW DOWN ***")
            lines.append(f"{names[opp-1]}: shows [{villain_hole[opp][0]} {villain_hole[opp][1]}] ({score_name(opp_score)})")
            lines.append(f"Hero: shows [{hero_hole[0]} {hero_hole[1]}] ({score_name(hero_score)})")
            winner = hero_seat if hero_win else opp
        else:
            better = last_aggressor if last_aggressor in active else active[0]
            bet = round(0.5 * pot, 2)
            lines.append(act(better, "bet", bet))
            for t in active:
                if t == better:
                    continue
                if t == hero_seat and RNG.random() < 0.5:
                    lines.append(act(t, "call", bet))
                else:
                    lines.append(act(t, "fold"))
            active = active_seats()
            winner = better if better in active else active[0]
    if winner is None:
        winner = hero_seat if hero_seat not in folded else active_seats()[0]

    rake = round(pot * 0.05, 2)
    net = round(pot - rake, 2)
    lines.append(f"{names[winner-1]}: collected ${net:.2f} from pot")
    lines.append("*** SUMMARY ***")
    lines.append(f"Total pot ${pot:.2f} | Rake ${rake:.2f}")
    return "\n".join(lines) + "\n"


def main() -> int:
    out_dir = Path("data/hand_history/synthetic_cash_100")
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(100):
        (out_dir / f"hand_{70000 + i}.txt").write_text(make_hand(i), encoding="utf-8")
    Path("data/hand_history/source.json").write_text(
        '{"source": "synthetic_seed_42", "license": "internal-synthetic",\n'
        ' "description": "确定性合成牌局（scripts/generate_synthetic_hands.py，seed=42）：100 手现金桌。仅用于 Pipeline 验证，不能用于策略有效性证明。",\n'
        ' "generated_by": "scripts/generate_synthetic_hands.py", "seed": 42, "hands": 100}',
        encoding="utf-8")
    print(f"generated {len(list(out_dir.glob('*.txt')))} hands → {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
