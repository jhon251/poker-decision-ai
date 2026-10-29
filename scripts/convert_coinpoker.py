"""CoinPoker 牌谱 → V1（PokerStars 风格）转换器。

用法：
  python scripts/convert_coinpoker.py <CoinPoker导出.txt> <输出.txt>

转换规则（CoinPoker 导出 ≈ PokerStars 风格，差异逐项归一）：
  * 头部：CoinPoker Hand #N: NLH (₮a/b) DATE CST → PokerStars Hand #N:
    No Limit Hold'em ($a/b USD) - DATE CST
  * ₮ 金额符号 → $；"posts auto big blind" → "posts big blind"
  * 空发牌行（Dealt to X，无牌）丢弃
  * X: ALLIN ₮A → 按街上下文转成 calls/bets/raises ... and is all-in
  * X: RETURN ₮A 丢弃（未获跟注返还；底池/赢家行已含真实结果）
  * SUMMARY 仅保留 Total pot/Rake 行；Splash Fee、Game ended、
    座位结算行、空 Board 行丢弃；"showed ... and won/lost with X"
    在 SHOWDOWN 段归一为 "shows [cards] (desc)"
"""
from __future__ import annotations

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MONEY = "₮"
_HEADER = re.compile(
    r"^CoinPoker Hand #(?P<id>[0-9A-Za-z]+):\s*NLH\s*\(\s*(?:₮|\$)?"
    r"(?P<sb>\d+\.?\d*)\s*/\s*(?:₮|\$)?(?P<bb>\d+\.?\d*)\s*\)"
    r"\s*(?P<ts>.+)$")
_STREET_MARKERS = ("*** HOLE CARDS ***", "*** FLOP ***", "*** TURN ***",
                   "*** RIVER ***", "*** SHOW DOWN ***", "*** SUMMARY ***")
_STREET_OF = {"*** HOLE CARDS ***": "preflop", "*** FLOP ***": "flop",
              "*** TURN ***": "turn", "*** RIVER ***": "river"}
_DEALT_EMPTY = re.compile(r"^Dealt to .+$")
_DEALT_CARDS = re.compile(r"^Dealt to (?P<name>.+?)\s*\[(?P<cards>[^\]]+)\]")
_SHOWDOWN_SHOW = re.compile(
    r"^(?P<player>.+?):\s+shows\s*\[(?P<cards>[^\]]+)\]\s*"
    r"\((?P<desc>.+?)\)\s*$")
_SUMMARY_SHOWED = re.compile(
    r"^Seat \d+:\s*(?P<player>.+?)\s+showed\s*\[(?P<cards>[^\]]+)\]"
    r"\s+and\s+(?P<res>won|lost)(?:\s+with\s+(?P<desc>.+?))?\s*$")
_ALLIN = re.compile(r"^(?P<player>.+?):\s+ALLIN\s+(?:₮|\$)(?P<a>\d+\.?\d*)\s*$")


def _money(line: str) -> str:
    return line.replace(MONEY, "$")


def _split_hands(text: str) -> list[list[str]]:
    hands: list[list[str]] = []
    cur: list[str] = []
    for line in text.splitlines():
        if line.startswith("CoinPoker Hand #"):
            if cur:
                hands.append(cur)
            cur = [line]
        elif cur is not None:
            cur.append(line)
    if cur:
        hands.append(cur)
    return hands


def convert_hand(lines: list[str]) -> tuple[str | None, list[str]]:
    """一手 CoinPoker 文本 → V1 文本；返回 (v1_text, issues)。"""
    out: list[str] = []
    issues: list[str] = []
    street = "preflop"
    committed: dict[str, float] = {}   # 本街每人已投入
    level = 0.0                        # 本街当前下注水平（preflop 含盲注）

    def new_street(name: str):
        nonlocal street, committed, level
        street = name
        committed = {}
        level = 0.0

    for raw in lines:
        line = raw.rstrip()
        if not line.strip():
            continue
        if any(m in line for m in _STREET_MARKERS):
            out.append(line)
            marker = next(m for m in _STREET_MARKERS if m in line)
            if marker in _STREET_OF:
                new_street(_STREET_OF[marker])
            continue

        m = _HEADER.match(line)
        if m:
            out.append(
                f"PokerStars Hand #{m['id']}: No Limit Hold'em "
                f"(${m['sb']}/${m['bb']} USD) - {m['ts']}")
            continue
        if line.startswith("Seat ") and MONEY in line:
            out.append(_money(line))
            continue
        if line.startswith("Table "):
            out.append(line)
            continue

        blind = re.match(
            r"^(?P<p>.+?):\s+posts\s+(?P<kind>auto )?big blind\s+"
            r"(?:₮|\$)(?P<a>\d+\.?\d*)\s*$", line)
        if blind:
            amount = float(blind["a"])
            out.append(f"{blind['p']}: posts big blind ${amount:.2f}")
            committed[blind["p"]] = committed.get(blind["p"], 0.0) + amount
            level = max(level, amount)
            continue
        blind_sb = re.match(
            r"^(?P<p>.+?):\s+posts\s+small blind\s+(?:₮|\$)(?P<a>\d+\.?\d*)\s*$",
            line)
        if blind_sb:
            amount = float(blind_sb["a"])
            out.append(f"{blind_sb['p']}: posts small blind ${amount:.2f}")
            committed[blind_sb["p"]] = committed.get(blind_sb["p"], 0.0) + amount
            level = max(level, amount)
            continue

        dealt = _DEALT_CARDS.match(line)
        if dealt:
            out.append(f"Dealt to {dealt['name']} [{dealt['cards']}]")
            continue
        if _DEALT_EMPTY.match(line):
            continue                              # 空发牌行（非 Hero）丢弃

        allin = _ALLIN.match(_money(line))
        if allin:
            player, a = allin["player"], float(allin["a"])
            put = committed.get(player, 0.0)
            to_call = max(level - put, 0.0)
            if a <= to_call:
                out.append(f"{player}: calls ${a:.2f} and is all-in")
            elif to_call > 0:
                out.append(f"{player}: raises ${a - put:.2f} to ${a:.2f}"
                           " and is all-in")
            else:
                out.append(f"{player}: bets ${a:.2f} and is all-in")
            committed[player] = committed.get(player, 0.0) + a
            level = max(level, committed[player])
            continue

        ret = re.match(r"^(?P<p>.+?):\s+RETURN\s+", _money(line))
        if ret:
            continue                              # 未获跟注返还行丢弃

        m_act = re.match(
            r"^(?P<p>.+?):\s+(?P<a>folds|checks|calls|bets|raises)"
            r"(?:\s+(?:₮|\$)(?P<amt>\d+\.?\d*)"
            r"(?:\s+to\s+(?:₮|\$)(?P<to>\d+\.?\d*))?)?\s*$", line)
        if m_act:
            p_, a_ = m_act["p"], m_act["a"]
            amt = float(m_act["amt"]) if m_act["amt"] else None
            to = float(m_act["to"]) if m_act["to"] else None
            newline = _money(line)
            if a_ == "calls" and amt:
                committed[p_] = committed.get(p_, 0.0) + amt
            elif a_ == "bets" and amt:
                committed[p_] = amt
                level = amt
            elif a_ == "raises" and to is not None:
                committed[p_] = to
                level = to
            out.append(newline)
            continue

        sd_show = _SHOWDOWN_SHOW.match(line)
        if sd_show:
            out.append(f"{sd_show['player']}: shows [{sd_show['cards']}] "
                       f"({sd_show['desc']})")
            continue
        sum_showed = _SUMMARY_SHOWED.match(line)
        if sum_showed:
            # SHOWDOWN 段已有 shows 行；SUMMARY 里的重复行丢弃
            continue

        pot = re.match(r"^Total pot\s+(?:₮|\$)(?P<p>\d+\.?\d*)\s*\|\s*"
                       r"Rake\s+(?:₮|\$)(?P<r>\d+\.?\d*)", line)
        if pot:
            out.append(f"Total pot ${pot['p']} | Rake ${pot['r']}")
            continue
        if line.startswith("Board ["):
            # 公共牌已由街道标记行承载；preflop 空板行丢弃
            continue
        if (line.startswith("Splash Fee") or "Splash Fee" in line
                or line.startswith("Hand was run once")
                or line.startswith("Game ended")):
            continue
        issues.append(line)
    return ("\n".join(out) + "\n") if out else None, issues


def convert_file(src: str, dst: str) -> tuple[int, int, int]:
    from pathlib import Path
    text = Path(src).read_text(encoding="utf-8", errors="replace")
    hands = _split_hands(text)
    ok = fail = 0
    outs: list[str] = []
    from poker_decision_ai.domain.poker.hand_history import parse_hand_history
    for lines in hands:
        v1, issues = convert_hand(lines)
        if not v1:
            fail += 1
            continue
        parsed = parse_hand_history(v1)
        # 严格校验：能解析出 hero 底牌与至少 1 条行动，才算转换成功
        if parsed.hole_cards and parsed.actions:
            outs.append(v1)
            ok += 1
            if parsed.warnings:
                issues.extend(f"W:{w}" for w in parsed.warnings[:2])
        else:
            fail += 1
            issues.extend(f"PARSE_FAIL:{w}" for w in parsed.warnings[:2])
    Path(dst).write_text("\n".join(outs), encoding="utf-8")
    return ok, fail, len(hands)


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    ok, fail, total = convert_file(sys.argv[1], sys.argv[2])
    print(f"CoinPoker → V1：共 {total} 手，转换成功 {ok}，失败 {fail}")
    print(f"输出：{sys.argv[2]}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
