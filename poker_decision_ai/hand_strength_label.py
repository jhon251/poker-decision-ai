"""成手类别与"诈唬风险"标注（P30-9-Z）。

只做**事实标注**：用既有 `domain/poker/evaluator` 判定 hero 的成手类别与听牌，
并在"无摊牌价值却建议 bet/raise"时明确提示这等于诈唬、且引擎未建模弃牌权益。

**不参与任何策略计算**（不改动作/尺寸/权重/EV），纯标注与可读性。
"""
from __future__ import annotations

from typing import Any

LABEL_VERSION = "p30-9-z-hand-label-v1"

#: evaluator 的 category → 中文名
CATEGORY_ZH = {
    "straight_flush": "同花顺", "four_of_a_kind": "四条", "full_house": "葫芦",
    "flush": "同花", "straight": "顺子", "three_of_a_kind": "三条",
    "two_pair": "两对", "one_pair": "一对", "high_card": "高牌",
}
#: 有摊牌价值的类别（可showdown）
_SHOWDOWN = ("one_pair", "two_pair", "three_of_a_kind", "straight", "flush",
             "full_house", "four_of_a_kind", "straight_flush")
_RANKS = "23456789TJQKA"
_RANK_VAL = {r: i + 2 for i, r in enumerate(_RANKS)}


def _draws(cards: list) -> dict:
    """听牌事实：同花听牌（某花色 4 张）/ 顺子听牌（某 5 连窗内有 4 张）。"""
    suits: dict[str, int] = {}
    vals: set[int] = set()
    for c in cards:
        t = str(c)
        if len(t) < 2:
            continue
        vals.add(_RANK_VAL.get(t[0].upper(), 0))
        suits[t[1].lower()] = suits.get(t[1].lower(), 0) + 1
    flush_draw = any(v == 4 for v in suits.values())
    straight_draw = None
    for high in range(14, 4, -1):
        window = {high - i for i in range(5)}
        hit = len(window & vals)
        if hit == 5:
            straight_draw = "made"
            break
        if hit == 4:
            straight_draw = "draw"
    return {"flush_draw": flush_draw, "straight_draw": straight_draw}


def hero_hand_class(obs: Any) -> dict:
    """成手类别（翻后；需 2 张底牌 + ≥3 张公共牌）。无法判定 ⇒ category=None。"""
    o = obs if isinstance(obs, dict) else {}
    hole = [str(c) for c in (o.get("hole_cards") or [])]
    board = [str(c) for c in (o.get("board") or [])]
    out = {"version": LABEL_VERSION, "category": None, "zh": None,
           "showdown_value": None, "draws": {}, "reason": None}
    if len(hole) != 2 or len(board) < 3:
        out["reason"] = "insufficient_cards"
        return out
    try:
        from itertools import combinations
        from .domain.poker.cards import parse_cards
        from .domain.poker.evaluator import (CATEGORY_NAMES, evaluate5,
                                              evaluate7)
        cards = parse_cards(hole + board)
        if len(cards) == 7:
            score = evaluate7(cards)
        elif len(cards) == 5:
            score = evaluate5(cards)
        else:
            # 6 张（turn）等：取**最佳 5 张组合**（此前误用前 5 张 ⇒ 真同花被判成高牌）
            score = max(evaluate5(list(c)) for c in combinations(cards, 5))
        cat = CATEGORY_NAMES.get(int(score) >> 20)

        # ── P30-9-AV：底牌是否**参与**成手（而非"7 张最好"） ──────────────
        hole_r = [str(c)[0].upper() for c in hole]
        board_r = [str(c)[0].upper() for c in board]
        pocket = len(hole_r) == 2 and hole_r[0] == hole_r[1]
        hole_hits_board = any(r in board_r for r in hole_r)
        board_pair_ranks = sorted({r for r in board_r if board_r.count(r) >= 2})
        # 顺子/同花级别是否需要底牌：比较"含底牌"与"纯公共牌"的最强类别
        hole_cards = parse_cards(hole)
        board_cards = parse_cards(board)
        all_cards = hole_cards + board_cards
        best_with_hole = max(
            evaluate5(list(c)) for c in combinations(all_cards, 5)
            if any(x in c for x in hole_cards))
        best_board_only = None
        if len(board_cards) >= 5:
            best_board_only = max(evaluate5(list(c))
                                   for c in combinations(board_cards, 5))
        uses_hole_upgrade = best_board_only is None or \
            best_with_hole > best_board_only
        hero_contributes = bool(pocket or hole_hits_board or
                                 (uses_hole_upgrade and
                                  (int(best_with_hole) >> 20) >= 4))
        plays_board = (not hero_contributes) and bool(board_pair_ranks)
        out["hero_contributes"] = hero_contributes
        out["plays_board"] = plays_board
        out["board_pair_ranks"] = board_pair_ranks
    except Exception as e:                             # noqa: BLE001
        out["reason"] = f"evaluator_error:{type(e).__name__}"
        return out
    out["category"] = cat
    out["zh"] = CATEGORY_ZH.get(cat)
    # 摊牌价值只看"自有成手"：仅打公共牌 ⇒ 无自有摊牌价值
    out["showdown_value"] = bool(out.get("hero_contributes")) and cat in _SHOWDOWN
    out["draws"] = _draws(hole + board)
    return out


def bluff_risk_note(fd: dict, obs: Any) -> dict:
    """无**自有**摊牌价值时的诚实标注（只加说明，不改动作/尺寸）。

    * `bet`/`raise`（无成手）⇒ 诈唬提示（原行为，但判据改为"底牌是否参与成手"）；
    * `call`（面对下注且无自有成手）⇒ P30-9-AV 新增：说明该跟注依赖对手范围很宽，
      且引擎未建模隐含赔率/弃牌权益（用户实测反馈："公对没中牌却让我 call"）。
    """
    if not isinstance(fd, dict):
        return fd
    action = str(fd.get("action") or "").strip().lower()
    if action not in ("bet", "raise", "call"):
        return fd
    hc = hero_hand_class(obs)
    if hc.get("showdown_value") is True:
        return fd                      # 有自有成手 ⇒ 不提示
    if action == "call":
        note = ("ℹ 跟注提示：你在本街**没有自有成手**"
                + ("（仅在打公共牌的对子）" if hc.get("plays_board") else "（高牌）")
                + "——该跟注**依赖对手下注范围足够宽**；引擎自述『单街近似/无弃牌权益"
                  "假设』且未建模隐含赔率，故本条无法被引擎证明，请自行判断。")
        fd["reasoning"] = f"{fd.get('reasoning') or ''}（{note}）"
        risk = dict(fd.get("advice_risk") or {})
        risk.setdefault("kind", "call_without_showdown_value")
        risk["call_without_showdown_value"] = {
            "hand_class": hc.get("category"), "hand_class_zh": hc.get("zh"),
            "plays_board": hc.get("plays_board"),
            "board_pair_ranks": hc.get("board_pair_ranks")}
        fd["advice_risk"] = risk
        sem = fd.get("action_semantics")
        fd["action_semantics"] = ({**sem, "hand_class": hc}
                                   if isinstance(sem, dict)
                                   else {"version": LABEL_VERSION,
                                          "hand_class": hc})
        return fd
    if hc.get("category") != "high_card" and not hc.get("plays_board"):
        return fd
    draws = hc.get("draws") or {}
    note = ("⚠ 本次为**无摊牌价值的激进建议**：你的成手是「高牌」"
            + ("（但存在听牌）" if (draws.get("flush_draw")
                                     or draws.get("straight_draw") == "draw") else "（无对子、无听牌）")
            + "——bet/raise 等于**诈唬**；引擎自述『单街近似/无弃牌权益假设』，"
              "即它**没有建模对手弃牌**，因此本条建议无法由引擎证明，请自行判断是否愿意诈唬。")
    fd["reasoning"] = f"{fd.get('reasoning') or ''}（{note}）"
    fd["advice_risk"] = {"kind": "bluff_no_showdown_value",
                          "hand_class": hc.get("category"),
                          "hand_class_zh": hc.get("zh"),
                          "draws": draws,
                          "note": "无摊牌价值的激进建议＝诈唬；引擎未建模弃牌权益"}
    sem = fd.get("action_semantics")
    if isinstance(sem, dict):
        fd["action_semantics"] = {**sem, "hand_class": hc}
    else:
        fd["action_semantics"] = {"version": LABEL_VERSION, "hand_class": hc}
    return fd
