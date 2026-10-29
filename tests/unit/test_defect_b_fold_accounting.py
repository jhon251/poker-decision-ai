"""Defect B 回归测试：AI 座弃牌记账（527 点差异的根因）。

Defect B = 遗留 runner 对 AI 座动作 `continue` 而未推进弃牌集合，导致
「已弃牌的 AI 座仍被视为在场对手」，主对手回退可能选中它。
S‴′ 运行所用 runner（`s3prime_runner`）以统一 `_adv_both` 记账 → 缺陷不存在。
本文件锁定该事实，防止回归。
"""
from __future__ import annotations

import ast
import pathlib

import pytest

from poker_decision_ai.pluribus_validation import AI_SEAT
from poker_decision_ai.research_unfreeze import s3_validation as s3
from poker_decision_ai.research_unfreeze import s3prime_runner as v2

H1, H2 = "MrBlue", "MrOrange"


def _fold(pid, folded, folded_all, hist):
    v2._adv_both({"action": "fold", "player": pid}, pid, "preflop",
                 folded, folded_all, hist)


def test_ai_fold_is_recorded_in_both_sets():
    folded, folded_all, hist = set(), set(), []
    _fold(AI_SEAT, folded, folded_all, hist)
    assert AI_SEAT in folded and AI_SEAT in folded_all
    assert folded == folded_all
    assert hist == []


def test_human_fold_records_hist_for_value_actions_only():
    folded, folded_all, hist = set(), set(), []
    v2._adv_both({"action": "call", "player": H1, "amount": 100.0}, H1,
                 "flop", folded, folded_all, hist)
    assert folded == folded_all == set()
    assert hist == [{"street": "flop", "player_id": H1, "action": "call",
                     "amount": 100.0}]
    _fold(H1, folded, folded_all, hist)
    assert folded == folded_all == {H1}


def test_folded_ai_is_not_selected_as_fallback_villain():
    """核心：AI 座弃牌后，主对手回退不得选中它。"""
    players = [AI_SEAT, H1, H2]
    hero = "Gogo"
    folded, folded_all, hist = set(), set(), []
    _fold(AI_SEAT, folded, folded_all, hist)
    picked = next((p for p in players if p != hero and p not in folded), None)
    assert picked == H1, "已弃牌的 AI 座被选为主对手（Defect B 复发）"


def test_legacy_sealed_runner_still_exhibits_defect_b():
    """留证：sealed S‴ runner 的 AI 分支确实未推进弃牌集合（历史缺陷）。"""
    src = pathlib.Path(s3.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    found = False
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and "AI_SEAT" in ast.unparse(node.test):
            body = ast.unparse(node)
            if "continue" in body and "_adv" not in body:
                found = True
    assert found, "sealed runner 的 AI 分支形态已变化，请重新审计 Defect B"


def test_v2_runner_has_no_ai_exception_in_fold_tracking():
    src = pathlib.Path(v2.__file__).read_text(encoding="utf-8")
    body = src[src.find("def _adv_both"):]
    body = body[:body.find("\n\n\n")]
    assert "AI_SEAT" not in body, "v2 记账路径出现 AI 例外（回归）"
    assert body.count("folded.add(pid)") == 1


def test_v2_uses_the_same_set_for_villain_and_reference():
    """同一 `folded` 同时驱动主对手回退与参照 → 不存在两套口径。"""
    src = pathlib.Path(v2.__file__).read_text(encoding="utf-8")
    assert "p not in folded" in src                      # 主对手回退
    assert "folded=folded_all, hero_cards=hole" in src    # 参照
    assert "build_villain_hands(h, pid, folded_all" in src
    # 两个集合只在 _adv_both 中同步写入，且写入条件相同
    assert src.count("folded_all.add(pid)") == 1
    assert src.count("folded.add(pid)") == 1


def test_no_ai_filter_in_fold_accounting_of_v2_loop():
    """循环体内不得出现「AI 座 continue 跳过记账」形态。"""
    src = pathlib.Path(v2.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    bad = []
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and "AI_SEAT" in ast.unparse(node.test):
            bad.append(ast.unparse(node.test))
    assert bad == [], f"v2 中出现 AI_SEAT 分支：{bad}"


@pytest.mark.parametrize("acts,expect", [
    ([("fold", AI_SEAT)], {AI_SEAT}),
    ([("call", H1), ("fold", AI_SEAT)], {AI_SEAT}),
    ([("fold", H1), ("fold", AI_SEAT)], {H1, AI_SEAT}),
    ([("raise", AI_SEAT)], set()),
])
def test_fold_set_sequences(acts, expect):
    folded, folded_all, hist = set(), set(), []
    for act, pid in acts:
        v2._adv_both({"action": act, "player": pid, "amount": 50.0}, pid,
                     "flop", folded, folded_all, hist)
    assert folded == folded_all == expect
