"""EXE-1 — play GUI 冒烟（事件渲染/按钮状态机；不起 mainloop）。

无显示环境自动跳过（Tk 需要 display）。
"""
from __future__ import annotations

import pytest

tk = pytest.importorskip("tkinter")


@pytest.fixture()
def app():
    try:
        root = tk.Tk()
    except tk.TclError:                                   # 无显示环境
        pytest.skip("no display for Tk")
    root.withdraw()
    from cash_game_ui.play_gui import PlayGuiApp
    a = PlayGuiApp(root)
    yield a
    root.destroy()


def _ctx(**kw):
    from poker_decision_ai.play.match import HeroTurnContext
    base = dict(hand_index=0, decision_index=0, street="flop",
                board=["2c", "7d", "Th"], pot_chips=120, to_call_chips=20,
                legal_actions=["fold", "call", "raise", "all-in"],
                min_raise_to_chips=60, max_raise_to_chips=2000,
                all_in_level_chips=2000, call_level_chips=20,
                stacks_chips=[1980] * 6, street_bets_chips=[20] * 6,
                folded_seats=[5], button=0, position_label="BB",
                hero_hole=["As", "Kd"], bb_chips=20, stake="nl2",
                stake_label="t")
    base.update(kw)
    return HeroTurnContext(**base)


def test_gui_renders_session_and_turn(app):
    app._dispatch({"type": "session_start", "stake": "nl2",
                   "stake_label": "$1/$2 测试桌", "n_hands": 3,
                   "lineup": [{"seat": i, "archetype": "recreational",
                               "bot_kind": "pool_bot", "bot_seed": 1}
                              for i in range(1, 6)]})
    app._dispatch({"type": "hand_start", "hand_index": 0, "n_hands": 3,
                   "button": 1, "hero_hole": ["As", "Kd"],
                   "stacks": [2000] * 6, "pot_chips": 30})
    app._dispatch({"type": "street", "street": "flop",
                   "board": ["2c", "7d", "Th"], "pot_chips": 120,
                   "stacks": [1980] * 6})
    app._dispatch({"type": "hero_turn", "context": _ctx(), "advisor": None})
    assert app.turn_ctx is not None
    assert str(app.btn_fold["state"]) == "normal"
    assert str(app.btn_check["state"]) == "disabled"      # 面对下注无过牌
    assert str(app.btn_raise["state"]) == "normal"
    # 加注滑条范围 = min_raise_to ~ max_raise_to（bb）
    assert app.scale.cget("from") == pytest.approx(3.0)
    assert app.scale.cget("to") == pytest.approx(100.0)


def test_gui_decision_and_hand_end(app):
    app._dispatch({"type": "hero_turn", "context": _ctx(), "advisor": None})
    app._dispatch({"type": "decision_result", "hand_index": 0,
                   "street": "flop", "user_action": "call",
                   "user_size_chips": None,
                   "advisor": {"available": True, "action": "raise",
                               "size_bb": 6.0, "latency_ms": 12.3,
                               "provenance": {"source": "solver_hit",
                                              "node": "flop/SRP"}},
                   "advisor_agree": False})
    content = app.advice_txt.get("1.0", "end")
    assert "AI 参考" in content and "不一致" in content and "solver" in content
    app._dispatch({"type": "hand_end", "hand_index": 0,
                   "board": ["2c", "7d", "Th", "9s", "2h"],
                   "net_bb": 2.5, "net_bb_raked": 2.3,
                   "terminal_reason": "showdown", "pot_chips": 300,
                   "revealed": [{"seat": 3, "hole": ["Qh", "Qc"]}],
                   "analysis": {"n_hero_decisions": 2, "n_advisor_available": 2,
                                "n_agree": 1,
                                "advisor_provenance_sources": ["solver_hit"]}})
    assert app.turn_ctx is None
    log = app.log_txt.get("1.0", "end")
    assert "本手结束" in log and "亮牌" in log
    # 动作按钮全部回到禁用
    assert str(app.btn_fold["state"]) == "disabled"


def test_gui_banned_imports():
    """红线：GUI 模块无捕捉/自动化 import（检查真实 import 语句而非注释）。"""
    import pathlib
    import re
    f = pathlib.Path(__file__).resolve().parent.parent / "cash_game_ui" / "play_gui.py"
    text = f.read_text(encoding="utf-8")
    imports = re.findall(r"^\s*(?:from\s+([\w.]+)\s+import|import\s+[\w.,\s]+)",
                         text, re.M)
    joined = " ".join(i or "" for i in imports)
    for b in ("capture_frame", "locate_coinpoker", "pyautogui", "win32gui",
              "cash_game_ui.runner", "vision2", "ctypes"):
        assert b not in joined, f"banned import {b!r} in play_gui"
