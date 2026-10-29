# -*- coding: utf-8 -*-
"""call/check 语义契约回归测试（P30-9-D：非冻结边界对齐）。

修复前：观察里 `to_call` 缺失（UNKNOWN）被当 0 ⇒ 已观察合法集明明是
fold/call/raise（面对下注），却输出 bet/check 且文案写「需跟注 0.00bb」
—— agent 分不清 call 与 check。

修复后契约（本测试固化）：
  1) 状态语义：to_call 缺失 ⇒ UNKNOWN；facing_bet 由已观察合法集推导；
  2) 输出边界：动作/备选必须落在已观察合法集内（跨世界按语义族改写）；
  3) 改写时尺寸置空（跨世界尺寸不可沿用）；
  4) 面对下注且 to_call 数值未知时，文案不得出现「需跟注 0.00bb」。

注意：`poker_decision_ai/agents/judge/agent.py` 是**冻结文件**（哈希
239436fae7299c47）——本修复**不触碰**它，只在非冻结边界对齐。
"""
from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from poker_decision_ai import action_semantics as asem                # noqa: E402
import poker_decision_ai.agents.state.agent as state_agent            # noqa: E402

SEATS = [{"seat": 1, "name": "Alice", "stack": 25.0, "in_hand": False},
         {"seat": 2, "name": "Bob", "stack": 31.25, "in_hand": True},
         {"seat": 3, "name": "Hero", "stack": 24.38, "in_hand": True},
         {"seat": 4, "name": "Carol", "stack": 20.5, "in_hand": False},
         {"seat": 5, "name": "Dave", "stack": 28.25, "in_hand": False},
         {"seat": 6, "name": "Eve", "stack": 25.0, "in_hand": False}]


def _obs(to_call, current_bet, actions, text="x"):
    return {"source": "mock", "hero_seat": 3, "button_seat": 2, "street": "flop",
            "hole_cards": ["As", "Kd"], "board": ["Qh", "7d", "2c"],
            "seats": SEATS, "pot": 10.10, "current_bet": current_bet,
            "to_call": to_call, "blinds": {"sb": 0.10, "bb": 0.25}, "ante": 0.0,
            "available_actions": actions, "action_text": text,
            "confidence": 0.85, "warnings": [], "extra": {"stakes": "NL10"}}


def _run(obs: dict) -> dict:
    """**真实 CLI 路径**（子进程）——与用户实际使用一致，含输出边界对齐。"""
    import subprocess
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False,
                                     encoding="utf-8") as fh:
        json.dump(obs, fh, ensure_ascii=False)
        path = fh.name
    try:
        r = subprocess.run(
            [sys.executable, "-m", "poker_decision_ai.cli", "analyze",
             "--vision", path],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            cwd=str(ROOT), timeout=600)
        text = r.stdout or ""
        start = text.find("{")
        assert start >= 0, text[-500:]
        return json.loads(text[start:])
    finally:
        pathlib.Path(path).unlink(missing_ok=True)


# ── 1. 状态语义层（非冻结） ─────────────────────────────────────────
def test_semantics_unknown_to_call_and_facing_from_evidence():
    sem = state_agent._field_semantics_from_observation(
        _obs(None, 3.75, ["fold", "call", "raise"]), 10.10, 3.75, 0.0)
    assert sem["to_call"] == "UNKNOWN"
    assert sem["facing_bet"] == "BET_PENDING"

    sem2 = state_agent._field_semantics_from_observation(
        _obs(None, 0.0, ["check", "bet"]), 10.10, 0.0, 0.0)
    assert sem2["to_call"] == "UNKNOWN" and sem2["facing_bet"] == "NO_BET"

    sem3 = state_agent._field_semantics_from_observation(
        _obs(0.0, 0.0, []), 10.10, 0.0, 0.0)
    assert sem3["to_call"] == "KNOWN_ZERO" and sem3["facing_bet"] == "NO_BET"


# ── 2/3/4. 输出边界对齐 ─────────────────────────────────────────────
def test_align_maps_bet_to_raise_check_to_call_and_nulls_size():
    fd = {"action": "bet", "alternative_action": "check", "recommended_size": 16.15,
          "reasoning": "底池 4.5bb，需跟注 0.00bb；最终动作 bet。"}
    out = asem.align_final_decision(
        final_decision=fd, obs=_obs(None, 3.75, ["fold", "call", "raise"]))
    assert out["action"] == "raise"
    assert out["alternative_action"] == "call"
    assert out["recommended_size"] is None
    assert "需跟注 0.00bb" not in out["reasoning"]
    assert out["action_semantics"]["facing_from_evidence"] == "BET_PENDING"
    assert out["action_semantics"]["rewrites"]


def test_align_is_noop_without_observation():
    fd = {"action": "bet", "alternative_action": None, "recommended_size": 3.0}
    out = asem.align_final_decision(final_decision=fd, obs={})
    assert out["action"] == "bet" and out["recommended_size"] == 3.0
    assert out["action_semantics"]["observed_legal_actions"] == []


def test_align_keeps_legal_action_untouched():
    fd = {"action": "call", "alternative_action": "fold", "recommended_size": None}
    out = asem.align_final_decision(
        final_decision=fd, obs=_obs(3.75, 3.75, ["fold", "call", "raise"]))
    assert out["action"] == "call" and out["alternative_action"] == "fold"
    assert out["action_semantics"]["rewrites"] == []


def test_family_and_facing_helpers():
    assert asem.facing_from_legal(["fold", "call", "raise"]) == "BET_PENDING"
    assert asem.facing_from_legal(["check", "bet"]) == "NO_BET"
    assert asem.facing_from_legal([]) == "UNKNOWN"
    assert asem.align_action("bet", ["fold", "call", "raise"]) == ("raise", True)
    assert asem.align_action("check", ["fold", "call", "raise"]) == ("call", True)
    assert asem.align_action("raise", ["check", "bet"]) == ("bet", True)
    assert asem.align_action("call", ["check", "bet"]) == ("check", True)
    assert asem.align_action("call", ["fold", "call"]) == ("call", False)
    assert asem.to_call_text(0.0, True) == "面对下注（to_call 数值未知）"
    # 文案替换必须容忍空格差异（实测冻结 EXE 输出「需跟注0.00bb」无空格）
    fd2 = {"action": "bet", "alternative_action": None, "recommended_size": 1.0,
           "reasoning": "底池 4.5bb，需跟注0.00bb；最终动作 bet。"}
    o2 = asem.align_final_decision(
        final_decision=fd2, obs=_obs(None, 3.75, ["fold", "call", "raise"]))
    assert "需跟注0.00bb" not in o2["reasoning"]
    assert "面对下注（to_call 数值未知）" in o2["reasoning"]
    assert "已按已观察合法动作集对齐" in o2["reasoning"]
    assert asem.to_call_text(1.5, True) == "需跟注 1.50bb"


def test_contract_declares_no_frozen_or_strategy_change():
    c = asem.action_semantics_contract()
    assert c["frozen_layers_modified"] is False
    assert c["candidate_domain_modified"] is False
    assert c["weights_or_ev_modified"] is False


# ── 端到端（CLI 路径）：三个世界各自合法 ────────────────────────────
def test_cli_c5_unknown_to_call_with_facing_evidence():
    d = _run(_obs(None, 3.75, ["fold", "call", "raise"], "Bob bets"))
    fd = d.get("final_decision") or {}
    legal = {"fold", "call", "raise"}
    assert fd.get("action") in legal
    assert fd.get("alternative_action") in legal or fd.get("alternative_action") is None
    reasoning = fd.get("reasoning") or ""
    assert "需跟注 0.00bb" not in reasoning
    assert fd.get("recommended_size") is None


def test_cli_checked_to_world_still_bet_or_check():
    d = _run(_obs(0.0, 0.0, ["check", "bet"], "Bob checks"))
    assert (d.get("final_decision") or {}).get("action") in {"check", "bet"}


def test_cli_facing_bet_world_still_call_or_raise():
    d = _run(_obs(3.75, 3.75, ["fold", "call", "raise"], "Bob bets 3.75"))
    assert (d.get("final_decision") or {}).get("action") in {"fold", "call", "raise"}
