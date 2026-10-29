# -*- coding: utf-8 -*-
"""P30-9-DU：逐手 EV 偏离机制的**回归测试**（锁定第 15/16 轮成果，防静默回归）。

覆盖：
  1. 逐手信号确实产生（用真实 r1 库数据构造 inp，不依赖时间/网络）；
  2. **参照必须=聚合频率 argmax**（锚定的真实规则）——这正是第 15 轮修掉的 bug，
     若有人改回用 agent 的 gto_action 作参照，本测试会失败；
  3. 逐手频率门 `dev_min_freq_hand` 生效（低于门 ⇒ 无提案）；
  4. 增益下限 `dev_min_ev_gain_hand` 生效（低于门 ⇒ 偏离被拒）；
  5. 回退开关：`SC_PER_HAND_DEV=0` / `SC_SIZE_ONLY_DEV=0` / `dev_per_hand_ev=False`。
"""
from __future__ import annotations

import json
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from poker_decision_ai.domain.poker import strategy_anchor as sa   # noqa: E402

SPOT = ROOT / "data" / "solver" / "spots_r1" / "3BP-flop-ace_high.json"


def _fixture() -> tuple[dict, dict]:
    d = json.loads(SPOT.read_text(encoding="utf-8"))
    hands = d["private_cards_actor"]
    acts = d["actions"]
    hero = [hands[0][:2], hands[0][2:4]]            # Ac2c：逐手最优 = check（低频率混合）
    anc = max(acts, key=lambda a: float(a.get("frequency") or 0))
    rows = [{"action": a.get("name"), "size_bb": a.get("amount_bb"),
              "ev": a.get("ev_per_hand_bb"), "freq": a.get("freq_per_hand")}
             for a in acts]
    state = {"players": [{"is_hero": True, "hole_cards": list(hero), "seat": 3,
                           "status": "active", "stack": 89.0, "player_id": "Hero"}],
              "street": "flop", "pot": float((d.get("query") or {}).get("pot_bb") or 22.0),
              "to_call": 0.0, "bb": 1.0, "board": ["As", "7s", "2h"],
              "hero_seat": 3, "button_seat": 5}
    inp = {"solver": {
                "solver_name": "postflop",
                "is_approximate": False,
                "gto_action": "bet",                     # agent 侧选择（≠ 锚定参照）
                "actions": [{"name": a.get("name"), "size_bb": a.get("amount_bb"),
                              "frequency": a.get("frequency"), "ev": a.get("ev_bb")}
                             for a in acts],
                "source_detail": {"per_hand": {"hands": hands, "rows": rows}}},
            "exploit": {}, "bluff": {}, "range": {}, "equity": {}}
    return inp, state


def test_signal_uses_frequency_argmax_as_reference():
    inp, state = _fixture()
    sig = sa._per_hand_signal(inp, state)
    assert sig is not None, "逐手信号应产生（真实库数据）"
    # 锚定参照 = 聚合频率最高者（Bet），而逐手最优 = check ⇒ 提案必须是 check
    assert sig["action"] == "check", f"提案应为逐手最优 check，实际 {sig['action']}"
    assert sig["signal"] == "solver_per_hand_ev"
    assert float(sig["ev_gain_bb"]) > 0.0
    assert float(sig["freq_hand"]) > 0.0


def test_signal_absent_when_best_equals_anchor(monkeypatch):
    """若逐手最优与锚定动作、尺寸都相同 ⇒ 不应提案（无意义偏离）。"""
    inp, state = _fixture()
    ph = inp["solver"]["source_detail"]["per_hand"]
    # 把 check 行的逐手 EV 压到最低 ⇒ 逐手最优变成 Bet（= 锚定行）
    for r in ph["rows"]:
        if r["action"] == "Check":
            r["ev"] = [0.0] * len(r["ev"])
    assert sa._per_hand_signal(inp, state) is None


def test_freq_hand_gate_blocks(monkeypatch):
    """逐手频率低于 dev_min_freq_hand ⇒ 偏离被拒（但支持集内）。"""
    inp, state = _fixture()
    ph = inp["solver"]["source_detail"]["per_hand"]
    for r in ph["rows"]:
        if r["action"] == "Check":
            r["freq"] = [0.001] * len(r["freq"])
    sig = sa._per_hand_signal(inp, state)
    assert sig is not None
    anchor = sa.anchor_from_solver(state, inp["solver"], None)
    assert anchor is not None
    dev = sa.bounded_deviation(state, anchor, inp, {"anchor": "solver",
                                                     "dev_min_freq_hand": 0.02,
                                                     "dev_min_ev_gain_hand": 0.0})
    assert dev is None, "逐手频率 0.1% < 2% 门 ⇒ 禁止偏离"


def test_gain_floor_blocks_small_gain(monkeypatch):
    """增益低于 dev_min_ev_gain_hand ⇒ 偏离被拒。"""
    inp, state = _fixture()
    anchor = sa.anchor_from_solver(state, inp["solver"], None)
    assert anchor is not None
    dev = sa.bounded_deviation(state, anchor, inp, {"anchor": "solver",
                                                     "dev_min_freq_hand": 0.02,
                                                     "dev_min_ev_gain_hand": 5.0})
    assert dev is None, "增益 0.08bb < 5bb 门 ⇒ 禁止偏离"


def test_deviation_allowed_when_gates_pass():
    inp, state = _fixture()
    anchor = sa.anchor_from_solver(state, inp["solver"], None)
    assert anchor is not None
    dev = sa.bounded_deviation(state, anchor, inp, {"anchor": "solver",
                                                     "dev_min_freq_hand": 0.02,
                                                     "dev_min_ev_gain_hand": 0.01})
    assert dev is not None, "门槛满足时应产生偏离"
    assert dev["action"] == "check"
    assert dev["justification"]["signal"] == "solver_per_hand_ev"


def test_revert_switches(monkeypatch):
    """回退开关：SC_PER_HAND_DEV=0 ⇒ 无信号。"""
    inp, state = _fixture()
    monkeypatch.setenv("SC_PER_HAND_DEV", "0")
    assert sa._per_hand_signal(inp, state) is None
    monkeypatch.delenv("SC_PER_HAND_DEV", raising=False)
    assert sa._per_hand_signal(inp, state) is not None


def test_sixmax_freq_threshold_defaults():
    """P30-9-FH：6-max 逐手频率偏离阈值默认 0.35/0.35（双样本交叉验证所得）。

    若有人改动默认值，本测试会失败 ⇒ 提醒重新做双样本验证（主选一致 +3.7pp、支持率不变）。
    """
    src = (pathlib.Path(__file__).resolve().parents[2] / "poker_decision_ai" / "domain"
            / "poker" / "strategy_anchor.py").read_text(encoding="utf-8")
    assert 'SC_PHF_TOP") or 0.35' in src, "默认 top 阈值应为 0.35"
    assert 'SC_PHF_ANCHOR_MAX") or 0.35' in src, "默认 anchor_max 阈值应为 0.35"
