"""P30-7 — fold 测试调试：引擎 fold 后 snapshot players 内容。"""
from __future__ import annotations
import sys
sys.path.insert(0, '.')
from poker_decision_ai.domain.poker.state_engine import PokerStateEngine

eng = PokerStateEngine("hero")
eng.apply({"type": "HAND_START", "seats": [
    {"seat": 1, "name": "p1", "stack": 1000},
    {"seat": 2, "name": "hero", "stack": 1000}],
    "button_seat": 1, "sb": 5, "bb": 10})
print("players after HAND_START:", eng.snapshot()["players"])
eng.apply({"type": "PLAYER_ACTION", "player_id": "p1", "street":
           "preflop", "action": "fold"})
snap = eng.snapshot()
print("players after fold:", snap["players"])
print("status:", snap.get("status"))
