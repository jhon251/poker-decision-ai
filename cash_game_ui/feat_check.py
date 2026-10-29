# 功能验证：弱点报告 / 导出 / 回放叠加 AI 建议 / i18n
import os
import sys
import time

sys.path.insert(0, r"<repo>")
os.chdir(r"<repo>")
from poker_decision_ai.play.history import PlayHistoryWriter

# 1) 造 3 手含决策/摊牌/时间线的会话（覆盖输赢与各街）
hist = PlayHistoryWriter("data/play_history", "featcheck", stake="nl2",
                         label="$1/$2 娱乐重仓桌", seed_base=9,
                         lineup=[{"seat": s, "archetype": "fish", "bot_kind": "pool_bot"}
                                 for s in range(1, 6)],
                         deterministic=True, table_size=6, stack_bb=100,
                         focus_position="auto")
hands = [
    {"hero": ["As", "Kh"], "board": ["Ts", "9d", "2h", "4c", "7s"],
     "net": 4.2, "net_r": 4.1, "term": "showdown", "button": 0,
     "tl": [("preflop", 1, "call", None, 50, [0, 20, 20, 0, 0, 0]),
            ("flop", 1, "bet", 46, 136, [0, 46, 0, 0, 0, 0])],
     "dec": [{"street": "preflop", "action": "raise", "amount_chips": 60,
              "advisor": {"available": True, "action": "raise", "size_bb": 3.0,
                          "provenance": {"source": "preflop_chart"},
                          "latency_ms": 12}},
             {"street": "flop", "action": "bet", "amount_chips": 46,
              "advisor": {"available": True, "action": "check", "size_bb": None,
                          "provenance": {"source": "solver_hit", "node": "flop-SBBTN"},
                          "latency_ms": 30}}],
     "rev": [{"seat": 1, "hole": ["Qd", "Qc"]}], "win": [1]},
    {"hero": ["7c", "2d"], "board": [], "net": -1.0, "net_r": -1.0,
     "term": "fold_out", "button": 1, "tl": [("preflop", 2, "raise", 60, 90,
                                              [0, 0, 20, 0, 0, 0])],
     "dec": [{"street": "preflop", "action": "fold", "amount_chips": None,
              "advisor": {"available": True, "action": "fold", "size_bb": None,
                          "provenance": {"source": "preflop_chart"},
                          "latency_ms": 8}}],
     "rev": [], "win": [2]},
    {"hero": ["Qd", "Qc"], "board": ["Qh", "7d", "3s"], "net": -6.0,
     "net_r": -5.8, "term": "showdown", "button": 2,
     "tl": [("preflop", 0, "raise", 60, 90, [60, 10, 20, 0, 0, 0]),
            ("flop", 1, "bet", 80, 250, [60, 0, 40, 0, 0, 0])],
     "dec": [{"street": "preflop", "action": "raise", "amount_chips": 60,
              "advisor": {"available": True, "action": "raise", "size_bb": 3.0,
                          "provenance": {"source": "preflop_chart"},
                          "latency_ms": 10}},
             {"street": "flop", "action": "call", "amount_chips": None,
              "advisor": {"available": True, "action": "bet", "size_bb": 4.0,
                          "provenance": {"source": "solver_hit", "node": "flop-BBvBTN"},
                          "latency_ms": 25}}],
     "rev": [{"seat": 1, "hole": ["8d", "8s"]}], "win": [1]},
]
for i, x in enumerate(hands):
    hist.write_hand(
        hand_index=i, deal_seed=i, button=x["button"], hero_hole=x["hero"],
        board=x["board"], street_reached="showdown" if x["board"] else "preflop",
        n_players_dealt=5,
        hero_decisions=x["dec"], net_bb=x["net"], net_bb_raked=x["net_r"],
        net_bb_raked_full=x["net_r"], rake_chips=0.2, pot_chips=180.0,
        terminal_reason=x["term"], aborted=None, hand_wall_ms=90.0,
        timeline=[{"street": st, "seat": se, "action": a, "amount_chips": am,
                   "pot_chips": po, "stacks": [2000] * 6,
                   "street_bets": sb} for st, se, a, am, po, sb in x["tl"]],
        winners=x["win"], revealed=x["rev"])

# 2) 弱点报告
from cash_game_ui.weakness_report import build_report
report = build_report("data/play_history")
print("=== 弱点报告（节选） ===")
for line in report.splitlines():
    if line.strip():
        print(" ", line[:90])
print()

# 3) 导出
from poker_decision_ai.play.history import load_session
from cash_game_ui.play_gui import ReplayBrowser
data = load_session("data/play_history/play_session_featcheck.jsonl")
exported = ReplayBrowser._format_session(object.__new__(ReplayBrowser), data)
print("=== 导出（节选） ===")
for line in exported.splitlines()[:14]:
    print(" ", line)

# 4) i18n
from cash_game_ui import i18n
i18n.set_lang("en")
print("=== i18n ===")
print(" en:", i18n.tr("new_session"), "/", i18n.tr("fold"), "/", i18n.tr("ready"))
i18n.set_lang("zh")
print(" zh:", i18n.tr("new_session"), "/", i18n.tr("fold"), "/", i18n.tr("ready"))

# 5) 回放叠加建议：_build_events 校验
head = {"stake_label": "$1/$2", "table_size": 6, "lineup": []}
hand = data["hands"][0]
events = list(ReplayBrowser._build_events(object.__new__(ReplayBrowser),
                                          head, hand, 1))
hero_advice = [e.get("replay_advice") for e in events
               if e["type"] == "action" and e.get("is_hero")]
print("=== 回放建议叠加 ===")
print(" hero 动作数:", len(hero_advice),
      "| 首个建议 available:", hero_advice[0].get("available"),
      "| AI 动作:", hero_advice[0].get("action"))

# 清理测试会话
os.remove("data/play_history/play_session_featcheck.jsonl")
print("cleanup ok")
