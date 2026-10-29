"""EXE-1 — 6-max 交互对打回路（产品核心；非实时辅助红线内）。

设计决策（任务书要求显式说明）：**引擎调用 external_harness，不复制**。

理由：
  1. harness 引擎（external_harness/engine N-seat 状态机 + vendor 底座）是
     GATE-MIN 判决的同一状态机——对打产品与基准共用一套规则语义，绝不分叉
     （分叉 = 自欺温床，harness 纪律的本意）；
  2. harness 依赖为纯 Python 零重资源（vendor 引擎 100KB），打包体积可忽略；
  3. 池 bot（pool_bots）、脱敏视图（guarded_player_view）、抽水（RakeModel）、
     judge 链适配（JudgeHeroController）全部直接复用——产品对打看到的就是
     GATE-MIN 同款的桌面与对手。
  复制的只有**交互驱动回路**（本文件 ~400 行）：harness 的 run_match_6max
  是批量基准回路（fail-closed 中止 + 聚合记录），没有逐事件交互钩子
  （街展示/摊牌亮牌/每决策参考建议/一手解析）；产品回路以事件回调
  （on_event）驱动 CLI/GUI 前端，动作施加走同一引擎 apply。

诚实边界：
  * 用户动作 = 人类输入，provenance 记 human_input；AI 参考建议单独记录
    （advisor.*），两者永不混淆；
  * 参考建议来自生产 judge 链（solver 锚定/图表/频率先验，provenance 如实
    透传；judge 不可用时如实标注 unavailable）；
  * 用户非法动作只重试（交互产品），绝不静默改判；
  * 本模块无任何屏幕捕捉/自动操作路径。
"""
from __future__ import annotations

import random
import secrets
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from external_harness.adapter.judge_adapter import JudgeHeroController
from external_harness.engine import NSeatHoldemState, fresh_nseat_state
from external_harness.engine.nseat_engine import position_labels
from external_harness.protocol.bench_match import _bot_act_dispatch
from external_harness.protocol.cash_match_6max import (
    BIG_BLIND_CHIPS, STAKE_TABLES)
from external_harness.protocol.rake_model import RakeModel
from external_harness.view import guarded_player_view

from .history import PlayHistoryWriter

PLAY_MATCH_VERSION = "exe1-play-match-v1"

#: 桌面阵容预设（构成与 M4 GATE 战役一致；种子由会话声明）。
LINEUP_PRESETS: dict[str, dict] = {
    "nl2": {"stake": "nl2", "name": "$1/$2 娱乐重仓桌（GATE-MIN 构成）",
            "archetypes": {1: "loose_passive_fish", 2: "loose_passive_fish",
                           3: "recreational", 4: "recreational",
                           5: "semi_reg"}},
    "nl5": {"stake": "nl5", "name": "$2/$5 常规桌（GATE-1 构成）",
            "archetypes": {1: "fish", 2: "recreational", 3: "semi_reg",
                           4: "semi_reg", 5: "reg"}},
    "nl10": {"stake": "nl10", "name": "$5/$10 混合桌",
             "archetypes": {1: "fish", 2: "recreational", 3: "semi_reg",
                            4: "reg", 5: "reg"}},
    "nl25": {"stake": "nl25", "name": "$10/$25 常客桌",
             "archetypes": {1: "recreational", 2: "semi_reg", 3: "semi_reg",
                            4: "reg", 5: "reg"}},
    "nl50": {"stake": "nl50", "name": "$25/$50 高额常客桌",
             "archetypes": {1: "semi_reg", 2: "semi_reg", 3: "reg",
                            4: "reg", 5: "reg"}},
}

#: 大师桌 reg 参数锐化（在 nl5 reg（全项目最强模拟对手）基础上：
#: 开池更宽、3bet 更频、诈唬更多、turn/river 不过度弃牌、下注更大）
_MAX_REG_PARAMS: dict = {
    "sb_raise_str": 0.67, "sb_raise_loose": 0.03, "defend": 0.10,
    "threebet": 0.20, "pb_bluff": 0.20, "pb_value_max": 0.85,
    "fc_fold_th": 0.30, "vs_raise_fold_bump": 0.02,
    "raise_th": 0.62, "raise_p": 0.55, "bet_cbb": 0.72,
    "street_fold_mult": [1.0, 1.05, 1.10],
}

#: AI 难度分级：微额→高额玩家画像 → 座位 bot 绑定（画像原型定义见
#: pool_bots.FACTION_SPECS；"normal" 不改阵容，保持档位 GATE 基准构成）。
#: label 中的 NL 范围为**对手强度参考**（等效真实牌桌级别，与当前
#: 盲注档位无关——同一难度在 nl2/nl5 下只是筹码量纲不同）。
DIFFICULTY_TIERS: dict[str, dict] = {
    "easy": {"hu_archetype": "recreational",
             "label": "新手桌（参考 NL2–NL10：鱼与疯子为主，送分）",
             "archetypes": {
                 "nl2": {1: "loose_passive_fish", 2: "maniac",
                         3: "recreational", 4: "loose_passive_fish",
                         5: "nit"},
                 "nl5": {1: "fish", 2: "maniac", 3: "recreational",
                         4: "nit", 5: "fish"},
             }},
    "normal": {"hu_archetype": "semi_reg",
               "label": "常规桌（参考 NL20–NL100：GATE 基准构成）",
               "archetypes": None},
    "hard": {"hu_archetype": "reg",
             "label": "高手桌（参考 NL100–NL500：紧凶 reg 桌）",
             "archetypes": {
                 "nl2": {1: "recreational", 2: "semi_reg", 3: "reg",
                         4: "reg", 5: "reg"},
                 "nl5": {1: "recreational", 2: "semi_reg", 3: "reg",
                         4: "reg", 5: "reg"},
             }},
    "max": {"hu_archetype": {"archetype": "reg", "params": dict(_MAX_REG_PARAMS)},
            "label": "大师桌（参考 NL2000+：职业/半职业常客）",
            "archetypes": {
                stake: {seat: {"archetype": "reg",
                               "params": dict(_MAX_REG_PARAMS)}
                        for seat in range(1, 6)}
                for stake in ("nl2", "nl5")
            }},
}

SEAT_NAMES = ("Hero", "Bot1", "Bot2", "Bot3", "Bot4", "Bot5")

# 事件回调：接收 dict（"type" 字段判别）；前端只渲染，不复算规则。
EventFn = Callable[[dict], None]


# ═══════════════════════════ 决策上下文快照 ═══════════════════════════
@dataclass
class HeroTurnContext:
    """hero 行动瞬间的快照（前端据此渲染 + 收集用户动作）。"""
    hand_index: int
    decision_index: int
    street: str
    board: list[str]
    pot_chips: int
    to_call_chips: int
    legal_actions: list[str]
    min_raise_to_chips: int | None
    max_raise_to_chips: int | None
    all_in_level_chips: int | None
    call_level_chips: int
    stacks_chips: list[int]
    street_bets_chips: list[int]
    folded_seats: list[int]
    button: int
    position_label: str
    hero_hole: list[str]
    bb_chips: int = BIG_BLIND_CHIPS
    stake: str = "nl2"
    stake_label: str = ""

    def bb(self, chips: float | None) -> float | None:
        return None if chips is None else round(chips / self.bb_chips, 2)


@dataclass
class HumanAction:
    """用户动作（raise-to 语义：amount_chips = 结束后本街水平）。"""
    action: str                       # fold|check|call|bet|raise|all-in
    amount_chips: int | None = None
    peeked_advice: bool = False       # 用户在行动前主动看过参考建议


# ═══════════════════════════ AI 参考建议（advisor） ═══════════════════
class Advisor:
    """生产 judge 链参考建议（后台线程跑；结果只记录，不代打）。"""

    def __init__(self, judge: JudgeHeroController | None, *,
                 hero_seat: int = 0):
        self.judge = judge
        self.hero_seat = int(hero_seat)
        self._thread: threading.Thread | None = None
        self._result: dict | None = None

    # ------------------------------------------------------------------
    def start(self, state: NSeatHoldemState, *, hand_index: int,
              decision_index: int) -> None:
        """在后台线程计算当前 spot 的 judge 建议（用户思考的同时跑）。"""
        if self.judge is None:
            self._thread = None
            self._result = {"available": False,
                            "reason": "advisor_disabled"}
            return
        self._result = None
        t0 = time.perf_counter()

        def work():
            env = None
            last_exc: Exception | None = None
            # 稳定性加固：judge 链偶发超时/异常时重试一次（退避 0.8s）
            for attempt in range(2):
                try:
                    env = self.judge.decide(
                        state, hero_seat=self.hero_seat,
                        hand_index=hand_index,
                        decision_index=decision_index)
                    break
                except Exception as exc:              # noqa: BLE001
                    last_exc = exc
                    if attempt == 0:
                        time.sleep(0.8)
            if env is None:
                self._result = {
                    "available": False,
                    "reason": (f"advisor_error:"
                               f"{type(last_exc).__name__ if last_exc else 'empty'}"),
                    "detail": str(last_exc)[:200]}
                return
            self._result = {
                "available": env.get("outcome") == "ACCEPT"
                             and bool(env.get("action")),
                "action": env.get("action"),
                "size_bb": env.get("recommended_size_bb"),
                "provenance": dict(env.get("provenance") or {}),
                "fallback_conservative": bool(env.get("fallback_conservative")),
                "frequency_prior_state": env.get("frequency_prior_state"),
                "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
                "claim": "AI 参考建议（生产 judge 链；非 GTO 承诺）",
            }

        self._thread = threading.Thread(target=work, daemon=True)
        self._thread.start()

    # ------------------------------------------------------------------
    def collect(self, timeout_s: float = 90.0) -> dict:
        """等建议完成（用户已行动/请求 peek）。失败 → 诚实 unavailable。"""
        th = self._thread
        if th is None:
            return self._result or {"available": False,
                                    "reason": "advisor_not_started"}
        th.join(timeout=timeout_s)
        if th.is_alive():
            return {"available": False, "reason": "advisor_timeout"}
        return self._result or {"available": False, "reason": "advisor_empty"}


# ═══════════════════════════ 交互对打回路 ═══════════════════════════
def run_play_session(*, stake: str, n_hands: int, seed_base: int,
                     get_human_action: Callable[[HeroTurnContext], HumanAction],
                     on_event: EventFn,
                     judge: JudgeHeroController | None = None,
                     rake: RakeModel | None = None,
                     history: PlayHistoryWriter | None = None,
                     hero_seat: int = 0,
                     lineup_archetypes: dict[int, str] | None = None,
                     session_id: str = "play",
                     shuffle_algorithm: str = "lcg",
                     difficulty: str = "normal",
                     table_size: int = 6,
                     stack_bb: int = 100,
                     focus_position: str = "auto") -> dict:
    """跑 n_hands 手交互对打（2-6 桌型可选）；返回会话聚合（与 harness 口径一致）。

    get_human_action  = 前端动作收集器（CLI 终端 / GUI 队列）。
    on_event          = 事件回调（session_start/hand_start/street/
                        hero_turn/decision_result/hand_end/session_end）。
    shuffle_algorithm = 洗牌随机源双轨：
                        "lcg"   确定性（种子复盘；跨解释器可复现）；
                        "python" 强随机（实战默认；每手 secrets 熵独立
                        播种 MT19937，手间无状态连续、不可复盘）。
    """
    if shuffle_algorithm not in ("lcg", "python"):
        raise ValueError(f"unknown shuffle_algorithm: {shuffle_algorithm!r}")
    n_seats = int(table_size)
    if n_seats not in (2, 3, 4, 5, 6):
        raise ValueError(f"unsupported table_size: {table_size!r} (2-6)")
    stack_bb = int(stack_bb)
    if not 20 <= stack_bb <= 500:
        raise ValueError(f"stack_bb out of range [20,500]: {stack_bb}")
    if focus_position not in ("auto", "btn", "sb", "bb"):
        raise ValueError(f"unknown focus_position: {focus_position!r}")
    if stake not in STAKE_TABLES:
        raise KeyError(f"unknown stake: {stake}")
    table = STAKE_TABLES[stake]
    rules = None
    from external_harness.protocol.cash_match_6max import stake_rules
    rules = stake_rules(stake)
    tier = DIFFICULTY_TIERS.get(str(difficulty))
    if tier is None:
        raise KeyError(f"unknown difficulty: {difficulty!r}")
    archetypes = dict(LINEUP_PRESETS[stake]["archetypes"]
                      if lineup_archetypes is None else lineup_archetypes)
    if lineup_archetypes is None and tier["archetypes"] is not None:
        # 难度画像按档位取；新档位（nl10+）回落到 nl5 的画像构成
        archetypes = dict(tier["archetypes"].get(stake)
                          or tier["archetypes"].get("nl5") or archetypes)
    if n_seats < 6:
        # 短桌：按座位号取画像映射的前 n-1 席；HU 用难度专属对手画像
        if n_seats == 2 and tier.get("hu_archetype") is not None:
            archetypes = {1: tier["hu_archetype"]}
        else:
            archetypes = dict(sorted(archetypes.items())[: n_seats - 1])
    seats, lineup = build_seat_configs(stake, seed_base, archetypes,
                                       n_seats=n_seats)
    advisor = Advisor(judge, hero_seat=hero_seat)
    stack_chips = int(BIG_BLIND_CHIPS * stack_bb)

    on_event({"type": "session_start", "version": PLAY_MATCH_VERSION,
              "session_id": session_id, "stake": stake,
              "stake_label": table["label"], "n_hands": int(n_hands),
              "seed_base": int(seed_base), "deterministic": True,
              "lineup": lineup, "bb_chips": BIG_BLIND_CHIPS,
              "starting_stack_chips": stack_chips,
              "difficulty": str(difficulty), "tier_label": tier["label"],
              "table_size": n_seats, "stack_bb": int(stack_bb),
              "focus_position": str(focus_position)})

    records: list[dict] = []
    counters = {"hero_decisions": 0, "advisor_available": 0,
                "advisor_agree": 0, "bot_decisions": 0, "hand_aborts": 0}
    session_t0 = time.perf_counter()
    # 现金桌规则：筹码跨手延续；筹码 ≤ 1bb 的座位在下一手自动补满
    stacks_now = [stack_chips] * n_seats

    for hand_index in range(int(n_hands)):
        stacks_now = [s if s > BIG_BLIND_CHIPS else stack_chips
                      for s in stacks_now]              # 破码自动补满
        if focus_position == "btn":
            button = hero_seat                                   # 固定 BTN
        elif focus_position == "sb":
            button = (hero_seat - 1) % n_seats                   # 固定 SB
        elif focus_position == "bb":
            button = (hero_seat - 2) % n_seats                   # 固定 BB
        else:
            button = hand_index % n_seats
        if shuffle_algorithm == "python":
            # 强随机实战：每手独立 128-bit 熵播种（手间无连续状态，
            # 观察牌面无法跨手预测）；deal_seed 仅作落盘标识。
            rng_seed = secrets.randbits(128)
            deal_seed = rng_seed & ((1 << 64) - 1)
            state = fresh_nseat_state(rng=random.Random(rng_seed),
                                      n_seats=n_seats,
                                      seat_stacks=list(stacks_now),
                                      button=button, rules=rules,
                                      shuffle_algorithm="python")
        else:
            deal_seed = (seed_base + hand_index * 0x9E3779B1) & ((1 << 64) - 1)
            state = fresh_nseat_state(seed=deal_seed, n_seats=n_seats,
                                      seat_stacks=list(stacks_now),
                                      button=button, rules=rules,
                                      shuffle_algorithm="lcg")
        rec: dict = {"hand_index": hand_index, "deal_seed": int(deal_seed),
                     "button": button, "aborted": None, "timeline": []}
        stacks_at_start = list(state.stacks)   # 本手起始筹码（net 基准）
        hero_hole = [str(c) for c in state.hole_cards[hero_seat]]
        hero_decisions: list[dict] = []
        hand_t0 = time.perf_counter()
        on_event({"type": "hand_start", "hand_index": hand_index,
                  "n_hands": int(n_hands), "button": button,
                  "hero_hole": hero_hole, "stacks": list(state.stacks),
                  "pot_chips": int(state.pot)})

        last_street = None
        while not state.terminal:
            street_now = str(state.street)
            if street_now != last_street:
                last_street = street_now
                on_event({"type": "street", "street": street_now,
                          "board": [str(c) for c in state.board],
                          "pot_chips": int(state.pot),
                          "stacks": list(state.stacks),
                          "street_bets": [int(b) for b in state.street_bets]})
            actor = state.current_player
            if actor == hero_seat:
                ctx = _turn_context(state, hero_seat=hero_seat,
                                    hand_index=hand_index,
                                    decision_index=len(hero_decisions),
                                    stake=stake, table=table)
                advisor.start(state, hand_index=hand_index,
                              decision_index=ctx.decision_index)
                on_event({"type": "hero_turn", "context": ctx,
                          "advisor": advisor})
                try:
                    human = get_human_action(ctx)
                    # 用户动作 → 引擎（非法 → 事件声明 + 重试，绝不静默改判）
                    mapped = _validate_human(state, human)
                except _QuitSession:
                    on_event({"type": "session_end", "reason": "user_quit",
                              "summary": _session_summary(
                                  records, counters, stake, session_t0)})
                    return _session_summary(records, counters, stake,
                                            session_t0)
                adv = advisor.collect()
                if not mapped.get("ok"):
                    on_event({"type": "invalid_action",
                              "reason": str(mapped.get("reason")),
                              "legal_actions": list(state.legal_actions())})
                    continue
                adv_action = adv.get("action") if adv.get("available") else None
                agree = (adv_action == mapped["action"]
                         if adv_action else None)
                if adv.get("available"):
                    counters["advisor_available"] += 1
                    if agree:
                        counters["advisor_agree"] += 1
                counters["hero_decisions"] += 1
                hero_decisions.append({
                    "street": street_now,
                    "pot_bb": ctx.bb(ctx.pot_chips),
                    "to_call_bb": ctx.bb(ctx.to_call_chips),
                    "legal_actions": list(ctx.legal_actions),
                    "position": ctx.position_label,
                    "action": mapped["action"],
                    "amount_chips": mapped["amount_chips"],
                    "size_bb": (round(mapped["amount_chips"]
                                      / BIG_BLIND_CHIPS, 2)
                                if mapped["amount_chips"] is not None
                                else None),
                    "peeked_advice": bool(human.peeked_advice),
                    "advisor": {k: v for k, v in adv.items()
                                if k in ("available", "action", "size_bb",
                                         "provenance", "fallback_conservative",
                                         "frequency_prior_state", "reason",
                                         "latency_ms")},
                    "advisor_agree": agree,
                })
                on_event({"type": "decision_result",
                          "hand_index": hand_index,
                          "street": street_now,
                          "user_action": mapped["action"],
                          "user_size_chips": mapped["amount_chips"],
                          "advisor": adv,
                          "advisor_agree": agree})
                state = state.apply(mapped["action"], mapped["amount_chips"])
                rec["timeline"].append({
                    "street": street_now, "seat": hero_seat,
                    "action": mapped["action"],
                    "amount_chips": mapped["amount_chips"],
                    "pot_chips": int(state.pot),
                    "stacks": list(state.stacks),
                    "street_bets": [int(b) for b in state.street_bets]})
                on_event({"type": "action", "seat": hero_seat, "is_hero": True,
                          "action": mapped["action"],
                          "amount_chips": mapped["amount_chips"],
                          "pot_chips": int(state.pot),
                          "stacks": list(state.stacks),
                          "street_bets": [int(b) for b in state.street_bets]})
            else:
                counters["bot_decisions"] += 1
                seat_cfg = seats.get(actor)
                view = guarded_player_view(state, seat=actor)
                res = _bot_act_dispatch(
                    view["view"], bot_kind=str(seat_cfg["bot_kind"]),
                    seed=int(seat_cfg["bot_seed"]), hand_index=hand_index,
                    decision_index=len(rec.get("bot_actions", [])),
                    freq_table=None, bot_params=seat_cfg["bot_params"])
                if not res.get("ok"):
                    counters["hand_aborts"] += 1
                    rec["aborted"] = f"BOT_ILLEGAL:seat{actor}:{res.get('reason')}"
                    break
                try:
                    prev_pot = int(state.pot)
                    state = state.apply(res["action"], res["amount_chips"])
                except Exception as exc:              # noqa: BLE001
                    counters["hand_aborts"] += 1
                    rec["aborted"] = f"BOT_ENGINE_REJECT:seat{actor}:{exc}"
                    break
                rec.setdefault("bot_actions", []).append(
                    {"seat": actor, "a": res["action"]})
                rec["timeline"].append({
                    "street": street_now, "seat": actor,
                    "action": res["action"],
                    "amount_chips": res["amount_chips"],
                    "pot_chips": int(state.pot),
                    "stacks": list(state.stacks),
                    "street_bets": [int(b) for b in state.street_bets]})
                on_event({"type": "action", "seat": actor, "is_hero": False,
                          "action": res["action"],
                          "amount_chips": res["amount_chips"],
                          "pot_chips": int(state.pot),
                          "stacks": list(state.stacks),
                          "street_bets": [int(b) for b in state.street_bets]})
            if len(state.history) > 300:
                counters["hand_aborts"] += 1
                rec["aborted"] = "HAND_DID_NOT_TERMINATE"
                break

        # ── 一手结束：结算 + 逐手解析 + 落盘 ──────────────────────────
        hand_wall_ms = (time.perf_counter() - hand_t0) * 1000.0
        if rec.get("aborted") is None and state.terminal:
            final = list(state.final_stacks())
            pot_chips = float(sum(state.contributions))
            n_dealt = len([c for c in state.contributions if c > 0])
            flop_dealt = len(tuple(state.board)) >= 3
            rake_chips = (rake.compute(pot_chips=pot_chips,
                                       n_players_dealt=n_dealt,
                                       flop_dealt=flop_dealt)
                          if rake is not None else 0.0)
            final_raked = _apply_rake(
                final, float(stacks_at_start[hero_seat]), rake_chips)
            net_free = (final[hero_seat]
                        - stacks_at_start[hero_seat]) / BIG_BLIND_CHIPS
            net_nfd = (final_raked[hero_seat]
                       - stacks_at_start[hero_seat]) / BIG_BLIND_CHIPS
            net_full = net_free if rake is None else net_nfd
            if rake is not None:
                saved = rake.no_flop_no_drop
                rake.no_flop_no_drop = False
                rake_full = rake.compute(pot_chips=pot_chips,
                                         n_players_dealt=n_dealt,
                                         flop_dealt=flop_dealt)
                rake.no_flop_no_drop = saved
                final_full = _apply_rake(
                    final, float(stacks_at_start[hero_seat]), rake_full)
                net_full = (final_full[hero_seat]
                            - stack_chips) / BIG_BLIND_CHIPS
            rec.update({
                "hero_hole": hero_hole,
                "board": [str(c) for c in state.board],
                "street_reached": str(state.street),
                "n_players_dealt": n_dealt,
                "pot_chips": pot_chips,
                "terminal_reason": state.terminal_reason,
                "net_bb": round(net_free, 4),
                "net_bb_raked": round(net_nfd, 4),
                "net_bb_raked_full": round(net_full, 4),
                "rake_chips": round(rake_chips, 2),
                "final_stacks": final,
            })
            reveal = _reveal(state)
            winners = [s for s in range(len(final))
                       if final[s] > stacks_at_start[s]]
            rec["winners"] = winners
            # 现金桌：筹码跨手延续（抽水后取整）
            stacks_now = [int(round(x)) for x in
                          _apply_rake(final,
                                      float(stacks_at_start[hero_seat]),
                                      rake_chips)]
            analysis = _hand_analysis(rec, hero_decisions, seats)
            on_event({"type": "hand_end", **{k: rec[k] for k in
                      ("hand_index", "hero_hole", "board", "street_reached",
                       "pot_chips", "terminal_reason", "net_bb",
                       "net_bb_raked", "net_bb_raked_full", "final_stacks")},
                      "winners": winners,
                      "revealed": reveal, "analysis": analysis,
                      "hand_wall_ms": round(hand_wall_ms, 1)})
        else:
            analysis = _hand_analysis(rec, hero_decisions, seats)
            rec["street_reached"] = str(state.street)
            rec["hero_hole"] = hero_hole
            rec["board"] = [str(c) for c in state.board]
            on_event({"type": "hand_end", "hand_index": hand_index,
                      "aborted": rec.get("aborted"),
                      "hero_hole": hero_hole,
                      "board": [str(c) for c in state.board],
                      "street_reached": str(state.street),
                      "analysis": analysis,
                      "hand_wall_ms": round(hand_wall_ms, 1)})
        records.append(rec)
        if history is not None:
            history.write_hand(
                hand_index=hand_index, deal_seed=int(deal_seed), button=button,
                hero_hole=hero_hole, board=rec.get("board") or [],
                street_reached=rec.get("street_reached") or "",
                n_players_dealt=rec.get("n_players_dealt") or 0,
                hero_decisions=hero_decisions,
                net_bb=rec.get("net_bb"), net_bb_raked=rec.get("net_bb_raked"),
                net_bb_raked_full=rec.get("net_bb_raked_full"),
                rake_chips=rec.get("rake_chips") or 0.0,
                pot_chips=rec.get("pot_chips") or 0.0,
                    terminal_reason=rec.get("terminal_reason"),
                    aborted=rec.get("aborted"), hand_wall_ms=hand_wall_ms,
                    timeline=rec.get("timeline"),
                    winners=rec.get("winners"),
                    revealed=reveal)

    summary = _session_summary(records, counters, stake, session_t0)
    on_event({"type": "session_end", "reason": "completed", "summary": summary})
    return summary


# ═══════════════════════════ 内部工具 ═══════════════════════════
class _QuitSession(Exception):
    """用户在输入动作时请求退出会话。"""


def build_seat_configs(stake: str, seed_base: int,
                       archetypes: dict[int, object],
                       *, n_seats: int = 6) -> tuple[dict, list]:
    """座位 bot 配置 + 阵容声明（seats 给回路；lineup 给事件/落盘声明）。

    archetypes 座位值支持两种形态：
      * str  —— 原型名（用 FACTION_SPECS 基础参数）；
      * dict —— {"archetype": 名, "params": 显式参数覆盖}（如大师桌锐化 reg）。
    """
    seats: dict[int, dict] = {}
    lineup: list[dict] = []
    from external_harness.protocol.pool_bots import FACTION_SPECS
    bot_stake = stake if stake in FACTION_SPECS else "nl5"   # 新档位复用 nl5 画像参数
    for seat in range(1, int(n_seats)):
        spec = archetypes.get(seat) or "recreational"
        if isinstance(spec, dict):
            arch = str(spec.get("archetype") or "recreational")
            overrides = spec.get("params") or None
        else:
            arch, overrides = str(spec), None
        bot_seed = (seed_base ^ (seat * 0x9E3779B1)) & ((1 << 32) - 1)
        bot_params: dict = {"stake": bot_stake, "archetype": arch}
        if overrides:
            bot_params["params"] = dict(overrides)
        seats[seat] = {"bot_kind": "pool_bot", "bot_seed": int(bot_seed),
                       "bot_params": bot_params}
        lineup.append({"seat": seat, "bot_kind": "pool_bot", "stake": bot_stake,
                       "archetype": arch, "bot_seed": int(bot_seed),
                       "params": dict(overrides) if overrides else None})
    return seats, lineup


def _turn_context(state: NSeatHoldemState, *, hero_seat: int, hand_index: int,
                  decision_index: int, stake: str, table: dict) -> HeroTurnContext:
    legal = list(state.legal_actions())
    aggressive = [a for a in legal if a in ("bet", "raise")]
    labels = position_labels(state.n_seats)
    return HeroTurnContext(
        hand_index=hand_index, decision_index=decision_index,
        street=str(state.street), board=[str(c) for c in state.board],
        pot_chips=int(state.pot), to_call_chips=state.to_call(hero_seat),
        legal_actions=legal,
        min_raise_to_chips=(state.min_aggressive_to(hero_seat)
                            if aggressive else None),
        max_raise_to_chips=(state.max_aggressive_to(hero_seat)
                            if aggressive else None),
        all_in_level_chips=state.all_in_level(hero_seat),
        call_level_chips=state.call_level(hero_seat),
        stacks_chips=list(state.stacks),
        street_bets_chips=list(state.street_bets),
        folded_seats=sorted(state.folded), button=state.button,
        position_label=labels[(hero_seat - state.button) % state.n_seats],
        hero_hole=[str(c) for c in state.hole_cards[hero_seat]],
        stake=stake, stake_label=str(table.get("label") or ""))


def _validate_human(state: NSeatHoldemState, human: HumanAction) -> dict:
    """用户动作校验（交互产品：非法 → ok=False 重试；绝不静默改判）。"""
    action = str(human.action or "").strip().lower()
    legal = state.legal_actions()
    if action == "quit":
        raise _QuitSession()
    if action == "all-in":
        action = "all-in" if "all-in" in legal else None
        if action:
            return {"ok": True, "action": "all-in", "amount_chips": None}
    if action not in legal:
        return {"ok": False,
                "reason": f"illegal:{action}not_in{list(legal)}"}
    if action in ("fold", "check", "call"):
        return {"ok": True, "action": action, "amount_chips": None}
    # bet / raise：amount = 结束后本街水平（raise-to）；缺省 = 最小加注
    level = human.amount_chips
    lo = state.min_aggressive_to(state.current_player)
    hi = state.max_aggressive_to(state.current_player)
    if level is None:
        level = lo
    level = int(level)
    if level < lo or level > hi:
        return {"ok": False, "reason": f"level_out_of_range:{level}"
                                       f"not_in[{lo},{hi}]"}
    return {"ok": True, "action": action, "amount_chips": level}


def _apply_rake(final: list[int], stack_chips: float, rake_chips: float) -> list[int]:
    """与 harness apply_rake_to_final_stacks 同式（按赢利比例扣水）。"""
    from external_harness.protocol.rake_model import apply_rake_to_final_stacks
    return list(apply_rake_to_final_stacks(list(final), float(stack_chips),
                                           float(rake_chips)))


def _reveal(state: NSeatHoldemState) -> list[dict]:
    """摊牌亮牌（只读；非摊牌终局如实声明无摊牌）。"""
    if not (state.terminal and state.terminal_reason != "fold_out"):
        return []
    out = []
    for seat in sorted(state.live_seats):
        out.append({"seat": seat, "hole": [str(c) for c in state.hole_cards[seat]]})
    return out


def _hand_analysis(rec: dict, hero_decisions: list[dict],
                   seats: dict[int, dict]) -> dict:
    """一手解析（结构化；前端渲染）：决策 provenance + 参考建议对比。"""
    n_dec = len(hero_decisions)
    n_adv = sum(1 for d in hero_decisions if (d.get("advisor") or {}).get("available"))
    n_agree = sum(1 for d in hero_decisions if d.get("advisor_agree"))
    sources = sorted({str((d.get("advisor") or {}).get("provenance", {})
                          .get("source") or "")
                      for d in hero_decisions
                      if (d.get("advisor") or {}).get("available")})
    return {
        "n_hero_decisions": n_dec,
        "n_advisor_available": n_adv,
        "n_agree": n_agree,
        "agreement": (round(n_agree / n_adv, 3) if n_adv else None),
        "advisor_provenance_sources": sources,
        "aborted": rec.get("aborted"),
        "net_bb": rec.get("net_bb"),
        "net_bb_raked": rec.get("net_bb_raked"),
        "note": ("参考建议=生产 judge 链（solver 锚定/图表/频率先验，"
                 "provenance 如实）；对比仅教学展示，不构成 EV 承诺"),
    }


def _session_summary(records: list[dict], counters: dict, stake: str,
                     t0: float) -> dict:
    done = [r for r in records if r.get("net_bb") is not None]
    n = max(1, len(done))
    bb100 = round(100.0 * sum(r["net_bb"] for r in done) / n, 3) if done else None
    net100 = round(100.0 * sum(r.get("net_bb_raked") or 0.0 for r in done)
                   / n, 3) if done else None
    return {
        "version": PLAY_MATCH_VERSION, "stake": stake,
        "n_hands": len(records), "n_completed": len(done),
        "n_aborted": len(records) - len(done),
        "hero_total_bb_rakefree": (round(sum(r["net_bb"] for r in done), 3)
                                   if done else 0.0),
        "bb_per_100_rakefree": bb100, "bb_per_100_net": net100,
        "counters": dict(counters),
        "wall_sec": round(time.perf_counter() - t0, 1),
    }
