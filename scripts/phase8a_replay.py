"""Phase 8A：最小确定性 Replay Engine + Information-Set Isolation + 两臂共享 RNG 契约。

严格依据 docs/PHASE8_STEP0_EV_REPLAY_AUDIT.md（已确认）。第一阶段范围：
  - 确定性 replay loop（只复用冻结 PokerStateEngine 的事件增量机制）
  - Information-Set Isolation（Hero 策略输入白名单 + 硬断言 + 反向注入测试）
  - Pluribus/WSOP（seats 全空）的 Position 推导冻结规则（orbit 规则；歧义 → excluded）
  - Baseline A / Exploit Arm E(k,ε) 两臂框架（共享 RNG 流契约；8A 仅引擎测试）
  - 数据级资格审计（Pluribus 10k / WSOP NT 11 / iPoker 全员真手牌 31,396）

本阶段**不做**：EV 计算、P2 verdict、profitability/ROI、strength 校准、post-hoc 臂选择。
Baseline A 的真实 train-fit 人口频率策略 = 8B 交付；8A 使用标注的合成占位策略
（PLACEHOLDER_POLICY）以测试引擎契约。

冻结边界：只 import 冻结 uoft_mapping / canonical / state_engine；不修改任何 Phase ≤7C 产物。
"""
from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).parent
_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_SCRIPT_DIR))

from poker_decision_ai.domain.poker.state_engine import (  # noqa: E402
    PokerStateEngine, StateEngineError)
from scripts.uoft_mapping import (  # noqa: E402
    derive_button_seat, map_row_to_canonical, position_label)

# ---- 冻结臂网格（Audit §F）：k × ε；主臂 (1,1)；test 上禁止臂选择 ----
FROZEN_K = (0.5, 1.0)
FROZEN_EPS = (0.25, 0.5, 1.0)
PRIMARY_ARM = (1.0, 1.0)

# ---- 决策映射（Audit §F 冻结表；无 equity 依赖的频率乘子精化）----
ACTION_MAPPINGS = {
    ("vpip", "under"): {"spot": "steal", "action_class": "raise", "sign": +1,
                        "hypothesis_id": "H-vpip-under"},
    ("pfr", "over"): {"spot": "facing_villain_open", "action_class": "raise",
                      "sign": +1, "hypothesis_id": "H-pfr-over"},
    ("fold_to_3bet", "over"): {"spot": "facing_villain_open", "action_class": "raise",
                               "sign": +1, "hypothesis_id": "H-ft3b-over"},
    ("fold_to_3bet", "under"): {"spot": "facing_villain_open", "action_class": "raise",
                                "sign": -1, "hypothesis_id": "H-ft3b-under"},
}

# ---- Information-Set Isolation 白名单/黑名单（Audit §C/§J-4）----
ALLOWED_OBS_KEYS = frozenset({
    "hand_id", "street", "board", "hero_hole_cards", "pot", "current_bet",
    "to_call", "min_raise_to", "blinds", "hero_stack", "hero_position",
    "button_seat", "players", "action_history", "available_actions", "game_type",
})
ALLOWED_PLAYER_KEYS = frozenset({"player_id", "stack", "status", "position",
                                 "is_hero"})
FORBIDDEN_OBS_KEYS = frozenset({
    "villain_hole_cards", "future_actions", "showdown", "winners", "net_winnings",
    "final_stack", "hero_net", "outcome", "winnings", "hole_cards",
})


def assert_observation_clean(obs: dict) -> None:
    """信息集隔离硬断言：白名单之外 / 黑名单字段出现 → AssertionError（不降级）。"""
    for key in obs:
        if key not in ALLOWED_OBS_KEYS:
            raise AssertionError(f"observation 含白名单外字段: {key!r}（泄漏）")
        if key in FORBIDDEN_OBS_KEYS:
            raise AssertionError(f"observation 含禁止字段: {key!r}（outcome 泄漏）")
    for p in obs.get("players", []):
        for key in p:
            if key not in ALLOWED_PLAYER_KEYS:
                raise AssertionError(f"players 条目含白名单外字段: {key!r}")
            if key in FORBIDDEN_OBS_KEYS:
                raise AssertionError(f"players 条目含禁止字段: {key!r}")
    if "hero_hole_cards" in obs and not isinstance(obs["hero_hole_cards"], list):
        raise AssertionError("hero_hole_cards 必须为 list")
    if "action_history" in obs and not isinstance(obs["action_history"], list):
        raise AssertionError("action_history 必须为 list")


# ---------------------------------------------------------------- position 推导
def derive_positions_orbit(players: list, blinds: list, preflop_actions: list
                           ) -> dict | None:
    """seats 全空（Pluribus/WSOP）时的位置推导（8A 冻结 orbit 规则）。

    规则：
      1. 盲注：恰好 2 个不同正盲注（否则 straddle 歧义 → None）；SB=最小正盲注玩家，
         BB=最大正盲注玩家。
      2. orbit = 翻前行动 token 中不同玩家的首次出现顺序（= 从 UTG 起的顺时针行动序）。
      3. n>=3：orbit 必须覆盖全部 n 人；BB=orbit[-1]、SB=orbit[-2]、BTN=orbit[-3]，
         且与盲注玩家索引交叉验证一致（不一致 → None）。
         n==2：BTN/SB = 最小正盲注玩家，BB = 另一人。
      4. 位置标签：距 BTN 的 orbit 距离 d：0=BTN，1=SB，2=BB，n-1=LP(CO)，3..n-2=EP。
      5. 任何无歧义推导失败 → None（调用方将手置 excluded，禁止猜测/补样本）。
    """
    n = len(players)
    blinds = list(blinds or [])
    pos_blinds = [float(b) for b in blinds if b is not None and float(b) > 0]
    if len({round(b, 6) for b in pos_blinds}) != 2:
        return None                       # 盲注数量不为 2（含 straddle）→ 歧义
    sb_idx = min(range(n), key=lambda i: float(blinds[i] or 0.0)
                 if i < len(blinds) and blinds[i] is not None and float(blinds[i]) > 0
                 else float("inf"))
    bb_idx = max(range(n), key=lambda i: float(blinds[i] or 0.0)
                 if i < len(blinds) and blinds[i] is not None and float(blinds[i]) > 0
                 else float("-inf"))
    if sb_idx == bb_idx:
        return None

    orbit: list[int] = []
    for a in (preflop_actions or []):
        pid = getattr(a, "player_id", None)
        act = getattr(a, "action", "")
        if act == "post":
            continue
        try:
            i = players.index(pid)
        except ValueError:
            continue
        if i not in orbit:
            orbit.append(i)
    if n == 2:
        btn_idx = sb_idx                     # HU：button = SB
        if bb_idx == btn_idx:
            return None
        if not orbit:
            orbit = [btn_idx, bb_idx]        # 无翻前行动时用盲注序确定 HU 顺序
    else:
        if len(orbit) != n:
            return None                      # orbit 不完整 → 歧义
        if orbit[-1] != bb_idx or orbit[-2] != sb_idx:
            return None                      # 与盲注交叉验证不一致 → 歧义
        btn_idx = orbit[-3]
    labels: dict[str, str] = {}
    for d, i in enumerate(orbit):
        if n == 2:
            labels[players[i]] = "BTN" if i == btn_idx else "BB"
            continue
        dist = (d - orbit.index(btn_idx)) % n
        if dist == 0:
            labels[players[i]] = "BTN"
        elif dist == 1:
            labels[players[i]] = "SB"
        elif dist == 2:
            labels[players[i]] = "BB"
        elif dist == n - 1:
            labels[players[i]] = "LP"
        else:
            labels[players[i]] = "EP"
    # seat 合成（orbit 顺序 = 1..n 顺时针），供冻结引擎 positions_for 使用
    seats = {players[i]: d + 1 for d, i in enumerate(orbit)}
    return {"labels": labels, "seats": seats,
            "button_seat": seats[players[btn_idx]], "sb": players[sb_idx],
            "bb": players[bb_idx], "orbit": [players[i] for i in orbit]}


def derive_position_map(row_players: list, row_seats: list, row_blinds: list,
                        preflop_actions: list) -> dict | None:
    """统一位置推导：seats 完整 → 冻结 derive_button_seat/position_label；
    seats 全空 → 8A orbit 规则；其余 → None（excluded）。"""
    n = len(row_players)
    seats = [int(s) for s in (row_seats or []) if s is not None]
    if len(seats) == n and n >= 2:
        btn = derive_button_seat(row_blinds, seats, n)
        if btn is None:
            return None
        labels = {}
        for i, pid in enumerate(row_players):
            lab = position_label(int(seats[i]), btn, seats) if n >= 3 else None
            if n == 2:
                lab = "BTN" if int(seats[i]) == btn else "BB"
            labels[str(pid)] = lab
        if any(v is None for v in labels.values()):
            return None
        return {"labels": labels, "seats": {str(row_players[i]): seats[i]
                                            for i in range(n)},
                "button_seat": btn}
    if len(seats) == 0:
        return derive_positions_orbit(list(row_players), row_blinds,
                                      preflop_actions)
    return None


# ---------------------------------------------------------------- 事件流
def events_from_canonical(hand, hero_id: str, pos: dict) -> list[dict]:
    """CanonicalHand → 冻结引擎事件流（8A 新实现，语义镜像冻结 events_from_parsed_hand）。

    关键隔离：HOLE_CARDS 事件**只发 hero 的手牌**；对手手牌仅存在于 replay 管理
    上下文（hand.players[].hole_cards），绝不进入引擎状态/观测（信息集隔离结构性保证）。
    """
    players = hand.players
    seats = [{"seat": pos["seats"][str(p.player_id)], "name": str(p.player_id),
              "stack": float(p.stack_start) if p.stack_start is not None else 0.0,
              "is_hero": str(p.player_id) == hero_id, "bounty": 0.0}
             for p in players]
    events: list[dict] = [{
        "type": "HAND_START", "hand_id": hand.hand_id,
        "game_type": hand.game_type or "cash",
        "sb": float(hand.sb or 0.0), "bb": float(hand.bb or 0.0),
        "ante": float(hand.ante or 0.0), "button_seat": pos["button_seat"],
        "seats": seats}]
    hero_p = next((p for p in players if str(p.player_id) == hero_id), None)
    if hero_p is not None and getattr(hero_p, "hole_cards", []):
        events.append({"type": "HOLE_CARDS", "player_id": hero_id,
                       "cards": list(hero_p.hole_cards)})

    streets = [("preflop", hand.preflop_actions), ("flop", hand.flop_actions),
               ("turn", hand.turn_actions), ("river", hand.river_actions)]
    all_in_set = set(getattr(hand, "all_in", []) or [])
    last_action_of_street: dict[str, set] = {}
    for street, acts in streets:
        last_action_of_street[street] = {a.player_id for a in acts
                                         if a.action not in ("post",)}
    for street, acts in streets:
        if street != "preflop" and not acts:
            continue
        if street == "flop" and acts:
            events.append({"type": "FLOP", "street": "flop",
                           "board": list(hand.flop or [])})
        elif street == "turn" and acts:
            events.append({"type": "TURN", "street": "turn",
                           "board": list(hand.flop or []) + list(hand.turn or [])})
        elif street == "river" and acts:
            events.append({"type": "RIVER", "street": "river",
                           "board": list(hand.flop or []) + list(hand.turn or [])
                           + list(hand.river or [])})
        for a in acts:
            pid = str(a.player_id)
            all_in = (pid in all_in_set
                      and pid in last_action_of_street[street])
            kind = a.action
            if kind == "post":
                events.append({"type": "PLAYER_ACTION", "street": street,
                               "player_id": pid, "action": "post",
                               "amount": float(a.amount or 0.0)})
            elif kind == "fold":
                events.append({"type": "FOLD", "street": street,
                               "player_id": pid})
            elif kind == "check":
                events.append({"type": "PLAYER_ACTION", "street": street,
                               "player_id": pid, "action": "check"})
            elif kind == "call":
                events.append({"type": "CALL", "street": street,
                               "player_id": pid,
                               "amount": float(a.amount or 0.0), "all_in": all_in})
            elif kind == "bet":
                events.append({"type": "BET", "street": street, "player_id": pid,
                               "amount": float(a.amount or 0.0), "to": a.to,
                               "all_in": all_in})
            elif kind == "raise":
                events.append({"type": "RAISE", "street": street,
                               "player_id": pid, "amount": float(a.amount or 0.0),
                               "to": a.to, "all_in": all_in})
    if getattr(hand, "showdown", None):
        events.append({"type": "SHOWDOWN", "street": "river",
                       "shows": [{"player_id": str(s.get("player_id"))}
                                 for s in hand.showdown.get("shows", [])]})
    if getattr(hand, "winners", None):
        events.append({"type": "HAND_END", "street": "river",
                       "winners": [{"player_id": str(w.get("player_id"))}
                                   for w in hand.winners]})
    return events


# ---------------------------------------------------------------- 决策点与观测
def is_decision_point(engine: PokerStateEngine, event: dict, hero_id: str) -> bool:
    """确定性决策点判定（语义镜像冻结 replay_engine._is_decision_point，8A 本地实现）。"""
    kind = (event.get("action") or event.get("type") or "").lower()
    if kind not in ("call", "check", "bet", "raise"):
        return False
    if event.get("player_id") == hero_id:
        return False
    s = engine.state
    hero = next((p for p in s["players"] if p["is_hero"]), None)
    if hero is None or hero["status"] not in ("active", "allin"):
        return False
    if s["to_call"] > 0:
        return True
    return engine._invested(s["street"], hero_id) == 0.0


def build_observation(engine: PokerStateEngine, pos_labels: dict,
                      hero_id: str) -> dict:
    """信息集隔离观测：只含 Hero 当时合法可观察信息（白名单，§ALLOWED_OBS_KEYS）。"""
    s = engine.snapshot()
    hero = next(p for p in s["players"] if p["is_hero"])
    obs = {
        "hand_id": s["hand_id"], "game_type": s["game_type"],
        "street": s["street"], "board": list(s["board"]),
        "hero_hole_cards": list(hero["hole_cards"]),
        "pot": round(float(s["pot"]), 4),
        "current_bet": round(float(s["current_bet"]), 4),
        "to_call": round(float(s["to_call"]), 4),
        "min_raise_to": round(float(s["min_raise_to"]), 4),
        "blinds": {"sb": s["sb"], "bb": s["bb"], "ante": s["ante"]},
        "hero_stack": round(float(hero["stack"]), 4),
        "hero_position": pos_labels.get(hero_id),
        "button_seat": s["button_seat"],
        "players": [{"player_id": p["player_id"], "stack": round(float(p["stack"]), 4),
                     "status": p["status"], "position": pos_labels.get(p["player_id"]),
                     "is_hero": bool(p["is_hero"])} for p in s["players"]],
        "action_history": [dict(r) for r in s["action_history"]],
        "available_actions": list(s["available_actions"]),
    }
    assert_observation_clean(obs)          # 白名单硬断言（构造即校验）
    return obs


# ---------------------------------------------------------------- 两臂（共享 RNG 契约）
def placeholder_policy(obs: dict) -> dict:
    """8A 占位策略（标注：真实 train-fit 人口频率策略 = 8B 交付）。

    确定性权重表，仅在 obs 合法行动上采样；两臂共用同一 RNG 流。
    """
    weights = {"fold": 2.0, "check": 1.0, "call": 1.0, "bet": 1.0, "raise": 1.0,
               "allin": 1.0}
    legal = [a for a in obs.get("available_actions", []) if a in weights] or ["fold"]
    total = sum(weights[a] for a in legal)
    return {a: weights[a] / total for a in legal}


class BaselineArm:
    """Baseline A：非剥削人口频率策略（8A 用占位表；8B 换 train-fit 表）。"""

    def __init__(self, rng: random.Random, policy=placeholder_policy) -> None:
        self.rng = rng
        self.policy = policy

    def action_probs(self, obs: dict) -> dict:
        return dict(self.policy(obs))

    def decide(self, obs: dict) -> str:
        probs = self.action_probs(obs)
        actions = sorted(probs)
        r = self.rng.random() * sum(probs[a] for a in actions)
        acc = 0.0
        for a in actions:
            acc += probs[a]
            if r <= acc:
                return a
        return actions[-1]


class ExploitArm(BaselineArm):
    """Exploit Arm E(k,ε)：同引擎、同 RNG 流；唯一差异 = exploit 信号乘子（Audit §F）。

    乘子公式（冻结）：p' = clip(p × (1 + sign·k·c·|m−μ|·ε), 0, 1)，作用于映射行动类。
    recommendation_provider(obs) → dict|None；8A 阶段传 None 或合成 rec（引擎契约测试）。
    """

    def __init__(self, rng: random.Random, k: float, eps: float,
                 recommendation_provider=None, policy=placeholder_policy) -> None:
        super().__init__(rng, policy)
        if k not in FROZEN_K:
            raise AssertionError(f"k={k} 不在冻结臂网格 {FROZEN_K}")
        if eps not in FROZEN_EPS:
            raise AssertionError(f"ε={eps} 不在冻结臂网格 {FROZEN_EPS}")
        self.k, self.eps = k, eps
        self.recommendation_provider = recommendation_provider or (lambda obs: None)

    def action_probs(self, obs: dict) -> dict:
        probs = super().action_probs(obs)
        rec = self.recommendation_provider(obs)
        if not rec or not rec.get("eligible"):
            return probs
        mapping = rec.get("mapping") or {}
        if mapping.get("action_class") not in probs:
            return probs
        sign = mapping.get("sign", 0)
        mult = 1.0 + sign * self.k * float(rec.get("confidence", 0.0)) \
            * float(rec.get("deviation_magnitude", 0.0)) * self.eps
        cls = mapping["action_class"]
        probs[cls] = max(0.0, min(1.0, probs[cls] * max(mult, 0.0)))
        total = sum(probs.values())
        if total <= 0:
            return {cls: 1.0}
        return {a: p / total for a, p in probs.items()}


# ---------------------------------------------------------------- replay loop
def replay_hand(hand, hero_id: str, pos: dict, seed: int,
                recommendation_provider=None, arm_k: float = PRIMARY_ARM[0],
                arm_eps: float = PRIMARY_ARM[1]) -> dict:
    """确定性线性回放：应用全部真实事件；在 hero 决策点记录观测与两臂行动。

    8A 阶段不做反事实分支（= 8B）。同一 (hand, hero, pos, seed, provider) →
    完全一致 trace（determinism 由测试断言）。
    """
    events = events_from_canonical(hand, hero_id, pos)
    engine = PokerStateEngine(hero_id=hero_id)
    # 共享 RNG 流契约：同一 seed → 两臂各自的流完全一致（同概率表 → 同行动；
    # 唯一差异来源 = exploit 乘子）。绝不共用同一 rng 实例（会顺序消耗导致漂移）。
    rng_base = random.Random(seed)
    rng_exploit = random.Random(seed)
    baseline = BaselineArm(rng_base)
    exploit = ExploitArm(rng_exploit, arm_k, arm_eps, recommendation_provider)
    trace = {"hand_id": hand.hand_id, "hero_id": hero_id, "seed": seed,
             "positions": pos["labels"], "button_seat": pos["button_seat"],
             "decision_points": [], "engine_warnings": [],
             "excluded": False, "exclude_reason": ""}
    for ev in events:
        try:
            engine.apply(ev)
        except StateEngineError as e:
            trace["excluded"] = True
            trace["exclude_reason"] = f"engine_error: {e}"
            return trace
        if is_decision_point(engine, ev, hero_id):
            obs = build_observation(engine, pos["labels"], hero_id)
            trace["decision_points"].append({
                "street": obs["street"],
                "obs_hash": hashlib.sha256(
                    json.dumps(obs, sort_keys=True, ensure_ascii=False)
                    .encode("utf-8")).hexdigest()[:16],
                "observation": obs,
                "baseline_action": baseline.decide(obs),
                "exploit_action": exploit.decide(obs),
            })
    trace["final_state"] = engine.snapshot()
    trace["engine_warnings"] = list(engine.warnings)
    return trace
