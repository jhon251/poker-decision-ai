"""State Agent (状态解析): 把 request/vision/手牌历史 转换为完整 PokerState dict。

Input : {request: dict|None, vision: dict|None}
Output: StateOutput —— 字段与 domain.poker.models.STATE_SCHEMA 一致，
        外加 confidence / evidence / warnings 元字段。

优先级（契约 §4 state）：
  1. request.vision_observation
  2. vision 输出里的 observation
  3. request.hand_history_text -> parse_hand_history -> 构建状态
三者皆无 -> applicable() 返回 (False, ...)。

关键确定性计算（写注释）：
  - 位置        : positions_for(players, button_seat)
  - stakes 反查 : 遍历 ctx.config.stakes_table() 找 sb/bb 匹配；否则 f"NL{100*bb:g}"
  - 重放        : pot = 累计所有下注/跟注/盲注；current_bet = 当前街最大 bet/raise 额；
                  to_call = current_bet - hero 本街已投入；last_aggressor = 最后 bet/raise 者
  - available_actions / min_raise_to 的推导见下方函数。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar

from poker_decision_ai.core.orchestration.base_agent import BaseAgent
from poker_decision_ai.core.orchestration.context import AgentContext
from poker_decision_ai.domain.poker.hand_history import ParsedHand, parse_hand_history
from poker_decision_ai.domain.poker.models import positions_for

_STREET_ORDER = ["preflop", "flop", "turn", "river"]
# 仅这些动作进入 action_history（盲注/ante 用 post 单独计入 pot，不进入 action_history）
_ACTION_KINDS = ("bet", "raise", "call", "fold", "check")


@dataclass
class StateOutput:
    hand_id: str = ""
    game_type: str = "cash"                      # cash | mtt | satellite
    street: str = "preflop"                      # preflop|flop|turn|river|showdown
    board: list = field(default_factory=list)
    pot: float = 0.0
    current_bet: float = 0.0
    to_call: float = 0.0
    min_raise_to: float = 0.0
    sb: float = 0.0
    bb: float = 0.0
    ante: float = 0.0
    players: list = field(default_factory=list)
    button_seat: int | None = None
    hero_id: str = ""
    hero_seat: int | None = None
    action_history: list = field(default_factory=list)
    street_actions: dict = field(default_factory=dict)
    available_actions: list = field(default_factory=list)
    last_aggressor: str | None = None
    rake: float = 0.0
    source: str = ""                             # vision | hand_history | manual
    cash: dict | None = None
    tournament: dict | None = None
    confidence: float = 1.0
    evidence: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    # V3F-P30-8-B：字段三态语义（UNKNOWN/KNOWN_ZERO/KNOWN_VALUE——
    # 不改变数值布局（V2 兼容）；消费者用 field_semantics 区分未知 vs 合法零）
    field_semantics: dict = field(default_factory=dict)
    # P30-9-CD：翻前节点上下文（加注层级 / 加注者位置 / 来源）——只增不改
    preflop_context: dict = field(default_factory=dict)


class StateAgent(BaseAgent):
    name: ClassVar[str] = "state"
    version: ClassVar[str] = "1.0.0"
    zh_name: ClassVar[str] = "状态解析 Agent"
    layer: ClassVar[str] = "perception"
    depends: ClassVar[list[str]] = ["vision"]
    Input: ClassVar[type] = dict
    Output: ClassVar[type] = StateOutput

    async def applicable(self, raw_input: dict, ctx: AgentContext) -> tuple[bool, str]:
        request = raw_input.get("request") or {}
        vision = raw_input.get("vision") or {}
        if request.get("vision_observation"):
            return True, ""
        if vision.get("observation"):
            return True, ""
        if request.get("hand_history_text"):
            return True, ""
        return False, "no vision_observation, vision output, or hand_history_text"

    async def handle(self, inp: dict, ctx: AgentContext) -> StateOutput:
        request = inp.get("request") or {}
        vision = inp.get("vision") or {}

        # ---- 来源优先级 ----
        if request.get("vision_observation"):
            return _from_observation(request["vision_observation"], request, ctx)
        if vision.get("observation"):
            return _from_observation(vision["observation"], request, ctx)
        if request.get("hand_history_text"):
            return _from_hand_history(request, ctx)

        # 防御：三者皆无（applicable 已拦截）
        return StateOutput(
            confidence=0.0, source="",
            warnings=[{"code": "NO_INPUT", "message": "no state source available"}],
        )


# --------------------------------------------------------------------------- #
# 手牌历史路径
# --------------------------------------------------------------------------- #
def _from_hand_history(request: dict, ctx: AgentContext) -> StateOutput:
    text = str(request.get("hand_history_text") or "")
    hand = parse_hand_history(text)

    warnings = [{"code": "PARSE_WARNING", "message": w} for w in hand.warnings]

    # ---- 街道 / 公共牌：按已出现的公共牌张数推导 ----
    street = "preflop"
    board: list[str] = []
    if hand.street_boards.get("flop"):
        street = "flop"
        board = list(hand.street_boards["flop"])
    if hand.street_boards.get("turn"):
        street = "turn"
        board = board + list(hand.street_boards["turn"])
    if hand.street_boards.get("river"):
        street = "river"
        board = board + list(hand.street_boards["river"])

    sb = float(hand.sb)
    bb = float(hand.bb)
    ante = _ante_from_request(request)
    game_type = str(hand.game or "cash")

    # ---- hero_id：request.hero_id 优先，其次 "Dealt to X" ----
    hero_id = str(request.get("hero_id") or hand.hero_id or "")
    hero_seat: int | None = None

    # ---- players：从 seats 构建 ----
    # 玩家 id 稳定约定：hero 用 hero_id；其他人用名字（手牌历史中名字是跨手牌的
    # 稳定身份键，座位号不是——同名玩家用 f"{name}@{seat}" 消歧）。
    players: list[dict] = []
    seen_names: dict[str, int] = {}
    for s in hand.seats:
        seat = int(s["seat"])
        name = str(s["name"])
        is_hero = bool(s.get("is_hero")) or (name == hero_id)
        if is_hero:
            pid = hero_id if hero_id else name
            hero_seat = seat
        else:
            seen_names[name] = seen_names.get(name, 0) + 1
            pid = name if seen_names[name] == 1 else f"{name}@{seat}"
        players.append({
            "player_id": pid, "name": name, "seat": seat,
            "stack": float(s["stack"]), "status": "active",
            "hole_cards": list(hand.hole_cards) if is_hero else [],
            "position": None, "is_hero": is_hero,
            "bounty": float(s.get("bounty") or 0.0),   # V1.1 赏金
        })

    # hero_id 可能仅由 request 指定而手牌里没有 "Dealt to"：按名字兜底
    if not hero_id:
        for s in hand.seats:
            if s.get("is_hero"):
                hero_id = str(s["name"])
                hero_seat = int(s["seat"])
                break

    button_seat = hand.button_seat

    # ---- 位置：positions_for 给每个 seat 一个 label ----
    pos_map = positions_for(players, button_seat) if button_seat is not None else {}
    for p in players:
        p["position"] = pos_map.get(p["seat"])

    # ---- 重放 action_history：pot / current_bet / to_call / last_aggressor / status ----
    # 动作按玩家名引用；映射 name -> player_id（hero 的 pid == hero_id）
    name_to_pid = {p["name"]: p["player_id"] for p in players}

    replay = _replay(hand, street, name_to_pid, hero_id, bb, players)
    pot = replay["pot"]
    current_bet = replay["current_bet"]
    to_call = replay["to_call"]
    last_aggressor = replay["last_aggressor"]
    action_history = replay["action_history"]
    street_actions = replay["street_actions"]

    # ---- hero 剩余筹码 & available_actions ----
    hero_stack = _hero_stack(players, hero_id, replay["total_invested"])
    available_actions = _available_actions(to_call, hero_stack, pot)

    # ---- min_raise_to = 2*current_bet - hero 本街已投入（preflop 用 bb 兜底）----
    hero_street_invested = replay["street_invested"].get((street, hero_id), 0.0)
    base = current_bet if current_bet > 0 else (bb if street == "preflop" else 0.0)
    min_raise_to = max(0.0, round(2.0 * base - hero_street_invested, 4))

    hand_id = str(hand.hand_id or request.get("hand_id") or f"hh-{_now_ts()}")

    # ---- cash / tournament 块 ----
    cash_block: dict | None = None
    tournament_block: dict | None = None
    if game_type == "cash":
        stakes = _stakes_name(ctx, sb, bb)
        cash_block = {
            "stakes": stakes, "sb": sb, "bb": bb,
            "currency": str(hand.currency or "USD"), "rake_percent": float(request.get("rake_percent") or 5.0),
        }
    else:
        tournament_block = _tournament_block(hand, game_type)

    evidence = [{
        "kind": "state", "source": "hand_history",
        "text": f"由手牌历史构建状态: street={street}, pot={pot}, to_call={to_call}",
        "weight": 1.0,
    }]
    if not hand.actions and not hand.street_boards:
        warnings.append({"code": "EMPTY_HAND", "message": "手牌历史未解析出任何动作"})

    return StateOutput(
        hand_id=hand_id, game_type=game_type, street=street, board=board,
        pot=round(pot, 4), current_bet=round(current_bet, 4), to_call=round(to_call, 4),
        min_raise_to=min_raise_to, sb=sb, bb=bb, ante=ante,
        players=players, button_seat=button_seat, hero_id=hero_id, hero_seat=hero_seat,
        action_history=action_history, street_actions=street_actions,
        available_actions=available_actions, last_aggressor=last_aggressor,
        rake=float(hand.rake or 0.0), source="hand_history",
        cash=cash_block, tournament=tournament_block,
        confidence=1.0, evidence=evidence, warnings=warnings,
    )


def _replay(hand: ParsedHand, decision_street: str, name_to_pid: dict, hero_id: str,
            bb: float, players: list) -> dict:
    """重放全部动作，返回 pot/current_bet/to_call/last_aggressor/action_history 等。

    解析器动作金额语义（见 domain/poker/hand_history.py）：
      - post    : amount = 盲注/ante 金额
      - bet     : amount = 本街总下注额（首注）
      - raise   : amount = 加注增量（"raises $1.10 to $1.20" -> 1.10 = 1.20 - 1.10? 否，
                  实为 1.20 - 前一个下注额），"to" 被解析器丢弃
      - call    : amount = 跟注额（= 本街当前下注 - 其已投入）
      - check/fold : 0

    因此 pot 累加必须按 "to" 语义重放：
      加注者新总投入 new_total = 本街当前下注 + 增量；实际进 pot = new_total - 其已投入。
    """
    street_invested: dict[tuple[str, str], float] = {}   # (street, pid) -> 本街累计投入
    total_invested: dict[str, float] = {}                # pid -> 全手累计投入
    street_bet: dict[str, float] = {}                    # street -> 本街当前最高总下注
    status: dict[str, str] = {}
    pot = 0.0
    last_aggressor: str | None = None
    seq = 0
    action_history: list[dict] = []
    street_actions: dict[str, list[dict]] = {}

    def contrib(st: str, pid: str) -> float:
        return street_invested.get((st, pid), 0.0)

    def add_contrib(st: str, pid: str, amt: float) -> None:
        street_invested[(st, pid)] = contrib(st, pid) + amt
        total_invested[pid] = total_invested.get(pid, 0.0) + amt

    for a in hand.actions:
        st = str(a["street"])
        pid = name_to_pid.get(str(a["player"]), str(a["player"]))
        act = str(a["action"])
        amt = float(a.get("amount") or 0.0)
        all_in = bool(a.get("all_in"))

        if act == "post":
            # 盲注/ante 直接进入 pot；bb 构成 preflop 的 baseline 当前下注
            pot += amt
            add_contrib(st, pid, amt)
            if st == "preflop" and amt >= bb - 1e-9:
                street_bet[st] = max(street_bet.get(st, 0.0), bb)
            continue

        if act == "bet":
            prior = contrib(st, pid)
            pot += amt - prior
            add_contrib(st, pid, amt - prior)
            street_bet[st] = max(street_bet.get(st, 0.0), amt)
            last_aggressor = pid
        elif act == "raise":
            # new_total = 本街当前下注 + 加注增量（"raises $0.20 to $0.30" -> 0.10 + 0.20 = 0.30）
            prior = contrib(st, pid)
            new_total = street_bet.get(st, 0.0) + amt
            pot += new_total - prior
            add_contrib(st, pid, new_total - prior)
            street_bet[st] = new_total
            last_aggressor = pid
        elif act == "call":
            pot += amt
            add_contrib(st, pid, amt)
        elif act == "fold":
            status[pid] = "folded"
        elif act == "check":
            pass  # 无金额变化

        if all_in:
            status[pid] = "allin"

        # action_history 只收录真实动作（不含 post），amount 用 "to" 语义（本街累计额）
        if act in _ACTION_KINDS:
            seq += 1
            to_amount = round(contrib(st, pid), 4) if act in ("bet", "raise", "call") else 0.0
            rec = {"street": st, "player_id": pid, "action": act,
                   "amount": to_amount, "all_in": all_in, "seq": seq}
            action_history.append(rec)
            street_actions.setdefault(st, []).append(rec)

    # 应用 fold/allin 状态到 players
    for p in players:
        if p["player_id"] in status:
            p["status"] = status[p["player_id"]]

    # current_bet = 决策街当前最高总下注（preflop 无加注时至少是 bb）
    current_bet = street_bet.get(decision_street, 0.0)
    if decision_street == "preflop" and current_bet < bb:
        current_bet = bb
    # to_call = current_bet - hero 本街已投入
    hero_street = contrib(decision_street, hero_id)
    to_call = max(0.0, current_bet - hero_street)

    return {
        "pot": pot, "current_bet": current_bet, "to_call": to_call,
        "last_aggressor": last_aggressor, "action_history": action_history,
        "street_actions": street_actions, "total_invested": total_invested,
        "street_invested": street_invested,
    }


# --------------------------------------------------------------------------- #
# vision_observation 路径
# --------------------------------------------------------------------------- #
def _preflop_context(obs: dict, players: list, bb: float,
                     hero_seat: int | None = None) -> dict:
    """从**逐座位投入**推导翻前节点上下文（不猜：证据不足 ⇒ {}）。

    返回 {level, raises, aggressor_position, aggressor_seat, max_commitment_bb,
          source, zone_seats}
      level ∈ open | vs_raise | vs_3bet | vs_4bet
      source ∈ action_history | commitment_geometry
    """
    if float(bb or 0.0) <= 0:
        return {}
    pot_bb = float(obs.get("pot") or 0.0) / float(bb)
    # 来源 1：观察自带真实动作历史（最可靠）
    hist = [r for r in (obs.get("action_history") or [])
            if isinstance(r, dict) and str(r.get("street")) == "preflop"
            and str(r.get("action")) in ("raise", "bet", "all-in")]
    if hist:
        n = len(hist)
        last = hist[-1]
        seat = last.get("seat")
        pos = next((p.get("position") for p in players
                     if seat is not None and p.get("seat") == int(seat)), None)
        return {"level": {1: "vs_raise", 2: "vs_3bet"}.get(n, "vs_4bet"),
                "raises": n, "aggressor_position": pos,
                "aggressor_seat": (int(seat) if seat is not None else None),
                "max_commitment_bb": None, "source": "action_history"}
    # 来源 2：逐座位投入几何
    try:
        from poker_decision_ai.line_tracker import CLOCKWISE_ZONES
        bets = ((obs.get("extra") or {}).get("v3f_bets") or {})
        if not isinstance(bets, dict) or not bets:
            return {}
        max_zone, max_val = None, 0.0
        for z in CLOCKWISE_ZONES:
            v = bets.get(z)
            if isinstance(v, (int, float)) and float(v) > max_val:
                max_zone, max_val = z, float(v)
        if max_zone is None:
            return {}
        max_bb = max_val / float(bb)
        # P30-9-CM：**主路径**——投注区标签与 canonical 座位标签同源
        # （CANONICAL_SEATS=("S1","S2","S3","S5","S6","S4")），
        # `ring_number(zone)` 即座位号；无需 SB 锚点几何（后者仅在可得时作交叉校验）。
        ring_number = None
        try:
            from poker_decision_ai.vision2.pixel.realclient.seat_position import (
                ring_number as _rn)
            ring_number = _rn
        except Exception:                                      # noqa: BLE001
            ring_number = None
        zone_seats: dict = {}
        if ring_number is not None:
            for z in CLOCKWISE_ZONES:
                n = ring_number(z)
                if n is not None:
                    zone_seats[z] = n
        seat = zone_seats.get(max_zone)
        # 交叉校验（可得时）：SB 锚点几何给出的 zone→seat 若与主路径冲突 ⇒ 不采用
        try:
            from poker_decision_ai.position_derive import derive_positions_by_zone
            derived = derive_positions_by_zone(obs, CLOCKWISE_ZONES)
            if derived:
                z2s = (derived.get("zone_seats") or derived.get("by_zone") or {})
                if isinstance(z2s, dict) and z2s:
                    zone_seats = {k: int(v) for k, v in z2s.items()
                                   if v is not None}
                    seat = zone_seats.get(max_zone)
        except Exception:                                      # noqa: BLE001
            pass
        # 加注者 = **非 hero** 的最大投入区（hero 自己的跟注/加注不算加注者）
        hero_zone = next((z for z, st_ in zone_seats.items()
                           if hero_seat is not None
                           and int(st_) == int(hero_seat)), None)
        agg_zone, agg_val = None, 0.0
        for z in CLOCKWISE_ZONES:
            if hero_zone is not None and z == hero_zone:
                continue
            v = bets.get(z)
            if isinstance(v, (int, float)) and float(v) > agg_val:
                agg_zone, agg_val = z, float(v)
        seat = zone_seats.get(agg_zone) if agg_zone is not None else None
        if seat is not None and hero_seat is not None and int(seat) == int(hero_seat):
            seat = None
        pos = next((p.get("position") for p in players
                     if seat is not None and p.get("seat") == int(seat)), None)
        # 加注者未能定位时，退回"最高投入者"（若其非 hero）
        if pos is None and hero_zone is None and agg_zone is None:
            seat2 = zone_seats.get(max_zone)
            if seat2 is not None and (hero_seat is None
                                       or int(seat2) != int(hero_seat)):
                seat = seat2
                pos = next((p.get("position") for p in players
                             if p.get("seat") == int(seat2)), None)
        if max_bb <= 1.2:
            level, raises = "open", 0
        elif max_bb <= 4.5:
            level, raises = "vs_raise", 1
        elif max_bb <= 16.0:
            level, raises = "vs_3bet", 2
        else:
            level, raises = "vs_4bet", 3
        return {"level": level, "raises": raises,
                "aggressor_position": pos,
                "aggressor_seat": (int(seat) if seat is not None else None),
                "max_commitment_bb": round(max_bb, 3),
                "source": "commitment_geometry",
                "zone_seats": zone_seats, "pot_bb": round(pot_bb, 2)}
    except Exception:                                          # noqa: BLE001
        return {}


def _from_observation(obs: dict, request: dict, ctx: AgentContext) -> StateOutput:
    warnings: list[dict] = []
    blinds = obs.get("blinds") or {}
    sb = float(blinds.get("sb") or 0.0)
    bb = float(blinds.get("bb") or 0.0)
    ante = float(obs.get("ante") or 0.0)
    game_type = _derive_game_type(request, obs)
    hero_seat = obs.get("hero_seat")
    button_seat = obs.get("button_seat")
    hero_id = str(request.get("hero_id") or (f"seat{hero_seat}" if hero_seat is not None else "") or "")

    players: list[dict] = []
    for s in obs.get("seats") or []:
        seat = int(s.get("seat"))
        is_hero = (hero_seat is not None and seat == int(hero_seat))
        pid = hero_id if is_hero else f"seat{seat}"
        players.append({
            "player_id": pid, "name": str(s.get("name") or f"seat{seat}"),
            "seat": seat, "stack": float(s.get("stack") or 0.0),
            "status": "active" if s.get("in_hand") else "folded",
            "hole_cards": list(obs.get("hole_cards") or []) if is_hero else [],
            "position": None, "is_hero": is_hero,
        })

    pos_map = positions_for(players, button_seat) if button_seat is not None else {}
    for p in players:
        p["position"] = pos_map.get(p["seat"])

    pot = float(obs.get("pot") or 0.0)
    current_bet = float(obs.get("current_bet") or 0.0)
    to_call = float(obs.get("to_call") or 0.0)
    # P30-9-C：to_call **数值缺失 = UNKNOWN**（≠ 合法零）。UNKNOWN 时不得
    # 由 to_call=0 反推 check/bet —— 那是默认值填充，违反 UNKNOWN-first 红线，
    # 并会导致「面对下注却输出 bet/check」（call/check 混淆）。
    to_call_unknown = obs.get("to_call") is None
    observed_actions = [str(a).strip().lower()
                        for a in (obs.get("available_actions") or [])]

    hero_stack = _hero_stack(players, hero_id, {})
    available_actions = (observed_actions
                         or ([] if to_call_unknown
                             else _available_actions(to_call, hero_stack, pot)))

    # M4（facing-bet 锚点）：observation 可携带**真实**动作历史行
    #（bench 引擎/宿主声明通道；player_id 缺失时按 seat+hero_seat 解析——
    # 与本函数的 seat→pid 约定一致）。绝不编造：无该字段 → 空历史（原状）。
    obs_history: list[dict] = []
    # P30-9-CR：名字 → pid（与下方 seats 构建同一约定）
    _name_to_pid_early: dict[str, str] = {}
    for _s in obs.get("seats") or []:
        if not isinstance(_s, dict):
            continue
        _nm = str(_s.get("name") or "")
        _si = _s.get("seat")
        if not _nm or _si is None:
            continue
        _name_to_pid_early[_nm] = (
            hero_id if (hero_seat is not None and int(_si) == int(hero_seat))
            else f"seat{int(_si)}")
    _unattributed = 0
    for row in obs.get("action_history") or []:
        if not isinstance(row, dict):
            continue
        seat = row.get("seat")
        pid = str(row.get("player_id") or "")
        if not pid:
            _nm = str(row.get("name") or row.get("player") or "")
            if _nm:
                pid = _name_to_pid_early.get(_nm, "")
        if not pid and seat is not None:
            seat_i = int(seat)
            pid = hero_id if (hero_seat is not None
                              and seat_i == int(hero_seat)) else f"seat{seat_i}"
        if not pid:
            _unattributed += 1
        act = str(row.get("action") or "")
        if act not in ("fold", "check", "call", "bet", "raise", "all-in"):
            continue
        _entry = {
            "street": str(row.get("street") or ""),
            "player_id": pid, "action": act,
            "amount": float(row.get("amount") or 0.0),
            "all_in": act == "all-in",
            "seq": len(obs_history) + 1,
        }
        if not pid:
            _entry["attribution"] = "unknown"
        obs_history.append(_entry)

    cash_block: dict | None = None
    tournament_block: dict | None = None
    if game_type == "cash":
        stakes = _stakes_name(ctx, sb, bb)
        cash_block = {"stakes": stakes, "sb": sb, "bb": bb, "currency": "USD",
                      "rake_percent": float(request.get("rake_percent") or 5.0)}
    else:
        tournament_block = {"tournament_id": "", "name": "", "total_players": 0,
                            "players_remaining": 0, "total_chips": 0.0, "payouts": [],
                            "next_payout": None, "itm_players": 0, "satellite_seats": None,
                            "stage": None, "avg_stack": None}

    semantics = _field_semantics_from_observation(obs, pot, current_bet,
                                                  to_call)
    # P30-9-B：min_raise_to 语义防护——current_bet UNKNOWN（bridge 标记）
    # 时 min_raise_to 为 2×0=0（V2 布局）但语义未知：标记语义；数值保留
    # （消费方需读取 field_semantics 区分——不静默当作合法 raise 底线）。
    if semantics.get("current_bet") == "UNKNOWN" and current_bet == 0.0:
        semantics["min_raise_to"] = "UNKNOWN"

    # P30-9-CD：翻前节点上下文（只增不改；证据不足 ⇒ {}）
    preflop_ctx: dict = {}
    if str(obs.get("street") or "preflop") == "preflop":
        preflop_ctx = _preflop_context(obs, players, bb, hero_seat=hero_seat)

    if _unattributed:
        warnings.append({
            "code": "HISTORY_ATTRIBUTION_MISSING",
            "message": (f"{_unattributed} 条动作历史无法归属到玩家"
                         "（缺 player_id/seat/可解析的名字）⇒ 这些行不参与线路识别")})

    # P30-9-CI：公共牌张数校验（不足 ⇒ 标注 INCOMPLETE，**不伪造**缺失的牌）
    _need = {"preflop": 0, "flop": 3, "turn": 4, "river": 5}
    _street = str(obs.get("street") or "preflop")
    _board_now = [str(c) for c in (obs.get("board") or [])]
    _expect = _need.get(_street)
    if _expect is not None and len(_board_now) < _expect:
        semantics["board"] = "INCOMPLETE"
        warnings.append({
            "code": "BOARD_INCOMPLETE",
            "message": (f"公共牌只识别到 {len(_board_now)}/{_expect} 张"
                         f"（street={_street}）⇒ 牌面相关计算不可信，建议重拍")})

    return StateOutput(
        hand_id=str(request.get("hand_id") or f"hh-{_now_ts()}"),
        game_type=game_type, street=str(obs.get("street") or "preflop"),
        board=[str(c) for c in (obs.get("board") or [])],
        pot=pot, current_bet=current_bet, to_call=to_call,
        min_raise_to=max(0.0, round(2.0 * current_bet - 0.0, 4)),
        sb=sb, bb=bb, ante=ante,
        players=players, button_seat=button_seat, hero_id=hero_id, hero_seat=hero_seat,
        action_history=obs_history, street_actions={},
        available_actions=available_actions, last_aggressor=None,
        rake=0.0, source="vision",
        cash=cash_block, tournament=tournament_block,
        confidence=1.0,
        evidence=[{"kind": "state", "source": "vision",
                   "text": f"由视觉观察映射状态: street={obs.get('street')}, pot={pot}",
                   "weight": 1.0}],
        warnings=warnings,
        field_semantics=semantics,
        preflop_context=preflop_ctx,
    )


def _field_semantics_from_observation(obs: dict, pot, current_bet,
                                      to_call) -> dict:
    """V3F-P30-8-B：从 observation（含 bridge extra 标记）构建三态语义。

    - current_bet：bridge 标记 v3f_current_bet_status=UNKNOWN → UNKNOWN
      （数值仍 0.0 为 V2 布局兼容——语义在此区分——P30-2D 冻结）；
    - action_history：v3f_action_history_status → UNKNOWN/RECOGNIZED；
    - pot/to_call：bridge RECOGNIZED 保证 → KNOWN_VALUE（合法）；
    - 旧 V2 手写 observation（无 extra 标记）→ 空 dict（默认——不影响）。
    """
    extra = (obs or {}).get("extra") or {}
    semantics: dict = {}
    cb_st = extra.get("v3f_current_bet_status")
    if cb_st == "UNKNOWN":
        semantics["current_bet"] = "UNKNOWN"     # 数值 0.0 但语义未知
    else:
        semantics["current_bet"] = ("KNOWN_ZERO" if (current_bet or 0.0) == 0
                                    else "KNOWN_VALUE")
    ah_st = extra.get("v3f_action_history_status")
    semantics["action_history"] = (ah_st if ah_st in ("UNKNOWN", "RECOGNIZED")
                                   else "UNKNOWN")
    semantics["pot"] = ("KNOWN_VALUE" if (pot or 0.0) != 0 else "KNOWN_ZERO")
    # to_call：显式 status 或**观察缺字段** ⇒ UNKNOWN（数值 0.0 但语义未知）
    tc_st = str(extra.get("v3f_to_call_status") or "").upper()
    obs_actions = [str(a).strip().lower()
                   for a in (obs.get("available_actions") or [])]
    if tc_st == "UNKNOWN" or obs.get("to_call") is None:
        semantics["to_call"] = "UNKNOWN"
    else:
        semantics["to_call"] = ("KNOWN_VALUE" if (to_call or 0.0) != 0
                                else "KNOWN_ZERO")
    # facing_bet：**证据优先** —— 数值 > 0，或已知动作集含 call 且不含 check
    if (to_call or 0.0) > 1e-9:
        semantics["facing_bet"] = "BET_PENDING"
    elif obs_actions:
        if "call" in obs_actions and "check" not in obs_actions:
            semantics["facing_bet"] = "BET_PENDING"
        elif "check" in obs_actions and "call" not in obs_actions:
            semantics["facing_bet"] = "NO_BET"
        else:
            semantics["facing_bet"] = "UNKNOWN"
    elif semantics["to_call"] == "KNOWN_ZERO":
        semantics["facing_bet"] = "NO_BET"
    else:
        semantics["facing_bet"] = "UNKNOWN"
    return semantics


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _derive_game_type(request: dict, obs: dict | None = None) -> str:
    gt = request.get("game_type")
    if gt in ("cash", "mtt", "satellite"):
        return str(gt)
    if obs and obs.get("extra"):
        extra = obs.get("extra") or {}
        if extra.get("game_type") in ("cash", "mtt", "satellite"):
            return str(extra["game_type"])
    kind = str(request.get("kind") or "")
    if kind == "tournament_hand":
        return "mtt"
    return "cash"


def _stakes_name(ctx: AgentContext, sb: float, bb: float) -> str:
    """反查级别名：遍历 stakes_table 找 sb/bb 匹配；否则 f"NL{100*bb:g}"。"""
    for name, info in ctx.config.stakes_table().items():
        if isinstance(info, dict):
            if float(info.get("sb", -1) or -1) == float(sb) and float(info.get("bb", -1) or -1) == float(bb):
                return str(name)
    return f"NL{100.0 * float(bb):g}"


def _tournament_block(hand: ParsedHand, game_type: str) -> dict:
    """由解析结果 tournament dict + SUMMARY 行构造 tournament 块（含 V1.1 赏金字段）。"""
    t = dict(hand.tournament or {})
    satellite_seats = hand.satellite_seats if hand.satellite_seats is not None else t.get("satellite_seats")
    if game_type == "satellite":
        satellite_seats = satellite_seats if satellite_seats is not None else 1
    return {
        "tournament_id": str(t.get("tournament_id") or ""),
        "name": str(t.get("name") or ""),
        "total_players": int(t.get("total_players") or 0),
        "players_remaining": int(t.get("players_remaining") or 0),
        "total_chips": float(t.get("total_chips") or 0.0),
        "payouts": list(t.get("payouts") or []),
        "next_payout": t.get("next_payout"),
        "itm_players": int(t.get("itm_players") or 0),
        "satellite_seats": satellite_seats,
        "stage": t.get("stage"),
        "avg_stack": t.get("avg_stack"),
        # ---- V1.1 Bounty ----
        "bounty_type": t.get("bounty_type"),
        "is_bounty": bool(t.get("is_bounty") or t.get("bounty_type")),
        "progressive": bool(t.get("progressive") or t.get("bounty_type") == "pko"),
        "bounty_rate": float(t.get("bounty_rate") or 0.5),
        "total_bounty_pool": float(t.get("total_bounty_pool") or 0.0),
        "prize_pool": float(t.get("prize_pool") or 0.0),
        "mystery_phase_active": bool(t.get("mystery_phase_active")),
    }


def _hero_stack(players: list, hero_id: str, total_invested: dict) -> float:
    """hero 剩余筹码 = seat stack - 全手已投入。"""
    for p in players:
        if p["player_id"] == hero_id:
            return max(0.0, float(p["stack"]) - total_invested.get(hero_id, 0.0))
    return 0.0


def _available_actions(to_call: float, hero_stack: float, pot: float) -> list[str]:
    """推导可用动作（契约 §4 state 的规则）。"""
    if to_call <= 0:
        actions = ["check", "bet"]
        if hero_stack <= pot:           # 深码短于底池 -> 可全下
            actions.append("allin")
        return actions
    if hero_stack <= to_call:           # 只剩跟注全下
        return ["fold", "call"]
    return ["fold", "call", "raise"]


def _ante_from_request(request: dict) -> float:
    try:
        return float(request.get("ante") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _now_ts() -> int:
    import time
    return int(time.time())
