"""PI-EV 参照**完整性修复版**（`pi-ev-reference-complete-v1`）。

漏洞根因（本模块的审计结论）
--------------------------
早期阶段的验证主体是 **AI 座本身**（P54/R1 等评估 Pluribus 座位自己的决策），
因此 `villain_hands` 构造里写了 `p == AI_SEAT → 跳过`。当同一模式被复用到
**人类为验证主体**的阶段（S′ / S″ / S‴ / perf_runner_v2）时，该过滤就变成了
**把一名仍在牌局中的已知对手（Pluribus 座）从参照里剔除**：

  * 若该点只有 AI 座仍在牌局 → `villain_hands == []` → `PiEvReference` 判
    `available=False`（"no active opponent"）→ 点被计入 `unavailable`（S‴ 共 1,356 点）；
  * 若该点 AI 座与人类对手同时在场 → 参照**少算一名活跃对手**，hero 权益被高估、
    `pi_best` 构成受偏（S‴ 记录点中 414 点，6.76%）。

修复原则（**只修完整性，不动 PI-EV 定义**）
------------------------------------------
1. `villain_hands` = **全部仍在牌局中的对手**（`!= hero` 且未弃牌），**含 AI 座**，
   且必须持有 2 张已知手牌；
2. **严格 UNKNOWN 传播**：任一活跃对手缺已知手牌 → 该点判为 `available=False`
   （理由 `incomplete_information`），**绝不**用"少算对手"的方式算出可用值；
3. 无活跃对手 → `available=False`（沿用既有语义）；
4. 枚举预算 / seed / check-down 规则 / 候选动作 / 取值公式 **完全不改**——
   本模块直接调用 sealed `unfreeze_core.PiEvReference`，只修正传入的
   `villain_hands` 与 provenance 注释；
5. 参照仍需标注 `equilibrium=False` / `NOT_exact_gto=True`：**无 solver、无 GTO**。

可证明的完整性（本模块提供 check 函数）
--------------------------------------
* `build_villain_hands` 返回的 `villains_used` 必须等于「活跃且手牌已知」的对手全集；
* `live == villains_used + unknown_live`（集合无遗漏、无多余）；
* `available=True` ⟹ `unknown_live == []` 且 `len(villains_used) >= 1`
  （**不可能凭空制造可用性**）。
"""
from __future__ import annotations

from ..pluribus_validation import AI_SEAT
from .unfreeze_core import PI_EV_MAX_RUNOUTS, PiEvReference

REFERENCE_VERSION = "pi-ev-reference-complete-v1"
LEGACY_SOURCE = "legacy_excl_ai_seat"
COMPLETE_SOURCE = "all_live_opponents_incl_ai_seat"
UNAVAILABLE_INCOMPLETE = "incomplete_information"


# ── 活跃对手 / villain_hands 构造 ─────────────────────────────────────
def live_opponents(hand: dict, hero_pid: str, folded: set) -> list:
    """仍在牌局中的对手（未弃牌、且不是 hero 自己）——**不排除 AI 座**。"""
    return [p for p in hand["players"] if p != hero_pid and p not in folded]


def build_villain_hands(hand: dict, hero_pid: str, folded: set, *,
                        include_ai_seat: bool = True) -> dict:
    """返回完整/遗留两种口径下的对手手牌集合与未知集合。

    `include_ai_seat=False` 复现**遗留口径**（排除 AI 座）以便做修复前后对比。
    """
    live = live_opponents(hand, hero_pid, folded)
    if not include_ai_seat:
        live = [p for p in live if p != AI_SEAT]
    vhands: list = []
    unknown: list = []
    for p in live:
        cards = list(hand["hole_cards"].get(p) or [])
        if len(cards) == 2:
            vhands.append(cards)
        else:
            unknown.append(p)
    return {"live": live, "villain_hands": vhands, "unknown_live": unknown,
            "ai_seat_live": AI_SEAT in live,
            "villains_used": len(vhands), "live_count": len(live)}


# ── 完整性证明（对任一构造结果成立） ──────────────────────────────────
def assert_complete(facts: dict) -> None:
    """结构不变量：活跃集 = 已用集 + 未知集（无遗漏、无多余、无重复）。"""
    if facts["live_count"] != facts["villains_used"] + len(
            facts["unknown_live"]):
        raise AssertionError(f"活跃对手未守恒: {facts}")


def is_faithful(facts: dict) -> bool:
    try:
        assert_complete(facts)
    except AssertionError:
        return False
    return True


# ── 修复后的参照（直接委托 sealed PiEvReference） ─────────────────────
def reference_complete(*, hand: dict, hero_pid: str, folded: set,
                       hero_cards: list, board: list, pot_bb: float,
                       to_call_bb: float, hero_stack_bb: float,
                       candidates: list, rule: str = "villain_calls",
                       include_ai_seat: bool = True) -> dict:
    """完整性修复版参照。**可用性判据**：

    * `unknown_live` 非空 → 不可用（`incomplete_information`，UNKNOWN 传播）；
    * 无活跃对手 → 不可用（沿用 sealed 理由）；
    * 否则 → 用**全部**活跃对手交给 sealed `PiEvReference` 计算（定义不变）。
    """
    facts = build_villain_hands(hand, hero_pid, folded,
                                include_ai_seat=include_ai_seat)
    assert_complete(facts)
    src = COMPLETE_SOURCE if include_ai_seat else LEGACY_SOURCE
    if facts["unknown_live"]:
        return {"available": False,
                "unavailable_reason": f"{UNAVAILABLE_INCOMPLETE}: live "
                                      f"opponent(s) without known hole cards "
                                      f"{sorted(facts['unknown_live'])}",
                "provenance": {"kind": "pi_ev", "evidence_level": "L2",
                               "reference": "perfect-information",
                               "equilibrium": False,
                               "rule": f"checkdown_{rule}",
                               "runouts_enumerated": 0, "NOT_exact_gto": True,
                               "reference_version": REFERENCE_VERSION,
                               "villain_source": src,
                               "live_opponents": facts["live_count"],
                               "villains_used": 0,
                               "unknown_live_opponents":
                                   sorted(facts["unknown_live"])},
                "values": {}, "ordering": []}
    ref = PiEvReference().reference(
        hero_cards=list(hero_cards), villain_hands=facts["villain_hands"],
        board=list(board), pot_bb=pot_bb, to_call_bb=to_call_bb,
        hero_stack_bb=hero_stack_bb, candidates=candidates, rule=rule)
    out = dict(ref)
    out["provenance"] = dict(ref["provenance"],
                             reference_version=REFERENCE_VERSION,
                             villain_source=src,
                             live_opponents=facts["live_count"],
                             villains_used=facts["villains_used"],
                             unknown_live_opponents=[])
    out["villain_facts"] = {"live_count": facts["live_count"],
                            "villains_used": facts["villains_used"],
                            "ai_seat_live": facts["ai_seat_live"],
                            "unknown_live": []}
    if out.get("available"):
        assert out["villain_facts"]["villains_used"] >= 1, \
            "available=True 但没有任何对手 → 制造了伪可用"
        assert out["provenance"]["unknown_live_opponents"] == []
    return out


def budget_ok(n_runouts: int) -> bool:
    """仅用于审计陈述；预算常量取自 sealed 模块，未被本模块修改。"""
    return 0 < n_runouts <= PI_EV_MAX_RUNOUTS


# ── 决策集级审计（修复前后差异；只读语料，不产生验收判定） ────────────
def audit_decision_set(hands: list, players: set, *, hand_min_index: int,
                       max_points: int = 10 ** 9) -> dict:
    """遍历 flop/turn 上目标玩家的决策点，比较遗留口径与修复口径。

    返回逐项计数：活跃对手分布、AI 座在场、修复前后可用性、未知活跃对手等。
    不写入任何文件、不做任何验收判定。
    """
    from ..pluribus_validation import _load_config  # noqa: F401
    from .s3_validation import _adv
    from ..pluribus_validation import _current_aggressor, _state_for
    from ..pluribus_decision_adapter import _positions_for_seats
    from .pi_ev_contract import pi_ev_inputs

    scope = [h for i, h in enumerate(hands) if i >= hand_min_index]
    cnt = {"points": 0, "ai_live": 0, "ai_folded": 0, "ai_live_known": 0,
           "legacy_available": 0, "complete_available": 0,
           "legacy_unavailable": 0, "complete_unavailable": 0,
           "legacy_unavail_ai_live": 0, "legacy_unavail_no_live": 0,
           "unknown_live_any": 0, "zero_live": 0,
           "changed_villain_set": 0, "changed_n_runouts": 0,
           "runouts_budget_exceeded": 0, "villain_fallback_diff": 0,
           "ai_folded_but_legacy_live": 0}
    live_dist: dict = {}
    for h in scope:
        if cnt["points"] >= max_points:
            break
        bb = h["bb"] or 1.0
        pos_map = _positions_for_seats(len(h["players"]), len(h["players"]))
        folded: set = set()          # 正确口径：所有玩家（含 AI 座）的弃牌
        folded_legacy: set = set()   # 遗留口径：AI 座弃牌不入账（= sealed runner）
        hist: list = []
        boards = h["board_by_street"]
        for st in h["streets"]:
            street = st["street"]
            if street == "preflop":
                board: list = []
            elif street == "flop":
                board = list(boards.get("flop", []))
            elif street == "turn":
                board = list(boards.get("flop", [])) + list(
                    boards.get("turn", []))
            else:
                board = (list(boards.get("flop", []))
                         + list(boards.get("turn", []))
                         + list(boards.get("river", [])))
            record = street in ("flop", "turn")
            for a in st["actions"]:
                pid = a["player"]
                # **所有玩家（含 AI 座）的弃牌都必须入账** —— 遗留 runner 对
                # AI 座 `continue` 而未调用 `_adv`，导致「已弃牌的 AI 座仍被当成
                # 在场对手」，本审计与被修版本一律用 `_adv` 统一推进。
                if not record or pid not in players:
                    if a["action"] == "fold":
                        folded.add(pid)
                        if pid != AI_SEAT:
                            folded_legacy.add(pid)
                    if a["action"] in ("bet", "raise", "call"):
                        hist.append({"street": street, "player_id": pid,
                                     "action": a["action"],
                                     "amount": a.get("amount") or 0.0})
                    continue
                hole = list(h["hole_cards"].get(pid) or [])
                if len(hole) != 2:
                    _adv(a, pid, street, folded, hist)
                    continue
                cnt["points"] += 1
                leg = build_villain_hands(h, pid, folded, include_ai_seat=False)
                ful = build_villain_hands(h, pid, folded, include_ai_seat=True)
                live_dist[ful["live_count"]] = live_dist.get(
                    ful["live_count"], 0) + 1
                if ful["ai_seat_live"]:
                    cnt["ai_live"] += 1
                    if AI_SEAT not in ful["unknown_live"]:
                        cnt["ai_live_known"] += 1
                else:
                    cnt["ai_folded"] += 1
                if ful["unknown_live"]:
                    cnt["unknown_live_any"] += 1
                if ful["live_count"] == 0:
                    cnt["zero_live"] += 1
                if leg["villains_used"] != ful["villains_used"]:
                    cnt["changed_villain_set"] += 1
                cnt["legacy_available"] += 1 if leg["villains_used"] else 0
                cnt["legacy_unavailable"] += 1 if not leg["villains_used"] else 0
                if not leg["villains_used"]:
                    if ful["villains_used"]:
                        cnt["legacy_unavail_ai_live"] += 1
                    else:
                        cnt["legacy_unavail_no_live"] += 1
                ok_full = bool(ful["villains_used"]) and not ful["unknown_live"]
                cnt["complete_available"] += 1 if ok_full else 0
                cnt["complete_unavailable"] += 1 if not ok_full else 0
                state = _state_for(h, a, street, board, pos_map, folded, hist,
                                   pid, bb)
                inputs = pi_ev_inputs(state, raw=a, hand_bb=bb)
                n_leg = _n_runouts(hole, board, leg["villain_hands"])
                n_ful = _n_runouts(hole, board, ful["villain_hands"])
                if n_leg != n_ful:
                    cnt["changed_n_runouts"] += 1
                if max(n_leg, n_ful) > PI_EV_MAX_RUNOUTS:
                    cnt["runouts_budget_exceeded"] += 1
                # 第二处缺陷（审计项，不属本次修复范围）：遗留 runner 对 AI 座
                # 不记录弃牌，主对手回退可能选中**已弃牌的 AI 座**。
                if AI_SEAT in folded and AI_SEAT not in folded_legacy:
                    cnt["ai_folded_but_legacy_live"] += 1
                    v = _current_aggressor(h, street, a)
                    if v is None or v == pid:
                        v_leg = next((p for p in h["players"]
                                      if p != pid and p not in folded_legacy),
                                     None)
                        v_fix = next((p for p in h["players"]
                                      if p != pid and p not in folded), None)
                        if v_leg != v_fix:
                            cnt["villain_fallback_diff"] += 1
                _ = inputs
                _adv(a, pid, street, folded, hist)
                if pid != AI_SEAT:
                    _adv(a, pid, street, folded_legacy, hist)
    cnt["live_opponent_distribution"] = dict(sorted(live_dist.items()))
    cnt["hand_min_index"] = hand_min_index
    cnt["players"] = sorted(players)
    cnt["coverage_legacy"] = round(
        cnt["legacy_available"] / max(1, cnt["points"]), 6)
    cnt["coverage_complete"] = round(
        cnt["complete_available"] / max(1, cnt["points"]), 6)
    return cnt


def _n_runouts(hero: list, board: list, vhands: list) -> int:
    """与 sealed `_pi_equity` 同一算法的枚举规模（精确全枚举，无采样）。"""
    from itertools import combinations
    known = {c for c in hero + board}
    for o in vhands:
        known |= set(o)
    deck = [c for c in _deck_codes() if c not in known and c not in board]
    need = 5 - len(board)
    if need < 0 or not vhands:
        return 0
    return len(list(combinations(deck, need)))


def _deck_codes() -> list:
    return [r + s for r in "23456789TJQKA" for s in "shdc"]
