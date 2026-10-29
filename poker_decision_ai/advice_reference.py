"""R14-EXE-3 — 参考建议通道（方案 Y）：legal_actions UNKNOWN 时的标准打法推荐。

定位（production_decision_loop.judge 调用路径的参考建议分支）
----------------------------------------------------------------
bridge readiness OK 但 `derive_legal_actions(...) is None`（to_call UNKNOWN——
静态单帧不知道"是否轮到 hero / 当前下注额 / 可做哪些动作"）时，本通道输出
**该 street+手牌+pot 的标准打法参考**（声明表/既有 spot 库查表），替换原来的
硬 `judge_safety_gate:legal_actions_unknown` UNAVAILABLE。

语义红线（与正常决策严格可区分）
--------------------------------
* 输出 action = 参考打法，**不是** judge 对当前下注的决策（judge 的 to_call
  输入是 V2 布局缺省 0.0，语义 UNKNOWN——其输出仅存
  `judge_action_not_consumed` 审计字段）；
* `advice_type="reference"` / `legal_actions_status="unknown"` /
  `action_validated=False`（legal_actions 未知——验证在语义上不可能）；
* strategy 恒 BASELINE（chart/声明表/spot 库——绝不 REAL/RULES）；
* 诚实 miss：场景未声明 / 表缺失 / 字段不足 → `available=False`，
  绝不编造"标准打法"（上游维持 DECISION_UNAVAILABLE）。

查表源（全部既有冻结资产，只读引用——零新策略发明）
----------------------------------------------------
1. preflop 首入（pot ≈ sb+bb+ante×在座数 容差带内）：
   - HU：configs/solver/preflop_chart_hu.yaml（SOLVER-1 chart provider）；
   - 6-max：configs/ranges.yaml open.<position> 成员资格（V3G 既有声明范围；
     raise 尺寸 = chart sb_open.size_bb=2.5 声明值复用）；
   - BB 首入 → miss（无声明开局表；walk/limp 单帧不可分）。
2. preflop facing（pot 高于首入带）：HU+BB → chart bb_vs_sb_open；
   6-max facing → miss（声明表不含，绝不套用 HU 解）。
3. postflop：core/providers/spot_table.py first_in range-set 查表
   （street+pot 档+纹理类；argmax 频率 + solver sizing）+ 显式"首入假设"
   （to_call 未知——若实际面对下注则不适用）。

本模块是确定性查表——绝不调用 LLM、不做对手建模、不做 EV 计算。
"""
from __future__ import annotations

from typing import Any

#: 版本（provenance 溯源用）
ADVICE_REFERENCE_VERSION = "r14-exe3-advice-reference-v1"
#: 输出契约字面量（frame_decision / production_decision_loop / GUI / CLI 共用）
ADVICE_TYPE_REFERENCE = "reference"
LEGAL_ACTIONS_STATUS_UNKNOWN = "unknown"
DECISION_STATUS_ADVICE_REFERENCE = "ADVICE_REFERENCE"

#: 开注尺寸（chart preflop_chart_hu.yaml sb_open.size_bb 声明值复用——
#: 6-max 开局表 ranges.yaml 未声明尺寸，不新发明常数）
OPEN_RAISE_SIZE_BB = 2.5
#: 首入判定容差（bb）：pot_bb ≤ 理论盲注+ante 池 + 容差 → 首入族。
#: 容差吸收死盲/短 ante/识别 ±1 bb 级误差；超过 → facing 族（诚实）。
FIRST_IN_TOLERANCE_BB = 0.5

#: 诚实警告文案模板（GUI/CLI/loop 同一模板——面向用户）
ADVICE_WARNING_ZH = (
    "参考打法：未识别轮到你的动作选项（legal_actions 未知）——此为该街/"
    "手牌/底池的标准打法参考，不是针对当前下注的决策，也不代表对当前对手"
    "的最优解；动作选项识别恢复后将自动升级为正常决策。")

_STREETS = ("preflop", "flop", "turn", "river")
#: street ↔ board 张数（与 production_bridge._STREET_NEED 同口径）
_STREET_NEED = {"preflop": 0, "flop": 3, "turn": 4, "river": 5}
#: 动作序（确定性 tie-break：频率同 → 动作序靠前者）
_ACTION_ORDER = ("fold", "check", "call", "bet", "raise", "allin")
#: 6-max 开局表有声明的位置（ranges.yaml open.*；BB 无开局声明）
_OPEN_RANGE_POSITIONS = ("UTG", "MP", "CO", "BTN", "SB")
_OPEN_RANGES_FILE = "configs/ranges.yaml"


def _num(v: Any) -> float | None:
    if isinstance(v, bool) or v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    return None


def _hand_class(hole_cards: Any) -> str | None:
    """手牌类（AhKd→AKo / AhKc→AKs... 同 chart 口径）；无法分类 → None。"""
    if not isinstance(hole_cards, (list, tuple)) or len(hole_cards) != 2:
        return None
    try:
        from .core.providers.preflop_chart import PreflopChartProvider
        cls = PreflopChartProvider._hand_class([str(c) for c in hole_cards])
    except Exception:
        return None
    return cls or None


def _range_contains(range_str: str, hand_class: str) -> bool | None:
    """声明式范围成员资格（chart 既有解析器；None=记法不可解析→诚实 miss）。"""
    from .core.providers.preflop_chart import PreflopChartProvider
    return PreflopChartProvider._range_contains(range_str, hand_class)


def _load_ranges() -> dict:
    """configs/ranges.yaml 已声明范围（只读；缺文件 → 空——诚实 miss）。"""
    import yaml
    from pathlib import Path
    try:
        raw = yaml.safe_load(
            Path(_OPEN_RANGES_FILE).read_text(encoding="utf-8")) or {}
    except Exception:
        return {}
    return raw.get("ranges") or {}


def _run_chart_solve(provider, query: dict):
    """chart provider.solve（async 签名、纯同步实现）的上下文安全驱动。

    frame_decision 的参考分支运行在 asyncio loop 内——asyncio.run 会
    RuntimeError；solve 为纯查表（无 await），线程池驱动一次即完成（毫秒级）。
    """
    import asyncio
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(provider.solve(query))
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(asyncio.run, provider.solve(query)).result(
            timeout=30)


def reference_advice(*, street: str | None, hole_cards, board, pot,
                     blinds, ante=None, hero_position: str | None = None,
                     n_seated: int | None = None,
                     hero_stack=None) -> dict:
    """参考建议通道主入口（确定性查表；miss → available=False 绝不编造）。

    返回（无论命中与否）恒携带：
      {"available", "version", "advice_type": "reference",
       "legal_actions_status": "unknown", "action", "size_bb", "confidence",
       "source", "scenario", "hand_class", "basis", "warnings", "reason"}
    """
    basis: dict[str, Any] = {}
    warnings: list[str] = [ADVICE_WARNING_ZH]
    head = {
        "version": ADVICE_REFERENCE_VERSION,
        "advice_type": ADVICE_TYPE_REFERENCE,
        "legal_actions_status": LEGAL_ACTIONS_STATUS_UNKNOWN,
        "available": False,
        "action": None,
        "size_bb": None,
        "confidence": None,
        "source": None,
        "scenario": None,
        "hand_class": None,
        "warnings": warnings,
        "reason": "",
        "basis": basis,
    }

    # ── 输入完整性（UNKNOWN-first：缺任一必需 → 诚实 miss）────────────
    if street not in _STREETS:
        head["reason"] = f"advice_missing_input:street({street!r})"
        return head
    hc = _hand_class(hole_cards)
    if hc is None:
        head["reason"] = "advice_missing_input:hole_cards"
        return head
    p = _num(pot)
    sb = _num((blinds or {}).get("sb")) if isinstance(blinds, dict) else None
    bb = _num((blinds or {}).get("bb")) if isinstance(blinds, dict) else None
    if p is None or p <= 0 or bb is None or bb <= 0 or sb is None or sb <= 0:
        head["reason"] = "advice_missing_input:pot_or_blinds"
        return head
    pot_bb = p / bb
    ante_v = _num(ante) or 0.0
    basis.update({
        "street": street,
        "hand_class": hc,
        "pot_bb": round(pot_bb, 4),
        "blinds_bb": {"sb": round(sb / bb, 4), "bb": 1.0},
        "ante_bb": round(ante_v / bb, 4) if ante_v else 0.0,
        "hero_position": hero_position,
        "n_seated": n_seated,
    })
    head["hand_class"] = hc

    if street == "preflop":
        return _preflop_advice(head, basis, warnings, pot_bb=pot_bb,
                               ante_bb=ante_v / bb, n_seated=n_seated,
                               hero_position=hero_position,
                               hole_cards=hole_cards)
    return _postflop_advice(head, basis, warnings, street=street,
                            board=board, pot_bb=pot_bb)


# ── preflop：首入/facing 场景二分 + 声明表查表 ──────────────────────────
def _preflop_advice(head: dict, basis: dict, warnings: list[str], *,
                    pot_bb: float, ante_bb: float, n_seated,
                    hero_position, hole_cards) -> dict:
    if n_seated is None or n_seated < 2:
        head["reason"] = "advice_missing_input:n_seated"
        return head
    sb_bb = basis["blinds_bb"]["sb"]
    unopened_bb = sb_bb + 1.0 + ante_bb * n_seated
    basis["unopened_pot_bb"] = round(unopened_bb, 4)
    first_in = pot_bb <= unopened_bb + FIRST_IN_TOLERANCE_BB
    basis["scenario_family"] = ("first_in" if first_in else "facing")
    basis["first_in_rule"] = (
        f"pot_bb({pot_bb:.2f}) <= unopened({unopened_bb:.2f})"
        f"+{FIRST_IN_TOLERANCE_BB} -> first_in")
    pos = str(hero_position or "").upper()
    hc = str(head["hand_class"])

    if first_in:
        # BB 首入：无声明开局表（walk/limp 单帧不可分）→ 诚实 miss
        if pos == "BB":
            head["reason"] = "advice_scenario_not_declared:bb_first_in"
            return head
        if n_seated == 2:
            # HU：SB/BTN 首入 → 既有 chart 场景 sb_open
            return _chart_scenario(head, basis, warnings,
                                   scenario="sb_open", hole_cards=hole_cards)
        # 6-max：ranges.yaml open.<position> 成员资格
        if pos not in _OPEN_RANGE_POSITIONS:
            head["reason"] = (
                f"advice_missing_input:hero_position({hero_position!r})")
            return head
        ranges = _load_ranges()
        open_rng = str((ranges.get("open") or {}).get(pos) or "")
        if not open_rng:
            head["reason"] = f"advice_scenario_not_declared:open.{pos}"
            return head
        inside = _range_contains(open_rng, hc)
        if inside is None:
            head["reason"] = f"advice_range_token_unparsable:open.{pos}"
            return head
        basis["range_ref"] = f"open.{pos}"
        basis["in_range"] = inside
        if inside:
            head.update({
                "available": True,
                "action": "raise",
                "size_bb": OPEN_RAISE_SIZE_BB,
                "confidence": 0.6,
                "source": "ranges_open_table",
                "scenario": f"open.{pos}",
            })
            warnings.append(
                f"场景=6-max 首入 open.{pos}（ranges.yaml 声明范围成员资格；"
                f"尺寸=chart sb_open 声明值 {OPEN_RAISE_SIZE_BB}bb）")
        else:
            head.update({
                "available": True,
                "action": "fold",
                "size_bb": None,
                "confidence": 0.6,
                "source": "ranges_open_table",
                "scenario": f"open.{pos}",
            })
            warnings.append(
                f"场景=6-max 首入 open.{pos}（不在声明开局范围 → 标准弃牌）")
        return head

    # facing 族（pot 高于首入带）
    if n_seated == 2 and pos == "BB":
        return _chart_scenario(head, basis, warnings,
                               scenario="bb_vs_sb_open",
                               hole_cards=hole_cards)
    head["reason"] = "advice_scenario_not_declared:facing_6max"
    warnings.append("6-max facing 场景无声明表（绝不套用 HU 解冒充）——"
                    "参考建议诚实 miss")
    return head


def _chart_scenario(head: dict, basis: dict, warnings: list[str], *,
                    scenario: str, hole_cards) -> dict:
    """HU 场景 → 既有 SOLVER-1 chart provider 查表（source=chart）。"""
    from .core.providers.preflop_chart import PreflopChartProvider
    provider = PreflopChartProvider()
    if not provider.is_configured():
        head["reason"] = "advice_chart_missing:preflop_chart_hu.yaml"
        return head
    cards = [str(c) for c in (hole_cards or [])]
    try:
        result = _run_chart_solve(provider, {
            "street": "preflop", "scenario": scenario, "hole_cards": cards})
    except Exception as e:                  # noqa: BLE001 - 查表失败=诚实 miss
        head["reason"] = f"advice_chart_miss:{scenario}:{str(e)[:80]}"
        return head
    raw = result.raw or {}
    chosen = str(raw.get("chosen") or "")
    if not chosen:
        head["reason"] = f"advice_chart_miss:{scenario}:no_chosen"
        return head
    size = None
    for a in result.actions or []:
        if str(a.name) == chosen and a.size_bb is not None:
            size = float(a.size_bb)
            break
    head.update({
        "available": True,
        "action": chosen,
        "size_bb": size,
        "confidence": 0.6,
        "source": "preflop_chart",
        "scenario": scenario,
    })
    basis["chart_hand_class"] = str(raw.get("hand_class") or "")
    warnings.append(f"场景=HU chart {scenario}（source=chart，非均衡解）")
    return head


# ── postflop：spot 库 first_in range-set 查表（首入假设显式标注）────────
def _postflop_advice(head: dict, basis: dict, warnings: list[str], *,
                     street: str, board, pot_bb: float) -> dict:
    board_cards = [str(c) for c in (board or [])] if isinstance(board, list) \
        else []
    if len(board_cards) != _STREET_NEED[street]:
        head["reason"] = (
            f"advice_missing_input:board_cardinality"
            f"({len(board_cards)}!={_STREET_NEED[street]})")
        return head
    basis["assumption_first_in"] = (
        "to_call UNKNOWN → 首入假设（若实际面对下注则本参考不适用）")
    warnings.append("postflop 参考按首入节点套用（to_call 未知——若实际面对"
                    "下注则不适用；spot 库为 HU 均衡解、多人桌面为近似）")
    try:
        from .core.providers.spot_table import SpotTable
        hit = SpotTable().lookup({"street": street, "board": board_cards,
                                  "pot_bb": pot_bb, "node": "first_in"})
    except Exception as e:                  # noqa: BLE001
        head["reason"] = f"advice_spot_miss:{str(e)[:80]}"
        return head
    if not hit:
        head["reason"] = "advice_spot_miss:no_matching_spot"
        return head
    from .domain.poker.strategy_anchor import normalize_action
    rows = []
    for r in (hit.get("actions") or []):
        if r.get("frequency") is None or float(r["frequency"]) <= 0.0:
            continue
        name = normalize_action(r.get("name"))
        if not name:
            continue                      # 未知动作名（诚实丢弃）
        rows.append({**r, "name": name})
    if not rows:
        head["reason"] = "advice_spot_miss:empty_support"
        return head
    best = max(rows, key=lambda r: (float(r["frequency"]),
                                    -_ACTION_ORDER.index(str(r["name"]))))
    size = best.get("size_bb")
    raw = hit.get("raw") or {}
    head.update({
        "available": True,
        "action": str(best["name"]),
        "size_bb": (float(size) if size is not None else None),
        "confidence": (0.9 if hit.get("is_approximate") is False else 0.75),
        "source": "solver_spot_table_first_in",
        "scenario": str(raw.get("spot_id") or ""),
    })
    basis["spot_lookup_mode"] = str(raw.get("lookup_mode") or "")
    basis["spot_is_approximate"] = bool(hit.get("is_approximate"))
    return head


def advice_from_observation(obs: dict) -> dict:
    """bridge observation → 参考建议（frame_decision judge 门分支用）。

    只读已识别字段（UNKNOWN-first）：street/hole_cards/board/pot/blinds/ante
    /extra.v3f_hero_position/seats。to_call 必须 UNKNOWN（legal=None 前置已保证
    ——此处不重复判断）。
    """
    obs = obs or {}
    seats = [s for s in (obs.get("seats") or []) if isinstance(s, dict)]
    hero_stack = None
    hero_seat = obs.get("hero_seat")
    for s in seats:
        if hero_seat is not None and s.get("seat") == hero_seat:
            hero_stack = _num(s.get("stack"))
            break
    extra = obs.get("extra") or {}
    return reference_advice(
        street=obs.get("street"),
        hole_cards=obs.get("hole_cards"),
        board=obs.get("board"),
        pot=obs.get("pot"),
        blinds=obs.get("blinds"),
        ante=obs.get("ante"),
        hero_position=extra.get("v3f_hero_position"),
        n_seated=len(seats),
        hero_stack=hero_stack,
    )


__all__ = ["ADVICE_REFERENCE_VERSION", "ADVICE_TYPE_REFERENCE",
           "DECISION_STATUS_ADVICE_REFERENCE",
           "LEGAL_ACTIONS_STATUS_UNKNOWN", "ADVICE_WARNING_ZH",
           "OPEN_RAISE_SIZE_BB", "FIRST_IN_TOLERANCE_BB",
           "reference_advice", "advice_from_observation"]
