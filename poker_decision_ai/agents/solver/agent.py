"""Solver Agent (GTO 基线): normalized solver query -> solver provider.

Design rule (spec §18): an LLM must NEVER pretend to be a solver.  The GTO
baseline comes from `ctx.providers.get_solver()` only; when no real solver is
configured (or it raises ProviderNotConfiguredError / ProviderError) we degrade
to `MockSolverProvider` and flag `is_approximate=True` with a warning.

confidence = 0.35 for approximate (mock) baselines, 0.9 for a real solver.
"""
from __future__ import annotations

import asyncio
import os

import random as _random
from dataclasses import dataclass, field
from typing import Any, ClassVar

from poker_decision_ai.core.errors import ProviderError, ProviderNotConfiguredError
from poker_decision_ai.core.orchestration.base_agent import BaseAgent
from poker_decision_ai.core.orchestration.context import AgentContext
from poker_decision_ai.core.providers.solver import (
    MockSolverProvider,
    SolverAction,
    SolverResult,
)


class _SafeMockSolverProvider(MockSolverProvider):
    """Local workaround for a core bug in MockSolverProvider.solve().

    The core implementation builds its RNG seed as
        hash(str(sorted((query.get("board") or [])) + str(query.get("hole_cards"))))
    i.e. `sorted(board) + str(hole_cards)` -> list + str -> TypeError for any
    non-empty board/hole_cards.  This subclass fixes only that expression to
        sorted((board or []) + (hole_cards or []))
    and is otherwise identical to the core mock (reported as a core bug; the
    core/ module is left untouched per the agent contract).
    """

    async def solve(self, query: dict) -> SolverResult:
        seed_key = hash(str(sorted((query.get("board") or []) + (query.get("hole_cards") or []))))
        rng = _random.Random(self.seed ^ seed_key % (2 ** 32))
        equity = float(query.get("hero_equity", 0.5))
        pot = float(query.get("pot", 1.0))
        eff = float(query.get("effective_stack", 20.0))
        spr = eff / max(pot, 1e-9)
        street = query.get("street", "flop")

        actions: list[SolverAction] = []
        if street == "preflop":
            if query.get("to_call", 0) > 0:
                actions = [SolverAction("call", None, 0.55 + 0.25 * equity),
                           SolverAction("raise", 3.0, 0.35),
                           SolverAction("fold", None, max(0.0, 0.45 - 0.25 * equity))]
            else:
                actions = [SolverAction("raise", 2.5, 0.85), SolverAction("call", None, 0.15)]
        else:
            if query.get("to_call", 0) > 0:
                actions = [
                    SolverAction("call", None, round(min(0.8, max(0.1, equity - 0.1)), 2)),
                    SolverAction("raise", round(0.75 * pot, 1), 0.15),
                    SolverAction("fold", None, round(max(0.0, 0.9 - equity), 2)),
                ]
            else:
                if equity > 0.66 and spr > 3:
                    actions = [SolverAction("bet", round(0.66 * pot, 1), 0.6),
                               SolverAction("bet", round(1.2 * pot, 1), 0.25),
                               SolverAction("check", None, 0.15)]
                elif equity > 0.45:
                    actions = [SolverAction("bet", round(0.33 * pot, 1), 0.35),
                               SolverAction("check", None, 0.65)]
                else:
                    actions = [SolverAction("check", None, 0.9),
                               SolverAction("bet", round(0.33 * pot, 1), 0.1)]
        total = sum(a.frequency for a in actions) or 1.0
        for a in actions:
            a.frequency = round(a.frequency / total, 3)
        return SolverResult(
            solver_name=self.name, model=f"mock-heuristic(seed={self.seed})",
            actions=actions, bet_sizings_bb=[0.33, 0.66, 0.75, 1.2],
            notes="Mock 启发式基线（非真实 GTO 求解，仅作弱先验）", is_approximate=True,
            raw={"equity": equity, "spr": round(spr, 2)},
        )


@dataclass
class SolverOutput:
    solver_name: str = "mock"
    model: str = ""
    is_approximate: bool = True
    solver_available: bool = False            # V2：真实 Solver 可用=可用；mock=不可用（诚实标记）
    cache_hit: bool = False                   # V2：本结果来自 SolverCache
    actions: list = field(default_factory=list)       # [{name, size_bb, frequency, ev}]
    gto_action: str = "check"
    gto_frequency: float = 0.0
    bet_sizings_bb: list = field(default_factory=list)
    notes: str = ""
    confidence: float = 0.35
    evidence: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    # SOLVER-2：输出来源（judge 锚定路径据此分派 provenance）
    #   postflop_spot（spot 库查表）/ postflop_live（DEEP 现场求解）/
    #   preflop_chart（图表基线）/ mock（降级/未配置）
    source: str = "mock"
    source_detail: dict = field(default_factory=dict)


def _find_hero(players: list) -> dict | None:
    for p in players:
        if isinstance(p, dict) and p.get("is_hero"):
            return p
    return None


def _effective_stack(state: dict) -> float:
    players = state.get("players", []) or []
    stacks: list[float] = []
    for p in players:
        if not isinstance(p, dict):
            continue
        if p.get("is_hero") or p.get("status") in ("active", "allin"):
            stacks.append(float(p.get("stack", 0.0) or 0.0))
    return min(stacks) if stacks else 0.0


def _range_str(dist: Any, max_combos: int = 80) -> str:
    """分布 {cat: weight} → postflop 范围串（按权重取前 N，忽略具体权重）。

    诚实说明：只表达"哪些组合在范围内"，不表达权重（CLI 亦按等权处理）。
    """
    if not isinstance(dist, dict) or not dist:
        return ""
    items = [(str(k), float(v)) for k, v in dist.items()
             if isinstance(v, (int, float)) and float(v) > 0]
    items.sort(key=lambda kv: -kv[1])
    cats = [k for k, _ in items[:max_combos]]
    return ",".join(cats)


_RANK_ORDER = "AKQJT98765432"


def _hand_class(cards: list) -> str | None:
    """[card, card] → 手牌类别（如 'AQs' / 'AKo' / 'QQ'）。"""
    if not isinstance(cards, (list, tuple)) or len(cards) != 2:
        return None
    c = [str(x) for x in cards]
    if any(len(x) < 2 for x in c):
        return None
    r1, s1, r2, s2 = c[0][0], c[0][1].lower(), c[1][0], c[1][1].lower()
    if r1 not in _RANK_ORDER or r2 not in _RANK_ORDER:
        return None
    if r1 == r2:
        return f"{r1}{r2}"
    hi, lo = (r1, r2) if _RANK_ORDER.index(r1) < _RANK_ORDER.index(r2) else (r2, r1)
    return f"{hi}{lo}{'s' if s1 == s2 else 'o'}"


def _per_hand_pick(actions: list, hole_cards: list,
                    hands: list) -> tuple[str, float, float | None] | None:
    """按**英雄手牌类别**在逐手 EV 数组上的均值选动作（返回 name, freq, size）。"""
    hero_cls = _hand_class(hole_cards)
    if not hero_cls or not isinstance(hands, list) or not hands:
        return None
    # ① 精确组合优先（手牌字符串规范化后比较：牌序无关）
    def _norm(cards: list) -> str:
        return "".join(sorted(str(c) for c in cards))

    hero_key = _norm(hole_cards)
    exact = [i for i, h in enumerate(hands)
             if _norm([str(h)[:2], str(h)[2:4]]) == hero_key]
    if exact:
        idx = exact
    else:
        idx = [i for i, h in enumerate(hands)
               if _hand_class([str(h)[:2], str(h)[2:4]]) == hero_cls]
    if not idx:
        return None
    best = None
    for a in actions or []:
        if not isinstance(a, dict):
            continue
        arr = a.get("ev_per_hand")
        if not isinstance(arr, list):
            continue
        vals = [float(arr[i]) for i in idx
                if i < len(arr) and isinstance(arr[i], (int, float))]
        if not vals:
            continue
        mean_ev = sum(vals) / len(vals)
        if best is None or mean_ev > best[0]:
            sz = a.get("size_bb")
            best = (mean_ev, str(a.get("name") or "check"),
                    float(a.get("frequency") or 0.0),
                    float(sz) if isinstance(sz, (int, float)) else None)
    if best is None:
        return None
    return best[1], best[2], best[3]


def _sa_fields() -> tuple[str, ...]:
    """SolverAction 的合法字段名（用于过滤多余键，避免 TypeError 致 agent 降级）。"""
    from poker_decision_ai.core.providers.solver import SolverAction
    import dataclasses as _dc
    return tuple(f.name for f in _dc.fields(SolverAction))


def _mk_action(a):
    """dict → SolverAction（过滤多余键；逐手数组等附加信息由调用方另行处理）。"""
    from poker_decision_ai.core.providers.solver import SolverAction
    if not isinstance(a, dict):
        return a
    ok = set(_sa_fields())
    return SolverAction(**{k: v for k, v in a.items() if k in ok})


def _pick_gto(actions: list, use_ev: bool = True,
              hole_cards: list | None = None,
              hands: list | None = None,
              per_hand: bool = True) -> tuple[str, float, bool, float | None, str]:
    """P30-9-CT：选 gto_action——**有逐动作 EV 时按 EV 排序**，否则按频率。

    依据：已解节点的逐动作 EV（bb）是与均衡一致的排序；频率 argmax 在强混合
    或近似无差别处是弱信号（实测面对下注节点：求解器 EV 为 Call > Raise，
    而频率 argmax 会取到别的动作）。
    纪律：judge 侧权重不变（0.2/0.35 × 0.5，次优 ×0.2）；
    标注 `ev_ranked`；`SOLVER_EV_RANKED=0` 可完全回退。
    """
    import os as _os
    rows = [a for a in (actions or []) if isinstance(a, dict) and a.get("name")]
    if not rows:
        return "check", 0.0, False, None, "frequency"
    # P30-9-DC：逐手 EV（与 r1 基准同口径）优先；不可得 ⇒ 退回聚合
    if per_hand and hole_cards and hands and str(
            _os.environ.get("SOLVER_PER_HAND_EV", "1")).lower() \
            not in ("0", "off", "false", "no"):
        ph = _per_hand_pick(rows, hole_cards, hands)
        if ph is not None:
            return ph[0], ph[1], True, ph[2], "per_hand"
    use_ev = bool(use_ev) and str(
        _os.environ.get("SOLVER_EV_RANKED", "1")).lower() \
        not in ("0", "off", "false", "no")
    evs: list[float | None] = []
    for a in rows:
        try:
            evs.append(float(a.get("ev")) if a.get("ev") is not None else None)
        except (TypeError, ValueError):
            evs.append(None)
    if use_ev and any(v is not None for v in evs):
        i = max(range(len(rows)),
                key=lambda k: ((evs[k] if evs[k] is not None else -1e18),
                                float(rows[k].get("frequency") or 0.0)))
        b = rows[i]
        _sz = b.get("size_bb")
        return (str(b.get("name") or "check"),
                float(b.get("frequency") or 0.0), True,
                float(_sz) if isinstance(_sz, (int, float)) else None,
                "aggregate_ev")
    b = max(rows, key=lambda a: float(a.get("frequency", 0.0) or 0.0))
    _sz = b.get("size_bb")
    return (str(b.get("name") or "check"),
            float(b.get("frequency") or 0.0), False,
            float(_sz) if isinstance(_sz, (int, float)) else None,
            "frequency")


def _norm_action_rows(actions: list) -> list[dict]:
    """SOLVER-2：动作名归一（CLI Debug 名 Check/Bet/AllIn → 小写规范名）。

    SOLVER-1 缺陷修复：spot 记录的动作名是大写 Debug 形态，judge 的
    ACTIONS 表是小写——旧加权路径从未真正消费 solver 动作（只有 sizings
    生效）。归一后 solver 动作可被 judge/锚定路径确定性消费。
    """
    from poker_decision_ai.domain.poker.strategy_anchor import normalize_action
    rows: list[dict] = []
    for a in actions or []:
        d = a.to_dict() if hasattr(a, "to_dict") else dict(a)
        name = normalize_action(d.get("name"))
        if name:
            d["name"] = name
        rows.append(d)
    return rows


#: preflop chart provider 进程内缓存（毫秒级查表；按 chart 路径去重）
_CHART_PROVIDERS: dict[str, Any] = {}
_CHART_LOCK = __import__("threading").Lock()


def _get_chart_provider(chart_path: str):
    from poker_decision_ai.core.providers.preflop_chart import PreflopChartProvider
    with _CHART_LOCK:
        if chart_path not in _CHART_PROVIDERS:
            _CHART_PROVIDERS[chart_path] = PreflopChartProvider(chart_path)
        return _CHART_PROVIDERS[chart_path]


def _hero_facing_role(state: dict, pot_class: str | None) -> tuple[str, str] | None:
    """M4：从翻前动作推导 hero 在 HU spot 套件中的 (pot 族, 角色)。

    套件约定（scripts/solve_spots_fb.py 范围声明）：
      SRP（1 次翻前加注）：oop=翻前跟注方、ip=加注方；
      3BP（2 次翻前加注）：oop=3bet 方、ip=跟注方；
      LIMP（0 次加注，平跟池——vs 被动池的典型局面）：oop=BB（翻后先行动）、
        ip=SB（HU 平跟后翻后 BB 先行动；与套件 oop=BB 声明一致）。
    规则只读翻前 raise 计数与 hero 参与 + button 归属；推导不出 → None
    （诚实 miss，绝不猜测）。返回 (family, role) 或 None。
    """
    if pot_class not in ("SRP", "3BP", "LIMP"):
        # P30-9-CU：realtime 调用方传入的是**尺寸档**（_pot_class: small/mid/big），
        # 不是底池**家族** ⇒ 此前恒返回 None ⇒ facing 库从未被查询、求解器通道在
        # 面对下注节点完全失效。此处改为**从翻前线路推导家族**（本函数本职）。
        pot_class = None
    history = state.get("action_history") or []
    raises = []
    for row in history:
        if not isinstance(row, dict):
            continue
        if str(row.get("street") or "") != "preflop":
            continue
        if str(row.get("action") or "") in ("raise", "all-in"):
            raises.append(str(row.get("player_id") or ""))
    hero = next((p for p in (state.get("players") or [])
                 if isinstance(p, dict) and p.get("is_hero")), None)
    if hero is None:
        return None
    pid = str(hero.get("player_id") or "")
    if len(raises) == 1:
        return ("SRP", "ip" if raises[0] == pid else "oop")
    if len(raises) == 2:
        return ("3BP", "oop" if raises[1] == pid else "ip")
    if len(raises) == 0:
        # 平跟池：HU 翻后 BB 先行动 = 套件 oop；SB = ip（按钮归属判定）。
        hero_seat = hero.get("seat")
        button_seat = state.get("button_seat")
        if hero_seat is None or button_seat is None:
            return None
        is_sb = int(hero_seat) == int(button_seat)
        return ("LIMP", "ip" if is_sb else "oop")
    return None


_POS_TO_SRC = {"UTG": "LJ", "MP": "HJ", "CO": "CO", "BTN": "BTN",
               "SB": "SB", "BB": "BB"}


def _size_tag(preflop_ctx: dict) -> str:
    """P30-9-DB：由**观测到的加注者投入**推开局尺寸标签（bb）。

    ≤2.2 ⇒ "@2"；≤2.75 ⇒ "@2.5"；其余 ⇒ "@3"。无观测 ⇒ ""（用默认键）。
    """
    if not isinstance(preflop_ctx, dict):
        return ""
    mx = preflop_ctx.get("max_commitment_bb")
    if not isinstance(mx, (int, float)) or mx <= 0 or mx < 1.5:
        return ""
    if mx <= 2.2:
        return "@2"
    if mx <= 2.75:
        return "@2.5"
    return "@3"


def _preflop_scenario_6max(state: dict, to_call: float) -> tuple[str | None, str]:
    """6-max 翻前状态 → (场景键, 来源)。

    优先用 `state.action_history` 的翻前加注层级（准确）；
    历史缺失时用 pot 尺寸启发式兜底，并把来源标为 `pot_heuristic`（**不冒充**）。
    场景键与外部 GTO 图表一致：`open:<pos>` / `vs_raise:<hero>:<villain>` /
    `vs_3bet:...` / `vs_4bet:...` / `vs_5bet:...`（来源 README 的组合规则）。
    """
    players = state.get("players") or []
    hero = _find_hero(players)
    if not hero:
        return None, "no_hero"
    hero_src = _POS_TO_SRC.get(str(hero.get("position") or "").upper())
    if not hero_src:
        return None, "hero_position_unknown"
    # P30-9-CD：优先使用 state 携带的翻前节点上下文（来自投入几何/真实历史）
    ctx = state.get("preflop_context") or {}
    if isinstance(ctx, dict) and ctx:
        level = str(ctx.get("level") or "")
        agg = _POS_TO_SRC.get(str(ctx.get("aggressor_position") or "").upper())
        src = str(ctx.get("source") or "preflop_context")
        if level == "open" and hero_src != "BB":
            return f"open:{hero_src}", src
        if level in ("vs_raise", "vs_3bet", "vs_4bet") and agg and agg != hero_src:
            return f"{level}:{hero_src}:{agg}", src
        if level in ("vs_raise", "vs_3bet", "vs_4bet") and not agg:
            return None, "aggressor_position_unknown"
    hist = [r for r in (state.get("action_history") or [])
            if isinstance(r, dict) and str(r.get("street")) == "preflop"
            and str(r.get("action")) in ("raise", "bet", "all-in")]
    if hist:
        n = len(hist)
        last = next((r for r in reversed(hist)
                      if str(r.get("player_id")) != str(hero.get("player_id"))), None)
        if n == 0 or last is None:
            return f"open:{hero_src}", "history"
        vpid = str(last.get("player_id"))
        vpos = next((p.get("position") for p in players
                      if str(p.get("player_id")) == vpid), None)
        v_src = _POS_TO_SRC.get(str(vpos or "").upper())
        if not v_src:
            return None, "villain_position_unknown"
        node = {1: "vs_raise", 2: "vs_3bet", 3: "vs_4bet"}.get(n, "vs_5bet")
        return f"{node}:{hero_src}:{v_src}", "history"
    # 兜底：pot 尺寸启发式（2.5bb≈首入；7–13bb≈面对 3bet；>13bb≈面对 4bet）
    # 注意：**BB 不可能"首入"**（大盲无开局动作）⇒ 不为 BB 生成 open 场景。
    bb = float(state.get("bb") or 1.0)
    pot_bb = float(state.get("pot") or 0.0) / bb
    to_call_bb = float(to_call or 0.0) / bb
    if to_call_bb <= 1e-9:
        if hero_src == "BB":
            return None, "bb_no_open_chart"      # 诚实 miss（图表无 BB 首入）
        return f"open:{hero_src}", "pot_heuristic"
    # 面对下注：需要对手位置才能选场景；缺历史 ⇒ 不套用（返回未知节点）
    if pot_bb <= 5.0:
        return f"vs_raise:{hero_src}:?", "pot_heuristic"
    if pot_bb <= 14.0:
        return f"vs_3bet:{hero_src}:?", "pot_heuristic"
    return f"vs_4bet:{hero_src}:?", "pot_heuristic"


def _preflop_scenario(state: dict, to_call: float) -> str | None:
    """HU preflop 状态 → 图表场景键（图表未声明的节点 → None 诚实 miss）。"""
    hero = _find_hero(state.get("players") or [])
    pos = str((hero or {}).get("position") or "")
    is_sb = ("SB" in pos.upper()) or ("BTN" in pos.upper())
    if to_call <= 1e-9:
        return "sb_open" if is_sb else None
    return "bb_vs_sb_open" if (not is_sb and "BB" in pos.upper()) else None


class SolverAgent(BaseAgent):
    name: ClassVar[str] = "solver"
    version: ClassVar[str] = "1.0.0"
    zh_name: ClassVar[str] = "GTO 基线 Agent"
    layer: ClassVar[str] = "analysis"
    depends: ClassVar[list[str]] = ["state", "equity"]
    Input: ClassVar[type] = dict
    Output: ClassVar[type] = SolverOutput

    async def applicable(self, raw_input: dict, ctx: AgentContext) -> tuple[bool, str]:
        if not raw_input.get("state"):
            return False, "state input is required for solver baseline"
        return True, ""

    async def handle(self, inp: dict, ctx: AgentContext) -> SolverOutput:
        state = inp.get("state") or {}
        equity_in = inp.get("equity") or {}
        warnings: list[dict] = []

        hero_p = _find_hero(state.get("players", []) or [])
        hole_cards = [str(c) for c in (hero_p.get("hole_cards") or [])] if hero_p else []
        # P34：胜率来源诚实标记——缺胜率时不再静默以 0.5 伪造（mock 先验
        # 保留既有语义；realengine 输入门见 hero_equity_source != computed
        # 会拒绝 → 诚实降级 BASELINE，绝不基于伪造 equity 输出 REAL）。
        hero_equity_raw = equity_in.get("hero_equity")
        if isinstance(hero_equity_raw, bool) or not isinstance(
                hero_equity_raw, (int, float)):
            hero_equity = 0.5                       # mock 先验（既有语义）
            eq_source = "missing"
        else:
            hero_equity = hero_equity_raw
            eq_source = "computed"

        # ---- normalized solver query (documented in providers/solver.py) ----
        # P34：补 bb（chips→bb 换算用；mock 忽略，realengine 需要）
        query = {
            "board": [str(c) for c in state.get("board", []) or []],
            "hole_cards": hole_cards,
            "pot": float(state.get("pot") or 0.0),
            "effective_stack": _effective_stack(state),
            "street": str(state.get("street", "flop")),
            "to_call": float(state.get("to_call") or 0.0),
            "hero_equity": float(hero_equity),
            "hero_equity_source": eq_source,
            "bet_sizings": [50, 75],
            "game": "cash" if state.get("game_type", "cash") != "mtt" else "tournament",
            "bb": float(state.get("bb") or 0.0),
        }
        # P30-9-BJ：求解预算（迭代数/子进程超时）可配置；默认沿用现状（1000 / 25s）。
        # 注意：config.root 是 NestedDict ⇒ 必须用点号 key（dict() 会抛错）。
        try:
            _mi = ctx.config.root.get("solver.max_iterations", None)
            _ts = ctx.config.root.get("solver.timeout_s", None)
            if _mi is not None:
                query["max_iterations"] = int(_mi)
            if _ts is not None:
                query["timeout_s"] = float(_ts)
        except Exception:                                      # noqa: BLE001
            pass

        provider = ctx.providers.get_solver()
        if provider is None:
            provider = _SafeMockSolverProvider(seed=42)
            warnings.append({"code": "NO_SOLVER_PROVIDER",
                             "message": "no solver provider configured; using mock baseline"})

        # ---- SOLVER-2：preflop 图表基线（realtime；HU only）---------
        # postflop solver 不覆盖 preflop（SOLVER-1 契约诚实拒绝）；preflop
        # 走 configs/solver/preflop_chart_hu.yaml 查表（source=chart，
        # 毫秒级）。命中 → 真实图表基线（solver_name=preflop_chart）；
        # 未声明场景 → 诚实降级 mock（绝不编造）。
        is_realtime = (ctx.extra or {}).get("realtime") is not None
        n_active = len([pl for pl in (state.get("players") or [])
                        if isinstance(pl, dict)
                        and pl.get("status") in ("active", "allin")])
        spot_pot_class = None
        if getattr(provider, "name", "") == "postflop" and n_active >= 2:
            from poker_decision_ai.core.providers.spot_table import _pot_class
            spot_pot_class = _pot_class(
                float(state.get("pot") or 0.0)
                / float(state.get("bb") or 1.0))
        to_call_chips = float(state.get("to_call") or 0.0)
        if is_realtime and str(state.get("street") or "") == "preflop" \
                and n_active >= 2:
            scenario = _preflop_scenario(state, to_call_chips)
            node_source = "hu"
            chart_path = "configs/solver/preflop_chart_hu.yaml"
            try:
                chart_path = str(ctx.config.solver_config().get(
                    "preflop_chart") or chart_path)
            except AttributeError:
                pass
            # P30-9-CB：若配置的是频率图表（外部 GTO），改用 6-max 场景解析
            try:
                _probe = _get_chart_provider(chart_path)
                if _probe is not None and _probe.chart_mode() == "frequency":
                    _sc6, node_source = _preflop_scenario_6max(state, to_call_chips)
                    if _sc6 and not _sc6.endswith(":?"):
                        scenario = _sc6
                        # P30-9-DB：面对加注节点按观测开局尺寸选标签
                        if not _sc6.startswith("open:"):
                            _tag = _size_tag(state.get("preflop_context") or {})
                            if _tag:
                                scenario = f"{_sc6}{_tag}"
                    elif _sc6:
                        warnings.append({
                            "code": "PREFLOP_NODE_HEURISTIC",
                            "message": (f"翻前历史缺失 ⇒ 用 pot 尺寸启发式得到 "
                                         f"{_sc6}（对手位置未知，不套用）")})
                        scenario = None
            except Exception:                          # noqa: BLE001
                pass
            provider_chart = None
            if scenario:
                try:
                    provider_chart = _get_chart_provider(chart_path)
                except Exception as e:                 # noqa: BLE001
                    warnings.append({"code": "SOLVER_CHART_PROVIDER_ERROR",
                                     "message": f"chart provider error: {e}"})
            if provider_chart is not None and provider_chart.is_configured():
                chart_query = dict(query)
                chart_query["scenario"] = scenario
                chart_query["hero_position"] = str(
                    (_find_hero(state.get("players") or []) or {}).get("position") or "")
                chart_query["node_source"] = node_source
                result = None
                try:
                    result = await provider_chart.solve(chart_query)
                except ProviderError as e:
                    warnings.append({"code": "SOLVER_CHART_MISS",
                                     "message": (f"preflop chart miss "
                                                 f"({scenario}): {e}")})
                except Exception as e:                 # noqa: BLE001
                    warnings.append({"code": "SOLVER_CHART_ERROR",
                                     "message": f"chart lookup failed: {e}"})
                if result is not None:
                    actions = _norm_action_rows(result.actions)
                    # P30-9-CC：区分"外部 GTO 频率图表"与"HU 成员资格基线"
                    _raw = dict(result.raw or {})
                    _is_gto = (str(_raw.get("source") or "") == "external_gto_json"
                                and not bool(result.is_approximate))
                    out = SolverOutput(
                        # 锚定契约要求 solver_name 恰为 preflop_chart
                        solver_name="preflop_chart", model=result.model,
                        is_approximate=(not _is_gto),
                        solver_available=bool(_is_gto),
                        cache_hit=False, actions=actions,
                        bet_sizings_bb=list(result.bet_sizings_bb or []),
                        notes=result.notes, source="preflop_chart",
                        source_detail={"scenario": scenario,
                                       "node_source": node_source,
                                       "equilibrium": bool(_is_gto),
                                       "chart": _raw.get("chart"),
                                       "raw": _raw},
                    )
                    if actions:
                        best = max(actions, key=lambda a: float(
                            a.get("frequency", 0.0) or 0.0))
                        out.gto_action = str(best.get("name", "check"))
                        out.gto_frequency = float(
                            best.get("frequency", 0.0) or 0.0)
                    # 外部 GTO 图表 = 真 solver 输出 ⇒ 置信更高；HU 成员图表 = 基线
                    out.confidence = 0.75 if _is_gto else 0.6
                    out.evidence = [{
                        "kind": "solver",
                        "text": (f"preflop chart hit: scenario={scenario} "
                                 f"hand={out.gto_action}@{out.gto_frequency} "
                                 + ("(source=external_gto_json，solver 输出)"
                                    if _is_gto else "(source=chart，非均衡解)")),
                        "source": "preflop_chart", "weight": 1.0,
                    }]
                    out.warnings = warnings
                    return out
            elif not scenario:
                warnings.append({"code": "SOLVER_CHART_MISS",
                                 "message": ("no declared preflop chart "
                                             "scenario for this node; "
                                             "degraded to mock baseline")})
            provider = _SafeMockSolverProvider(seed=42)
            warnings.append({"code": "SOLVER_SPOT_MISS",
                             "message": ("preflop not covered by solved "
                                         "spots/chart; degraded to mock "
                                         "baseline")})
            result = await provider.solve(query)
            actions = _norm_action_rows(result.actions)
            out = SolverOutput(
                solver_name=result.solver_name, model=result.model,
                is_approximate=bool(result.is_approximate),
                solver_available=False, cache_hit=False, actions=actions,
                bet_sizings_bb=list(result.bet_sizings_bb or []),
                notes=result.notes,
            )
            if actions:
                best = max(actions, key=lambda a: float(
                    a.get("frequency", 0.0) or 0.0))
                out.gto_action = str(best.get("name", "check"))
                out.gto_frequency = float(best.get("frequency", 0.0) or 0.0)
            out.confidence = 0.35
            warnings.append({"code": "APPROXIMATE",
                             "message": "Mock 基线非真实 GTO（仅作弱先验）"})
            out.evidence = [{"kind": "solver",
                             "text": "solver=preflop chart-miss → mock fallback",
                             "source": "solver", "weight": 0.35}]
            out.warnings = warnings
            return out

        # ---- SOLVER-1：postflop 真求解器的路径分派 -----------------
        # realtime：只允许 spot 库查表（毫秒级；现场求解秒级起步会打死
        #   3s deadline）——命中即真实均衡策略；miss → 诚实降级 mock。
        # SOLVER-2 节点兼容门（M4 扩展）：首入（to_call=0）查 first-in 库；
        #   面对下注（to_call>0）查 M4 facing-bet 库（node=facing_bet），
        #   尺寸桶（bet_ratio）不匹配 → 诚实 miss，绝不跨节点/跨尺寸套用。
        # M4 多人规则（TARGET_SPEC「HU 解锚定 + 多人简化」）：
        #   * 首入 multiway → 允许套用 HU 解，显式 multiway_simplified 标注
        #     （简化级别由 anchor provenance 透传）；
        #   * facing-bet multiway → 诚实 miss（HU facing 节点不适用多人池）。
        # DEEP（非 realtime）：现场 provider.solve（分钟级可接受）。
        # 两条路径都绝不使用 LLM/启发式冒充 solver 输出（mock 显式
        # is_approximate=True，见既有降级路径）。
        # ---- P30-9-CG：6-max 位置键查表（外部 GTO 解；与 HU 并存）----
        if (getattr(provider, "name", "") == "postflop" and is_realtime
                and str(state.get("street") or "") in ("flop", "turn", "river")):
            try:
                from poker_decision_ai.core.providers.spot_table_6max import (
                    get_spot_table_6max)
                _pl = [pl for pl in (state.get("players") or [])
                       if isinstance(pl, dict)
                       and pl.get("status") in ("active", "allin")]
                _hero = next((pl for pl in _pl if pl.get("is_hero")), None)
                _vill = next((pl for pl in _pl if not pl.get("is_hero")), None)
                # P30-9-CW：6-max 外部库**只适用于 ≥3 人**局面——HU（2 人）时拦截
                # 会以"近似"命中抢先返回，使 HU 的 fb 精确库无法生效（范畴错误）。
                _allow_hu6 = str(__import__("os").environ.get(
                    "SPOT6_ALLOW_HU", "0")).lower() in ("1", "on", "true", "yes")
                # P30-9-DZ：门 = 牌桌规模（就座条目数）或 在池人数；HU 桌不介入
                _n_seated = len([p_ for p_ in (state.get("players") or [])
                                 if isinstance(p_, dict)])
                _min_seats = int(__import__("os").environ.get(
                    "SPOT6_TABLE_MIN_SEATS", "5"))
                # P30-9-EM：**多人守卫**——外部 6-max 库是"翻后单挑"解，
                # ≥3 人在池时无参考数据 ⇒ 诚实 miss（避免把单挑解套给多人局面）
                _allow_mw6 = str(__import__("os").environ.get(
                    "SPOT6_ALLOW_MULTIWAY", "0")).lower() in ("1", "on", "true", "yes")
                if len(_pl) >= 3 and not _allow_mw6:
                    warnings.append({"code": "SPOT6_MULTIWAY_SKIPPED",
                                     "message": ("多人底池（≥3 人在池）无 6-max 单挑解参考"
                                                 "⇒ 诚实 miss（不套用单挑策略）")})
                elif (_hero is not None and _vill is not None
                        and (len(_pl) >= 3 or _n_seated >= _min_seats or _allow_hu6)):
                    # P30-9-EA：修正静默失效——_PO（POS_ORDER 别名）此前不在作用域，
                    # 本分支每次抛错并被 except 吞掉 ⇒ 6-max 通道**从未生效**
                    from poker_decision_ai.core.providers.spot_table_6max import (
                        POS_ORDER as _PO)
                    hp, vp = (str(_hero.get("position") or "").upper(),
                              str(_vill.get("position") or "").upper())
                    if hp in _PO and vp in _PO:
                        _oop, _ip = (hp, vp) if _PO[hp] < _PO[vp] else (vp, hp)
                        _node = ("facing_bet" if to_call_chips > 1e-9 else "first_in")
                        _pot_bb = (float(state.get("pot") or 0.0)
                                   / float(state.get("bb") or 1.0))
                        # P30-9-EB：6-max 外部库的 `bet_ratio` 口径是 **to_call / pot**
                        # （HU 库才是 to_call/(pot−to_call)）⇒ 此处必须用**该库口径**，
                        # 否则尺寸桶永不匹配（实测导致 6-max 命中率与锚定率恒为 0）。
                        _ratio = None
                        if _node == "facing_bet":
                            _pot_now = float(state.get("pot") or 0.0)
                            _ratio = to_call_chips / max(_pot_now, 1e-6)
                        _stk_bb = ((_hero.get("stack") or 0.0)
                                    / float(state.get("bb") or 1.0))
                        # P30-9-FR：翻前家族（3bp/srp）——由翻前投入推导；推不出则不传
                        # （不约束 ⇒ 向后兼容）。缺失会**跨族撞键**：深层 srp 节点底池
                        # 涨入 big 档后与 3bp 节点同键（留出验证实测命中错族 ✗）。
                        _fam6 = None
                        try:
                            _pc6 = state.get("preflop_context") or {}
                            _mc6 = float(_pc6.get("max_commitment_bb") or 0.0)
                            if _mc6 <= 0.0:
                                # 回退：由**翻前历史**的最大投入推导（实测 preflop_context
                                # 常缺失 ⇒ 若不回退，查询不带 family ⇒ 跨族撞键 ✗）
                                _bb6 = float(state.get("bb") or 1.0) or 1.0
                                _amts = [float(h.get("amount") or 0.0)
                                          for h in (state.get("action_history") or [])
                                          if str(h.get("street") or "") == "preflop"
                                          and float(h.get("amount") or 0.0) > 0.0]
                                if _amts:
                                    _mc6 = max(_amts) / _bb6
                            if _mc6 >= 7.0:
                                _fam6 = "3bp"
                            elif _mc6 > 0.0:
                                _fam6 = "srp"
                        except Exception:                       # noqa: BLE001
                            _fam6 = None
                        _q6 = {
                            # P30-9-EK：行动者 = hero 位置
                            "actor": str(_hero.get("position") or ""),
                            "street": str(state.get("street") or "flop"),
                            "board": [str(c) for c in (state.get("board") or [])],
                            "pot_bb": _pot_bb, "node": _node,
                            "positions": {"oop": _oop, "ip": _ip},
                            "bet_ratio": _ratio,
                            "to_call_bb": (to_call_chips
                                            / float(state.get("bb") or 1.0)),
                            "stack_bb": _stk_bb}
                        if _fam6:
                            _q6["family"] = _fam6
                        _hit6 = get_spot_table_6max().lookup(_q6)
                        if _hit6 is not None:
                            _acts6 = _norm_action_rows(_hit6.get("actions") or [])
                            # P30-9-CL：**近似命中不得锚定**——anchor_from_solver 只查
                            # solver_name/街/节点，不查 is_approximate；故近似时用不满足
                            # 其契约的 solver_name 阻断锚定，仅作弱先验（可审计）。
                            _approx6 = bool(_hit6.get("is_approximate"))
                            # P30-9-GI：近似命中（两人池）⇒ **后台**解一次并写缓存，供后续精确命中。
                            #   实测 flop 现场解 ~7.5s（无法进实时）⇒ 解一次、复用；决策**不等待** ✓。
                            try:
                                from poker_decision_ai.core.providers import live_cache as _lc
                                if _approx6 and n_active == 2:
                                    _dr9 = _lc.resolve_ranges(
                                        str(state.get("street") or ""),
                                        float(_pot_bb), _fam6)
                                    if _dr9:
                                        _lc.schedule(
                                            provider=provider,
                                            family=(_fam6 or "3bp"),
                                            oop=_oop, ip=_ip,
                                            street=str(state.get("street") or ""),
                                            board=[str(c) for c in (state.get("board") or [])],
                                            pot_bb=float(_pot_bb),
                                            stack_bb=float(_stk_bb or 100.0),
                                            ranges={"oop": _dr9[0], "ip": _dr9[1]},
                                            timeout_s=float(__import__("os").environ.get("SOLVER_LIVE_CACHE_TIMEOUT_S", "3600")),
                                            max_iterations=300)
                                        # P30-9-HF：**facing 局面**按**实际下注比例**另解一次
                                        #   （CLI 采纳 bet_sizes ✓ ⇒ 可解出与实际尺寸一致的 facing 节点 ✓）
                                        _tc9 = float(state.get("to_call") or 0.0)
                                        _bb9 = float(state.get("bb") or 1.0) or 1.0
                                        if _tc9 > 1e-9:
                                            _lc.schedule_facing(
                                                family=(_fam6 or "3bp"), oop=_oop, ip=_ip,
                                                street=str(state.get("street") or ""),
                                                board=[str(c) for c in (state.get("board") or [])],
                                                pot_bb=float(_pot_bb),
                                                stack_bb=float(_stk_bb or 100.0),
                                                ranges={"oop": _dr9[0], "ip": _dr9[1]},
                                                bet_ratio=(_tc9 / max(float(_pot_bb) - _tc9, 1e-9)),
                                                to_call_bb=(_tc9 / _bb9))
                            except Exception:                      # noqa: BLE001
                                pass
                            # P30-9-FT：**桶内共识**命中（近似，但提供逐手频率）也透出，
                            # 供"逐手频率偏离"通道使用；来源在 source_detail 里标注 ✓。
                            _cn6 = int(((_hit6.get("raw") or {}).get("consensus_n") or 0))
                            # P30-9-EG：透出逐手频率（6-max 库无逐手 EV）
                            if ((not _approx6) or _cn6) and (_hit6.get("hands") or []):
                                from poker_decision_ai.domain.poker.strategy_anchor import (
                                    normalize_action as _na6)
                                _fph = dict(_hit6.get("freq_per_hand") or {})
                                _rows6 = []
                                for _a in (_hit6.get("actions") or []):
                                    if not isinstance(_a, dict):
                                        continue
                                    _nm = _na6(_a.get("name"))
                                    _arr = _fph.get(_nm) or _fph.get(str(_a.get("name")))
                                    _rows6.append({"action": _nm,
                                                    "size_bb": _a.get("size_bb"),
                                                    "ev": None,
                                                    "freq": ([float(x) if isinstance(x, (int, float))
                                                               else 0.0 for x in _arr]
                                                              if isinstance(_arr, list) else None)})
                                _hit6["_per_hand_rows"] = _rows6
                            # P30-9-HG：**turn/river 的 facing 局面 ⇒ 同步解出真解** ✓
                            #   实测 turn 69ms / river 18ms（实时预算 1200ms 内 ✓）；
                            #   此前现场真解只出街根节点（OOP 首行动 ✗）⇒ 面对下注拿不到真解 ✗。
                            #   仅在近似命中且两人池时尝试；失败/超时保持原先验（诚实 ✓）。
                            try:
                                from poker_decision_ai.core.providers import live_cache as _lcG
                                _tcG = float(state.get("to_call") or 0.0)
                                _stG = str(state.get("street") or "")
                                if (_approx6 and _tcG > 1e-9 and n_active == 2
                                        and _stG in ("turn", "river")):
                                    _drG = _lcG.resolve_ranges(_stG, float(_pot_bb), _fam6)
                                    if _drG:
                                        _fg = await _lcG.solve_facing_now(
                                            oop=_oop, ip=_ip, street=_stG,
                                            board=[str(c) for c in (state.get("board") or [])],
                                            pot_bb=float(_pot_bb),
                                            stack_bb=float(_stk_bb or 100.0),
                                            ranges={"oop": _drG[0], "ip": _drG[1]},
                                            bet_ratio=_tcG / max(float(_pot_bb) - _tcG, 1e-9),
                                            timeout_s=float(__import__("os").environ.get(
                                                "SOLVER_FACING_TIMEOUT_S", "1.5")))
                                        if _fg and _fg.get("actions"):
                                            from poker_decision_ai.core.providers.solver import (
                                                SolverAction as _SA, SolverOutput as _SO)
                                            _actsG = [_SA(str(a["name"]), a.get("size_bb"),
                                                           float(a.get("frequency") or 0)) 
                                                       for a in _fg["actions"]]
                                            _bG = max(_actsG, key=lambda a: float(
                                                getattr(a, "frequency", 0.0) or 0.0))
                                            out6 = _SO(
                                                solver_name="postflop", model="live_facing",
                                                is_approximate=False, solver_available=True,
                                                cache_hit=False, actions=_actsG,
                                                bet_sizings_bb=[],
                                                notes=("两人池 turn/river **facing 现场真解**"
                                                        "（按实际下注比例构造菜单；范围=声明范围）"),
                                                source="postflop_live_facing",
                                                source_detail={
                                                    "live_facing": True,
                                                    "ranges": "declared_ranges",
                                                    "range_assumption": True,
                                                    "bet_ratio": _fg.get("bet_ratio"),
                                                    "to_call_bb": _fg.get("to_call_bb"),
                                                    "size_pct": _fg.get("size_pct"),
                                                    "ms": _fg.get("ms"),
                                                    "library_prior": (_hit6.get("raw") or {}).get("spot_id")})
                                            out6.gto_action = str(getattr(_bG, "name", "check"))
                                            out6.gto_frequency = float(
                                                getattr(_bG, "frequency", 0.0) or 0.0)
                                            warnings.append({"code": "SOLVER_LIVE_FACING",
                                                             "message": ("面对下注局面已升级为**现场真解**"
                                                                           "（turn/river，按实际尺寸）")})
                                            out6.warnings = warnings
                                            return out6
                            except Exception:                      # noqa: BLE001
                                pass
                            # P30-9-GG：**近似命中（含桶内共识）时优先现场真解**——
                            #   实测 provider 端到端 turn 44ms / river 18ms；flop 7.5s ⇒ 仅 turn/river。
                            #   仅在两人池且拿得到声明范围时尝试；失败/超时保持原近似先验（诚实）。
                            _live_ok = False
                            if (_approx6
                                    and str(state.get("street") or "") in ("turn", "river")
                                    and n_active == 2
                                    and str(__import__("os").environ.get(
                                        "SOLVER_LIVE_TURN_RIVER", "1")).lower()
                                        not in ("0", "off", "false", "no")
                                    and getattr(provider, "name", "") == "postflop"):
                                try:
                                    from poker_decision_ai.core.providers import (
                                        live_cache as _lc6x)
                                    _drx = _lc6x.resolve_ranges(
                                        str(state.get("street") or ""),
                                        float(_pot_bb), _fam6)
                                    if _drx:
                                        _q6x = {"street": str(state.get("street") or ""),
                                                 "board": [str(c) for c in (state.get("board") or [])],
                                                 "ranges": {"oop": _drx[0], "ip": _drx[1]},
                                                 "pot": float(state.get("pot") or 0.0),
                                                 "effective_stack": float(_stk_bb * float(state.get("bb") or 1.0)),
                                                 "bb": float(state.get("bb") or 1.0),
                                                 "max_iterations": 300,
                                                 "target_exploitability_pot_pct": 1.0}
                                        _tmo6 = float(__import__("os").environ.get(
                                            "SOLVER_LIVE_TIMEOUT_MS", "1500")) / 1000.0
                                        _res6 = await asyncio.wait_for(
                                            provider.solve(_q6x), timeout=_tmo6)
                                        _acts6 = _norm_action_rows(_res6.actions)
                                        if _acts6:
                                            out6 = SolverOutput(
                                                solver_name="postflop",
                                                model=f"live:{_res6.model}",
                                                is_approximate=False,
                                                solver_available=True, cache_hit=False,
                                                actions=_acts6,
                                                bet_sizings_bb=list(_res6.bet_sizings_bb or []),
                                                notes=("两人池 turn/river **现场真解**"
                                                        "（范围=套件声明范围）"),
                                                source="postflop_live",
                                                source_detail={
                                                    "live": True,
                                                    "ranges": "declared_ranges",
                                                    "range_assumption": True,
                                                    "street": str(state.get("street") or ""),
                                                    "pot_bb": round(_pot_bb, 3),
                                                    "node": _node,
                                                    "library_prior": (_hit6.get("raw") or {}).get("spot_id")})
                                            try:
                                                _b6 = max(_acts6, key=lambda a: float(
                                                    a.get("frequency", 0.0) or 0.0))
                                                out6.gto_action = str(_b6.get("name", "check"))
                                                out6.gto_frequency = float(
                                                    _b6.get("frequency", 0.0) or 0.0)
                                            except Exception:            # noqa: BLE001
                                                pass
                                            warnings.append({"code": "SOLVER_LIVE_TURN_RIVER",
                                                             "message": ("近似命中已升级为**现场真解**"
                                                                           "（turn/river，两人池）")})
                                            _live_ok = True
                                except Exception as _e6:                # noqa: BLE001
                                    warnings.append({"code": "SOLVER_LIVE_FAILED",
                                                     "message": (f"现场求解失败（{type(_e6).__name__}）"
                                                                   "⇒ 保留库近似先验（诚实）")})
                            if _live_ok:
                                out6.warnings = warnings
                                return out6
                            out6 = SolverOutput(
                                solver_name=("postflop_approx" if _approx6
                                              else "postflop"),
                                model=str(_hit6.get("model") or ""),
                                is_approximate=_approx6,
                                solver_available=not _approx6,
                                cache_hit=False, actions=_acts6,
                                bet_sizings_bb=[], notes="6-max 外部 GTO 解（位置键查表）",
                                source=("postflop_6max_spot_approx" if _approx6
                                         else "postflop_6max_spot"),
                                source_detail={
                                    "node": _node,
                                    "spot_id": (_hit6.get("raw") or {}).get("spot_id"),
                                    "positions": {"oop": _oop, "ip": _ip},
                                    "pot_bb": round(_pot_bb, 3),
                                    "bet_ratio": (round(_ratio, 4)
                                                   if _ratio is not None else None),
                                    "pot_approximate": bool(
                                        _hit6.get("pot_approximate")),
                                    "size_approximate": bool(
                                        _hit6.get("size_approximate")),
                                    "anchored": not _approx6,
                                    # P30-9-EG：逐手频率结构（偏离层消费）
                                    "per_hand": ({"hands": [str(h_) for h_ in _hit6["hands"]],
                                                   "rows": _hit6["_per_hand_rows"]}
                                                  if _hit6.get("_per_hand_rows") else None),
                                    "source": "external_gto_json",
                                    "attribution": ((_hit6.get("raw") or {})
                                                     .get("source_note", "")[:160])})
                            if _acts6:
                                # P30-9-CT：有逐动作 EV 时按 EV 排序（否则频率）
                                # P30-9-ED：_pick_gto 现返回 5 值（P30-9-DI 起）
                                _ga6, _gf6, _evr6, _gsz6, _gkind6 = _pick_gto(
                                    _acts6, use_ev=(_node == "facing_bet"))
                                out6.gto_action, out6.gto_frequency = _ga6, _gf6
                                if _evr6:
                                    out6.source_detail = {
                                        **(out6.source_detail or {}),
                                        "ev_ranked": True}
                                if _gsz6 is not None:
                                    out6.bet_sizings_bb = [float(_gsz6)]
                                    out6.source_detail = {
                                        **(out6.source_detail or {}),
                                        "size_from_action": True}
                            out6.confidence = 0.8
                            out6.evidence = [{
                                "kind": "solver",
                                "text": (f"6-max GTO 解命中（spot="
                                          f"{(_hit6.get('raw') or {}).get('spot_id')}, "
                                          f"node={_node}, 位置={_oop}/{_ip}）"),
                                "source": "postflop_6max_spot", "weight": 1.0}]
                            out6.warnings = warnings
                            return out6
            except Exception as e:                             # noqa: BLE001
                warnings.append({"code": "SPOT6_LOOKUP_ERROR",
                                 "message": f"6-max table lookup failed: {e}"})

        if getattr(provider, "name", "") == "postflop" and is_realtime:
            from poker_decision_ai.core.providers.spot_table import get_spot_table
            node_type = ("facing_bet" if to_call_chips > 1e-9
                         else "first_in")
            multiway = n_active > 2
            node_supported = True
            if multiway and node_type == "facing_bet":
                node_supported = False
                warnings.append({"code": "MULTIWAY_FACING_BET_UNANCHORED",
                                 "message": ("multiway facing-bet node has no "
                                             "HU-applicable anchor; degraded to "
                                             "mock baseline (honest miss)")})
            elif multiway:
                warnings.append({"code": "SOLVER_SPOT_MULTIWAY_SIMPLIFIED",
                                 "message": ("HU first-in anchor applied to "
                                             "multiway node; simplification "
                                             "labeled in provenance")})
            hit = None
            if node_supported and spot_pot_class is not None:
                query["pot_bb"] = (float(state.get("pot") or 0.0)
                                   / float(state.get("bb") or 1.0))
                query["range_set"] = spot_pot_class
                query["node"] = node_type
                if node_type == "facing_bet":
                    pot_now = float(state.get("pot") or 0.0)
                    query["bet_ratio"] = (to_call_chips
                                          / max(pot_now - to_call_chips, 1e-6))
                    role_pair = (_hero_facing_role(state, spot_pot_class)
                                 if not multiway else None)
                    if role_pair is None:
                        # 角色推导不出（历史缺失/非常规线）→ 诚实 miss
                        node_supported = False
                        query["facing_actor"] = ""
                        query["pot_family"] = ""
                        warnings.append({"code": "FACING_ROLE_UNKNOWN",
                                         "message": ("cannot derive hero "
                                                     "suite role from "
                                                     "preflop history; "
                                                     "honest miss")})
                    else:
                        family, role = role_pair
                        query["facing_actor"] = role
                        query["pot_family"] = family
                # P30-9-ES：HU 单挑中按钮（SB）为 IP ⇒ 行动者 = ip/oop
                # （否则"IP 过牌后决策"会命中 OOP 的策略 = 范畴错误）
                try:
                    _hero_seat = (state.get("hero_seat")
                                   if isinstance(state.get("hero_seat"), int)
                                   else (next((p_.get("seat") for p_ in _pl
                                                if p_.get("is_hero")), None)))
                    _btn_seat = state.get("button_seat")
                    if _hero_seat is not None and isinstance(_btn_seat, int):
                        query["actor"] = ("ip" if int(_hero_seat) == int(_btn_seat)
                                           else "oop")
                except Exception:                       # noqa: BLE001
                    pass
                try:
                    hit = get_spot_table().lookup(query)
                except Exception as e:                     # noqa: BLE001
                    warnings.append({"code": "SOLVER_SPOT_TABLE_ERROR",
                                     "message": f"spot table lookup failed: {e}"})
            if hit is not None:
                if multiway and node_type == "first_in":
                    hit_raw = dict(hit.get("raw") or {})
                    hit_raw["multiway_simplified"] = True
                    hit_raw["n_active_players"] = n_active
                    hit["raw"] = hit_raw
                result = SolverResult(
                    solver_name=str(hit.get("solver_name") or "postflop"),
                    model=str(hit.get("model") or ""),
                    actions=[_mk_action(a)
                             for a in (hit.get("actions") or [])],
                    bet_sizings_bb=list(hit.get("bet_sizings_bb") or []),
                    notes=str(hit.get("notes") or ""),
                    is_approximate=bool(hit.get("is_approximate", False)),
                    raw=dict(hit.get("raw") or {}),
                )
                actions = _norm_action_rows(result.actions)
                solver_available = bool(
                    not result.is_approximate
                    and result.solver_name not in ("mock", ""))
                best = None
                _evr = False
                if actions:
                    # P30-9-CT：有逐动作 EV 时按 EV 排序（否则频率）
                    # P30-9-DD：逐手 EV 只在**精确命中**时可用（近似命中取的是
                    # 代表 spot 的逐手 EV——另一个牌面 ⇒ 可能更差）；聚合 EV 仍限 facing。
                    _exact_hit = not bool(hit.get("is_approximate"))
                    # P30-9-DJ：选择器要用**未过滤**的原始行（含 ev_per_hand/freq_per_hand），
                    # 因为 _mk_action 会按 dataclass 字段过滤掉这些数组（实测导致逐手路径从不触发）。
                    from poker_decision_ai.domain.poker.strategy_anchor import (
                        normalize_action as _na)
                    _pick_rows = [{**a, "name": _na(a.get("name"))}
                                   for a in (hit.get("actions") or [])
                                   if isinstance(a, dict)]
                    _ga, _gf, _evr, _gsz, _gkind = _pick_gto(
                        _pick_rows or actions, use_ev=(node_type == "facing_bet"),
                        hole_cards=list(hole_cards or []),
                        hands=list(hit.get("hands") or []),
                        per_hand=_exact_hit)
                    best = {"name": _ga, "frequency": _gf}
                raw = dict(result.raw or {})
                out = SolverOutput(
                    solver_name=result.solver_name,
                    model=result.model,
                    is_approximate=bool(result.is_approximate),
                    solver_available=solver_available,
                    cache_hit=False,
                    actions=actions,
                    bet_sizings_bb=list(result.bet_sizings_bb or []),
                    notes=result.notes,
                    source="postflop_spot",
                    source_detail={"lookup_mode": str(raw.get("lookup_mode")
                                                      or ""),
                                   "spot_id": str(raw.get("spot_id") or ""),
                                   "node": node_type,
                                   "bet_bucket": str(raw.get("bet_bucket")
                                                     or ""),
                                   "actor_role": str(raw.get("actor_role")
                                                     or ""),
                                   "family": str(raw.get("family") or "")},
                )
                if best is not None:
                    out.gto_action = str(best.get("name", "check"))
                    out.gto_frequency = float(
                        best.get("frequency", 0.0) or 0.0)
                    if _evr:
                        out.source_detail = {**(out.source_detail or {}),
                                              "ev_ranked": True,
                                              "ev_kind": _gkind}
                    # P30-9-DI/DK：架构边界——SolverOutput 无 raw 字段，冻结锚定只读
                    # raw.chosen（恒空）⇒ 锚定必取频率 argmax；逐手 EV 偏好改由
                    # **偏离层**（非冻结）消费。此处把逐手数据放进 source_detail 透传。
                    if _exact_hit and (hit.get("hands") or []):
                        _per_hand_rows = []
                        for _a in (hit.get("actions") or []):
                            if not isinstance(_a, dict):
                                continue
                            _ev = _a.get("ev_per_hand")
                            _fq = _a.get("freq_per_hand")
                            if not isinstance(_ev, list):
                                continue
                            _per_hand_rows.append({
                                "action": str(_na(_a.get("name"))),
                                "size_bb": _a.get("size_bb"),
                                "ev": [float(x) if isinstance(x, (int, float)) else None
                                        for x in _ev],
                                "freq": ([float(x) if isinstance(x, (int, float)) else 0.0
                                           for x in _fq] if isinstance(_fq, list) else None)})
                        if _per_hand_rows:
                            out.source_detail = {
                                **(out.source_detail or {}),
                                "per_hand": {"hands": [str(h) for h in (hit.get("hands") or [])],
                                              "rows": _per_hand_rows}}
                    if _gkind == "per_hand" and _ga:
                        out.source_detail = {**(out.source_detail or {}),
                                              "per_hand_pick": _ga,
                                              "per_hand_size": _gsz,
                                              "per_hand_bypassed_by_anchor": True}
                    # P30-9-DN：精确命中 + 逐手数据 ⇒ 用**该手牌最优行**的尺寸
                    # （r1 残余 regret 主要来自尺寸：逐手最优与频率 argmax 常同为 bet）
                    import os as _os3
                    if (_exact_hit and (hit.get("hands") or [])
                            and str(_os3.environ.get("SOLVER_SIZE_FROM_PER_HAND", "1")
                                     ).lower() not in ("0", "off", "false", "no")):
                        _ph_rows = []
                        for _a in (hit.get("actions") or []):
                            if not isinstance(_a, dict):
                                continue
                            _ev = _a.get("ev_per_hand")
                            if isinstance(_ev, list):
                                _ph_rows.append(_a)
                        if _ph_rows:
                            _ph_pick = _per_hand_pick(
                                _ph_rows, list(hole_cards or []),
                                list(hit.get("hands") or []))
                            if _ph_pick is not None and _ph_pick[2] is not None:
                                out.bet_sizings_bb = [float(_ph_pick[2])]
                                out.source_detail = {
                                    **(out.source_detail or {}),
                                    "size_from_per_hand": True,
                                    "per_hand_row": [_ph_pick[0], _ph_pick[2]]}
                    # P30-9-CZ：sizings 收敛为所选动作那一行的尺寸（否则 judge 会按
                    # "最接近 0.66×pot" 选尺寸，忽略逐手 EV ⇒ 行级 regret 上升）
                    import os as _os2
                    if (str(_os2.environ.get("SOLVER_SIZE_FROM_ACTION", "1")).lower()
                            not in ("0", "off", "false", "no")
                            and _gsz is not None):
                        out.bet_sizings_bb = [float(_gsz)]
                        out.source_detail = {**(out.source_detail or {}),
                                              "size_from_action": True}
                out.confidence = 0.9 if not result.is_approximate else 0.75
                if result.is_approximate:
                    warnings.append({"code": "SOLVER_SPOT_NEAREST",
                                     "message": "nearest spot hit; "
                                                "approximate application"})
                out.evidence = [{
                    "kind": "solver",
                    "text": (f"spot lookup hit: {result.raw.get('spot_id')} "
                             f"mode={result.raw.get('lookup_mode')} "
                             f"exploitability={result.raw.get('exploitability_bb')}bb"),
                    "source": "postflop_spot_lookup", "weight": 1.0,
                }]
                out.warnings = warnings
                return out
            # P30-9-GF：实时路径**对两人池 turn/river 放行现场求解**——
            #   实测 provider 端到端：turn 44ms / river 18ms（远低于实时预算），
            #   而 flop 约 7.5s ⇒ **不**放行 flop（仍走预解库/降级）。
            #   范围用套件声明范围（**范围假设如实标注**，不伪造）。
            _live_tr = (str(os.environ.get("SOLVER_LIVE_TURN_RIVER", "1")).lower()
                         not in ("0", "off", "false", "no"))
            _street_now = str(state.get("street") or "")
            _n_act_now = len([pl for pl in (state.get("players") or [])
                               if isinstance(pl, dict)
                               and pl.get("status") in ("active", "allin")])
            if (_live_tr and _street_now in ("turn", "river") and _n_act_now == 2
                    and getattr(provider, "name", "") == "postflop"):
                if "ranges" not in query:
                    try:
                        from poker_decision_ai.core.providers.spot_table import (
                            _pot_class, get_spot_table)
                        from poker_decision_ai.core.providers import live_cache as _lc9
                        _dr = _lc9.resolve_ranges(
                            _street_now,
                            float(state.get("pot") or 0.0) / float(state.get("bb") or 1.0))
                        if _dr:
                            query["ranges"] = {"oop": _dr[0], "ip": _dr[1]}
                            query["range_source"] = "declared_ranges"
                    except Exception:                          # noqa: BLE001
                        pass
                if "ranges" in query:
                    # P30-9-HG：**面对下注**局面 ⇒ 先试**同步 facing 真解**（turn 69ms / river 18ms ✓）
                    #   此前现场真解只出街根节点（OOP 首行动 ✗）⇒ facing 拿不到真解 ✗。
                    _tcG = float(state.get("to_call") or 0.0)
                    _fg = None
                    if _tcG > 1e-9:
                        try:
                            from poker_decision_ai.core.providers import live_cache as _lcF
                            _potb = float(state.get("pot") or 0.0) / float(state.get("bb") or 1.0)
                            _stkb = (float((state.get("effective_stack") or 0) or 0)
                                      / float(state.get("bb") or 1.0)) or 100.0
                            _fg = await _lcF.solve_facing_now(
                                oop=str(query.get("positions", {}).get("oop")
                                         or query.get("actor") or ""),
                                ip=str(query.get("positions", {}).get("ip") or ""),
                                street=_street_now,
                                board=[str(c) for c in (state.get("board") or [])],
                                pot_bb=_potb, stack_bb=_stkb,
                                ranges=query.get("ranges"),
                                bet_ratio=_tcG / max(_potb - _tcG, 1e-9),
                                timeout_s=float(os.environ.get("SOLVER_FACING_TIMEOUT_S", "1.5")))
                        except Exception:                      # noqa: BLE001
                            _fg = None
                    if _fg and _fg.get("actions"):
                        _actsF = [SolverAction(str(a["name"]), a.get("size_bb"),
                                                float(a.get("frequency") or 0))
                                   for a in _fg["actions"]]
                        _bestF = max(_actsF, key=lambda a: float(
                            getattr(a, "frequency", 0.0) or 0.0))
                        warnings.append({"code": "SOLVER_LIVE_FACING",
                                         "message": (f"两人池 {_street_now} **facing 现场真解**"
                                                      "（按实际下注比例；范围=声明范围）")})
                        out_live_f = SolverOutput(
                            solver_name="postflop", model="live_facing",
                            is_approximate=False, solver_available=True, cache_hit=False,
                            actions=_actsF, bet_sizings_bb=[],
                            notes="两人池 turn/river facing 现场真解（按实际下注比例构造菜单）",
                            source="postflop_live_facing",
                            source_detail={"live_facing": True,
                                            "ranges": "declared_ranges",
                                            "range_assumption": True,
                                            "bet_ratio": _fg.get("bet_ratio"),
                                            "to_call_bb": _fg.get("to_call_bb"),
                                            "size_pct": _fg.get("size_pct"),
                                            "ms": _fg.get("ms")})
                        out_live_f.gto_action = str(getattr(_bestF, "name", "check"))
                        out_live_f.gto_frequency = float(getattr(_bestF, "frequency", 0.0) or 0.0)
                        out_live_f.warnings = warnings
                        return out_live_f
                    try:
                        _tmo = float(os.environ.get("SOLVER_LIVE_TIMEOUT_MS", "1500")) / 1000.0
                        result = await asyncio.wait_for(provider.solve(query), timeout=_tmo)
                        warnings.append({"code": "SOLVER_LIVE_TURN_RIVER",
                                         "message": (f"两人池 {_street_now} **现场真解**"
                                                      "（范围=套件声明范围，假设已标注）")})
                    except Exception as e:                      # noqa: BLE001
                        provider = _SafeMockSolverProvider(seed=42)
                        warnings.append({"code": "SOLVER_LIVE_FAILED",
                                         "message": (f"现场求解失败（{type(e).__name__}）"
                                                      "⇒ 降级 mock 基线（诚实）")})
                        result = await provider.solve(query)
                else:
                    provider = _SafeMockSolverProvider(seed=42)
                    warnings.append({"code": "SOLVER_LIVE_NO_RANGES",
                                     "message": "无可用声明范围 ⇒ 不现场求解（不伪造范围）"})
                    result = await provider.solve(query)
            else:
                provider = _SafeMockSolverProvider(seed=42)
                warnings.append({"code": "SOLVER_SPOT_MISS",
                                 "message": ("no solved spot for this state in "
                                             "realtime path; degraded to mock "
                                             "baseline (live solving is DEEP-only)")})
                result = await provider.solve(query)
            actions = _norm_action_rows(result.actions)
            out = SolverOutput(
                solver_name=result.solver_name, model=result.model,
                is_approximate=bool(result.is_approximate),
                solver_available=False, cache_hit=False, actions=actions,
                bet_sizings_bb=list(result.bet_sizings_bb or []),
                notes=result.notes,
            )
            if actions:
                best = max(actions, key=lambda a: float(
                    a.get("frequency", 0.0) or 0.0))
                out.gto_action = str(best.get("name", "check"))
                out.gto_frequency = float(best.get("frequency", 0.0) or 0.0)
            out.confidence = 0.35
            warnings.append({"code": "APPROXIMATE",
                             "message": "Mock 基线非真实 GTO（仅作弱先验）"})
            out.evidence = [{"kind": "solver",
                             "text": "solver=postflop spot-miss → mock fallback",
                             "source": "solver", "weight": 0.35}]
            out.warnings = warnings
            return out

        # ---- V2 SolverCache：归一化局面命中则跳过求解 ----
        cache = (ctx.extra or {}).get("solver_cache")
        cache_hit = False
        if cache is not None and getattr(provider, "name", "") != "mock":
            cached = cache.get(query)
            if cached is not None:
                cache_hit = True
                result = SolverResult(
                    solver_name=str(cached.get("solver_name") or provider.name),
                    model=str(cached.get("model") or ""),
                    actions=[_mk_action(a)
                             for a in (cached.get("actions") or [])],
                    bet_sizings_bb=list(cached.get("bet_sizings_bb") or []),
                    notes=str(cached.get("notes") or ""),
                    is_approximate=bool(cached.get("is_approximate", True)),
                    raw=dict(cached.get("raw") or {}),
                )
            else:
                # P30-9-DY：**多人局面不做 HU 实时求解**——CLI 求解器是 HU-only，
                # 对 ≥3 人局面既超时（fast 路径 1200ms）又语义不符（范畴错误）。
                _n_act = len([pl for pl in (state.get("players") or [])
                               if isinstance(pl, dict)
                               and pl.get("status") in ("active", "allin")])
                _allow_mw = str(__import__("os").environ.get(
                    "SOLVER_ALLOW_MULTIWAY_LIVE", "0")).lower() in ("1", "on", "true", "yes")
                if _n_act >= 3 and not _allow_mw:
                    provider = _SafeMockSolverProvider(seed=42)
                    warnings.append({"code": "MULTIWAY_LIVE_SOLVE_SKIPPED",
                                     "message": ("多人局面不做 HU 实时求解（范畴不符）"
                                                 "⇒ 已标注弱先验（不超时）")})
                    result = await provider.solve(query)
                else:
                    try:
                        result = await provider.solve(query)
                        cache.put(query, result.to_dict())
                    except (ProviderNotConfiguredError, ProviderError) as e:
                        provider = _SafeMockSolverProvider(seed=42)
                        warnings.append({"code": "SOLVER_DEGRADED",
                                         "message": f"solver unavailable ({e}); degraded to mock baseline"})
                        result = await provider.solve(query)
        else:
            # ---- spot 库声明范围（原有逻辑，优先） ----
            # P30-9-CV：**facing_bet 节点不塞 ranges**——declared_ranges() 只按
            # **尺寸档**返回（与家族 SRP/3BP 无关）⇒ 塞进去会让 facing 查表走
            # "需精确匹配范围" 分支而恒为 nearest（代表节点策略套用）。
            # 让表走**自带套件范围**的 range-set 路径 ⇒ 牌面可精确命中。
            # 可用 SOLVER_FACING_RANGE_SET=0 回退。
            _facing_rangeset = (str(__import__("os").environ.get(
                "SOLVER_FACING_RANGE_SET", "1")).lower()
                not in ("0", "off", "false", "no"))
            _is_facing_node = float(state.get("to_call") or 0.0) > 1e-9
            if (getattr(provider, "name", "") == "postflop"
                    and spot_pot_class is not None
                    and "ranges" not in query
                    and not (_is_facing_node and _facing_rangeset)):
                from poker_decision_ai.core.providers.spot_table import (
                    get_spot_table)
                dr = get_spot_table().declared_ranges(
                    str(query.get("street") or ""), spot_pot_class)
                if dr:
                    query["ranges"] = {"oop": dr[0], "ip": dr[1]}

            # ---- P30-9-BK：spot 库未命中时，用**范围 agent 的真实输出**（默认关闭）----
            # 开关：config `solver.range_from_agent` 或环境变量 SOLVER_RANGE_FROM_AGENT=1
            try:
                _ron = (str(os.environ.get("SOLVER_RANGE_FROM_AGENT", "0")).lower()
                        not in ("0", "", "false", "no")) or bool(
                    ctx.config.root.get("solver.range_from_agent", False))
            except Exception:                                  # noqa: BLE001
                _ron = False
            if _ron and "ranges" not in query and \
                    getattr(provider, "name", "") == "postflop":
                _act = [pl for pl in (state.get("players") or [])
                        if isinstance(pl, dict)
                        and pl.get("status") in ("active", "allin")]
                _rng_in = inp.get("range") or {}
                if len(_act) == 2 and isinstance(_rng_in, dict):
                    _order = {"SB": 0, "BB": 1, "UTG": 2, "MP": 3, "CO": 4, "BTN": 5}
                    _hero = next((pl for pl in _act if pl.get("is_hero")), None)
                    _vill = next((pl for pl in _act if not pl.get("is_hero")), None)
                    if _hero is not None and _vill is not None:
                        _hero_report = _rng_in.get("hero_range") or {}
                        _vill_report = ((_rng_in.get("villains") or {})
                                        .get(_vill.get("player_id"))) or {}
                        _hs = _range_str(_hero_report.get("distribution"))
                        _vs = _range_str(_vill_report.get("distribution"))
                        if _hs and _vs:
                            if _order.get(str(_hero.get("position")), 9) < \
                                    _order.get(str(_vill.get("position")), 9):
                                query["ranges"] = {"oop": _hs, "ip": _vs}
                            else:
                                query["ranges"] = {"oop": _vs, "ip": _hs}
                            query["range_source"] = "range_agent"
                            warnings.append({
                                "code": "SOLVER_RANGE_FROM_AGENT",
                                "message": ("solver 范围来自 range agent 真实输出"
                                             "（非 spot 库声明）")})
                        else:
                            warnings.append({
                                "code": "SOLVER_RANGE_EMPTY",
                                "message": "range agent 输出为空 ⇒ 不伪造范围（保持降级）"})

            try:
                result = await provider.solve(query)
            except (ProviderNotConfiguredError, ProviderError) as e:
                provider = _SafeMockSolverProvider(seed=42)
                warnings.append({"code": "SOLVER_DEGRADED",
                                 "message": f"solver unavailable ({e}); degraded to mock baseline"})
                result = await provider.solve(query)

        actions = _norm_action_rows(result.actions)

        # V2 诚实标记：真实 Solver（非 mock、非近似）才算可用
        solver_available = bool(
            not result.is_approximate and result.solver_name not in ("mock", "")
        )

        out = SolverOutput(
            solver_name=result.solver_name,
            model=result.model,
            is_approximate=bool(result.is_approximate),
            solver_available=solver_available,
            cache_hit=cache_hit,
            actions=actions,
            bet_sizings_bb=list(result.bet_sizings_bb or []),
            notes=result.notes,
            # SOLVER-2 来源标记：DEEP 现场求解 = postflop_live（真实求解器）
            source=("postflop_live" if solver_available
                    and getattr(provider, "name", "") == "postflop"
                    else ("realengine" if solver_available else "mock")),
        )

        if actions:
            best = max(actions, key=lambda a: float(a.get("frequency", 0.0) or 0.0))
            out.gto_action = str(best.get("name", "check"))
            out.gto_frequency = float(best.get("frequency", 0.0) or 0.0)

        if result.is_approximate:
            out.confidence = 0.35
            warnings.append({"code": "APPROXIMATE",
                             "message": "Mock 基线非真实 GTO（仅作弱先验）"})
        else:
            out.confidence = 0.9

        out.evidence = [{"kind": "solver",
                         "text": f"solver={result.solver_name}, approximate={result.is_approximate}, "
                                 f"gto_action={out.gto_action}@{out.gto_frequency}",
                         "source": "solver", "weight": 1.0}]
        out.warnings = warnings
        return out
