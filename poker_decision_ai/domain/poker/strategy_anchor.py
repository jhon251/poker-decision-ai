"""SOLVER-2 — 求解器锚定 + 有界剥削偏离（策略核心基座反转）。

任务书（round4_SOLVER-2）：把生产决策装配从"启发式加权投票"翻转为
"**求解器锚定 + 有界剥削偏离**"：

  1. 基座反转：spot 库命中（HU flop/turn/river，to_call=0 首入节点）时
     base_decision = solver 策略（argmax 频率或按频率采样，配置选择）；
     下注尺寸用 solver sizings（替换 BET_FRACTIONS/RAISE_TO_MULTS 常数——
     那两个常数仅在 solver miss 的启发式降级路径继续使用）。
  2. 有界偏离：exploit / villain_frequency_prior / bluff 信号只能在
     solver 基线附近做**有预算的偏离**：
       * 目标动作必须在 solver 显著支持内（freq ≥ dev_min_freq）；
       * solver 预测 EV 损失 ≤ dev_max_ev_loss_bb（偏离预算，bb/决策）；
       * 信号置信 ≥ dev_min_signal_conf；
       * 每次偏离记录 justification（信号、预测 EV 增量、预算消耗）。
     禁止无锚点的大幅偏离（不在 solver 支持内 / 预算外 → 不偏离）。
  3. preflop → 图表基线（source=chart，非均衡解，任务书明确）；
     chart 无 EV 锚点，chart 偏离的量化门 = exploit 预测 EV 增量 ≥
     dev_min_ev_gain_bb，且目标动作必须是图表场景声明过的动作。

诚实边界（"不是什么"）：
  * spot 库只覆盖 HU to_call=0 首入节点；facing-bet 节点不在库内 →
    不锚定（诚实 miss，走启发式路径）。
  * chart 是查表基线（1/0 成员资格，无混合频率），不是均衡解。
  * 本模块是确定性数学，绝不调用 LLM。

provenance 类别（SOLVER-2 验收）：solver_hit / preflop_chart /
solver_miss_fallback / exploit_deviation。
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

# ── provenance 常量（验收要求四类可区分）─────────────────────────────
ANCHOR_SOLVER_HIT = "solver_hit"
ANCHOR_PREFLOP_CHART = "preflop_chart"
ANCHOR_MISS_FALLBACK = "solver_miss_fallback"
ANCHOR_EXPLOIT_DEVIATION = "exploit_deviation"

POSTFLOP_STREETS: tuple[str, ...] = ("flop", "turn", "river")
ACTIONS: tuple[str, ...] = ("fold", "check", "call", "bet", "raise", "allin")
#: to_call==0（首入节点）合法动作；to_call>0（面对下注）合法动作
NODE_ACTIONS_NO_BET: tuple[str, ...] = ("check", "bet", "allin")
NODE_ACTIONS_FACING: tuple[str, ...] = ("fold", "call", "raise", "allin")

#: strategy_core 配置缺省值（生产值在 configs/config.yaml::strategy_core）
ANCHOR_DEFAULTS: dict[str, Any] = {
    "anchor": "solver",            # solver | off（off=旧加权融合路径）
    "anchor_action_mode": "argmax",  # argmax | sample
    "sample_seed": 42,
    "dev_min_freq": 0.10,          # 偏离目标必须在 solver 显著支持内
    "dev_max_ev_loss_bb": 0.05,    # solver 预测 EV 损失预算（bb/决策）
    "dev_min_signal_conf": 0.55,   # 剥削信号置信下限
    "dev_min_ev_gain_bb": 0.10,    # chart 锚定的量化偏离门（exploit 预测增量）
}

#: anchor 置信（exact 命中 > nearest 近似 > chart 查表）
_CONF_EXACT = 0.9
_CONF_NEAREST = 0.75
_CONF_CHART = 0.6


def anchor_config(cfg: dict | None) -> dict:
    """合并缺省值（配置缺项回退到声明缺省，代码不写死生产数值）。"""
    out = dict(ANCHOR_DEFAULTS)
    out.update({k: v for k, v in (cfg or {}).items() if v is not None})
    return out


def normalize_action(name: Any) -> str:
    """动作名归一：CLI Debug 名（Check/Bet/AllIn/all-in）→ 小写规范名。"""
    n = str(name or "").strip().lower().replace("-", "").replace("_", "")
    if n == "allin":
        return "allin"
    return n if n in ACTIONS else ""


def _num(v: Any) -> float | None:
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    return None


def _clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return float(max(lo, min(hi, x)))


def normalized_support(actions: Any) -> list[dict]:
    """solver/chart 动作行 → 归一支持集 [{action,size_bb,frequency,ev}]。

    同名同尺寸行去重（频率/EV 取和/最大）；未知动作名丢弃（诚实）。
    """
    rows: list[dict] = []
    for a in actions or []:
        if not isinstance(a, dict):
            continue
        name = normalize_action(a.get("name"))
        if not name:
            continue
        size = _num(a.get("size_bb"))
        if size is None:
            size = _num(a.get("amount_bb"))
        freq = _num(a.get("frequency")) or 0.0
        ev = _num(a.get("ev"))
        if ev is None:
            ev = _num(a.get("ev_bb"))
        hit = None
        for r in rows:
            if r["action"] == name and (
                    (r["size_bb"] is None and size is None)
                    or (r["size_bb"] is not None and size is not None
                        and abs(r["size_bb"] - size) < 1e-9)):
                hit = r
                break
        if hit is None:
            rows.append({"action": name, "size_bb": size,
                         "frequency": freq, "ev": ev})
        else:
            hit["frequency"] += freq
            if ev is not None and (hit["ev"] is None or ev > hit["ev"]):
                hit["ev"] = ev
    return rows


def _hero_stack_bb(state: dict) -> float | None:
    for p in state.get("players") or []:
        if isinstance(p, dict) and p.get("is_hero"):
            st = _num(p.get("stack"))
            bb = _num(state.get("bb")) or 1.0
            if st is not None:
                return st / bb
    return None


def _pick_anchor_row(support: list[dict], mode: str,
                     rng_seed: Any, state: dict) -> dict | None:
    """锚定动作行：argmax 频率（默认，确定性）或按频率采样（种子可控）。"""
    if not support:
        return None
    if mode == "sample":
        seed_key = hash(str(sorted(state.get("board") or []))
                        + str(state.get("hand_id") or ""))
        seed = int(_num(rng_seed) or 42)
        rng = random.Random(seed ^ (seed_key % (2 ** 32)))
        roll = rng.random() * sum(max(r["frequency"], 0.0) for r in support)
        acc = 0.0
        for r in sorted(support, key=lambda r: (-r["frequency"],
                                                ACTIONS.index(r["action"]),
                                                r["size_bb"] or 0.0)):
            acc += max(r["frequency"], 0.0)
            if roll <= acc:
                return r
        return support[0]
    # argmax：频率降序、动作序、尺寸升序（确定性 tie-break）
    return max(support,
               key=lambda r: (r["frequency"],
                              -ACTIONS.index(r["action"]),
                              -(r["size_bb"] or 0.0)))


@dataclass
class AnchorDecision:
    """锚定基座（base_decision 的求解器形态）。"""
    action: str = "check"
    recommended_size_bb: float | None = None
    confidence: float = _CONF_EXACT
    source: str = ANCHOR_SOLVER_HIT          # solver_hit | preflop_chart
    support: list[dict] = field(default_factory=list)
    provenance: dict = field(default_factory=dict)


# ── 基座 1：postflop spot 库命中 ─────────────────────────────────────
def _state_node_type(state: dict) -> str:
    """状态节点类型：to_call>0 → facing_bet；否则 first_in（M4 契约键）。"""
    return "facing_bet" if (_num(state.get("to_call")) or 0.0) > 1e-9 \
        else "first_in"


def _solver_src_is_approximate(solver_src: dict) -> bool:
    """P30-9-CS：命中是否为**近似**（nearest 桶 / 代表节点 / 尺寸或底池近似）。

    纪律：只有**精确**命中才允许锚定覆盖 EV 核；近似命中一律降级为弱先验。
    （此前 `anchor_from_solver` 不检查 is_approximate ⇒ `range_set_nearest`
     的代表节点策略会**覆盖** EV 核，实测在 facing 节点造成系统性过度加注。）
    """
    solver_src = solver_src or {}
    if bool(solver_src.get("is_approximate")):
        return True
    detail = solver_src.get("source_detail")
    detail = detail if isinstance(detail, dict) else {}
    mode = str(detail.get("lookup_mode") or solver_src.get("lookup_mode") or "")
    if mode and ("nearest" in mode.lower() or "approx" in mode.lower()):
        return True
    for k in ("anchored",):
        if detail.get(k) is False:
            return True
    return bool(detail.get("pot_approximate") or detail.get("size_approximate"))


def _solver_src_node(solver_src: dict) -> str:
    """solver_src 声明的节点类型（raw.node / source_detail.node；缺省 first_in）。

    SolverOutput 是 dataclass → judge 侧收到的是 to_dict 后的普通 dict，
    `raw` 字段不存在——节点声明以 `source_detail.node` 为准（solver agent
    在 spot 命中路径写入）；`raw.node` 兼容直接传 spot 查表结果的调用方。
    """
    raw = solver_src.get("raw") if isinstance(solver_src.get("raw"), dict) else {}
    detail = solver_src.get("source_detail")
    detail = detail if isinstance(detail, dict) else {}
    node = str(detail.get("node") or raw.get("node")
               or solver_src.get("node") or "first_in")
    return node if node in ("first_in", "facing_bet") else "first_in"


def _n_active_players(state: dict) -> int:
    return len([p for p in (state.get("players") or [])
                if isinstance(p, dict) and p.get("status") in ("active", "allin")])


def anchor_from_solver(state: dict, solver_src: dict,
                       cfg: dict | None = None) -> AnchorDecision | None:
    """spot 库命中（真实 postflop 输出）→ 锚定基座；否则 None。

    节点兼容门（M4 扩展）：solver_src 必须声明与状态一致的节点类型——
      * to_call=0（首入）→ 只接受 node=first_in（旧库全部是首入节点）；
      * to_call>0（面对下注）→ 只接受 node=facing_bet（M4 facing 库）。
    类型不匹配（把首入策略套到面对下注、或反向）→ 不锚定（诚实 miss）。
    非 postflop 街 → 不锚定。
    """
    cfg = anchor_config(cfg)
    if str(cfg.get("anchor") or "solver") != "solver":
        return None
    solver_src = solver_src or {}
    if str(solver_src.get("solver_name") or "") != "postflop":
        return None
    # P30-9-CS：近似命中不锚定（只允许精确命中覆盖 EV 核）
    if _solver_src_is_approximate(solver_src):
        return None
    street = str(state.get("street") or "")
    if street not in POSTFLOP_STREETS:
        return None
    to_call = _num(state.get("to_call")) or 0.0
    state_node = _state_node_type(state)
    if _solver_src_node(solver_src) != state_node:
        return None
    support = normalized_support(solver_src.get("actions"))
    support = [r for r in support if r["frequency"] > 0.0]
    # 节点合法性（防御性）：首入节点不可能有 call/fold，面对下注不可能有
    # check/bet——越界动作行（stub/脏数据/nearest 桶错配）直接剔除；剔除后
    # 为空 → 不锚定（诚实 miss），绝不输出非法动作。
    legal = NODE_ACTIONS_FACING if to_call > 1e-9 else NODE_ACTIONS_NO_BET
    support = [r for r in support if r["action"] in legal]
    if not support:
        return None
    row = _pick_anchor_row(support, str(cfg.get("anchor_action_mode")),
                           cfg.get("sample_seed"), state)
    if row is None:
        return None
    action = row["action"]
    size = row["size_bb"]
    if action == "allin":
        stack = _hero_stack_bb(state)
        size = stack if stack is not None else size
    detail = dict(solver_src.get("source_detail") or {})
    approx = bool(solver_src.get("is_approximate"))
    raw = solver_src.get("raw") if isinstance(solver_src.get("raw"), dict) else {}
    lookup_mode = str(detail.get("lookup_mode")
                      or raw.get("lookup_mode") or "live_solve")
    # M4：multiway 简化标注（TARGET_SPEC「HU 解锚定 + 多人简化」——multiway
    # 首入允许套用 HU 解但必须显式降级标注；facing-bet multiway 不在此路径，
    # 由 solver agent 诚实 miss）。
    n_active = _n_active_players(state)
    multiway = n_active > 2
    node_label = ("facing_bet(to_call>0)" if to_call > 1e-9
                  else "first_in(to_call=0)")
    provenance = {
        "source": ANCHOR_SOLVER_HIT,
        "street": street,
        "anchor_action": action,
        "anchor_action_freq": round(row["frequency"], 4),
        "anchor_action_ev_bb": row["ev"],
        "lookup_mode": lookup_mode,
        "spot_id": str(detail.get("spot_id") or raw.get("spot_id") or ""),
        "is_approximate": approx,
        "node": node_label,
        "node_type": state_node,
    }
    if state_node == "facing_bet":
        provenance["bet_bucket"] = str(raw.get("bet_bucket")
                                       or detail.get("bet_bucket") or "")
        provenance["actor_role"] = str(raw.get("actor_role")
                                       or detail.get("actor_role") or "")
        provenance["pot_family"] = str(raw.get("family")
                                       or detail.get("family") or "")
    if multiway:
        provenance["multiway_simplified"] = True
        provenance["n_active_players"] = n_active
        provenance["simplification"] = (
            "hu_first_in_anchor_multiway（HU 均衡解套用到多人首入节点；"
            "简化级别=HU 锚定近似，非多人精确解）")
    return AnchorDecision(
        action=action,
        recommended_size_bb=size,
        confidence=_CONF_NEAREST if approx else _CONF_EXACT,
        source=ANCHOR_SOLVER_HIT,
        support=support,
        provenance=provenance,
    )


# ── 基座 2：preflop 图表（source=chart，非均衡解）────────────────────
def anchor_from_chart(state: dict, solver_src: dict,
                      cfg: dict | None = None) -> AnchorDecision | None:
    """preflop 图表命中（source=chart）→ 锚定基座；否则 None。"""
    cfg = anchor_config(cfg)
    if str(cfg.get("anchor") or "solver") != "solver":
        return None
    solver_src = solver_src or {}
    if str(state.get("street") or "") != "preflop":
        return None
    if str(solver_src.get("solver_name") or "") != "preflop_chart":
        return None
    support = normalized_support(solver_src.get("actions"))
    if not support:
        return None
    chosen = None
    raw = solver_src.get("raw") if isinstance(solver_src.get("raw"), dict) else {}
    chosen_name = normalize_action(raw.get("chosen"))
    for r in support:
        if r["action"] == chosen_name:
            chosen = r
            break
    if chosen is None:
        chosen = max(support, key=lambda r: r["frequency"])
    detail = dict(solver_src.get("source_detail") or {})
    return AnchorDecision(
        action=chosen["action"],
        recommended_size_bb=chosen["size_bb"],
        confidence=_CONF_CHART,
        source=ANCHOR_PREFLOP_CHART,
        support=support,
        provenance={
            "source": ANCHOR_PREFLOP_CHART,
            "street": "preflop",
            "anchor_action": chosen["action"],
            "anchor_action_freq": chosen["frequency"],
            "scenario": str(raw.get("scenario")
                            or detail.get("scenario") or ""),
            "hand_class": str(raw.get("hand_class")
                              or detail.get("hand_class") or ""),
            "is_approximate": True,
            "equilibrium": False,
            "note": "chart 查表基线（非 solver 产物、非均衡解）",
        },
    )


# ── 有界剥削偏离 ─────────────────────────────────────────────────────
def _fe_signal_mode() -> bool:
    """P30-9-BS：弃牌权益信号是否启用（默认 off = 旧行为）。"""
    import os
    return str(os.environ.get("SC_FOLD_SIGNALS", "off")).lower() not in (
        "off", "0", "", "false", "no")


def _fe_gain_k() -> float:
    import os
    try:
        return float(os.environ.get("SC_FOLD_GAIN_K") or 1.0)
    except (TypeError, ValueError):
        return 1.0


def _fe_low_threshold() -> float:
    import os
    try:
        return float(os.environ.get("SC_FOLD_LOW") or 0.30)
    except (TypeError, ValueError):
        return 0.30


def _hero_cards(state: dict) -> list[str]:
    """英雄手牌（players[].hole_cards）。"""
    for pl in (state.get("players") or []):
        if isinstance(pl, dict) and pl.get("is_hero"):
            return [str(c) for c in (pl.get("hole_cards") or [])]
    return []


def _hand_class(cards: list) -> str | None:
    order = "AKQJT98765432"
    if not isinstance(cards, (list, tuple)) or len(cards) != 2:
        return None
    c = [str(x) for x in cards]
    if any(len(x) < 2 for x in c):
        return None
    r1, s1, r2, s2 = c[0][0], c[0][1].lower(), c[1][0], c[1][1].lower()
    if r1 not in order or r2 not in order:
        return None
    if r1 == r2:
        return f"{r1}{r2}"
    hi, lo = ((r1, r2) if order.index(r1) < order.index(r2) else (r2, r1))
    return f"{hi}{lo}{'s' if s1 == s2 else 'o'}"


def _per_hand_signal(inp: dict, state: dict) -> dict | None:
    """P30-9-DM：solver 逐手 EV 最优 → 偏离提案（含逐手频率/EV，供预算门使用）。"""
    import os as _os
    if str(_os.environ.get("SC_PER_HAND_DEV", "1")).lower() in (
            "0", "off", "false", "no"):
        return None
    sol = inp.get("solver") or {}
    sd = sol.get("source_detail") if isinstance(sol.get("source_detail"), dict) else {}
    ph = sd.get("per_hand") if isinstance(sd, dict) else None
    if not isinstance(ph, dict):
        return None
    hands = ph.get("hands") or []
    rows = ph.get("rows") or []
    if not hands or not rows:
        return None
    hero = _hero_cards(state)
    if len(hero) != 2:
        return None

    def norm(cards) -> str:
        return "".join(sorted(str(c) for c in cards))

    idx = [i for i, h in enumerate(hands)
           if norm([str(h)[:2], str(h)[2:4]]) == norm(hero)]
    if not idx:
        cls = _hand_class(hero)
        idx = [i for i, h in enumerate(hands)
               if _hand_class([str(h)[:2], str(h)[2:4]]) == cls] if cls else []
    if not idx:
        return None

    def mean(arr) -> float | None:
        vals = [float(arr[i]) for i in idx
                if isinstance(arr, list) and i < len(arr)
                and isinstance(arr[i], (int, float))]
        return (sum(vals) / len(vals)) if vals else None

    scored = []
    for r in rows:
        ev = mean(r.get("ev"))
        fq = mean(r.get("freq"))
        if ev is None:
            continue
        scored.append({"action": normalize_action(r.get("action")),
                        "size_bb": r.get("size_bb"), "ev_hand": ev,
                        "freq_hand": (fq if fq is not None else 0.0)})
    if not scored:
        return None
    # P30-9-DR：锚定的**真实**规则 = 支持行中聚合频率最高者（不是 agent 的 gto_action）
    _srows = [r_ for r_ in (sol.get("actions") or []) if isinstance(r_, dict)]
    _freq_row = (max(_srows, key=lambda r_: float(r_.get("frequency") or 0.0))
                  if _srows else None)
    anchor_act = normalize_action(((_freq_row or {}).get("name")
                                    or sol.get("gto_action") or ""))
    # P30-9-DP：锚定的**尺寸**来自频率行；同动作不同尺寸也允许偏离（仅尺寸偏离）
    _anc_size = None
    _sd_rows = sol.get("actions") if isinstance(sol.get("actions"), list) else []
    _anc_row = max((r_ for r_ in _sd_rows if isinstance(r_, dict)
                     and normalize_action(r_.get("name")) == anchor_act),
                    key=lambda r_: float(r_.get("frequency") or 0.0), default=None)
    if isinstance(_anc_row, dict):
        _anc_size = _anc_row.get("size_bb")
    best = max(scored, key=lambda r: r["ev_hand"])
    anc = next((r for r in scored if r["action"] == anchor_act
                 and (_anc_size is None or r.get("size_bb") == _anc_size)), None)
    if anc is None:
        anc = next((r for r in scored if r["action"] == anchor_act), None)
    if best["action"] == (anc or {}).get("action"):
        import os as _os2
        if str(_os2.environ.get("SC_SIZE_ONLY_DEV", "1")).lower() in (
                "0", "off", "false", "no"):
            return None
        if best.get("size_bb") == (anc or {}).get("size_bb"):
            return None                   # 动作与尺寸都相同 ⇒ 无需偏离
    return {"action": best["action"], "signal": "solver_per_hand_ev",
            "confidence": 1.0,
            "ev_gain_bb": (best["ev_hand"] - anc["ev_hand"]) if anc else None,
            "size_bb": best.get("size_bb"),
            "freq_hand": best.get("freq_hand"),
            "ev_hand": best.get("ev_hand"),
            "ev_hand_anchor": (anc or {}).get("ev_hand"),
            "source": "solver_per_hand_ev",
            "signal_detail": {"hand_indices": idx[:6], "n_idx": len(idx)}}


def _per_hand_freq_signal(inp: dict, state: dict) -> dict | None:
    """P30-9-EG：6-max 逐手**频率**偏离提案（库无逐手 EV ⇒ 用频率口径）。

    条件：逐手最频繁动作 ≠ 锚定动作，且其频率 ≥0.5、锚定动作频率 ≤0.2。
    """
    import os as _os
    if str(_os.environ.get("SC_PER_HAND_FREQ", "1")).lower() in (
            "0", "off", "false", "no"):
        return None
    sol = inp.get("solver") or {}
    sd = sol.get("source_detail") if isinstance(sol.get("source_detail"), dict) else {}
    # P30-9-EI：**限定 6-max 库来源**——HU r1/fb 库有逐手 EV（走 EV 口径），
    # 频率口径仅作"无逐手 EV 库"（6-max 外部解）的回退；否则会劣化 HU 侧
    # （实测 r1 turn/river regret_max 0.018 → 0.153）。
    _src = str(sol.get("source") or "")
    if not ("6max" in _src or (sd or {}).get("attribution")):
        return None
    ph = (sd or {}).get("per_hand") if isinstance(sd, dict) else None
    if not isinstance(ph, dict) or not ph.get("hands") or not ph.get("rows"):
        return None
    # 若存在可用的逐手 EV ⇒ 交给 EV 口径（此处不介入）
    if any(isinstance(r.get("ev"), list) and any(
            isinstance(x, (int, float)) for x in r["ev"])
            for r in (ph.get("rows") or [])):
        return None
    rows = [r for r in ph.get("rows") or [] if isinstance(r.get("freq"), list)]
    if not rows:
        return None
    hero = _hero_cards(state)
    if len(hero) != 2:
        return None

    def norm(c) -> str:
        return "".join(sorted(str(x) for x in c))

    hands = ph.get("hands") or []
    idx = [i for i, h in enumerate(hands) if norm([str(h)[:2], str(h)[2:4]]) == norm(hero)]
    if not idx:
        cls = _hand_class(hero)
        idx = [i for i, h in enumerate(hands)
               if _hand_class([str(h)[:2], str(h)[2:4]]) == cls] if cls else []
    if not idx:
        return None

    def mean(arr) -> float | None:
        vals = [float(arr[i]) for i in idx
                if isinstance(arr, list) and i < len(arr)
                and isinstance(arr[i], (int, float))]
        return (sum(vals) / len(vals)) if vals else None

    scored = []
    for r in rows:
        f = mean(r.get("freq"))
        if f is None:
            continue
        scored.append({"action": normalize_action(r.get("action")), "size_bb": r.get("size_bb"),
                        "freq_hand": f})
    if not scored:
        return None
    best = max(scored, key=lambda x: x["freq_hand"])
    _srows = [x for x in (sol.get("actions") or []) if isinstance(x, dict)]
    _freq_row = max(_srows, key=lambda x: float(x.get("frequency") or 0.0)) if _srows else None
    anchor_act = normalize_action(((_freq_row or {}).get("name") or ""))
    anc = next((x for x in scored if x["action"] == anchor_act), None)
    if best["action"] == anchor_act:
        return None
    # P30-9-FH：阈值可配置（默认 = 现网值），供证据驱动调优
    import os as _os3
    # P30-9-FH：默认 0.35/0.35 —— 双样本交叉验证：主选一致 +3.7pp、支持率不变
    _top = float(_os3.environ.get("SC_PHF_TOP") or 0.35)
    _amax = float(_os3.environ.get("SC_PHF_ANCHOR_MAX") or 0.35)
    if best["freq_hand"] < _top or (anc or {}).get("freq_hand", 0.0) > _amax:
        return None
    return {"action": best["action"], "signal": "solver_per_hand_freq",
            "confidence": round(float(best["freq_hand"]), 4),
            "size_bb": best.get("size_bb"), "freq_hand": best["freq_hand"],
            "freq_margin": best["freq_hand"] - float((anc or {}).get("freq_hand", 0.0)),
            "source": "solver_per_hand_freq",
            "signal_detail": {"anchor_action": anchor_act,
                               "anchor_freq_hand": (anc or {}).get("freq_hand"),
                               "n_idx": len(idx)}}


def _deviation_signals(inp: dict, state: dict) -> list[dict]:
    """偏离信号提案（确定性读取；语义与 judge 既有来源一致）。

    返回 [{action, signal, confidence, ev_gain_bb(可 None), source}]。
    """
    proposals: list[dict] = []
    to_call = _num(state.get("to_call")) or 0.0
    pot_bb = (_num(state.get("pot")) or 0.0) / (_num(state.get("bb")) or 1.0)

    # 1) exploit agent：唯一自带预测 EV 增量（bb）的信号
    best = ((inp.get("exploit") or {}).get("best_exploit") or {})
    act = normalize_action(best.get("suggested_action"))
    if act:
        conf = _num(best.get("confidence"))
        if conf is None:
            conf = _num((inp.get("exploit") or {}).get("confidence"))
        proposals.append({
            "action": act,
            "signal": f"exploit:{str(best.get('type') or best.get('source') or 'rule')}",
            "confidence": _clip(conf if conf is not None else 0.5),
            "ev_gain_bb": _num(best.get("ev_gain_est_bb")),
            "source": "exploit_agent",
        })

    # 2) villain_frequency_prior 通道（与 judge 既有通道同构；不改变其权重）
    freq_src = inp.get("villain_frequency_prior") or {}
    if isinstance(freq_src, dict) and freq_src.get("available"):
        sel_pid = str((freq_src.get("selection") or {}).get("player_id") or "")
        sel_prior = (freq_src.get("villains") or {}).get(sel_pid) or {}
        freqs = sel_prior.get("frequencies") or {}
        conf = _clip(_num(freq_src.get("confidence")) or 0.5)
        f_fold = _num((freqs.get("fold") or {}).get("rate"))
        if f_fold is not None and f_fold > 0:
            # P30-9-BS：把弃牌权益**量化**，并在低弃牌率时**抑制**加注（默认 off）
            _mode = _fe_signal_mode()
            _k = _fe_gain_k()
            _low = _fe_low_threshold()
            if _mode and f_fold < _low and to_call > 0:
                proposals.append({
                    "action": "call",
                    "signal": "frequency_prior:low_fold_equity",
                    "confidence": conf, "ev_gain_bb": None,
                    "source": "villain_frequency_prior",
                    "signal_detail": {"f_fold": f_fold, "pot_bb": round(pot_bb, 2),
                                       "note": ("对手弃牌率低（<阈值）⇒ 加注缺乏"
                                                 "弃牌权益，倾向跟注")},
                })
            else:
                _gain = (round(f_fold * pot_bb * _k, 4) if _mode else None)
                for a in ("bet", "raise"):
                    proposals.append({
                        "action": a, "signal": "frequency_prior:fold_equity",
                        "confidence": conf,
                        "ev_gain_bb": _gain,   # off 时保持 None（与旧行为一致）
                        "source": "villain_frequency_prior",
                        "signal_detail": {"f_fold": f_fold,
                                           "pot_bb": round(pot_bb, 2),
                                           "ev_gain_bb": _gain},
                    })
        p_cont = _num(sel_prior.get("p_continue"))
        if to_call > 0 and p_cont is not None:
            proposals.append({
                "action": "call", "signal": "frequency_prior:call_channel",
                "confidence": conf, "ev_gain_bb": None,
                "source": "villain_frequency_prior",
                "signal_detail": {"p_continue": p_cont},
            })

    # 3) bluff 信号（bluff 概率 → 顶住；value 概率 → 放弃）
    bluff_src = inp.get("bluff") or {}
    bp = _num(bluff_src.get("bluff_probability"))
    vp = _num(bluff_src.get("value_probability"))
    bconf = _clip(_num(bluff_src.get("confidence")) or 0.5)
    if bp is not None and bp > 0.5 and to_call > 0:
        proposals.append({
            "action": "call", "signal": "bluff:bluff_heavy",
            "confidence": bconf, "ev_gain_bb": None, "source": "bluff_agent",
            "signal_detail": {"bluff_probability": bp},
        })
    if vp is not None and vp > 0.6 and to_call > 0:
        proposals.append({
            "action": "fold", "signal": "bluff:value_heavy",
            "confidence": bconf, "ev_gain_bb": None, "source": "bluff_agent",
            "signal_detail": {"value_probability": vp},
        })
    # 5) P30-9-DM：solver 逐手 EV 偏好（精确命中且逐手数据可得）
    ph_sig = _per_hand_signal(inp, state)
    if ph_sig is not None:
        proposals.append(ph_sig)
    # 6) P30-9-EG：6-max 逐手**频率**偏好（库无逐手 EV）
    pf_sig = _per_hand_freq_signal(inp, state)
    if pf_sig is not None:
        proposals.append(pf_sig)

    return proposals


def bounded_deviation(state: dict, anchor: AnchorDecision, inp: dict,
                      cfg: dict | None = None) -> dict | None:
    """在 solver 锚定基线附近做**有预算**的偏离；不满足预算 → None。

    预算（全部来自 config::strategy_core）：
      * 目标动作 ∈ 节点合法动作 ∩ solver/chart 声明支持；
      * postflop：solver 预测 EV 损失 = ev(anchor) − ev(target)
        ≤ dev_max_ev_loss_bb（chart 无 EV → 改用 exploit 预测增量 ≥
        dev_min_ev_gain_bb 作量化门）；
      * 目标动作频率 ≥ dev_min_freq（postflop；chart 行频率恒 1）；
      * 信号置信 ≥ dev_min_signal_conf；
      * 每决策最多一个偏离（取量化增益/置信最高者）。
    """
    cfg = anchor_config(cfg)
    if str(cfg.get("anchor") or "solver") != "solver":
        return None
    to_call = _num(state.get("to_call")) or 0.0
    legal = NODE_ACTIONS_FACING if to_call > 1e-9 else NODE_ACTIONS_NO_BET
    bb = _num(state.get("bb")) or 1.0
    stack_bb = _hero_stack_bb(state)

    anchor_row = None
    for r in anchor.support:
        if r["action"] == anchor.action and (
                r["size_bb"] is None and anchor.recommended_size_bb is None
                or (r["size_bb"] is not None
                    and anchor.recommended_size_bb is not None
                    and abs(r["size_bb"] - anchor.recommended_size_bb) < 1e-6)):
            anchor_row = r
            break
    if anchor_row is None:
        anchor_row = next((r for r in anchor.support
                           if r["action"] == anchor.action), None)

    candidates: list[dict] = []
    for sig in _deviation_signals(inp, state):
        target = sig["action"]
        if target not in legal:
            continue
        _size_only = (target == anchor.action
                      and str(sig.get("signal") or "") == "solver_per_hand_ev"
                      and _num(sig.get("size_bb")) is not None
                      and _num(sig.get("size_bb")) != anchor.recommended_size_bb)
        if target == anchor.action and not _size_only:
            continue
        if _size_only and not bool(cfg.get("dev_size_only", True)):
            continue
        row = next((r for r in anchor.support if r["action"] == target), None)
        if row is None:
            continue                                   # 无锚点：禁止偏离
        if _clip(_num(sig.get("confidence")) or 0.0) < float(
                cfg["dev_min_signal_conf"]):
            continue
        ev_loss = None
        # P30-9-EG：频率类信号的门（该手牌最频繁动作 vs 锚定动作的**频率余量**）
        if (str(sig.get("signal") or "") == "solver_per_hand_freq"
                and bool(cfg.get("dev_per_hand_freq", True))):
            _fh2 = _num(sig.get("freq_hand"))
            _mg = _num(sig.get("freq_margin"))
            import os as _os4
            _mm = float(_os4.environ.get("SC_PHF_MARGIN")
                        or cfg.get("dev_min_freq_margin", 0.30))
            if (_fh2 is None or _fh2 < float(cfg.get("dev_min_freq_hand", 0.02))
                    or _mg is None or _mg < _mm):
                continue
            candidates.append({**sig, "target_row": row,
                                "solver_ev_loss_bb": None, "per_hand_gate": True,
                                "freq_only": True})
            continue
        if anchor.source == ANCHOR_SOLVER_HIT:
            _fh = _num(sig.get("freq_hand"))
            _eh = _num(sig.get("ev_hand"))
            _eah = _num(sig.get("ev_hand_anchor"))
            if (str(sig.get("signal") or "") == "solver_per_hand_ev"
                    and _fh is not None and _eh is not None and _eah is not None
                    and bool(cfg.get("dev_per_hand_ev", True))):
                if _size_only:
                    # P30-9-DP/DQ：仅尺寸偏离 ⇒ 门 = **逐手 EV 增益** ≥ dev_min_ev_gain_bb
                    if _fh < float(cfg.get("dev_min_freq_hand", 0.02)):
                        continue
                    _gain = _eh - _eah
                    if _gain < float(cfg.get("dev_min_ev_gain_bb", 0.02)):
                        continue
                    candidates.append({**sig, "target_row": row,
                                       "solver_ev_loss_bb": max(0.0, -_gain),
                                       "per_hand_gate": True, "size_only": True})
                    continue
                # P30-9-DM/DQ：**逐手口径**（与 r1 基准同源）——逐手频率门 + 逐手 EV 预算
                if _fh < float(cfg.get("dev_min_freq_hand", 0.02)):
                    continue
                # P30-9-DT：增益下限（避免追逐噪声级差异）
                _min_gain = float(cfg.get("dev_min_ev_gain_hand", 0.02))
                if (_eh - _eah) < _min_gain:
                    continue
                ev_loss = max(0.0, _eah - _eh)
                if ev_loss > float(cfg["dev_max_ev_loss_bb"]):
                    continue
                candidates.append({**sig, "target_row": row,
                                   "solver_ev_loss_bb": ev_loss,
                                   "per_hand_gate": True})
                continue
            if row["frequency"] < float(cfg["dev_min_freq"]):
                continue                               # 显著支持之外
            if anchor_row is not None and anchor_row["ev"] is not None \
                    and row["ev"] is not None:
                ev_loss = max(0.0, anchor_row["ev"] - row["ev"])
                if ev_loss > float(cfg["dev_max_ev_loss_bb"]):
                    continue                           # 预算外：禁止偏离
            else:
                continue      # EV 锚点缺失 → 不做"有预算"偏离（诚实）
        else:  # chart 锚定：无 EV 锚点 → 量化门 = exploit 预测增量
            gain = _num(sig.get("ev_gain_bb"))
            if gain is None or gain < float(cfg["dev_min_ev_gain_bb"]):
                continue
        candidates.append({**sig, "target_row": row,
                           "solver_ev_loss_bb": ev_loss})

    if not candidates:
        return None

    def _rank(c: dict) -> tuple:
        gain = _num(c.get("ev_gain_bb")) or 0.0
        return (gain, _num(c.get("confidence")) or 0.0)

    best = max(candidates, key=_rank)
    row = best["target_row"]
    # P30-9-DM：逐手信号自带尺寸（同名多尺寸时避免取错行）
    size = (_num(best.get("size_bb"))
            if str(best.get("signal") or "") == "solver_per_hand_ev"
            and _num(best.get("size_bb")) is not None else row["size_bb"])
    # P30-9-DP：仅尺寸偏离时，动作名不变但尺寸取自目标行
    if best["action"] == "allin" and size is None and stack_bb is not None:
        size = stack_bb
    justification = {
        "signal": best["signal"],
        "signal_source": best.get("source"),
        "signal_confidence": round(_clip(_num(best.get("confidence")) or 0.0), 4),
        "predicted_ev_gain_bb": (round(_num(best.get("ev_gain_bb")), 4)
                                 if _num(best.get("ev_gain_bb")) is not None
                                 else None),
        "signal_detail": best.get("signal_detail") or {},
        "solver_predicted_ev_loss_bb": (round(best["solver_ev_loss_bb"], 4)
                                        if best["solver_ev_loss_bb"] is not None
                                        else None),
        "budget": {"dev_max_ev_loss_bb": float(cfg["dev_max_ev_loss_bb"]),
                   "dev_min_freq": float(cfg["dev_min_freq"]),
                   "dev_min_signal_conf": float(cfg["dev_min_signal_conf"])},
        "anchor_action": anchor.action,
        "anchor_action_freq": anchor.provenance.get("anchor_action_freq"),
        "anchor_source": anchor.source,
        "deviated_size_bb": round(size, 3) if size is not None else None,
        "note": ("预算内偏离（solver 锚点保持；预测 EV 损失 ≤ 预算）"
                 if anchor.source == ANCHOR_SOLVER_HIT else
                 "chart 锚定偏离（无 EV 锚点；量化门=exploit 预测增量）"),
    }
    return {"action": best["action"], "size_bb": size,
            "justification": justification}
