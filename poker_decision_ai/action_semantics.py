"""动作语义对齐（P30-9-D，**非冻结边界模块**）。

问题（用户可见）
----------------
当观察中 `to_call` **数值缺失（UNKNOWN）** 时，内部把它当 0.0 用（`field_semantics`
修复前标成 `KNOWN_ZERO`），于是"无人下注"世界被套到"面对下注"的牌局上：
已观察合法动作明明是 `[fold, call, raise]`，输出却是 `bet` / `check`，
文案还写「需跟注 0.00bb」——即 **agent 分不清 call 与 check**。

本模块的职责（**只做语义对齐，零策略改动**）
--------------------------------------------
* 以**已观察到的 `available_actions`** 为准（证据优先；无观察则原样返回，不发明约束）；
* 动作不在合法集内时按**语义族**改写：bet↔raise（激进）、check↔call（被动）；
* 改写发生时**尺寸置空**（另一世界尺寸不可沿用）并记录 provenance/uncertainty；
* 面对下注但 to_call 数值未知时，修正面向用户的文案（不得写「需跟注 0.00bb」）。

不修改：候选域、EV、权重、solver 锚点、频率先验、任何冻结层。
"""
from __future__ import annotations

import re

from typing import Any

ACTION_SEMANTICS_VERSION = "p30-9-d-action-semantics-v1"

#: 语义族：跨"面对下注 / 无人下注"两个世界时保持激进↔激进、被动↔被动
FACING_FAMILY: dict[str, tuple[str, ...]] = {
    "bet": ("raise", "call", "fold", "check"),          # 面对下注：bet ⇒ raise
    "raise": ("raise", "bet", "call", "fold", "check"),  # 无人下注：raise ⇒ bet
    "check": ("call", "check", "fold", "raise", "bet"),  # 面对下注：check ⇒ call
    "call": ("check", "call", "raise", "fold", "bet"),   # 无人下注：call ⇒ check
    "fold": ("fold", "check", "call", "bet", "raise"),
    "allin": ("allin", "raise", "bet", "call", "fold"),
}

#: 需跟注 0.00bb 的各种写法（含无空格/多位小数）
_WRONG_TO_CALL_RE = re.compile(r"需跟注\s*0(?:\.0+)?\s*bb")


def observed_legal(obs: Any) -> list[str]:
    """已观察到的合法动作集（空列表 = 无证据）。"""
    o = obs if isinstance(obs, dict) else {}
    return [str(a).strip().lower() for a in (o.get("available_actions") or [])]


def facing_from_legal(legal: list[str]) -> str:
    """由合法动作集判定是否面对下注（证据）。"""
    if not legal:
        return "UNKNOWN"
    if "call" in legal and "check" not in legal:
        return "BET_PENDING"
    if "check" in legal and "call" not in legal:
        return "NO_BET"
    return "UNKNOWN"


def align_action(action: Any, legal: list[str]) -> tuple[str, bool]:
    """把动作对齐到合法集；返回 (动作, 是否发生改写)。无证据 ⇒ 原样。"""
    a = str(action or "").strip().lower()
    if not a or not legal or a in legal:
        return a, False
    for cand in FACING_FAMILY.get(a, ()):
        if cand in legal:
            return cand, True
    return str(legal[0]), True


def to_call_text(to_call_bb: Any, facing: bool) -> str:
    """面向用户的跟注文案（数值未知但已判定面对下注时不得写 0.00bb）。"""
    if facing and (to_call_bb is None or float(to_call_bb) <= 1e-9):
        return "面对下注（to_call 数值未知）"
    if to_call_bb is None:
        return "跟注额未知"
    return f"需跟注 {float(to_call_bb):.2f}bb"


def align_final_decision(*, final_decision: Any, obs: Any,
                         to_call_unknown: bool | None = None) -> dict:
    """把一条 `final_decision` 对齐到已观察合法集。

    返回**新 dict**（不修改入参）：
      {"action", "alternative_action", "recommended_size", "reasoning", ...,
       "action_semantics": {...}}
    无观察合法集 ⇒ 逐字段原样返回（只补 provenance）。
    """
    fd = dict(final_decision) if isinstance(final_decision, dict) else {}
    legal = observed_legal(obs)
    # P30-9-E：观察未直接给出合法动作集时，由**已识别证据**推导
    # （to_call / extra.v3f_bets 逐座位投入）——修复"不知道前面有人下注"。
    evidence = None
    if not legal:
        try:
            from .facing_evidence import derived_facing
            evidence = derived_facing(obs)
            legal = list(evidence.get("legal") or [])
        except Exception:                     # noqa: BLE001
            evidence = None
    facing_state = (str(evidence.get("facing")) if evidence and evidence.get("facing")
                    else facing_from_legal(legal))
    facing = facing_state == "BET_PENDING"
    if to_call_unknown is None:
        o = obs if isinstance(obs, dict) else {}
        to_call_unknown = o.get("to_call") is None
    notes: list[str] = []
    # ── P30-9-ID：**弃牌安全不变量** ✗✗ ──────────────────────────────────
    #   用户实测：界面**能过牌**（免费）却给出 **弃牌** ✗ —— 这在任何模型下都是**被支配**的 ✓。
    #   根因：`legal` 为空（未观察到按钮集 ✗）⇒ `align_action` 无法把 fold 对齐成 check ✗，
    #         而 `to_call` 又是 None（不知道有没有注要跟 ✗）⇒ fold 被原样输出 ✗。
    #   ⇒ 不变量：**只有在"确实面对下注"被确立时，才允许输出弃牌** ✓；
    #     否则：若已知能过牌 ⇒ 改写为 `check` ✓；仍不确定 ⇒ **撤回动作**（诚实 UNKNOWN ✓）
    #     并在文案里写明原因 ✓（宁可说"信息不足"，也不给被支配的建议 ✓）。
    try:
        _act0 = str(fd.get("action") or "").strip().lower()
    except Exception:                                          # noqa: BLE001
        _act0 = ""
    # P30-9-IE：把不变量**推广到让渡动作**：`fold` 与 **`call`** 都以"面对下注"为前提 ✓。
    #   用户实测第二例：界面是 **盖牌/过牌/下注**（**没有跟注** ✓）却建议 **CALL** ✗✗
    #   —— 与"能过牌却让弃牌"同类，都是**被支配**的 ✓。
    if _act0 in ("fold", "盖牌", "弃牌", "call", "跟注"):
        _legal_now = list(legal or [])
        _facing_proven = bool(facing)
        if not _facing_proven:
            _o = obs if isinstance(obs, dict) else {}
            _tc = _o.get("to_call")
            # P30-9-IG：**合理性门** ✓ —— 跟注额必然小于底池 ✓（底池含那笔下注 ✓）。
            #   实测用户帧：把底池 11,600 读成跟注额 ✗（`to_call == pot` ✓）⇒ 不能据此证明面对下注 ✗。
            if _tc is not None and float(_tc) > 1e-9:
                _pot = _o.get("pot")
                _bad = (_pot is not None and float(_pot) > 0
                        and float(_tc) >= float(_pot) - 1e-9)
                if not _bad:
                    _facing_proven = True                      # 有明确且合理的跟注额 ⇒ 允许 ✓
                else:
                    fd.setdefault("notes_extra", [])
                    fd["to_call_implausible"] = {"to_call": float(_tc), "pot": float(_pot)}
        if not _facing_proven:
            # P30-9-IG：**只改写为"已确认合法"的动作** ✓ ——
            #   已知 `check` 合法 ⇒ 改写为 `check` ✓；
            #   合法集未知（`legal` 为空 ✗）⇒ **撤回**动作 ✓
            #   （把 `call` 猜成 `check` 在真面对下注时是**非法**动作 ✗✗，比被支配更糟 ✓，
            #     所以宁可按诚实 UNKNOWN 处理 ✓）。
            if "check" in _legal_now:
                fd["action"] = "check"                         # 能过牌 ⇒ 免费选项优先 ✓
                fd["recommended_size"] = None
                notes.append(f"action:{_act0}→check（已确认能过牌 ⇒ {_act0} 被支配，已改写 ✓）")
            else:
                fd["action"] = None                            # 撤回：诚实 UNKNOWN ✓
                fd["recommended_size"] = None
                fd["blocked_action"] = _act0
                fd["block_reason"] = "facing_unproven_action_withdrawn"
                notes.append(f"action:{_act0}→∅（无法确立面对下注、也无法确认可过牌 ⇒ 撤回 ✓）")
            fd["reasoning"] = (f"{fd.get('reasoning') or ''}"
                                "（安全不变量：**无法确立面对下注时不给弃牌/跟注建议** ✓ —— "
                                "若可过牌，这两者都被支配 ✗；此处按诚实 UNKNOWN / 过牌处理 ✓）")
    if legal:
        for key in ("action", "alternative_action"):
            raw = fd.get(key)
            if raw is None:
                continue
            new, remapped = align_action(raw, legal)
            if remapped:
                fd[key] = new
                notes.append(f"{key}:{raw}→{new}")
                if key == "action" and fd.get("recommended_size") is not None:
                    fd["recommended_size"] = None
                    notes.append("recommended_size 置空（跨世界尺寸不可沿用）")
        if fd.get("alternative_action") == fd.get("action"):
            fd["alternative_action"] = None
    # 文案修正：面对下注且 to_call 未知时不得写「需跟注 0.00bb」
    reasoning = fd.get("reasoning")
    if isinstance(reasoning, str) and facing and to_call_unknown:
        fixed = _WRONG_TO_CALL_RE.sub("面对下注（to_call 数值未知）", reasoning)
        if fixed != reasoning:
            fd["reasoning"] = fixed
            notes.append("reasoning 文案修正（需跟注 0.00bb → 面对下注）")
    # 有改写时，在文案里显式说明（否则内部"过牌/下注"措辞会与对齐后的动作冲突）
    action_rewrites = [n for n in notes if n.startswith("action:")]
    if action_rewrites:
        fd["reasoning"] = f"{fd.get('reasoning') or ''}" \
            f"（已按已观察合法动作集对齐：{action_rewrites[0]}；" \
            f"内部求解在 to_call 数值未知的节点给出，仅供参考）"
    fd["action_semantics"] = {
        "version": ACTION_SEMANTICS_VERSION,
        "observed_legal_actions": legal,
        "legal_source": (evidence.get("source") if evidence else "available_actions"),
        "facing_from_evidence": facing_state,
        "to_call_unknown": bool(to_call_unknown),
        "evidence_detail": (evidence.get("detail") if evidence else None),
        "rewrites": notes,
        "rule": ("动作必须落在已观察合法集内；跨世界按语义族改写"
                  "（bet↔raise / check↔call）；无观察不改写"),
    }
    # P30-9-V：无法确认是否面对下注（面板跟注额与对手投入都不可见）时，
    # 不得给出"下注/加注"尺寸——那可能完全是错的（实测第二帧：应面对加注，
    # 却被按"无人下注"算成 bet 8bb）。
    try:
        from .facing_evidence import facing_unverifiable
        fv = facing_unverifiable(obs)
    except Exception:                              # noqa: BLE001
        fv = {"unverifiable": False}
    if fv.get("unverifiable"):
        detail = fv.get("detail") or {}
        hero_bet = float(detail.get("hero_bet") or 0.0)
        act = str(fd.get("action") or "").strip().lower()
        fd["state_incomplete"] = {
            "facing": "unverifiable",
            "reason": fv.get("reason"),
            "detail": detail}
        if hero_bet > 0 and act in ("bet", "check"):
            # P30-9-W：`bet`/`check` 在"面对下注"的世界里**根本不存在（不合法）**。
            # 你座位已有投入 + 看不到对手投入 ⇒ 无法确认是否面对加注 ⇒ **不给动作**，
            # 而不是给一个可能非法的动作（实测用户反馈：翻前有人下注却输出 bet/check）。
            fd["action"] = None
            fd["alternative_action"] = None
            fd["recommended_size"] = None      # 状态矛盾 ⇒ 尺寸一并撤下
            fd["advice_quality"] = "unverifiable_facing"
            fd["reasoning"] = (
                f"{fd.get('reasoning') or ''}"
                f"（⚠ 无法给出动作：你的座位已有投入 {hero_bet:g}，但未识别到任何对手投入，"
                f"因此**无法确认你是面对下注还是无人下注**——`bet`/`check` 仅在无人下注时合法，"
                f"此处不给动作以免误导。请确认：动作面板的「跟注/加注」金额是否可见、"
                f"以及牌桌是否完整未被遮挡）")
        else:
            fd["reasoning"] = (
                f"{fd.get('reasoning') or ''}"
                f"（⚠ 未能确认是否面对下注：面板跟注额与对手投入均未识别，"
                f"而你的座位已有投入 {hero_bet:g} ——本建议可能基于「无人下注」假设，"
                f"请以客户端按钮为准）")
    # P30-9-Y：街道路词法（翻前无 bet；翻前 check 仅 BB 无人加注时合法）
    fd = street_vocabulary_fix(fd, obs)
    # P30-9-I：尺寸下限校验在 action_semantics 落盘**之后**（合并 size_check）
    fd = apply_size_floor(fd, obs)
    # P30-9-N：策略尺寸被撤下时，补**规则下限尺寸**（具体尺度，明确标注非策略）
    fd = apply_rule_size(fd, obs)
    # P30-9-AI：多人底池胜率高估 ⇒ 标注（只标注，不改动作/尺寸）
    try:
        from .multiway_equity import equity_consistency
        _eng = None
        _fi = fd.get("final_inputs_summary")
        for _src in (fd.get("key_evidence") or []):
            _s = str(_src)
            if "hero_equity=" in _s:
                try:
                    _eng = float(_s.split("hero_equity=")[1].split()[0].rstrip(",;"))
                except (ValueError, IndexError):
                    _eng = None
                break
        _mc = equity_consistency(obs, _eng)
        if _mc.get("inflated"):
            fd["advice_risk"] = fd.get("advice_risk") or {
                "kind": "multiway_equity_inflated"}
            fd["advice_risk"]["multiway_equity"] = {
                "engine": _mc.get("engine_equity"),
                "multiway_estimate": _mc.get("multiway_estimate"),
                "n_villains": _mc.get("n_villains"), "ratio": _mc.get("ratio")}
            fd["reasoning"] = f"{fd.get('reasoning') or ''}（{_mc.get('note')}）"
    except Exception:                              # noqa: BLE001
        pass
    # P30-9-AC：时机提示（只标注）——hero 已是当前最大投入者 ⇒ 很可能未轮到
    try:
        from .facing_evidence import hero_is_max_contributor
        mc = hero_is_max_contributor(obs)
    except Exception:                              # noqa: BLE001
        mc = {"is_max": False}
    if mc.get("is_max"):
        fd["timing_hint"] = {
            "kind": "hero_is_current_max_contributor",
            "hero_bet": mc.get("hero_bet"), "opp_max": mc.get("opp_max"),
            "note": ("你已是当前最大投入者（或与最高投入相同）——若正等待对手行动，"
                      "本建议的时机可能不适用（引擎无法识别 current_player）")}
        fd["reasoning"] = (
            f"{fd.get('reasoning') or ''}"
            f"（⚠ 时机提示：你已是当前最大投入者（{mc.get('hero_bet'):g} ≥ "
            f"对手最高 {mc.get('opp_max'):g}）——若此刻并未轮到你，本建议仅适用于"
            f"轮到你且无人下注的情形；引擎无法识别『当前行动者』）")
    # P30-9-Z：成手类别 + 诈唬风险标注（**只标注，不改策略/尺寸/动作**）
    try:
        from .hand_strength_label import bluff_risk_note
        return bluff_risk_note(fd, obs)
    except Exception:                              # noqa: BLE001
        return fd


#: 尺寸下限（bb）——单位统一为 bb，故下限与盲注数值无关
MIN_BET_FLOOR_BB = 1.0        # 最小下注 = 1 BB（客户端规则）
MIN_RAISE_TO_FLOOR_BB = 2.0   # 加注到 ≥ 2 BB（翻前仅有盲注时的最小加注）


def size_floor(action: str) -> dict:
    """动作对应的**可证明**尺寸下限（bb）。与策略无关，仅规则下限。"""
    a = str(action or "").strip().lower()
    if a == "bet":
        return {"floor_bb": MIN_BET_FLOOR_BB, "kind": "min_bet",
                "rationale": "最小下注 = 1 BB"}
    if a == "raise":
        return {"floor_bb": MIN_RAISE_TO_FLOOR_BB, "kind": "min_raise_to",
                "rationale": ("加注到 ≥ 2 BB（翻前仅有盲注时的最小加注；实际还须"
                              "大于当前注额，单帧未识别 current_bet 时无法进一步验证）")}
    return {"floor_bb": None, "kind": "not_applicable", "rationale": ""}


def apply_size_floor(fd: dict, obs: Any = None) -> dict:
    """尺寸低于下限 ⇒ **撤下尺寸**（不改成别的尺寸，绝不发明）并写明原因。

    只影响 `recommended_size` 与说明文字；动作、置信、来源一律不动。
    """
    if not isinstance(fd, dict):
        return fd
    action = str(fd.get("action") or "").strip().lower()
    size = fd.get("recommended_size")
    info = size_floor(action)
    verdict = "not_applicable"
    detail = {"action": action, "size_bb_raw": size,
              "floor_bb": info.get("floor_bb"), "floor_kind": info.get("kind"),
              "rationale": info.get("rationale")}
    if action in ("bet", "raise") and isinstance(size, (int, float)) \
            and info.get("floor_bb") is not None and float(size) < float(info["floor_bb"]):
        verdict = "withheld_below_floor"
        fd["recommended_size"] = None
        label = "下注" if action == "bet" else "加注"
        note = (f"建议尺寸低于最小{label}下限（{float(size):.2f}bb < "
                f"{float(info['floor_bb']):.1f}bb）⇒ 未提供尺寸，"
                f"请按客户端最小{label}执行；{info.get('rationale')}")
        fd["reasoning"] = f"{fd.get('reasoning') or ''}（{note}）"
        detail["note"] = note
    elif action in ("bet", "raise") and isinstance(size, (int, float)):
        verdict = "ok"
    sem = fd.get("action_semantics")
    if not isinstance(sem, dict):
        sem = {"version": ACTION_SEMANTICS_VERSION}
    sem = {**sem, "size_check": {**detail, "verdict": verdict}}
    fd["action_semantics"] = sem
    return fd


def _bb_chips(obs: Any) -> float | None:
    """盲注 BB 的 chips 值（用于把 bb 尺寸换算成客户端可输入的筹码数）。"""
    o = obs if isinstance(obs, dict) else {}
    b = o.get("blinds")
    if isinstance(b, dict):
        try:
            bb = float(b.get("bb"))
            return bb if bb > 0 else None
        except (TypeError, ValueError):
            return None
    return None


def rule_size_suggestion(fd: dict, obs: Any = None) -> dict:
    """**规则下限尺寸**（非策略尺寸）——让被撤下尺寸的建议仍可执行。

    * `bet`  ⇒ 1 × BB（最小下注）；
    * `raise`⇒ 2 × BB（翻前最小加注到；postflop 的**绝对**下限）；
    * `chips` 由 `obs.blinds.bb` 换算（客户端的下注滑条就是筹码单位）；
    * 明确标注 `source="rule_floor"`、`is_strategy_size=False`，绝不冒充策略尺寸。
    """
    if not isinstance(fd, dict):
        return {}
    action = str(fd.get("action") or "").strip().lower()
    if action not in ("bet", "raise"):
        return {}
    if fd.get("recommended_size") is not None:
        return {}                       # 已有策略尺寸 ⇒ 不加规则尺寸
    # P30-9-V：连"是否面对下注"都无法确认时，规则尺寸同样会误导 ⇒ 不给
    if (fd.get("state_incomplete") or {}).get("facing") == "unverifiable":
        return {}
    bb_units = MIN_BET_FLOOR_BB if action == "bet" else MIN_RAISE_TO_FLOOR_BB
    bb_chips = _bb_chips(obs)
    su = {"action": action, "bb": bb_units,
          "chips": (round(bb_units * bb_chips, 4) if bb_chips else None),
          "bb_chips": bb_chips,
          "source": "rule_floor", "is_strategy_size": False,
          "basis": ("最小下注 = 1×BB" if action == "bet"
                     else "最小加注到 = 2×BB（翻前；postflop 为绝对下限）"),
          "caveat": ("实际加注到必须大于当前注额；单帧未识别 current_bet 时"
                      "本值仅为下限")}
    return su


def apply_rule_size(fd: dict, obs: Any = None) -> dict:
    """把规则下限尺寸挂到结果上并在文案中说明（仅在策略尺寸缺失时）。"""
    su = rule_size_suggestion(fd, obs)
    if not su:
        return fd
    fd["rule_size_suggestion"] = su
    chips_txt = (f"{su['chips']:g} chip" if su.get("chips") is not None else "chip 未知")
    label = "下注" if su["action"] == "bet" else "加注到"
    fd["reasoning"] = (f"{fd.get('reasoning') or ''}"
                       f"（规则下限参考：{label} {su['bb']:.2f}bb = {chips_txt}；"
                       f"非策略尺寸，仅为可执行下限）")
    sem = fd.get("action_semantics")
    if isinstance(sem, dict):
        fd["action_semantics"] = {**sem, "rule_size": su}
    return fd


def _num_safe(v: Any) -> float | None:
    if isinstance(v, bool) or v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    return None


def street_vocabulary_fix(fd: dict, obs: Any = None) -> dict:
    """**街道路词法**校正（纯规则，不含任何策略/尺寸推断）。

    * 翻前（preflop）**不存在 `bet`**：盲注即为场上的注 ⇒ 首个激进动作是 `raise`。
      实测用户反馈：翻前输出 `BET 1.9bb`（客户端根本无法执行）。
    * 翻前 `check` **仅**在"hero 是 BB 且无人加注"时合法；否则应为 `call`（跟注盲注）。

    只在**能证明前**才改写；每次改写记入 `action_semantics.rewrites`，动作仍不发明尺寸。
    """
    if not isinstance(fd, dict):
        return fd
    o = obs if isinstance(obs, dict) else {}
    street = str(o.get("street") or "").lower()
    if street != "preflop":
        return fd
    act = str(fd.get("action") or "").strip().lower()
    notes: list = []
    if act == "bet":
        fd["action"] = "raise"
        notes.append("street:bet→raise（翻前无「下注」）")
    elif act == "check":
        extra = o.get("extra") if isinstance(o.get("extra"), dict) else {}
        bets = extra.get("v3f_bets") if isinstance(extra, dict) else None
        blinds = o.get("blinds") if isinstance(o.get("blinds"), dict) else {}
        bb = _num_safe(blinds.get("bb"))
        button = None
        try:
            sb_seat_val = o.get("button_seat")
            button = int(str(sb_seat_val).upper().lstrip("S")) \
                if sb_seat_val is not None else None
        except (TypeError, ValueError):
            button = None
        hero_seat = None
        try:
            hero_seat = int(str(o.get("hero_seat")).upper().lstrip("S")) \
                if o.get("hero_seat") is not None else None
        except (TypeError, ValueError):
            hero_seat = None
        seats = o.get("seats") if isinstance(o.get("seats"), list) else []
        n = len(seats) or 6
        bb_seat = ((button % n + 1) % n + 1) if button is not None else None
        hero_is_bb = (hero_seat is not None and bb_seat is not None
                       and hero_seat == bb_seat)
        opp_above_bb = False
        if isinstance(bets, dict) and bb:
            for k, v in bets.items():
                val = _num_safe(v)
                if val is None or val <= 0:
                    continue
                try:
                    seat = int(str(k).upper().lstrip("S"))
                except (TypeError, ValueError):
                    seat = None
                if seat is not None and hero_seat is not None and seat == hero_seat:
                    continue
                if val > bb + 1e-9:
                    opp_above_bb = True
        to_call = _num_safe(o.get("to_call"))
        raise_pending = bool(opp_above_bb) or (to_call is not None and to_call > 1e-9)
        if not (hero_is_bb and not raise_pending):
            fd["action"] = "call"
            notes.append("street:check→call（翻前仅有 BB 无人加注时可过牌）")
    if notes:
        alt = fd.get("alternative_action")
        if isinstance(alt, str) and alt.strip().lower() == str(fd.get("action")):
            fd["alternative_action"] = None
        sem = fd.get("action_semantics")
        sem = dict(sem) if isinstance(sem, dict) else {
            "version": ACTION_SEMANTICS_VERSION}
        sem["rewrites"] = list(sem.get("rewrites") or []) + notes
        sem["street_vocabulary"] = {"street": street, "notes": notes}
        fd["action_semantics"] = sem
    return fd


def action_semantics_contract() -> dict:
    return {
        "version": ACTION_SEMANTICS_VERSION,
        "size_floors_bb": {"bet": MIN_BET_FLOOR_BB, "raise": MIN_RAISE_TO_FLOOR_BB},
        "frozen_layers_modified": False,
        "candidate_domain_modified": False,
        "weights_or_ev_modified": False,
        "family": {k: list(v) for k, v in FACING_FAMILY.items()},
        "rule": ("证据优先：仅当观察提供 available_actions 时才约束/改写动作；"
                  "改写时尺寸置空并记录 provenance"),
    }
