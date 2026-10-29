"""P30-9-E：从**已识别证据**推导「是否面对下注」并合成合法动作集。

证据来源（按优先级；全部来自观察，无凭空发明）
------------------------------------------------
1. `available_actions`（若视觉/桥直接给出）——最高优先；
2. `to_call`：已知 > 0 ⇒ 面对下注；已知 == 0 ⇒ 无人下注；
3. `extra.v3f_bets`（逐座位本街投入，实测可识别）+ `hero_seat`：
   存在**非 hero 座位**投入 > hero 自身投入 ⇒ 面对下注；
   所有座位投入均为 0/未知且 to_call 已知为 0 ⇒ 无人下注；
4. 以上皆无 ⇒ UNKNOWN（**不改写、不发明**）。

合成规则（与 `decision_contract.derive_legal_actions` 同口径）：
  面对下注 ⇒ [fold, call, raise]；无人下注 ⇒ [check, bet]。
"""
from __future__ import annotations

import re

from typing import Any

EVIDENCE_VERSION = "p30-9-e-facing-evidence-v1"

BET_SEATS_KEY = "v3f_bets"


def _num(v: Any) -> float | None:
    if isinstance(v, bool) or v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    return None


def _seat_of_bet_key(key: Any) -> int | None:
    """`"S3"`/`"3"`/`"seat3"` → 座位号。"""
    s = str(key).strip().upper().lstrip("S").replace("EAT", "")
    try:
        return int(s)
    except ValueError:
        return None


def derived_facing(obs: Any) -> dict:
    """由观察证据推导 `{facing, legal, source, detail}`（无证据 ⇒ UNKNOWN）。"""
    o = obs if isinstance(obs, dict) else {}
    extra = o.get("extra") if isinstance(o.get("extra"), dict) else {}

    # 1) 显式 available_actions
    explicit = [str(a).strip().lower() for a in (o.get("available_actions") or [])]
    if explicit:
        facing = ("BET_PENDING" if ("call" in explicit and "check" not in explicit)
                  else "NO_BET" if ("check" in explicit and "call" not in explicit)
                  else "UNKNOWN")
        return {"facing": facing, "legal": explicit, "source": "available_actions",
                "detail": {"source": "observed"}}

    # 2) to_call（数值已知）——P30-9-IG：**算术合理性门** ✓
    #   实测（用户帧）：识别器把**底池 11,600** 读成了跟注额 ✗（`to_call == pot` ✓），
    #   而该帧 hero 的按钮明确是 **盖牌/过牌/下注**（能过牌 ✓）⇒ 并不面对下注 ✓。
    #   算术事实：**跟注额必然小于底池** ✓（底池本身已含那笔下注 ✓）
    #   ⇒ `to_call >= pot` 直接判为**不可信** ✗（保留 detail 供审计 ✓，但不用它断言 facing ✓）。
    tc = _num(o.get("to_call"))
    if tc is not None:
        _pot = _num(o.get("pot"))
        _implausible = (_pot is not None and _pot > 0 and tc >= _pot - 1e-9)
        if _implausible:
            return {"facing": "UNKNOWN", "legal": [],
                    "source": "to_call_implausible_ge_pot",
                    "detail": {"to_call": tc, "pot": _pot,
                                "note": "跟注额 ≥ 底池（算术上不可能 ✓，疑似把底池读成跟注额 ✗）"
                                         "⇒ 不据此断言面对下注 ✓"}}
        if tc > 1e-9:
            return {"facing": "BET_PENDING", "legal": ["fold", "call", "raise"],
                    "source": "to_call", "detail": {"to_call": tc}}
        return {"facing": "NO_BET", "legal": ["check", "bet"],
                "source": "to_call", "detail": {"to_call": tc}}

    # 3) extra.v3f_bets（逐座位投入）——**几何归属**（P30-9-AB，与 to_call 判定同源）
    bets = extra.get(BET_SEATS_KEY)
    if isinstance(bets, dict) and bets:
        hb = hero_opp_bets(o)
        hero_bet, opp_max, seen = hb["hero_bet"], hb["opp_max"], hb["n_readable"]
        detail = {**hb["detail"], "hero_bet": hero_bet,
                  "max_opponent_bet": opp_max, "n_readable": seen}
        if hb.get("mapping") == "hero_seat_unknown":
            return {"facing": "UNKNOWN", "legal": [],
                    "source": "v3f_bets_hero_seat_unknown",
                    "detail": {**detail, "hero_seat": None}}
        if not seen:
            return {"facing": "UNKNOWN", "legal": [],
                    "source": "v3f_bets_unreadable", "detail": detail}
        if opp_max > hero_bet + 1e-9:
            # ── P30-9-IG：**单凭"对手桌前筹码更多"不足以断定"现在必须跟"** ✗ ──────────
            #   用户实测（真实帧）：画面里那枚 11,600 与「底池 11,600」同值 ✓，
            #   很可能是**底池筹码展示**或是**已被跟过的旧注**（客户端会把本街投入留在桌前 ✓），
            #   而 hero 的按钮明确是 **盖牌 / 过牌 / 下注**（**能过牌** ✓ ⇒ 没有待跟注 ✓）。
            #   此前该分支直接断言 BET_PENDING ✗ ⇒ legal=[fold,call,raise] ✗ ⇒ 输出**弃牌** ✗✗
            #   （若可过牌，弃牌是**被支配**的 ✗）。
            #   ⇒ 改为**保守**：仅在**另有旁证**时才断言面对下注 ✓
            #     （`to_call` 已知 >0 ✓ 或显式合法集 ✓ —— 二者都已在上面优先处理 ✓）；
            #     否则如实标 UNKNOWN ✓（不给被支配建议 ✓，且保留 detail 供审计 ✓）。
            return {"facing": "UNKNOWN", "legal": [],
                    "source": "v3f_bets_uncorroborated",
                    "detail": {**detail,
                                "note": ("对手桌前投入多于 hero，但**无 to_call / 无合法集旁证** ⇒ "
                                          "可能是底池展示或已跟过的旧注 ⇒ 不据此断言面对下注 ✓")}}
        if hero_bet > 0 and opp_max <= 0:
            # hero 已有投入但**看不到任何对手投入** ⇒ 无法区分"已行动/面对加注(漏读)"
            return {"facing": "UNKNOWN", "legal": [],
                    "source": "v3f_bets_hero_invested_no_opponent",
                    "detail": detail}
        return {"facing": "NO_BET", "legal": ["check", "bet"],
                "source": "v3f_bets", "detail": detail}

    return {"facing": "UNKNOWN", "legal": [], "source": "no_evidence", "detail": {}}


def derive_blinds(obs: Any) -> dict:
    """盲注证据推导（header 金额不可读时的诚实替代来源）。

    优先级：
      1) `obs["blinds"]` 双值 > 0 ⇒ source="observed"（视觉已识别）；
      2) preflop + `button_seat` + `extra.v3f_bets` ⇒ 庄家后两位座位
         （SB = button+1、BB = button+2，1..n 环形）的投入即盲注
         ⇒ source="derived_from_posted_blinds"；
      3) 否则 ⇒ sb/bb = None（**不猜**，调用方维持阻断）。

    返回：{"sb", "bb", "source", "detail"}（detail 含证据明细，便于审计）。
    """
    o = obs if isinstance(obs, dict) else {}
    b = o.get("blinds")
    if isinstance(b, dict):
        sb, bb = _num(b.get("sb")), _num(b.get("bb"))
        if sb is not None and bb is not None and sb > 0 and bb > 0:
            return {"sb": sb, "bb": bb, "source": "observed",
                    "detail": {"blinds": dict(b)}}

    extra = o.get("extra") if isinstance(o.get("extra"), dict) else {}
    bets = extra.get(BET_SEATS_KEY)
    button = _seat_of_bet_key(o.get("button_seat"))
    street = str(o.get("street") or "").lower()
    seats = o.get("seats") if isinstance(o.get("seats"), list) else []
    n = len(seats) or 6
    detail = {"v3f_bets": ({str(k): v for k, v in bets.items()}
                           if isinstance(bets, dict) else None),
              "button_seat": button, "n_seats": n, "street": street}
    if street != "preflop" or button is None or not isinstance(bets, dict) or not bets:
        return {"sb": None, "bb": None, "source": "no_evidence", "detail": detail}

    sb_seat = button % n + 1
    bb_seat = (button % n + 1) % n + 1
    found: dict[int, float] = {}
    for k, v in bets.items():
        s = _seat_of_bet_key(k)
        val = _num(v)
        if s is not None and val is not None:
            found[s] = val
    sb, bb = found.get(sb_seat), found.get(bb_seat)
    detail.update({"sb_seat": sb_seat, "bb_seat": bb_seat,
                   "sb_posted": sb, "bb_posted": bb})
    if sb is None or bb is None or sb <= 0 or bb <= 0 or bb < sb:
        return {"sb": None, "bb": None,
                "source": "derived_insufficient", "detail": detail}
    return {"sb": float(sb), "bb": float(bb),
            "source": "derived_from_posted_blinds", "detail": detail}


#: 盲注会话记忆 TTL（秒）——现金桌盲注不随牌局变化；仅**同桌同局**复用
BLINDS_MEMO_TTL_S = 900.0


def memo_store(memo: dict, key: str, ev: dict, *, now: float) -> None:
    """记录一次**已识别/已推导**的盲注（仅当 sb/bb 有效）。"""
    if not isinstance(memo, dict) or not key:
        return
    if not (isinstance(ev, dict) and ev.get("sb") and ev.get("bb")):
        return
    memo[key] = {"sb": float(ev["sb"]), "bb": float(ev["bb"]),
                 "origin": str(ev.get("source") or "unknown"),
                 "detail": ev.get("detail"), "ts": float(now)}


def memo_lookup(memo: dict, key: str, *, now: float,
                ttl_s: float = BLINDS_MEMO_TTL_S) -> dict | None:
    """取同桌同局的盲注记忆（过期 ⇒ None；**绝不跨桌猜测**）。"""
    if not isinstance(memo, dict) or not key:
        return None
    rec = memo.get(key)
    if not isinstance(rec, dict):
        return None
    age = float(now) - float(rec.get("ts") or 0.0)
    if age < 0 or age > float(ttl_s):
        return None
    return {"sb": rec.get("sb"), "bb": rec.get("bb"), "source": "session_memo",
            "detail": {"memo_origin": rec.get("origin"),
                       "memo_age_s": round(age, 1),
                       "memo_detail": rec.get("detail")}}


#: 窗口标题里的注额对（如 "NL5 0.02/0.05"、"$0.02/$0.05"、"0,02/0,05"）
#: 允许货币符号与空格出现在数字前；不允许前后紧邻数字/小数点（防误读版本号）
_STAKES_RE = re.compile(
    r"(?<![\d.,])[$€£¥]?\s*(\d+(?:[.,]\d+)?)\s*/\s*[$€£¥]?\s*"
    r"(\d+(?:[.,]\d+)?)(?![\d.,])")

#: 常见 BB 白名单（盲注是标准化数值；不在表内 ⇒ 视为误读，不采用）
_BB_WHITELIST = (0.01, 0.02, 0.04, 0.05, 0.1, 0.2, 0.25, 0.5, 1.0, 2.0, 3.0,
                 5.0, 10.0, 20.0, 25.0, 50.0, 100.0, 200.0, 500.0, 1000.0)
#: 允许的 SB:BB 比例区间（1:2、1:2.5、1:3 等；超出 ⇒ 不采用）
_MIN_RATIO, _MAX_RATIO = 1.5, 3.0


def parse_stakes_from_title(title: Any) -> dict:
    """从**窗口标题**（OS 元数据）解析盲注；不自洽 ⇒ 不采用。

    接受 `sb/bb` 形式的数值对；要求 0 < sb < bb 且 bb/sb ≤ 4（防误读日期/版本号）。
    """
    if not isinstance(title, str) or not title:
        return {"sb": None, "bb": None, "source": "no_evidence", "detail": {}}
    for m in _STAKES_RE.finditer(title):
        try:
            sb = float(m.group(1).replace(",", "."))
            bb = float(m.group(2).replace(",", "."))
        except ValueError:
            continue
        if not (0 < sb < bb):
            continue
        ratio = bb / sb
        if not (_MIN_RATIO <= ratio <= _MAX_RATIO):
            continue
        if not any(abs(bb - w) < 1e-9 for w in _BB_WHITELIST):
            continue          # BB 不在常见盲注表内 ⇒ 视为误读（如 "6/9"、"1.2/3.4"）
        return {"sb": sb, "bb": bb, "source": "window_title",
                "detail": {"title": title, "matched": m.group(0)}}
    return {"sb": None, "bb": None, "source": "no_evidence",
            "detail": {"title": title}}


def suggest_blinds(obs: Any) -> dict:
    """给出盲注**建议值**（供 GUI 预填人工确认；**绝不自动采用**）。

    分层：
      ① observed / derived_from_posted_blinds（盲注座投入齐全）⇒ strong=True；
      ② 单一投入水平 + pot 算术：
         bb = 该水平（某位玩家在翻前跟到/投出的一注，通常等于 BB），
         sb = (pot − 可读投入之和) − bb，要求 sb 在盲注白名单内且 0 < sb < bb
         ⇒ strong=False（供人工确认）；
      ③ 否则 ⇒ 无建议（sb/bb 均 None）。

    实测例（用户 18:01 帧）：pot=0.12、可读投入 {S5: 0.05}
      ⇒ bb=0.05、sb=0.12−0.05−0.05=0.02 ⇒ 与真实 0.02/0.05 完全一致。
    """
    o = obs if isinstance(obs, dict) else {}
    ev = derive_blinds(o)
    if ev.get("sb") and ev.get("bb"):
        return {**ev, "strong": True}
    extra = o.get("extra") if isinstance(o.get("extra"), dict) else {}
    bets = extra.get(BET_SEATS_KEY)
    pot = _num(o.get("pot"))
    street = str(o.get("street") or "").lower()
    detail = {"v3f_bets": ({str(k): v for k, v in bets.items()}
                           if isinstance(bets, dict) else None), "pot": pot}
    if street != "preflop" or not isinstance(bets, dict) or pot is None:
        return {"sb": None, "bb": None, "source": "no_evidence",
                "strong": False, "detail": detail}
    vals = [_num(v) for v in bets.values()]
    pos = [v for v in vals if v is not None and v > 0]
    if not pos or len(set(round(v, 9) for v in pos)) != 1:
        return {"sb": None, "bb": None, "source": "no_evidence",
                "strong": False, "detail": detail}
    level = float(pos[0])
    if not any(abs(level - w) < 1e-9 for w in _BB_WHITELIST):
        return {"sb": None, "bb": None, "source": "level_not_whitelisted",
                "strong": False, "detail": {**detail, "level": level}}
    sb = round(float(pot) - sum(pos) - level, 9)
    detail.update({"level": level, "readable_sum": sum(pos), "sb_derived": sb})
    if 0 < sb < level and any(abs(sb - w) < 1e-9 for w in _BB_WHITELIST):
        return {"sb": sb, "bb": level, "source": "partial_posted_level",
                "strong": False, "detail": detail}
    # SB 无法复原 ⇒ 仅建议 BB（下游容忍 sb 未知）
    return {"sb": None, "bb": level, "source": "bb_only_from_posted_level",
            "strong": False, "detail": detail}


def blinds_consistent(obs: Any) -> dict:
    """观察里的 `blinds` 是否与**其他已识别证据**自洽。

    规则（只用已识别值，不做任何猜测）：
      1) `pot < sb + bb` ⇒ **不可能**（任何街道底池都不小于一个盲注轮）；
      2) 存在正的逐座位投入且 `bet < bb / 2` ⇒ 不可能（最小下注=1BB；盲注轮除外），
         实测误读案例：header 读 2/5 而桌上投入是 0.05；
      3) 两条都不触发 ⇒ ok=True（不构成反证）。

    返回 {"ok": bool, "reason": str, "detail": {...}}。
    """
    o = obs if isinstance(obs, dict) else {}
    b = o.get("blinds")
    if not isinstance(b, dict):
        return {"ok": True, "reason": "no_blinds", "detail": {}}
    sb, bb = _num(b.get("sb")), _num(b.get("bb"))
    pot = _num(o.get("pot"))
    if bb is None or bb <= 0:
        return {"ok": True, "reason": "bb_missing", "detail": {}}
    detail = {"sb": sb, "bb": bb, "pot": pot}
    # 规则 ①：底池**不足半个盲注轮**才是硬矛盾（底池识别本身可能略偏小，
    # 故不用 pot < sb+bb 这种过严判据）
    if pot is not None and pot > 0 and sb is not None and sb > 0 \
            and pot + 1e-9 < (sb + bb) / 2.0:
        return {"ok": False, "reason": "pot_less_than_half_blind_round",
                "detail": {**detail, "pot_minus_blinds": round(pot - (sb + bb), 6)}}
    extra = o.get("extra") if isinstance(o.get("extra"), dict) else {}
    bets = extra.get(BET_SEATS_KEY)
    if isinstance(bets, dict) and bets:
        vals = [v for v in (_num(x) for x in bets.values()) if v is not None and v > 0]
        if vals:
            mx = max(vals)
            small = min(sb, bb) if (sb and sb > 0) else bb
            detail["max_observed_bet"] = mx
            detail["ratio_blind_over_bet"] = round(small / mx, 3)
            # 规则 ②：桌上金额比盲注小 **20 倍以上** 才判矛盾。
            # （盲注投入本身可等于 SB < BB，甚至 ante 更小；实测误读案例为 100×）
            if mx > 0 and (small / mx) >= 20.0:
                return {"ok": False, "reason": "observed_amounts_far_below_blinds",
                        "detail": detail}
    return {"ok": True, "reason": "consistent", "detail": detail}


#: 全角/易混字符 → ASCII（中文输入法常把 '.' 打成 '。'）
_DECIMAL_FIX = {"。": ".", "．": ".", "｡": ".", "，": ".", ",": ".",
                "：": ".", "·": ".", "　": ""}


def normalize_decimal_text(s: Any) -> str:
    """把用户输入的数字文本归一为 `[0-9.]`（最多一个小数点）。

    处理：全角句点/逗号（中文输入法）、千分位逗号、空格、多小数点、前后缀杂字符。
    例：`"0。05"` → `"0.05"`；`"0,05"` → `"0.05"`；`"0.0.5"` → `"0.05"`。
    """
    if s is None:
        return ""
    import unicodedata
    # NFKC：全角数字/句点/逗号 → ASCII（'０.０５' → '0.05'；'。' 另由下表处理）
    txt = unicodedata.normalize("NFKC", str(s))
    for bad, good in _DECIMAL_FIX.items():
        txt = txt.replace(bad, good)
    out: list[str] = []
    seen_dot = False
    for ch in txt:
        if ch.isdigit():
            out.append(ch)
        elif ch == ".":
            if not seen_dot:
                out.append(ch)
                seen_dot = True
        # 其余字符（字母/符号/空格）一律丢弃
    res = "".join(out)
    if res.startswith("."):
        res = "0" + res
    return res


def parse_decimal_text(s: Any) -> float | None:
    """归一后转 float；空/不可解析 ⇒ None。"""
    txt = normalize_decimal_text(s)
    if not txt or txt == ".":
        return None
    try:
        return float(txt)
    except ValueError:
        return None


def writeback_decision(cur_sb: Any, cur_bb: Any, new_sb: str, new_bb: str,
                       user_edited: bool) -> str:
    """是否把识别到的盲注回写输入框：'noop' | 'skip_user_edited' | 'write'。

    规则：已相同 ⇒ noop；用户手填过且框内非空 ⇒ skip_user_edited（绝不覆盖）；
    否则 ⇒ write（自我纠正）。
    """
    cs, cb = str(cur_sb or "").strip(), str(cur_bb or "").strip()
    if cs == str(new_sb) and cb == str(new_bb):
        return "noop"
    if user_edited and (cs or cb):
        return "skip_user_edited"
    return "write"


#: 6-max 位置名（庄家座之后的顺序：SB → BB → UTG → MP → CO → BTN）
_POS_6MAX = ("SB", "BB", "UTG", "MP", "CO", "BTN")


def annotate_seat_bets(obs: Any) -> dict:
    """把 `extra.v3f_bets` 标注为可读形式。

    每座输出：`{"seat", "label", "amount", "readable", "role"}`；
    label 例：`"S5(你·CO)"`、`"S1(SB)"`、`"S6(BTN)"`、`"S3"`；
    amount 为 None ⇒ **未识别或无投入**（`readable=False`，绝不当作 0 已确认）。

    角色推导只用**已识别**的庄家座（`button_seat`）与 hero 座（`hero_seat`）：
    庄家后第一位 = SB、第二位 = BB（与 `derive_blinds` 同口径）；不猜其他座位。
    """
    o = obs if isinstance(obs, dict) else {}
    extra = o.get("extra") if isinstance(o.get("extra"), dict) else {}
    bets = extra.get(BET_SEATS_KEY)
    hero_seat = _seat_of_bet_key(o.get("hero_seat"))
    button = _seat_of_bet_key(o.get("button_seat"))
    seats = o.get("seats") if isinstance(o.get("seats"), list) else []
    n = len(seats) or 6
    sb_seat = (button % n + 1) if button is not None else None
    bb_seat = ((button % n + 1) % n + 1) if button is not None else None
    hero_pos = extra.get("v3f_hero_position")
    items: list[dict] = []
    for key, val in (bets or {}).items():
        seat = _seat_of_bet_key(key)
        amt = _num(val)
        tags: list[str] = []
        role = None
        if seat is not None and hero_seat is not None and seat == hero_seat:
            tags.append("你")
            if hero_pos:
                tags.append(str(hero_pos))
        if seat is not None and sb_seat is not None and seat == sb_seat:
            role = "SB"
        elif seat is not None and bb_seat is not None and seat == bb_seat:
            role = "BB"
        elif seat is not None and button is not None and seat == button:
            role = "BTN"
        if role:
            tags.append(role)
        label = f"S{seat}" + (f"({'·'.join(tags)})" if tags else "")
        items.append({"seat": seat, "label": label, "amount": amt,
                      "readable": amt is not None, "role": role,
                      "is_hero": bool(seat is not None and hero_seat is not None
                                       and seat == hero_seat)})
    items.sort(key=lambda x: (x["seat"] is None, x["seat"] or 0))
    parts = [f"{it['label']}: " + (f"{it['amount']:g}" if it["readable"]
                                    else "未识别/无投入") for it in items]
    legend = {"S<数字>": "座位号", "None": "未识别或无投入（≠ 已确认的 0）",
              "SB/BB/BTN": "由已识别庄家座推导", "你": "hero 座"}
    return {"items": items, "text": " · ".join(parts) if parts else "（无座位投入证据）",
            "legend": legend,
            "hero_seat": hero_seat, "button_seat": button}


def to_call_lower_bound(obs: Any) -> dict:
    """由**逐座位投入**推 `to_call` 的**下界**（面板跟注额读不到时的诚实替代）。

    下界 = max(非 hero 座可读投入) − hero 自身可读投入（<0 记 0）。
    语义：对手至少投了这么多 ⇒ hero 若要继续至少需补这么多。
    **只是下界**（若对手加注额漏读则偏小），故返回值必须标注 `is_lower_bound=True`，
    调用方需在文案/provenance 里说明"非面板识别值"。

    返回 {"to_call", "is_lower_bound", "source", "detail"}；无证据 ⇒ to_call=None。
    """
    o = obs if isinstance(obs, dict) else {}
    extra = o.get("extra") if isinstance(o.get("extra"), dict) else {}
    bets = extra.get(BET_SEATS_KEY)
    if not isinstance(bets, dict) or not bets:
        return {"to_call": None, "is_lower_bound": True,
                "source": "no_evidence", "detail": {}}
    hb = hero_opp_bets(o)
    hero_bet, opp_max, seen = hb["hero_bet"], hb["opp_max"], hb["n_readable"]
    detail = {**hb["detail"], "hero_bet": hero_bet,
              "max_opponent_bet": opp_max, "n_readable": seen}
    if not seen or opp_max <= 0:
        return {"to_call": None, "is_lower_bound": True,
                "source": "no_opponent_bet", "detail": detail}
    lb = round(max(0.0, opp_max - hero_bet), 6)      # 去浮点噪声（0.12−0.05）
    if lb <= 0:
        return {"to_call": None, "is_lower_bound": True,
                "source": "hero_already_matched", "detail": detail}
    return {"to_call": lb, "is_lower_bound": True,
            "source": "lower_bound_from_seat_bets", "detail": detail}


def facing_unverifiable(obs: Any) -> dict:
    """是否**无法确认**"hero 是否面对下注"（且 hero 桌上已有投入）。

    场景（实测第二张真实帧）：hero 已加注（桌上有 0.15 属于 hero）、
    对手的加注筹码（0.25）**漏读** ⇒ `to_call` 未知、对手投入不可见
    ⇒ 引擎按"无人下注"算 ⇒ 输出 `bet 8bb`（对当时状态毫无意义）。

    判据：`to_call` 未知 **且** 无任何非 hero 座正投入 **且** hero 有正投入。
    返回 {"unverifiable": bool, "reason": str, "detail": {...}}。
    """
    o = obs if isinstance(obs, dict) else {}
    if _num(o.get("to_call")) is not None:
        return {"unverifiable": False, "reason": "to_call_known", "detail": {}}
    extra = o.get("extra") if isinstance(o.get("extra"), dict) else {}
    bets = extra.get(BET_SEATS_KEY)
    hb = hero_opp_bets(o)
    hero_bet, opp_max = hb["hero_bet"], hb["opp_max"]
    detail = {**hb["detail"], "hero_bet": hero_bet,
              "max_opponent_bet": opp_max}
    _st = stale_blind_bet(o)
    if _st.get("stale"):
        detail["stale_blind_bet"] = _st
        return {"unverifiable": True, "reason": "postflop_bet_looks_like_leftover",
                "detail": detail, "version": EVIDENCE_VERSION}
    if opp_max > 0:
        return {"unverifiable": False, "reason": "has_opponent_bet_evidence",
                "detail": detail}
    if hero_bet > 0:
        # hero 已有投入但看不到对手投入 ⇒ 可能是"已行动/面对加注"且对手筹码漏读
        return {"unverifiable": True,
                "reason": "hero_bet_present_no_opponent_bet_visible",
                "detail": detail}
    # 完全看不到任何投入：可能"无人下注"，也可能整体漏读 ⇒ 不可确认（温和）。
    # 但若观察里连状态都没有（空观察）⇒ 属"没有观察"，不在此判定（调用方自会处理）。
    if o.get("street") or o.get("hero_seat") or o.get("pot") is not None:
        return {"unverifiable": True,
                "reason": "no_bet_visible_to_call_unknown", "detail": detail}
    return {"unverifiable": False, "reason": "no_observation", "detail": detail}


#: 1876 bet 环 → **座位几何角色**（三帧裁图实测，2026-10-10）：
#:   S1(440,455)=左上 · S2(680,390)=上中 · S6(875,790)=**左下**
#:   S3(1316,450)=右上 · S4(1036,933)=**底部中间（hero）** · S5(1350,886)=右下
#: 说明：bet 环的键与 `hero_seat` 的座位环**不是同一套编号**（实测 hero_seat=5
#: 而 hero 的筹码由 bet 环的 "S4" 位置读出）⇒ 归属必须按几何，不能按编号。
BET_ZONE_ROLE_1876 = {
    "S1": "top_left", "S2": "top_center", "S3": "top_right",
    "S6": "bottom_left", "S4": "bottom_center_hero", "S5": "bottom_right",
}
#: 1876 中 hero 的投注区键（底部中间）
HERO_BET_KEYS_1876 = ("S4",)


def looks_like_1876_bet_ring(bets: Any) -> bool:
    """公开别名：见 `_looks_like_1876_bet_ring`（严格判据）。"""
    return _looks_like_1876_bet_ring(bets)


def _looks_like_1876_bet_ring(bets: Any) -> bool:
    """是否 1876 的六区 bet 环：键 ⊆ {S1..S6} 且 **至少 4 个区**。

    （收紧原因：{S2,S5} 这类不完整环在旧判据下会被误当作 1876 环 ⇒ 归属错位。）
    """
    if not isinstance(bets, dict) or len(bets) < 4:
        return False
    return set(str(k) for k in bets) <= {"S1", "S2", "S3", "S4", "S5", "S6"}


def hero_opp_bets(obs: Any) -> dict:
    """区分 **hero 自身投入** 与 **对手最高投入**（按几何归属；非按座位编号）。

    1876：hero 恒在底部中间 ⇒ 其投注区键为 `HERO_BET_KEYS_1876`；
    其它布局：退化为按 `hero_seat` 匹配（原行为）。

    返回 {"hero_bet", "opp_max", "n_readable", "mapping", "detail"}。
    """
    o = obs if isinstance(obs, dict) else {}
    extra = o.get("extra") if isinstance(o.get("extra"), dict) else {}
    bets = extra.get(BET_SEATS_KEY)
    detail = {"v3f_bets": ({str(k): v for k, v in bets.items()}
                            if isinstance(bets, dict) else None)}
    if not isinstance(bets, dict) or not bets:
        return {"hero_bet": 0.0, "opp_max": 0.0, "n_readable": 0,
                "mapping": "none", "detail": detail}
    hero_bet, opp_max, seen = 0.0, 0.0, 0
    _bad = set(implausible_preflop_posts(o).get("seats") or [])
    if _bad:
        detail["implausible_preflop_posts"] = sorted(_bad)
    if _looks_like_1876_bet_ring(bets):
        mapping = "geometry_1876"
        for k, v in bets.items():
            n = _num(v)
            if n is None:
                continue                      # None = 该区读不到（≠ 0）
            seen += 1                         # 显式 0 = 可读的"无投入"
            if n <= 0:
                continue
            if str(k) in _bad:                # 翻前 < 小盲 ⇒ 误读，排除
                continue
            if str(k) in HERO_BET_KEYS_1876:
                hero_bet = max(hero_bet, n)
            else:
                opp_max = max(opp_max, n)
    else:
        mapping = "hero_seat"
        hero_seat = _seat_of_bet_key(o.get("hero_seat"))
        if hero_seat is None:
            return {"hero_bet": 0.0, "opp_max": 0.0, "n_readable": 0,
                    "mapping": "hero_seat_unknown", "detail": detail}
        for k, v in bets.items():
            n = _num(v)
            if n is None:
                continue
            seen += 1
            if n <= 0:
                continue
            if str(k) in _bad:
                continue
            if _seat_of_bet_key(k) == hero_seat:
                hero_bet = max(hero_bet, n)
            else:
                opp_max = max(opp_max, n)
    # P30-9-AG：hero 的**强制投入下限**——翻前 SB/BB 的投入不可能低于对应盲注。
    # 实测用户帧：hero=BB、bb=0.15，却把 hero 区读成 0.05（=SB 金额）⇒
    # 若直接用会得到 to_call = 0.2-0.05 = 0.15（偏大 3 倍）。故用**已知盲注**做下限，
    # 并如实记录（不发明数值：盲注来自盲注证据）。
    try:
        street = str(o.get("street") or "").lower()
        pos = str((o.get("extra") or {}).get("v3f_hero_position") or "").upper()
        bl = o.get("blinds") if isinstance(o.get("blinds"), dict) else {}
        floor = None
        if street == "preflop" and pos == "SB":
            floor = _num(bl.get("sb"))
        elif street == "preflop" and pos == "BB":
            floor = _num(bl.get("bb"))
        if floor and hero_bet < floor - 1e-9:
            detail["hero_post_floor_applied"] = {
                "read": hero_bet, "floor": floor, "position": pos,
                "note": "翻前强制投入 ≥ 盲注；读数偏低按盲注计（已记录）"}
            hero_bet = float(floor)
    except Exception:                              # noqa: BLE001
        pass
    detail.update({"hero_seat": o.get("hero_seat"), "mapping": mapping,
                   "hero_keys": list(HERO_BET_KEYS_1876) if mapping == "geometry_1876"
                   else None})
    return {"hero_bet": round(hero_bet, 6), "opp_max": round(opp_max, 6),
            "n_readable": seen, "mapping": mapping, "detail": detail}


def stale_blind_bet(obs: Any) -> dict:
    """翻后是否只有"与盲注等值"的对手投入（疑似**上一街残留/名牌筹码**）。

    实测用户反馈：「翻后公对、对手没下注，却让我 fold」——最可能的解释是引擎把
    **盲注等值的残留筹码**当成了对手下注。此函数只做判定，用于**保守化**：
    翻后且（面板 to_call 未知）且对手最高投入**恰等于 bb**（或恰等于 hero 自己的投入）
    ⇒ 视为**无法确认面对下注**（不据此 fold）。
    """
    o = obs if isinstance(obs, dict) else {}
    street = str(o.get("street") or "").lower()
    out = {"stale": False, "reason": None}
    if street in ("", "preflop"):
        return out
    hb = hero_opp_bets(o)
    blind = o.get("blinds") if isinstance(o.get("blinds"), dict) else {}
    bb = _num(blind.get("bb"))
    opp, hero = float(hb["opp_max"]), float(hb["hero_bet"])
    if opp <= 0:
        return out
    if bb and abs(opp - bb) < 1e-9:
        out.update({"stale": True, "reason": "opponent_bet_equals_big_blind",
                    "opp_max": opp, "bb": bb})
    elif hero > 0 and abs(opp - hero) < 1e-9:
        out.update({"stale": True, "reason": "opponent_bet_equals_hero_commit",
                    "opp_max": opp, "hero_bet": hero})
    return out


def hero_is_max_contributor(obs: Any) -> dict:
    """hero 是否**当前最大投入者**（⇒ 很可能还没轮到他，或在等对手行动）。

    仅依据**已识别**的投入（几何归属）；无法判断 ⇒ False（不臆断）。
    返回 {"is_max": bool, "hero_bet", "opp_max", "detail"}。
    """
    o = obs if isinstance(obs, dict) else {}
    hb = hero_opp_bets(o)
    hero_bet, opp_max = float(hb["hero_bet"]), float(hb["opp_max"])
    is_max = hero_bet > 0 and hero_bet >= opp_max
    return {"is_max": bool(is_max), "hero_bet": hero_bet, "opp_max": opp_max,
            "detail": hb["detail"]}


def implausible_preflop_posts(obs: Any) -> dict:
    """翻前任何人的投入都**不可能小于小盲**（SB 强制投入）。

    实测用户帧：`blinds 0.05/0.15` 而某座读到 `0.02` ⇒ 属误读，必须**排除**，
    否则会污染 hero_bet / 对手最高投入 / to_call 下界。
    返回 {"seats": [..], "min_ok": sb, "detail": {...}}（只读，不改任何观察字段）。
    """
    o = obs if isinstance(obs, dict) else {}
    street = str(o.get("street") or "").lower()
    extra = o.get("extra") if isinstance(o.get("extra"), dict) else {}
    bets = extra.get(BET_SEATS_KEY) if isinstance(extra, dict) else None
    blinds = o.get("blinds") if isinstance(o.get("blinds"), dict) else {}
    sb = _num(blinds.get("sb"))
    out = {"seats": [], "min_ok": sb, "detail": {"street": street, "sb": sb}}
    if street != "preflop" or not sb or not isinstance(bets, dict):
        return out
    for k, v in bets.items():
        n = _num(v)
        if n is None or n <= 0:
            continue
        if n < sb - 1e-9:
            out["seats"].append(str(k))
    out["detail"]["bad"] = {k: bets.get(k) for k in out["seats"]}
    return out


def hero_cards_diagnosis(obs: Any) -> dict:
    """hero 牌识别诊断（**只读派生**，供 GUI/CLI 显示"为什么没识别到"）。"""
    o = obs if isinstance(obs, dict) else {}
    cards = o.get("hole_cards")
    n = len(cards) if isinstance(cards, (list, tuple)) else 0
    fc = o.get("field_confidences") if isinstance(o.get("field_confidences"), dict) else {}
    conf = _num(fc.get("hero_cards"))
    hero_seat = o.get("hero_seat")
    extra = o.get("extra") if isinstance(o.get("extra"), dict) else {}
    dealer = _num(o.get("button_seat"))
    reason = None
    if n == 2:
        reason = None
    elif hero_seat is None:
        reason = "hero_seat_unrecognized"
    elif dealer is None:
        reason = "dealer_button_unrecognized"
    elif n == 0:
        reason = "hero_cards_region_no_match"
    elif n == 1:
        reason = "hero_cards_partial_match"
    # P30-9-AF：透出**观察者自身**的原因（no_card_pair_present /
    # card_recognition_incomplete / glyph_gate_failed / red_suit_color_mismatch …）
    observer_reason = None
    for wr in (o.get("warnings") or []):
        s = str(wr)
        if s.startswith("hero_cards") and "reason" in s:
            observer_reason = s.split("reason", 1)[1].lstrip("=: ").strip()
            break
    if observer_reason is None:
        for wr in (o.get("warnings") or []):
            s = str(wr)
            if "hero_cards" in s:
                observer_reason = s
                break
    return {
        "n_cards": n, "confidence": conf, "hero_seat": hero_seat,
        "button_seat": dealer,
        "hero_position": extra.get("v3f_hero_position"),
        "action_panel_status": extra.get("v3f_status"),
        "reason": reason,
        "observer_reason": observer_reason,
        "ok": n == 2,
        "version": EVIDENCE_VERSION,
    }


def evidence_contract() -> dict:
    return {
        "version": EVIDENCE_VERSION,
        "blinds_priority": ["observed", "derived_from_posted_blinds",
                             "user_setting（GUI 手动输入）", "session_memo（同桌同局，TTL 900s）",
                             "window_title（OS 元数据）", "no_evidence"],
        "blinds_memo_ttl_s": BLINDS_MEMO_TTL_S,
        "blinds_consistency_guard": True,
        "priority": ["available_actions", "to_call", "extra.v3f_bets", "UNKNOWN"],
        "synthesis": {"BET_PENDING": ["fold", "call", "raise"], "NO_BET": ["check", "bet"]},
        "fabricates": False,
        "frozen_layers_modified": False,
    }
