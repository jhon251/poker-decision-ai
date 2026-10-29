# -*- coding: utf-8 -*-
"""P30-9-HY：**对手档案库**（每次点击识别 ⇒ 按对手 ID 记录逐座位证据；本地 SQLite ✓）。

设计红线（照 P30 纪律）：
  * **只记录真实可见的字段** ✓（普查结论：`seats[]` 有 seat/name/stack/in_hand ✓；
    `extra.v3f_bets` 有逐座位**前置下注额** ✓；而 `v3f_action_history` **恒 None** ✗、
    `v3f_status` 值**恒 UNKNOWN** ✗、`seats[].status` **不存在** ✗）
    ⇒ **单帧无法得到 fold/check/raise/call 标签** ⇒ 绝不虚构动作序列 ✗；
  * 动作线索只能**跨遭遇推导**（筹码减少=投入、前置下注增加、in_hand→消失=弃牌证据 ✓），
    每条都带 `basis`（依据）+ `confidence`（置信）+ 标为**推断** ✓；
  * 身份键用可见 `name` ✓；缺失 ⇒ `seat:<n>` 且 `name_missing=1` ✓（**不编造** ✓）；
  * 严格**旁路**：失败静默、不阻塞点击 ✓、可 `OPPONENT_DB=0` 关闭 ✓；
  * DB 默认 `data/opponents.db`（EXE 下即 `dist/data/opponents.db` ✓，**仅本机** ✓）。

表结构：
  `encounters`   —— 一次点击 = 一行（street/board/pot/to_call/blinds/button/hero/来源/帧哈希 ✓）
  `observations` —— 一次点击 × 每个可见座位 = 一行（身份/位置/筹码/在池/前置下注 ✓）
  `players`      —— 身份聚合（首次/最近出现、遭遇数 ✓）
  `transitions`  —— 同一手牌内**相邻遭遇**的迁移推断（投入/弃牌证据 ✓，带依据与置信 ✓）
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import sqlite3
import time
from typing import Any

POS6 = ["BTN", "SB", "BB", "UTG", "MP", "CO"]
DEFAULT_PATH = "data/opponents.db"


def enabled() -> bool:
    return str(os.environ.get("OPPONENT_DB", "1")).lower() not in (
        "0", "off", "false", "no")


def db_path() -> pathlib.Path:
    p = pathlib.Path(os.environ.get("OPPONENT_DB_PATH") or DEFAULT_PATH)
    if p.is_absolute():
        return p
    base = pathlib.Path(__file__).resolve().parents[2]        # 仓库/EXE 根 ✓
    return base / p


def source_tag() -> str:
    """来源三级判定（与证据采集一致 ✓）：显式 → pytest ⇒ test → runtime ✓。"""
    return (os.environ.get("DSH_EVIDENCE_SOURCE")
             or ("test" if os.environ.get("PYTEST_CURRENT_TEST") else "runtime"))


def _connect(path: pathlib.Path | None = None) -> sqlite3.Connection:
    p = pathlib.Path(path or db_path())
    p.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(p), timeout=3.0)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    return con


SCHEMA = """
CREATE TABLE IF NOT EXISTS players (
  player_key TEXT PRIMARY KEY, name TEXT, name_missing INTEGER DEFAULT 0,
  first_seen TEXT, last_seen TEXT, encounters INTEGER DEFAULT 0);
CREATE TABLE IF NOT EXISTS encounters (
  click_id TEXT PRIMARY KEY, ts TEXT, source TEXT, street TEXT, board TEXT,
  pot REAL, to_call REAL, sb REAL, bb REAL, button_seat INTEGER, hero_seat INTEGER,
  n_seats INTEGER, frame_sha16 TEXT, warnings TEXT);
CREATE TABLE IF NOT EXISTS observations (
  obs_id INTEGER PRIMARY KEY AUTOINCREMENT, click_id TEXT, player_key TEXT, name TEXT,
  name_missing INTEGER, seat INTEGER, is_hero INTEGER, position TEXT, stack REAL,
  in_hand INTEGER, bet_front REAL, field_conf REAL, street TEXT, board TEXT, ts TEXT);
CREATE INDEX IF NOT EXISTS ix_obs_player ON observations(player_key);
CREATE INDEX IF NOT EXISTS ix_obs_click ON observations(click_id);
CREATE TABLE IF NOT EXISTS transitions (
  trans_id INTEGER PRIMARY KEY AUTOINCREMENT, player_key TEXT, from_click TEXT, to_click TEXT,
  from_street TEXT, to_street TEXT, stack_delta REAL, betfront_delta REAL,
  inhand_to_out INTEGER, kind TEXT, basis TEXT, confidence REAL, ts TEXT);
CREATE INDEX IF NOT EXISTS ix_trans_player ON transitions(player_key);
-- P30-9-IA：**动作标签**（从可见筹码证据推导 ✓，每条带依据 + 置信 + 证据 JSON ✓）
CREATE TABLE IF NOT EXISTS actions (
  act_id INTEGER PRIMARY KEY AUTOINCREMENT, player_key TEXT, from_click TEXT, to_click TEXT,
  street TEXT, action TEXT, basis TEXT, confidence REAL, evidence TEXT, ts TEXT);
CREATE INDEX IF NOT EXISTS ix_act_player ON actions(player_key);
"""


# ── 提取（只取真实可见字段 ✓，缺失一律 None ✓）──────────────────────────
def _num(v: Any) -> float | None:
    if isinstance(v, bool) or v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    try:
        return float(str(v))
    except (TypeError, ValueError):
        return None


def _pos_for(seat: int | None, button: int | None, n: int) -> str | None:
    if seat is None or button is None or not n:
        return None
    off = (int(seat) - int(button)) % n
    return POS6[off] if 0 <= off < len(POS6) else None


def _bets_map(obs: dict) -> dict:
    """`extra.v3f_bets` ⇒ {seat_int: chips}（键可能是 `S1` 或 `1` ✓，值可为 null ✓）。"""
    ex = obs.get("extra") or {}
    raw = ex.get("v3f_bets")
    if not isinstance(raw, dict):
        return {}
    out: dict[int, float] = {}
    for k, v in raw.items():
        ks = str(k)
        if ks.startswith(("S", "s")):
            ks = ks[1:]
        try:
            seat = int(ks)
        except ValueError:
            continue
        val = _num(v)
        if val is not None:
            out[seat] = val
    return out


def extract(obs: dict, frame_rgba: bytes | None = None) -> dict:
    """把一次观测抽成可入库的结构（**不推断、不补值** ✓）。"""
    seats = [s for s in (obs.get("seats") or []) if isinstance(s, dict)]
    bets = _bets_map(obs)
    btn = obs.get("button_seat")
    hero = obs.get("hero_seat")
    n = len(seats) or 6
    conf = obs.get("field_confidences")
    rows = []
    for s in seats:
        seat = s.get("seat")
        name = s.get("name")
        name_s = str(name).strip() if name is not None else ""
        missing = 0 if name_s else 1
        key = name_s if name_s else (f"seat:{seat}" if seat is not None else "seat:?")
        rows.append({
            "player_key": key, "name": name_s or None, "name_missing": missing,
            "seat": int(seat) if isinstance(seat, int) else None,
            "is_hero": int(seat == hero) if (seat is not None and hero is not None) else 0,
            "position": _pos_for(seat if isinstance(seat, int) else None,
                                  int(btn) if isinstance(btn, int) else None, n),
            "stack": _num(s.get("stack")),
            "in_hand": int(bool(s.get("in_hand"))),
            "bet_front": bets.get(seat) if isinstance(seat, int) else None,
            "field_conf": (conf if isinstance(conf, (int, float)) else None),
        })
    blinds = obs.get("blinds") or {}
    sha = hashlib.sha256(bytes(frame_rgba)).hexdigest()[:16] if frame_rgba else None
    return {"obs": obs, "rows": rows, "sha16": sha,
             "enc": {"street": str(obs.get("street") or ""),
                      "board": "".join(str(c) for c in (obs.get("board") or [])),
                      "pot": _num(obs.get("pot")), "to_call": _num(obs.get("to_call")),
                      "sb": _num(blinds.get("sb")), "bb": _num(blinds.get("bb")),
                      "button_seat": int(btn) if isinstance(btn, int) else None,
                      "hero_seat": int(hero) if isinstance(hero, int) else None,
                      "n_seats": n,
                      "warnings": json.dumps([str((w or {}).get("code"))
                                               for w in (obs.get("warnings") or [])
                                               if isinstance(w, dict)],
                                              ensure_ascii=False)[:500]}}


# ── 同一手牌判定（启发式，**显式写出依据** ✓）───────────────────────────
def _same_hand(prev_enc: dict, prev_ts: float, enc: dict, ts: float,
                window_s: float = 900.0) -> bool:
    """同一手牌：**庄家位与 hero 位相同** 且 时间在窗口内 ✓（翻牌延续 ⇒ 按钮不变 ✓）。"""
    if not prev_enc or abs(ts - prev_ts) > window_s:
        return False
    return (prev_enc.get("button_seat") == enc.get("button_seat")
             and prev_enc.get("hero_seat") == enc.get("hero_seat"))


# ── 动作标签分类（P30-9-IA）──────────────────────────────────────────────
# 只依据**可见筹码证据**推导 ✓；每条带 basis（依据）+ confidence（置信）+ evidence（原文数字）✓。
# 规则（与 `docs/OPPONENT_DB.md` 一致 ✓）：
#   fold  : 在池 1→0 ✓（0.6）
#   allin : 筹码 ≈ 0 ✓（0.8），或前置下注 ≥ 上一帧全部筹码 ✓（0.7）
#   bet   : 本街前置下注 0→x，且**上一帧他人无下注** ✓（0.7；筹码池同向增长 +0.1）
#   raise : 前置下注增加且**超过**上一帧他人最大下注 ✓（0.7）
#   call  : 前置下注增加且**等于**上一帧他人最大下注 ✓（0.7）
#   blind : **翻前** SB/BB 的强制投入 ✓（0.5，**不算** call ✗）
#
# ⚠️ **刻意不支持 `check`** ✗：实测两帧稀疏序列**无法**可靠判定过牌 ——
#   `bet_front` 每街清零 ⇒ 翻前跟注者在翻牌显示 0 ✗，而单帧也看不到**行动顺序** ✗
#   ⇒ 强行标注会产出**假阳性**（冒烟实测：没参与的玩家被标成 check ✗✗）。
#   ⇒ 宁可少一个标签，也不编造 ✓。要支持它需要**动作历史/多帧序列**（视觉未提供 ✗）。
ACTION_LABELS = ("fold", "allin", "bet", "raise", "call", "blind")


def classify_action(prev: dict, cur: dict, *, prev_max_bet_others: float | None = None,
                     cur_max_bet_others: float | None = None,
                     street_changed: bool = False, pot_delta: float | None = None,
                     bb: float | None = None, others_acted: bool = False) -> list[dict]:
    """从两次相邻遭遇的可见证据推导动作标签（**可多条**：如 allin 亦可同时是 bet ✓）。

    ⚠️ **换街处理** ✓：`bet_front` 每街清零 ⇒ 换街时**上一街的下注数值不可用于本街比较** ✗，
    故本街起点按 0 计 ✓；本街的 bet/call 判断改用**本帧他人最大下注**（`cur_max_bet_others` ✓）
    并降低置信（0.6/0.5 ✓），因为**同帧的行动顺序不可见** ✗。
    `others_acted` 目前不参与判定 ✓（保留参数以备将来接入动作历史 ✓）。
    """
    out: list[dict] = []
    tol = max(1.0, (bb or 1.0) * 0.05)                       # 容差：5% 大盲或 1 筹码 ✓
    p_stack = _num(prev.get("stack"))
    c_stack = _num(cur.get("stack"))
    p_bet_raw = _num(prev.get("bet_front")) or 0.0
    c_bet = _num(cur.get("bet_front")) or 0.0
    p_in = int(prev.get("in_hand") or 0)
    c_in = int(cur.get("in_hand") or 0)
    street = str(cur.get("street") or "")
    pos = str(cur.get("position") or prev.get("position") or "")
    mx = _num(prev_max_bet_others)
    mx_cur = _num(cur_max_bet_others)
    # 换街 ⇒ 本街起点按 0 计 ✓（上一街数值不可比 ✗）
    p_bet = 0.0 if street_changed else p_bet_raw
    ev = {"prev_stack": p_stack, "cur_stack": c_stack, "prev_bet": p_bet_raw, "cur_bet": c_bet,
           "prev_in_hand": p_in, "cur_in_hand": c_in, "prev_max_bet_others": mx,
           "cur_max_bet_others": mx_cur, "street_changed": bool(street_changed),
           "pot_delta": pot_delta, "street": street, "position": pos, "tol": tol}

    # ① 弃牌 ✓（最强证据：在池 → 不在池）
    if p_in == 1 and c_in == 0:
        out.append({"action": "fold", "basis": "in_hand_to_out", "confidence": 0.6,
                     "evidence": ev})
        return out                                          # 已弃牌 ⇒ 不再推其它 ✓

    # ② 翻前强制投入 ⇒ blind（**不得**误记为 call ✗）
    if (street == "preflop" and pos in ("SB", "BB") and p_bet_raw <= tol
            and prev.get("bet_front") is None):
        out.append({"action": "blind", "basis": "forced_bet_preflop", "confidence": 0.5,
                     "evidence": ev})

    # ③ 全下 ✓（两条独立证据任一条成立 ✓）
    if bb and c_stack is not None and c_stack <= 0.5 * bb:
        out.append({"action": "allin", "basis": "stack_near_zero", "confidence": 0.8,
                     "evidence": ev})
    elif p_stack is not None and c_bet > 0 and c_bet >= p_stack - tol:
        out.append({"action": "allin", "basis": "bet_covers_whole_stack", "confidence": 0.7,
                     "evidence": ev})

    # ④ 本街投入增加 ⇒ bet / raise / call ✓
    if c_bet > p_bet + tol:
        if street_changed:
            # 换街：同帧顺序不可见 ✗ ⇒ 用**本帧他人最大下注**判断，置信略低 ✓
            if mx_cur is None or mx_cur <= tol or c_bet > mx_cur + tol:
                out.append({"action": "bet",
                             "basis": "street_open_bet_no_higher_bet_current",
                             "confidence": 0.6, "evidence": ev})
            elif abs(c_bet - mx_cur) <= tol:
                out.append({"action": "call", "basis": "street_bet_matches_current_max",
                             "confidence": 0.5, "evidence": ev})
            # 介于两者之间 ⇒ **不标** ✓（宁可漏，不编造 ✗）
        elif mx is None or mx <= tol:                        # 同街且此前无人下注 ⇒ 首个下注 ✓
            out.append({"action": "bet", "basis": "bet_front_opened_no_prior_bet",
                         "confidence": 0.7 + (0.1 if (pot_delta or 0) > 0 else 0.0),
                         "evidence": ev})
        elif c_bet > mx + tol:                               # 超过他人下注 ⇒ 加注 ✓
            out.append({"action": "raise", "basis": "bet_exceeds_prev_max_bet",
                         "confidence": 0.7, "evidence": ev})
        elif abs(c_bet - mx) <= tol:                          # 与他人下注持平 ⇒ 跟注 ✓
            out.append({"action": "call", "basis": "bet_matches_prev_max_bet",
                         "confidence": 0.7, "evidence": ev})
    return out


# ── 记录（一次性写 encounter + observations + transitions ✓）────────────
def record_encounter(obs: dict, *, frame_rgba: bytes | None = None,
                      click_id: str | None = None,
                      path: pathlib.Path | None = None) -> dict:
    """记录一次"点击识别"。返回统计（写入行数等 ✓）；**任何失败都静默**（旁路 ✓）。"""
    res = {"ok": False, "observations": 0, "transitions": 0, "players": 0}
    if not enabled():
        res["reason"] = "disabled(OPPONENT_DB=0)"
        return res
    try:
        ex = extract(obs, frame_rgba)
        if not ex["rows"]:
            res["reason"] = "no_seats"
            return res
        cid = click_id or f"{int(time.time() * 1000)}-{ex['sha16'] or 'noframe'}"
        ts_iso = time.strftime("%Y-%m-%dT%H:%M:%S")
        ts_epoch = time.time()
        con = _connect(path)
        try:
            con.executescript(SCHEMA)
            con.execute(
                "INSERT OR REPLACE INTO encounters(click_id, ts, source, street, board, pot,"
                " to_call, sb, bb, button_seat, hero_seat, n_seats, frame_sha16, warnings)"
                " VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (cid, ts_iso, source_tag(), ex["enc"]["street"], ex["enc"]["board"],
                 ex["enc"]["pot"], ex["enc"]["to_call"], ex["enc"]["sb"], ex["enc"]["bb"],
                 ex["enc"]["button_seat"], ex["enc"]["hero_seat"], ex["enc"]["n_seats"],
                 ex["sha16"], ex["enc"]["warnings"]))
            # 上一遭遇（用于同一手牌判定 ✓）
            prev = con.execute(
                "SELECT click_id, ts, street, button_seat, hero_seat FROM encounters"
                " WHERE click_id != ? ORDER BY ts DESC, rowid DESC LIMIT 1", (cid,)).fetchone()
            prev_enc = ({"button_seat": prev[3], "hero_seat": prev[4], "street": prev[2]}
                         if prev else {})
            prev_obs: dict[str, dict] = {}
            prev_pot = None
            if prev:
                prev_pot = con.execute("SELECT pot FROM encounters WHERE click_id=?",
                                        (prev[0],)).fetchone()
                prev_pot = prev_pot[0] if prev_pot else None
                for r in con.execute(
                        "SELECT player_key, stack, in_hand, bet_front, street, position"
                        " FROM observations WHERE click_id = ?", (prev[0],)):
                    prev_obs[str(r[0])] = {"stack": r[1], "in_hand": r[2], "bet_front": r[3],
                                            "street": r[4], "position": r[5]}
            # P30-9-IA：本帧"别人是否行动"（用于 check 的低置信判定 ✓）
            cur_by_key = {str(r["player_key"]): r for r in ex["rows"]}
            acted_now: set[str] = set()
            for k, r in cur_by_key.items():
                p0 = prev_obs.get(k)
                if not p0:
                    continue
                b0 = _num(p0.get("bet_front")) or 0.0
                b1 = _num(r.get("bet_front")) or 0.0
                s0 = _num(p0.get("stack"))
                s1 = _num(r.get("stack"))
                if b1 > b0 + 1e-9 or (s0 is not None and s1 is not None and s1 < s0 - 1e-9):
                    acted_now.add(k)
            pot_delta = None
            if prev_pot is not None and ex["enc"]["pot"] is not None:
                pot_delta = ex["enc"]["pot"] - float(prev_pot)
            prev_ts = ts_epoch - 3600.0
            if prev:
                try:
                    prev_ts = time.mktime(time.strptime(str(prev[1]), "%Y-%m-%dT%H:%M:%S"))
                except Exception:                              # noqa: BLE001
                    prev_ts = ts_epoch - 3600.0
            for row in ex["rows"]:
                con.execute(
                    "INSERT INTO observations(click_id, player_key, name, name_missing, seat,"
                    " is_hero, position, stack, in_hand, bet_front, field_conf, street, board, ts)"
                    " VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (cid, row["player_key"], row["name"], row["name_missing"], row["seat"],
                     row["is_hero"], row["position"], row["stack"], row["in_hand"],
                     row["bet_front"], row["field_conf"], ex["enc"]["street"],
                     ex["enc"]["board"], ts_iso))
                res["observations"] += 1
                con.execute(
                    "INSERT INTO players(player_key, name, name_missing, first_seen, last_seen,"
                    " encounters) VALUES(?,?,?,?,?,1)"
                    " ON CONFLICT(player_key) DO UPDATE SET last_seen=excluded.last_seen,"
                    " encounters=players.encounters+1,"
                    " name=COALESCE(players.name, excluded.name)",
                    (row["player_key"], row["name"], row["name_missing"], ts_iso, ts_iso))
                res["players"] += 1
                # ── 迁移推断（仅同一手牌 ✓；每条带依据与置信 ✓，绝不虚构 ✗）──
                p = prev_obs.get(str(row["player_key"]))
                if p and _same_hand(prev_enc, prev_ts, ex["enc"], ts_epoch):
                    p_stack = p.get("stack")
                    p_in = p.get("in_hand")
                    p_bet = p.get("bet_front")
                    p_street = p.get("street")
                    trans = []
                    if (p_stack is not None and row["stack"] is not None
                            and row["stack"] < p_stack - 1e-9):
                        trans.append(("invest", "stack_decrease",
                                       min(1.0, (p_stack - row["stack"]) / max(p_stack, 1.0)),
                                       (p_stack - row["stack"])))
                    if (p_bet is not None and row["bet_front"] is not None
                            and row["bet_front"] > p_bet + 1e-9):
                        trans.append(("invest", "bet_front_increase",
                                       min(1.0, (row["bet_front"] - p_bet)
                                            / max(abs(p_bet) + 1.0, 1.0)),
                                       (row["bet_front"] - p_bet)))
                    if int(p_in or 0) == 1 and int(row["in_hand"] or 0) == 0:
                        trans.append(("fold_evidence", "in_hand_to_out", 0.6, None))
                    for kind, basis, conf_v, delta in trans:
                        con.execute(
                            "INSERT INTO transitions(player_key, from_click, to_click,"
                            " from_street, to_street, stack_delta, betfront_delta,"
                            " inhand_to_out, kind, basis, confidence, ts)"
                            " VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                            (row["player_key"], prev[0], cid, p_street, ex["enc"]["street"],
                             (delta if basis == "stack_decrease" else None),
                             (delta if basis == "bet_front_increase" else None),
                             (1 if kind == "fold_evidence" else 0), kind, basis, conf_v,
                             ts_iso))
                        res["transitions"] += 1
                    # ── P30-9-IA：**动作标签**（可见筹码证据 ⇒ fold/check/call/bet/raise/allin ✓）──
                    _others = [(_num(o.get("bet_front")) or 0.0)
                                for k2, o in prev_obs.items() if k2 != str(row["player_key"])]
                    _mx = max(_others) if _others else 0.0
                    _others_cur = [(_num(o.get("bet_front")) or 0.0)
                                    for k2, o in cur_by_key.items()
                                    if k2 != str(row["player_key"])]
                    _mx_cur = max(_others_cur) if _others_cur else 0.0
                    _labels = classify_action(
                        p, {"stack": row["stack"], "in_hand": row["in_hand"],
                             "bet_front": row["bet_front"], "street": ex["enc"]["street"],
                             "position": row["position"]},
                        prev_max_bet_others=_mx,
                        cur_max_bet_others=_mx_cur,
                        street_changed=(str(p_street or "") != ex["enc"]["street"]),
                        pot_delta=pot_delta,
                        bb=ex["enc"]["bb"],
                        others_acted=any(k2 in acted_now
                                          for k2 in acted_now
                                          if k2 != str(row["player_key"])))
                    for lab in _labels:
                        con.execute(
                            "INSERT INTO actions(player_key, from_click, to_click, street,"
                            " action, basis, confidence, evidence, ts) VALUES(?,?,?,?,?,?,?,?,?)",
                            (row["player_key"], prev[0], cid, ex["enc"]["street"],
                             lab["action"], lab["basis"], lab["confidence"],
                             json.dumps(lab["evidence"], ensure_ascii=False), ts_iso))
                        res.setdefault("actions", 0)
                        res["actions"] += 1
                        res.setdefault("action_labels", {})
                        res["action_labels"][lab["action"]] = \
                            res["action_labels"].get(lab["action"], 0) + 1
            con.commit()
        finally:
            con.close()
        res["ok"] = True
        res["click_id"] = cid
    except Exception as e:                                     # noqa: BLE001
        res["reason"] = f"{type(e).__name__}"                  # 旁路：静默失败 ✓
    return res


def stats(path: pathlib.Path | None = None, min_sample: int = 20) -> dict:
    """读库汇总（**每项都带样本量** ✓；样本不足显式标注 ✓）。"""
    p = pathlib.Path(path or db_path())
    if not p.is_file():
        return {"db": str(p), "exists": False}
    con = _connect(p)
    try:
        con.executescript(SCHEMA)
        n_enc = con.execute("SELECT COUNT(*) FROM encounters").fetchone()[0]
        n_obs = con.execute("SELECT COUNT(*) FROM observations").fetchone()[0]
        n_pl = con.execute("SELECT COUNT(*) FROM players").fetchone()[0]
        n_tr = con.execute("SELECT COUNT(*) FROM transitions").fetchone()[0]
        by_src = dict(con.execute("SELECT source, COUNT(*) FROM encounters GROUP BY source"))
        rows = []
        for r in con.execute(
                "SELECT o.player_key, COUNT(*) AS enc, SUM(o.in_hand) AS inhand,"
                " (SELECT COUNT(*) FROM transitions t WHERE t.player_key=o.player_key"
                "   AND t.kind='fold_evidence') AS folds,"
                " (SELECT COUNT(*) FROM transitions t WHERE t.player_key=o.player_key"
                "   AND t.kind='invest') AS invests,"
                " MAX(o.name), MAX(o.position), MAX(o.stack)"
                " FROM observations o GROUP BY o.player_key ORDER BY enc DESC LIMIT 50"):
            _pk = r[0]
            # P30-9-HZ：**各位置 / 各街分布** ✓（每项都带分母 = 该对手遭遇数 ✓；
            #   这是**遭遇分布**，不是频率推断 ✗）
            by_pos = {str(k) if k is not None else "UNKNOWN": v for k, v in con.execute(
                "SELECT position, COUNT(*) FROM observations WHERE player_key=?"
                " GROUP BY position ORDER BY 2 DESC", (_pk,))}
            by_street = {str(k) if k is not None else "UNKNOWN": v for k, v in con.execute(
                "SELECT street, COUNT(*) FROM observations WHERE player_key=?"
                " GROUP BY street ORDER BY 2 DESC", (_pk,))}
            by_pos_inhand = {str(k) if k is not None else "UNKNOWN": v for k, v in con.execute(
                "SELECT position, SUM(in_hand) FROM observations WHERE player_key=?"
                " GROUP BY position ORDER BY 2 DESC", (_pk,))}
            # P30-9-IA：**动作分布**（带分母 ✓；每条标签都有依据与置信可查 ✓）
            by_action = {str(k): v for k, v in con.execute(
                "SELECT action, COUNT(*) FROM actions WHERE player_key=?"
                " GROUP BY action ORDER BY 2 DESC", (_pk,))}
            by_conf = con.execute(
                "SELECT ROUND(AVG(confidence), 3) FROM actions WHERE player_key=?",
                (_pk,)).fetchone()[0]
            rows.append({"player": _pk, "encounters": r[1], "in_hand": r[2] or 0,
                          "fold_evidence": r[3], "invest_events": r[4],
                          "name": r[5], "last_position": r[6], "last_stack": r[7],
                          "by_position": by_pos, "by_street": by_street,
                          "in_hand_by_position": by_pos_inhand,
                          "by_action": by_action, "actions_total": sum(by_action.values()),
                          "mean_action_confidence": by_conf,
                          "enough_sample": bool((r[1] or 0) >= min_sample)})
        return {"db": str(p), "exists": True, "encounters": n_enc, "observations": n_obs,
                 "players": n_pl, "transitions": n_tr, "by_source": by_src,
                 "min_sample": min_sample, "rows": rows,
                 "note": ("所有比例/计数均附样本量 ✓；encounters < min_sample 的对手标"
                           " enough_sample=False ⇒ **不可作结论** ✓；"
                           "by_position/by_street 是**遭遇分布**（分母=该对手遭遇数 ✓），"
                           "**不是**频率或倾向推断 ✗；此处**只有证据计数**，"
                           "不做 VPIP/PFR 等推断 ✗")}
    finally:
        con.close()
