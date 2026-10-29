"""UofT CPRG PHH 行 → CanonicalHand 映射（Phase 5A Step 0，共享纯函数模块）。

只读、无副作用；与 `scripts/analyze_uoft_cprg.py` 的 `map_row_to_canonical` 保持
逐字段一致（一致性由 tests/test_phase5a_uoft.py 保证）。

新增（相对 analyzer）：
  - derive_button_seat: 由盲注 posting 确定性反推 button 座位
  - position_label:     粗位置标签（BTN/SB/BB/EP/LP），供 RFI 位置敏感度特征使用
  - stakes_bucket / table_type: 分层抽样 + 桌型条件化用
  - map_row_full:       在 map_row_to_canonical 基础上附加 button_seat + 派生 meta

冻结边界：只 import 冻结的 canonical.py，不修改任何冻结文件。
"""
from __future__ import annotations

import collections
from typing import Any

from poker_decision_ai.domain.poker.canonical import (
    CanonicalAction, CanonicalHand, CanonicalPlayer)

STREETS = ("preflop", "flop", "turn", "river")


def _cards_from(s: str) -> list[str]:
    """PHH 牌串 -> 2 字符牌码列表（'8sTc' -> ['8s','Tc']）。"""
    s = (s or "").strip()
    if not s or "?" in s:
        return []
    return [s[i:i + 2] for i in range(0, len(s) - 1, 2)]


def _stakes_label(min_bet: float | None) -> str | None:
    if min_bet is None:
        return None
    return f"NL{round(min_bet * 100)}"


def stakes_bucket(min_bet: float | None) -> str:
    """盲注级别分桶（三档 + other），用于分层抽样与特征条件化。"""
    if min_bet is None:
        return "other"
    if 0.0 < min_bet <= 1.0:
        return "low"          # NL25–NL100
    if 1.0 < min_bet <= 4.0:
        return "mid"          # NL200–NL400
    if min_bet > 4.0:
        return "high"         # NL600–NL1000+
    return "other"


def table_type(n_players: int) -> str:
    """桌型：hu=2 人，6max=3~6，full=7+。主验证口径 = 非 hu（6max+）。"""
    if n_players <= 0:
        return "unknown"
    if n_players == 2:
        return "hu"
    if n_players <= 6:
        return "6max"
    return "full"


def derive_button_seat(blinds: list, seats: list, n_players: int) -> int | None:
    """由盲注 posting 确定性反推 button 座位（不猜测，推导失败 → None）。

    规则：
      - SB = 最小正盲注 posting 的玩家（HU 时列表顺序可能是 [BB, SB]）
      - HU（2 人）：button = SB（button 贴小盲）
      - 6-max+：button = 座位环中 SB 逆时针前一席（seats 按座位号升序=顺时针）
    """
    if not blinds or not seats:
        return None
    pos = [(i, b) for i, b in enumerate(blinds) if b is not None and b > 0]
    if not pos:
        return None
    sb_idx, sb_val = min(pos, key=lambda x: x[1])
    if sb_idx >= len(seats) or seats[sb_idx] is None:
        return None
    sb_seat = int(seats[sb_idx])
    if n_players == 2:
        return sb_seat
    occ = sorted(int(s) for s in seats if s is not None)
    if sb_seat not in occ:
        return None
    i = occ.index(sb_seat)
    return occ[(i - 1) % len(occ)]


def position_label(seat: int | None, button_seat: int | None,
                   seats: list) -> str | None:
    """粗位置标签（RFI 位置敏感度用）：BTN/SB/BB/EP/LP；不可推 → None。

    顺时针距离 d（button=0，SB=1，BB=2，…，CO=n-1）：
      d=0 → BTN；d=1 → SB；d=2 → BB；d=n-1 → LP(CO)；3<=d<=n-2 → EP
    HU（2 人）与缺失信息 → None（主口径只做 6-max+）。
    """
    if seat is None or button_seat is None or not seats:
        return None
    occ = sorted(int(s) for s in seats if s is not None)
    if seat not in occ or button_seat not in occ or len(occ) < 3:
        return None
    n = len(occ)
    d = (occ.index(seat) - occ.index(button_seat)) % n
    if d == 0:
        return "BTN"
    if d == 1:
        return "SB"
    if d == 2:
        return "BB"
    if d == n - 1:
        return "LP"
    return "EP"


def map_row_to_canonical(row: dict, row_idx: int) -> tuple[CanonicalHand | None, dict]:
    """PHH 结构化行 -> CanonicalHand；返回 (hand, meta)。

    与 scripts/analyze_uoft_cprg.map_row_to_canonical 逐字段一致（不含 button_seat）。
    非 NLHE 变体返回 (None, meta)；缺失字段一律 null/空，不猜测。
    """
    meta: dict[str, Any] = {"malformed_tokens": 0, "ignored_sd": 0,
                            "unknown_player_refs": 0}
    variant = row["variant"]
    if variant != "NT":
        return None, {"variant": variant}

    players: list[str] = row["players"] or []
    seats: list = row["seats"] or []
    blinds: list = row["blinds_or_straddles"] or []
    antes: list = row["antes"] or []
    starts: list = row["starting_stacks"] or []
    finishes: list = row["finishing_stacks"] or []
    winnings: list = row["winnings"] or []
    n = len(players)

    cplayers = []
    for i, pid in enumerate(players):
        cplayers.append(CanonicalPlayer(
            player_id=str(pid),
            seat=int(seats[i]) if i < len(seats) and seats[i] is not None else None,
            stack_start=float(starts[i]) if i < len(starts) and starts[i] is not None else None,
            stack_end=float(finishes[i]) if i < len(finishes) and finishes[i] is not None else None,
        ))

    actions = row["actions"] or []
    street_actions: dict[str, list[CanonicalAction]] = {s: [] for s in STREETS}
    committed: dict[int, float] = collections.defaultdict(float)
    invested_total: dict[int, float] = collections.defaultdict(float)
    current_bet = 0.0
    seq = 0
    street = "preflop"
    db_seen = 0
    board_cards: list[str] = []
    hole_by_player: dict[int, list[str]] = {}
    showdown_shows: list[dict] = []
    muck_count = 0

    for i in range(n):
        b = float(blinds[i]) if i < len(blinds) and blinds[i] is not None else 0.0
        if b > 0:
            street_actions["preflop"].append(CanonicalAction(
                street="preflop", seq=seq, player_id=str(players[i]),
                action="post", amount=b))
            committed[i] = b
            invested_total[i] += b
            current_bet = max(current_bet, b)
            seq += 1
        a = float(antes[i]) if i < len(antes) and antes[i] is not None else 0.0
        if a > 0:
            street_actions["preflop"].append(CanonicalAction(
                street="preflop", seq=seq, player_id=str(players[i]),
                action="post", amount=a))
            committed[i] += a
            invested_total[i] += a
            seq += 1

    for tok in actions:
        parts = tok.split()
        if not parts:
            meta["malformed_tokens"] += 1
            continue
        head = parts[0]
        if head == "d" and len(parts) >= 2 and parts[1] == "dh":
            try:
                pidx = int(parts[2][1:]) - 1
            except (ValueError, IndexError):
                meta["malformed_tokens"] += 1
                continue
            cards = _cards_from(parts[3]) if len(parts) > 3 else []
            if pidx < 0 or pidx >= n:
                meta["unknown_player_refs"] += 1
                continue
            if cards:
                hole_by_player[pidx] = cards
            continue
        if head == "d" and len(parts) >= 2 and parts[1] == "db":
            db_seen += 1
            cs = _cards_from(parts[2]) if len(parts) > 2 else []
            board_cards.extend(cs)
            street = ("preflop", "flop", "turn", "river")[min(db_seen, 3)]
            committed.clear()
            current_bet = 0.0
            continue
        if head == "[":
            meta["malformed_tokens"] += 1
            continue
        if not head.startswith("p"):
            meta["malformed_tokens"] += 1
            continue
        try:
            pidx = int(head[1:]) - 1
        except ValueError:
            meta["malformed_tokens"] += 1
            continue
        if pidx < 0 or pidx >= n:
            meta["unknown_player_refs"] += 1
            continue
        pid = str(players[pidx])
        code = parts[1] if len(parts) > 1 else ""

        if code == "f":
            street_actions[street].append(CanonicalAction(
                street=street, seq=seq, player_id=pid, action="fold"))
        elif code == "cc":
            if current_bet > committed[pidx] + 1e-9:
                amt = round(current_bet - committed[pidx], 4)
                street_actions[street].append(CanonicalAction(
                    street=street, seq=seq, player_id=pid, action="call",
                    amount=amt))
                invested_total[pidx] += amt
                committed[pidx] = current_bet
            else:
                street_actions[street].append(CanonicalAction(
                    street=street, seq=seq, player_id=pid, action="check"))
        elif code == "cbr" and len(parts) > 2:
            try:
                x = float(parts[2])
            except ValueError:
                meta["malformed_tokens"] += 1
                continue
            inc = round(x - committed[pidx], 4)
            if inc <= 1e-9:
                street_actions[street].append(CanonicalAction(
                    street=street, seq=seq, player_id=pid, action="check"))
            elif current_bet <= 0:
                street_actions[street].append(CanonicalAction(
                    street=street, seq=seq, player_id=pid, action="bet",
                    amount=x, to=x))
                invested_total[pidx] += inc
                committed[pidx] = x
                current_bet = x
            elif x > current_bet:
                street_actions[street].append(CanonicalAction(
                    street=street, seq=seq, player_id=pid, action="raise",
                    amount=inc, to=x))
                invested_total[pidx] += inc
                committed[pidx] = x
                current_bet = x
            else:
                street_actions[street].append(CanonicalAction(
                    street=street, seq=seq, player_id=pid, action="call",
                    amount=inc))
                invested_total[pidx] += inc
                committed[pidx] = x
        elif code == "sm":
            cs = _cards_from(parts[2]) if len(parts) > 2 else []
            if cs:
                showdown_shows.append({"player_id": pid, "cards": cs})
            else:
                muck_count += 1
        elif code == "pb" and len(parts) > 2:
            try:
                amt = float(parts[2])
            except ValueError:
                meta["malformed_tokens"] += 1
                continue
            street_actions[street].append(CanonicalAction(
                street=street, seq=seq, player_id=pid, action="post", amount=amt))
            committed[pidx] += amt
            invested_total[pidx] += amt
        elif code == "sd":
            meta["ignored_sd"] += 1
        else:
            meta["malformed_tokens"] += 1
        seq += 1

    flop = board_cards[:3] if len(board_cards) >= 3 else []
    turn = board_cards[3:4] if len(board_cards) >= 4 else []
    river = board_cards[4:5] if len(board_cards) >= 5 else []

    hole_cards: list[str] = []
    for pidx, cards in hole_by_player.items():
        cplayers[pidx].hole_cards = cards

    posted_vals = [float(v) for v in blinds[:2] if v]
    if len(posted_vals) >= 2:
        sb, bb = min(posted_vals), max(posted_vals)
    else:
        sb, bb = None, None
    if sb is None:
        sb = float(row["small_bet"]) if row.get("small_bet") is not None else None
    if bb is None:
        bb = float(row["big_bet"]) if row.get("big_bet") is not None else None
        if bb is None:
            bb = float(row["min_bet"]) if row.get("min_bet") is not None else None
    ante = None
    for v in antes:
        if v:
            ante = float(v)
            break

    winners = []
    win_from_delta = False
    for i in range(min(n, len(winnings))):
        w = float(winnings[i]) if winnings[i] is not None else 0.0
        if w > 0:
            winners.append({"player_id": str(players[i]), "amount": round(w, 4)})
    if not winners and len(finishes) == n and len(starts) == n:
        for i in range(n):
            if starts[i] is not None and finishes[i] is not None:
                d = round(float(finishes[i]) - float(starts[i]), 4)
                if d > 0:
                    winners.append({"player_id": str(players[i]),
                                    "amount": round(d + invested_total[i], 4)})
        win_from_delta = bool(winners)
    meta["win_from_delta"] = win_from_delta

    showdown = None
    if showdown_shows:
        showdown = {"happened": True, "shows": showdown_shows}

    started_at = None
    if (row.get("day") is not None and row.get("month") is not None
            and row.get("year") is not None):
        y, m, d = int(row["year"]), int(row["month"]), int(row["day"])
        t = row.get("time") or "00:00:00"
        started_at = f"{y:04d}-{m:02d}-{d:02d}T{t}"

    event = row.get("event") or ""
    if "World Series of Poker" in event:
        game_mode, game_type = "tournament", "mtt"
    elif row.get("venue"):
        game_mode, game_type = "cash", "cash"
    else:
        game_mode, game_type = "cash", "cash"
    meta["game_mode"] = game_mode
    meta["source_kind"] = "handhq" if row.get("venue") else (
        "pluribus" if "pluribus" in (row.get("source_file") or "") else (
            "wsop" if "World Series" in event else "historical"))

    hand_id = str(row["hand"]) if row.get("hand") else (
        f"{row.get('source_file')}#{row.get('hand_number_in_file') or row_idx}")

    hand = CanonicalHand(
        hand_id=hand_id,
        source=str(row.get("source_file") or ""),
        source_format="parquet_phh",
        game_mode=game_mode,
        game_type=game_type,
        tournament_type="mtt" if game_mode == "tournament" else None,
        stakes=_stakes_label(row.get("min_bet")),
        sb=sb, bb=bb, ante=ante,
        currency=str(row["currency"]) if row.get("currency") else None,
        table_size=n,
        started_at=started_at,
        players=cplayers,
        hero_id=None,
        hole_cards=hole_cards,
        preflop_actions=street_actions["preflop"],
        flop=flop, flop_actions=street_actions["flop"],
        turn=turn, turn_actions=street_actions["turn"],
        river=river, river_actions=street_actions["river"],
        showdown=showdown,
        winners=winners,
        meta={"venue": row.get("venue"), "mucks": muck_count,
              "ante_trimming_status": row.get("ante_trimming_status")},
    )
    return hand, meta


def map_row_full(row: dict, row_idx: int) -> tuple[CanonicalHand | None, dict]:
    """map_row_to_canonical + button_seat 反推 + 派生字段（写入 hand.meta，自包含）。"""
    hand, meta = map_row_to_canonical(row, row_idx)
    if hand is None:
        return None, meta
    seats = row["seats"] or []
    blinds = row["blinds_or_straddles"] or []
    btn = derive_button_seat(blinds, seats, len(hand.players))
    hand.button_seat = btn
    sk = meta.get("source_kind", "unknown")
    ven = row.get("venue") or ""
    hnf = row.get("hand_number_in_file")
    hnf_s = f"{int(hnf)}" if hnf is not None else "?"
    src = row.get("source_file") or ""
    # iPoker 内 hand id 非全局唯一（是牌桌/游戏 id），须用 source_file + 文件内序号消歧
    hand.meta.update({
        "button_seat": btn,
        "stakes_bucket": stakes_bucket(row.get("min_bet")),
        "table_type": table_type(len(hand.players)),
        "source_kind": sk,
        "hand_key": f"{sk}|{ven}|{src}|{hnf_s}|{hand.hand_id}",
    })
    meta["button_seat"] = btn
    meta["stakes_bucket"] = stakes_bucket(row.get("min_bet"))
    meta["table_type"] = table_type(len(hand.players))
    return hand, meta


def derive_player_events(hand: CanonicalHand) -> list[dict]:
    """CanonicalHand → 每玩家事件表行（feature_ready 层）。

    相对冻结 dataset_tool.canonical_to_player_records 的差异（不修改冻结，另行实现）：
      - three_bet 精确为「第 2 次加注」（而非「非首个加注者」粗糙代理）
      - cbet 与 cbet_opportunity 分离（冻结把二者都设为同一布尔）
      - 新增 rfi_opp / rfi_did（raise-first-in，位置敏感度探针用）
      - 新增 position / stakes_bucket / table_type / venue / hand_key

    返回 HAND_RECORD_FIELDS 超集的行列表（含全部玩家，调用方自行过滤）。
    """
    pf = list(hand.preflop_actions)
    fl = list(hand.flop_actions)
    acts = pf + list(hand.flop_actions) + list(hand.turn_actions) \
        + list(hand.river_actions)
    winner_ids = {w.get("player_id") for w in hand.winners}
    posted = {a.player_id for a in pf if a.action == "post"}
    folded_preflop = {a.player_id for a in pf if a.action == "fold"}
    folded_any = {a.player_id for a in acts if a.action == "fold"}

    # 加注顺序（preflop）
    raisers = [a.player_id for a in pf if a.action == "raise"]
    preflop_aggressor = raisers[-1] if raisers else None

    # RFI（raise-first-in）：首个自愿入池动作是加注的玩家
    voluntary_yet = False
    rfi_did: dict[str, bool] = {}
    rfi_opp_seen: set[str] = set()
    rfi_opp: dict[str, bool] = {}
    for a in pf:
        pid = a.player_id
        if pid not in rfi_opp_seen:
            rfi_opp_seen.add(pid)
            if pid not in posted and not voluntary_yet:
                rfi_opp[pid] = True
        if a.action in ("call", "raise"):
            if not voluntary_yet:
                voluntary_yet = True
                if a.action == "raise":
                    rfi_did[pid] = True
        # 注：raise 归入 raisers 已在上面统一收集

    # 玩家在 preflop 加注序列中的序位（1-based）
    raise_rank: dict[str, int] = {}
    for rank, pid in enumerate(raisers, start=1):
        raise_rank.setdefault(pid, rank)

    # flop aggressor 是否做 cbet
    flop_aggressor_bet = any(a.player_id == preflop_aggressor
                             and a.action in ("bet", "raise") for a in fl)
    flop_aggressor_folded = preflop_aggressor in folded_any
    reached_flop = bool(hand.flop) or bool(fl)
    showdown_happened = bool(hand.showdown and hand.showdown.get("happened"))
    bb = hand.bb or 1.0

    rows = []
    for p in hand.players:
        pid = p.player_id
        pf_p = [a for a in pf if a.player_id == pid]
        fl_p = [a for a in fl if a.player_id == pid]
        vpip = any(a.action in ("call", "raise") for a in pf_p)
        pfr = any(a.action == "raise" for a in pf_p)
        rank = raise_rank.get(pid)
        three_bet = rank == 2
        four_bet = rank == 3
        faced_raise = any(a.action == "raise" and a.player_id != pid for a in pf)
        folded_to_raise = faced_raise and any(a.action == "fold" for a in pf_p)
        saw_flop = pid not in folded_preflop and reached_flop
        is_aggr = pid == preflop_aggressor
        cbet_opp = is_aggr and saw_flop and not flop_aggressor_folded
        cbet = is_aggr and any(a.action in ("bet", "raise") for a in fl_p)
        # 面对 cbet：非加注者、翻牌仍在场（saw_flop 已保证未翻前弃牌）、aggressor 在 flop 下注
        faced_cbet = (not is_aggr) and saw_flop and flop_aggressor_bet
        folded_to_cbet = faced_cbet and any(a.action == "fold" for a in fl_p)
        went_to_sd = pid not in folded_any and showdown_happened
        won_sd = pid in winner_ids if went_to_sd else None
        won = pid in winner_ids
        invested = sum(float(a.amount or 0.0) for a in acts
                       if a.player_id == pid and a.action in ("call", "raise", "bet", "post"))
        won_amount = sum(float(w.get("amount") or 0.0) for w in hand.winners
                         if w.get("player_id") == pid)
        seat = p.seat
        pos = position_label(seat, hand.button_seat, [q.seat for q in hand.players])
        rows.append({
            "hand_id": hand.meta.get("hand_key", hand.hand_id),
            "hand_id_raw": hand.hand_id,
            "player_id": pid, "game": hand.game_type or "cash",
            "stakes": hand.stakes or "", "n_players": len(hand.players),
            "position": pos,
            "vpip": vpip, "pfr": pfr,
            "three_bet": three_bet, "faced_3bet": faced_raise,
            "folded_to_3bet": folded_to_raise, "four_bet": four_bet,
            "cbet_opportunity": cbet_opp, "cbet": cbet,
            "faced_cbet": faced_cbet, "folded_to_cbet": folded_to_cbet,
            "saw_flop": saw_flop, "went_to_showdown": went_to_sd,
            "won_at_showdown": won_sd, "won_hand": won,
            "net_bb": round((won_amount - invested) / bb, 4),
            "bets": sum(1 for a in acts if a.player_id == pid and a.action in ("bet", "raise")),
            "calls": sum(1 for a in acts if a.player_id == pid and a.action == "call"),
            "folds": sum(1 for a in acts if a.player_id == pid and a.action == "fold"),
            "ts": hand.started_at,
            "rfi_opp": bool(rfi_opp.get(pid)),
            "rfi_did": bool(rfi_did.get(pid)),
            "stakes_bucket": hand.meta.get("stakes_bucket"),
            "table_type": hand.meta.get("table_type"),
            "venue": hand.meta.get("venue"),
        })
    return rows
