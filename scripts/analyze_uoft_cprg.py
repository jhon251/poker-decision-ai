"""UofT CPRG Poker Dataset —— 只读 Dataset Qualification 分析（Real Dataset Validation）。

本脚本只读分析 C:\\PokerDataset\\UofT_CPRG\\data\\*.parquet：
  - 不写 data/ 任何分层（不进入 Normalize 落盘）
  - 复用冻结的 canonical.py / validator.py（import 使用，不修改）
  - 逐分片、逐行映射 PHH → CanonicalHand → 10 项检查，聚合统计
  - 输出 13 项报告 + 数据充分性判断（docs/DATASET_QUALIFICATION_REPORT_UOFT_CPRG.md）

用法:
  .\\.venv\\Scripts\\python.exe scripts/analyze_uoft_cprg.py --dir C:\\PokerDataset\\UofT_CPRG\\data --workers 12
  （调试/校准: 加 --limit-shards 1）
"""
from __future__ import annotations

import argparse
import collections
import json
import multiprocessing as mp
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pyarrow.parquet as pq  # noqa: E402

from poker_decision_ai.domain.poker.canonical import (  # noqa: E402
    CanonicalAction, CanonicalHand, CanonicalPlayer)
from poker_decision_ai.domain.poker.validator import validate_hand  # noqa: E402

STREETS = ("preflop", "flop", "turn", "river")
NON_NLHE_VARIANTS = ()  # 见 main 聚合


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


def map_row_to_canonical(row: dict, row_idx: int) -> tuple[CanonicalHand | None, dict]:
    """PHH 结构化行 -> CanonicalHand；返回 (hand, meta)。meta 收集映射层异常计数。

    非 NLHE 变体返回 (None, meta)（canonical 为 NLHE schema）。
    缺失字段一律 null/空，不猜测。
    """
    meta = {"malformed_tokens": 0, "ignored_sd": 0, "unknown_player_refs": 0}
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

    # ---- PHH 行动解析 ----
    actions = row["actions"] or []
    street_actions: dict[str, list[CanonicalAction]] = {s: [] for s in STREETS}
    committed: dict[int, float] = collections.defaultdict(float)   # 本街已投入
    invested_total: dict[int, float] = collections.defaultdict(float)  # 全手总投入
    current_bet: float = 0.0
    seq = 0
    street = "preflop"
    db_seen = 0
    board_cards: list[str] = []
    hole_by_player: dict[int, list[str]] = {}
    showdown_shows: list[dict] = []
    muck_count = 0

    # 盲注/ante 先入账（盲注=最高 posted 作为 current_bet）
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
            # PHH 语义: cbr X = 本街该玩家总投入 X（call/bet/raise 合一）。
            # 增量 = X - 本街已投入；X > current_bet → bet/raise；否则 call。
            inc = round(x - committed[pidx], 4)
            if inc <= 1e-9:                                  # 纯 check/无追加
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

    # 街拆分
    flop = board_cards[:3] if len(board_cards) >= 3 else []
    turn = board_cards[3:4] if len(board_cards) >= 4 else []
    river = board_cards[4:5] if len(board_cards) >= 5 else []

    # hero：数据无 hero 视角 → 不猜
    hole_cards: list[str] = []
    # 有手牌记录时保留在玩家对象（Pluribus 子集），canonical hero 字段留空
    for pidx, cards in hole_by_player.items():
        cplayers[pidx].hole_cards = cards

    # sb/bb：heads-up 列表顺序为 [BB, SB]，取前两个 posted 的 min/max
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

    # winners：优先 winnings>0（HandHQ）；否则由筹码两端确定性推导
    # （Pluribus/WSOP 无 winnings）：won = 净增 + 总投入 = 从底池收回总额，
    # 使冻结的 7_stacks 检查保持可核验（数据推导非猜测）。
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

    # showdown：有牌示出的 sm 才算可验证摊牌；纯 muck 不构造空 shows（避免误判 invalid）
    showdown = None
    if showdown_shows:
        showdown = {"happened": True, "shows": showdown_shows}

    # started_at：HandHQ 有 day/month/year + time；缺失 → None（不猜）
    started_at = None
    if (row.get("day") is not None and row.get("month") is not None
            and row.get("year") is not None):
        y, m, d = int(row["year"]), int(row["month"]), int(row["day"])
        t = row.get("time") or "00:00:00"
        started_at = f"{y:04d}-{m:02d}-{d:02d}T{t}"

    # game mode：venue 存在 = 现金局；WSOP event = 锦标赛；Pluribus = 现金研究
    event = row.get("event") or ""
    if "World Series of Poker" in event:
        game_mode, game_type = "tournament", "mtt"
    elif row.get("venue"):
        game_mode, game_type = "cash", "cash"
    else:
        game_mode, game_type = "cash", "cash"      # Pluribus / 历史局按现金
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


def analyze_shard(path: str) -> dict:
    """逐分片：映射 + 冻结 validate_hand + 聚合。返回可 pickle 的统计 dict。"""
    t0 = time.time()
    p = pq.ParquetFile(path)
    agg = {
        "path": path, "rows": 0, "nt_hands": 0,
        "status": collections.Counter(),           # valid/warning/invalid
        "checks": collections.Counter(),           # check_name+status
        "invalid_reasons": collections.Counter(),  # check name -> 次数（invalid）
        "replayable": 0,
        "player_hands": collections.Counter(),     # player_id -> 手数
        "action_types": collections.Counter(),     # (street, action) -> 次数
        "stakes": collections.Counter(),           # min_bet 值 -> 手数
        "modes": collections.Counter(),
        "venues": collections.Counter(),
        "source_kinds": collections.Counter(),
        "variants": collections.Counter(),
        "dates": [],
        "board_reach": collections.Counter(),      # db 数 -> 手数
        "showdown_hands": 0,
        "muck_only_showdown": 0,                   # sm 存在但无有牌示出
        "hole_known_hands": 0,                     # 至少一名玩家有已知手牌
        "nplayers": collections.Counter(),
        "ante_hands": 0,
        "straddle_hands": 0,                       # blinds 列表长度>2
        "mapping": collections.Counter(),          # malformed/ignored/unknown
        "sec": 0.0,
    }
    dates_min, dates_max = None, None

    for rg in range(p.num_row_groups):
        t = p.read_row_group(rg).to_pydict()
        for i in range(len(t["variant"])):
            agg["rows"] += 1
            row = {k: t[k][i] for k in t}
            agg["variants"][row["variant"] or "(null)"] += 1
            hand, meta = map_row_to_canonical(row, i)
            if hand is None:
                continue
            agg["nt_hands"] += 1
            agg["mapping"]["malformed"] += meta.get("malformed_tokens", 0)
            agg["mapping"]["ignored_sd"] += meta.get("ignored_sd", 0)
            agg["mapping"]["unknown_refs"] += meta.get("unknown_player_refs", 0)
            agg["modes"][meta["game_mode"]] += 1
            agg["source_kinds"][meta["source_kind"]] += 1
            if hand.meta.get("venue"):
                agg["venues"][hand.meta["venue"]] += 1
            mb = row.get("min_bet")
            agg["stakes"][mb if mb is not None else "(null)"] += 1
            agg["nplayers"][len(hand.players)] += 1
            if hand.ante is not None:
                agg["ante_hands"] += 1
            nonzero_blinds = sum(1 for v in (row["blinds_or_straddles"] or []) if v)
            if nonzero_blinds > 2:
                agg["straddle_hands"] += 1
            ndb = sum(1 for a in (row["actions"] or []) if a.startswith("d db"))
            agg["board_reach"][ndb] += 1
            nsm = sum(1 for a in (row["actions"] or [])
                      if a.split()[0].startswith("p") and len(a.split()) > 1
                      and a.split()[1] == "sm")
            if nsm:
                agg["showdown_hands"] += 1
                if hand.showdown is None:
                    agg["muck_only_showdown"] += 1
            if any(p.hole_cards for p in hand.players):
                agg["hole_known_hands"] += 1
            for pl in hand.players:
                agg["player_hands"][pl.player_id] += 1
            for street, acts in (("preflop", hand.preflop_actions),
                                 ("flop", hand.flop_actions),
                                 ("turn", hand.turn_actions),
                                 ("river", hand.river_actions)):
                for a in acts:
                    agg["action_types"][(street, a.action)] += 1
            if hand.started_at:
                agg["dates"].append(hand.started_at)
                d = hand.started_at[:10]
                if dates_min is None or d < dates_min:
                    dates_min = d
                if dates_max is None or d > dates_max:
                    dates_max = d
            # 冻结 10 项校验
            rep = validate_hand(hand)
            agg["status"][rep.status] += 1
            for c in rep.checks:
                agg["checks"][f"{c.name}|{c.status}"] += 1
                if c.status == "invalid":
                    agg["invalid_reasons"][c.name] += 1
            # 冻结 _replayable 语义: hand_id+hero_id+hole_cards+stack_start。
            # 本数据无 hero → hero_id None → 恒 False；单独统计"若指定 hero 可回放"。
            if (hand.hand_id and any(pl.hole_cards for pl in hand.players)
                    and any(pl.stack_start is not None for pl in hand.players)):
                agg["replayable"] += 1       # 结构可回放（若指定 hero）
    agg["dates_min"], agg["dates_max"] = dates_min, dates_max
    agg.pop("dates", None)
    agg["sec"] = round(time.time() - t0, 1)
    # 大 Counter 转 dict（可 pickle / JSON 序列化）
    agg["action_types"] = {f"{k[0]}|{k[1]}": v for k, v in agg["action_types"].items()}
    for k in ("player_hands", "stakes", "modes", "venues",
              "source_kinds", "variants", "board_reach", "nplayers",
              "checks", "status", "invalid_reasons", "mapping"):
        agg[k] = dict(agg[k])
    return agg


def merge(aggs: list[dict]) -> dict:
    out = None
    for a in aggs:
        if out is None:
            out = a
            continue
        for k, v in a.items():
            if isinstance(v, int):
                out[k] = out.get(k, 0) + v
            elif isinstance(v, dict):
                d = out.setdefault(k, {})
                for kk, vv in v.items():
                    d[kk] = d.get(kk, 0) + vv
            elif isinstance(v, float):
                out[k] = out.get(k, 0.0) + v
            elif k in ("path", "dates_min", "dates_max"):
                if k == "path":
                    out[k] = out[k] + ";" + v
                elif k == "dates_min":
                    if v and (out[k] is None or v < out[k]):
                        out[k] = v
                elif k == "dates_max":
                    if v and (out[k] is None or v > out[k]):
                        out[k] = v
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=r"C:\PokerDataset\UofT_CPRG\data")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit-shards", type=int, default=0)
    ap.add_argument("--out", default="docs/DATASET_QUALIFICATION_REPORT_UOFT_CPRG.md")
    args = ap.parse_args()

    files = sorted(str(p) for p in Path(args.dir).glob("train-*.parquet"))
    if args.limit_shards:
        files = files[: args.limit_shards]
    print(f"shards: {len(files)}", flush=True)

    t0 = time.time()
    results = []
    if len(files) == 1 or args.workers <= 1:
        for f in files:
            r = analyze_shard(f)
            results.append(r)
            print(f"  done {Path(f).name} rows={r['rows']} "
                  f"sec={r['sec']}", flush=True)
    else:
        with mp.Pool(args.workers) as pool:
            for r in pool.imap_unordered(analyze_shard, files):
                results.append(r)
                print(f"  done {Path(r['path']).name} rows={r['rows']} "
                      f"sec={r['sec']}", flush=True)
    total = merge(results)
    total["wall_sec"] = round(time.time() - t0, 1)
    print(json.dumps({k: total[k] for k in
                      ("rows", "nt_hands", "status", "replayable", "wall_sec")},
                     ensure_ascii=False))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(total, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"aggregate written: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
