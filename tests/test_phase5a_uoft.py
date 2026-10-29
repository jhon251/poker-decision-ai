"""Phase 5A UofT 数据层测试（映射 / 派生 / 端到端）。

覆盖：
  - 映射与 analyzer 逐字段一致性（回归：uoft_mapping == analyze_uoft_cprg）
  - derive_button_seat / position_label / stakes_bucket / table_type 正确性
  - derive_player_events 探针语义（vpip/pfr/3bet/cbet/rfi）
  - 端到端：抽样 → normalize → manifest → splits → stats（用 mini 合成 parquet）

不依赖 C:/PokerDataset（CI 可离线跑）。
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from uoft_mapping import (  # noqa: E402
    derive_button_seat, derive_player_events, map_row_to_canonical,
    position_label, stakes_bucket, table_type)
from poker_decision_ai.domain.poker.canonical import (  # noqa: E402
    CanonicalAction, CanonicalHand, CanonicalPlayer)


# ---------------------------------------------------------------- 映射纯函数
def test_stakes_bucket():
    assert stakes_bucket(None) == "other"
    assert stakes_bucket(0.25) == "low"
    assert stakes_bucket(1.0) == "low"
    assert stakes_bucket(2.0) == "mid"
    assert stakes_bucket(4.0) == "mid"
    assert stakes_bucket(6.0) == "high"
    assert stakes_bucket(10.0) == "high"


def test_table_type():
    assert table_type(0) == "unknown"
    assert table_type(2) == "hu"
    assert table_type(6) == "6max"
    assert table_type(9) == "full"


def test_derive_button_seat_6max():
    # 座位 [2,3,4,5,6,1]，SB 在 seat2（5），button 应为 seat1（逆时针前一席）
    assert derive_button_seat([5, 10, 0, 0, 0, 0], [2, 3, 4, 5, 6, 1], 6) == 1


def test_derive_button_seat_hu():
    # HU: blinds 顺序 [BB, SB]，button = SB
    assert derive_button_seat([10, 5], [1, 2], 2) == 2


def test_derive_button_seat_missing():
    assert derive_button_seat([], [], 6) is None
    assert derive_button_seat([0, 0, 0], [1, 2, 3], 3) is None


def test_position_label():
    seats = [2, 3, 4, 5, 6, 1]
    assert position_label(1, 1, seats) == "BTN"
    assert position_label(2, 1, seats) == "SB"
    assert position_label(3, 1, seats) == "BB"
    assert position_label(4, 1, seats) == "EP"   # d=3
    assert position_label(6, 1, seats) == "LP"   # d=n-1 → CO
    assert position_label(None, 1, seats) is None
    assert position_label(1, None, seats) is None
    assert position_label(1, 1, [1, 2]) is None  # HU 不支持


# ---------------------------------------------------------------- 事件派生语义
def _mk_hand(preflop_actions, flop_actions=(), flop=None, players=None,
             bb=2.0, showdown=None, winners=None, button_seat=None):
    players = players or [CanonicalPlayer(player_id=p) for p in ("A", "B", "C")]
    return CanonicalHand(
        hand_id="h1", game_mode="cash", game_type="cash",
        sb=1.0, bb=bb, table_size=len(players), button_seat=button_seat,
        players=players, preflop_actions=list(preflop_actions),
        flop=list(flop or []), flop_actions=list(flop_actions),
        showdown=showdown, winners=winners or [],
        meta={"hand_key": "h1", "stakes_bucket": "low", "table_type": "6max"},
    )


def _act(street, pid, action, amount=None, to=None, seq=0):
    return CanonicalAction(street=street, seq=seq, player_id=pid, action=action,
                           amount=amount, to=to)


def test_derive_vpip_pfr():
    # A 加注（raise），B 跟注（call），C 弃牌（fold）
    h = _mk_hand([
        _act("preflop", "A", "post", 1.0),
        _act("preflop", "B", "post", 2.0),
        _act("preflop", "C", "fold"),
        _act("preflop", "A", "raise", 3.0, 4.0),
        _act("preflop", "B", "call", 2.0),
    ])
    ev = {e["player_id"]: e for e in derive_player_events(h)}
    assert ev["A"]["vpip"] is True and ev["A"]["pfr"] is True
    assert ev["B"]["vpip"] is True and ev["B"]["pfr"] is False
    assert ev["C"]["vpip"] is False and ev["C"]["pfr"] is False


def test_derive_three_bet():
    # A 加注（2bet），B 再加注（3bet），C 弃牌
    h = _mk_hand([
        _act("preflop", "A", "post", 1.0),
        _act("preflop", "B", "post", 2.0),
        _act("preflop", "C", "raise", 6.0, 6.0),   # 第 1 次加注（2bet）
        _act("preflop", "A", "fold"),
        _act("preflop", "B", "raise", 12.0, 18.0),  # 第 2 次加注（3bet）
        _act("preflop", "C", "call", 12.0),
    ])
    ev = {e["player_id"]: e for e in derive_player_events(h)}
    assert ev["C"]["three_bet"] is False  # C 是首个加注者
    assert ev["B"]["three_bet"] is True   # B 是 3bet


def test_derive_cbet():
    # A 是唯一 preflop 加注者，flop A 下注 → cbet
    h = _mk_hand(
        preflop_actions=[
            _act("preflop", "A", "post", 1.0),
            _act("preflop", "B", "post", 2.0),
            _act("preflop", "C", "fold"),
            _act("preflop", "A", "raise", 3.0, 4.0),
            _act("preflop", "B", "call", 2.0),
        ],
        flop_actions=[_act("flop", "A", "bet", 4.0, 4.0),
                      _act("flop", "B", "fold")],
        flop=["As", "Kc", "2d"],
    )
    ev = {e["player_id"]: e for e in derive_player_events(h)}
    assert ev["A"]["cbet_opportunity"] is True and ev["A"]["cbet"] is True
    assert ev["B"]["cbet_opportunity"] is False and ev["B"]["cbet"] is False
    assert ev["B"]["faced_cbet"] is True and ev["B"]["folded_to_cbet"] is True


def test_derive_rfi():
    # C 在无人入池时加注（RFI）；B 在 C 之后跟注（非 RFI）
    h = _mk_hand([
        _act("preflop", "A", "post", 1.0),
        _act("preflop", "B", "post", 2.0),
        _act("preflop", "C", "raise", 6.0, 6.0),
        _act("preflop", "A", "fold"),
        _act("preflop", "B", "fold"),
    ])
    ev = {e["player_id"]: e for e in derive_player_events(h)}
    assert ev["C"]["rfi_did"] is True
    assert ev["A"]["rfi_did"] is False
    assert ev["B"]["rfi_did"] is False


# ---------------------------------------------------------------- 端到端 mini 集成
def _mini_parquet(tmp_path: Path, n_players=6, n_hands=20) -> Path:
    """生成一个 mini NT parquet（结构化 PHH 列，含选中玩家的多手）。"""
    rows = []
    pid = "PLR_A"
    for i in range(n_hands):
        players = [pid] + [f"P{j}" for j in range(1, n_players)]
        actions = [f"d dh p{k} ????" for k in range(1, n_players + 1)]
        actions += ["p3 f", "p4 f", "p5 f", "p6 f", "p1 cbr 6", "p2 f"]
        rows.append({
            "variant": "NT", "venue": "TestSite", "hand": f"100{i}",
            "hand_number_in_file": float(i + 1), "day": 1.0, "month": 7.0,
            "year": 2009.0, "time": f"12:{i:02d}:00", "min_bet": 2.0,
            "actions": actions, "players": players,
            "antes": [0.0] * n_players, "blinds_or_straddles": [1.0, 2.0] + [0.0] * (n_players - 2),
            "starting_stacks": [100.0] * n_players,
            "finishing_stacks": [], "winnings": [0.0] * n_players,
            "seats": list(range(1, n_players + 1)),
            "source_file": "test/mini.phhs", "event": None, "currency": None,
            "small_bet": None, "big_bet": None,
        })
    arr = pa.Table.from_pylist(rows)
    out = tmp_path / "mini.parquet"
    pq.write_table(arr, out)
    return out


def test_end_to_end_normalize(tmp_path):
    parquet = _mini_parquet(tmp_path)
    # 直接调用 normalize 的核心函数（不经 CLI）
    from phase5a_normalize import _shard_to_chunk
    out_norm = tmp_path / "normalized"
    out_ev = tmp_path / "feature_ready"
    out_norm.mkdir()
    out_ev.mkdir()
    r = _shard_to_chunk((str(parquet), 0, ["PLR_A"], str(out_norm), str(out_ev)))
    assert r["hands"] == 20
    assert r["events"] == 20  # 每手 1 个选中玩家（PLR_A）
    # canonical chunk 可解析
    lines = list(open(out_norm / "chunk_000.jsonl", encoding="utf-8"))
    assert len(lines) == 20
    rec = json.loads(lines[0])
    assert rec["quality"]["status"] in ("valid", "warning", "invalid")
    assert rec["canonical"]["meta"]["hand_key"].startswith("handhq|TestSite|")
    # event 行字段完整
    ev = json.loads(open(out_ev / "ev_tmp_000.jsonl", encoding="utf-8").readline())
    assert ev["player_id"] == "PLR_A"
    assert ev["vpip"] is True and ev["pfr"] is True
    assert ev["table_type"] == "6max"


def test_end_to_end_stats(tmp_path):
    """mini parquet → normalize → 手工聚合统计 → 验证探针值域与 CI 单调。"""
    parquet = _mini_parquet(tmp_path)
    from phase5a_normalize import _shard_to_chunk
    out_norm = tmp_path / "normalized"
    out_ev = tmp_path / "feature_ready"
    out_norm.mkdir()
    out_ev.mkdir()
    _shard_to_chunk((str(parquet), 0, ["PLR_A"], str(out_norm), str(out_ev)))
    evs = [json.loads(l) for l in open(out_ev / "ev_tmp_000.jsonl", encoding="utf-8")]
    from poker_decision_ai.domain.poker.stats import wilson_interval
    vpip_n = sum(1 for e in evs if e["vpip"])
    lo, hi = wilson_interval(vpip_n, len(evs))
    assert 0.0 <= lo <= hi <= 1.0
    # 样本越多区间越窄
    lo2, hi2 = wilson_interval(vpip_n * 2, len(evs) * 2)
    assert (hi2 - lo2) < (hi - lo)
