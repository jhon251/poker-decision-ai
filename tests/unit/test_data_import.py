"""Phase 11 (V2) 真实数据导入接口测试。"""
from __future__ import annotations

import json
from pathlib import Path

from poker_decision_ai.data_import import DataImporter, hand_to_records
from poker_decision_ai.domain.poker.hand_history import parse_hand_history

ROOT = Path(__file__).parent.parent.parent
FULL = (ROOT / "examples/hands/cash_full_hand.txt").read_text(encoding="utf-8")
SHOWDOWN = (ROOT / "examples/hands/cash_showdown_hand.txt").read_text(encoding="utf-8")


def test_hand_to_records_derives_core_flags(cfg):
    hand = parse_hand_history(FULL)
    records = hand_to_records(hand, cfg)
    by_name = {r["player_id"]: r for r in records}
    assert len(records) == 6
    # Hero：翻前加注（pfr=True、vpip=True），翻牌下注（cbet opportunity 且 cbet=True）
    assert by_name["Hero"]["pfr"] is True and by_name["Hero"]["vpip"] is True
    assert by_name["Hero"]["cbet"] is True and by_name["Hero"]["cbet_opportunity"] is True
    # Bob：limp-call（vpip=True，pfr=False），翻牌 check-call 面对 cbet 未弃牌
    assert by_name["Bob"]["vpip"] is True and by_name["Bob"]["pfr"] is False
    assert by_name["Bob"]["faced_cbet"] is True and by_name["Bob"]["folded_to_cbet"] is False
    # 弃牌玩家
    assert by_name["Eve"]["vpip"] is False and by_name["Eve"]["saw_flop"] is False
    # 级别反查
    assert by_name["Hero"]["stakes"] == "NL10"


def test_hand_to_records_showdown_flags(cfg):
    hand = parse_hand_history(SHOWDOWN)
    by_name = {r["player_id"]: r for r in hand_to_records(hand, cfg)}
    assert by_name["Hero"]["went_to_showdown"] is True
    assert by_name["Hero"]["won_at_showdown"] is False
    assert by_name["Bob"]["won_at_showdown"] is True
    assert by_name["Bob"]["won_hand"] is True
    assert by_name["Hero"]["net_bb"] < 0


def test_import_hand_text_writes_repo(runtime):
    imp = DataImporter(runtime)
    report = imp.import_hand_text(FULL)
    assert report.hands_imported == 1
    assert report.records_imported == 6
    assert runtime.repos.hands.count() >= 7        # 1 parsed + 6 记录


def test_import_normalized_json_validates(runtime):
    imp = DataImporter(runtime)
    records = [
        {"hand_id": "h1", "player_id": "A", "game": "cash", "vpip": True},
        {"hand_id": "h2"},                          # 缺 player_id → 拒绝
        {"hand_id": "h3", "player_id": "B", "game": "bad"},   # 非法 game → 拒绝
    ]
    report = imp.import_normalized_json(json.dumps(records))
    assert report.records_imported == 1
    assert report.rejected == 2
    assert any("missing player_id" in w for w in report.warnings)
    assert any("bad game" in w for w in report.warnings)


def test_import_csv(runtime):
    imp = DataImporter(runtime)
    csv_text = (
        "hand_id,player_id,game,stakes,vpip,pfr,net_bb\n"
        "c1,Alice,cash,NL10,true,false,-3.5\n"
        "c1,Bob,cash,NL10,false,true,12.0\n"
    )
    report = imp.import_csv(csv_text)
    assert report.records_imported == 2
    recs = [r for r in runtime.repos.hands.list() if r.get("hand_id") == "c1"]
    alice = next(r for r in recs if r["player_id"] == "Alice")
    assert alice["vpip"] is True and alice["net_bb"] == -3.5


def test_import_directory(runtime, tmp_path):
    d = tmp_path / "hands"
    d.mkdir()
    (d / "a.txt").write_text(FULL, encoding="utf-8")
    (d / "b.txt").write_text(SHOWDOWN, encoding="utf-8")
    report = DataImporter(runtime).import_directory(str(d))
    assert report.hands_imported == 2
    assert report.records_imported == 12


def test_missing_file_warns(runtime):
    report = DataImporter(runtime).import_file("no/such/file.txt")
    assert report.hands_imported == 0
    assert report.warnings
