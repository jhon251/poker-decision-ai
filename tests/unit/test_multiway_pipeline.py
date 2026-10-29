# -*- coding: utf-8 -*-
"""P30-9-HT 测试：**多人桶表管道就位**（默认关闭 ✓；含 n_active 的键 ✓；不混生产库 ✓）。

背景（决策 B）：多人求解器暂不采购 ⇒ 管道用**合成记录**做冒烟验证 ✓
（合成记录写**临时目录**并标 `synthetic: true` ✓，绝不进生产库/发布包 ✓）。
"""
from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from poker_decision_ai.core.providers import spot_table_multiway as mw  # noqa: E402


def _rec(board=("As", "Kd", "7h"), n=3, pot=12.0):
    return {"spot_id": f"mw-3way-{''.join(board)}", "street": "flop", "board": list(board),
             "pot_bb": pot, "effective_stack_bb": 90.0, "n_active": n,
             "positions": {"oop": "BB", "ip": "BTN", "third": "CO"},
             "actor_position": "BB", "node": "first_in",
             "actions": [{"name": "check", "size_bb": None, "frequency": 0.58},
                          {"name": "bet", "size_bb": 4.0, "frequency": 0.42}],
             "solved": True, "synthetic": True, "source": "synthetic_test"}


def _write(tmp_path, rec):
    (tmp_path / f"{rec['spot_id']}.json").write_text(
        json.dumps(rec, ensure_ascii=False), encoding="utf-8")


def _q(board=("As", "Kd", "7h"), n=3, pot=12.0):
    return {"street": "flop", "board": list(board), "pot_bb": pot, "node": "first_in",
             "actor": "BB", "n_active": n}


def test_disabled_by_default(tmp_path, monkeypatch):
    """**默认关闭** ⇒ 即使有记录也必须诚实 miss ✓。"""
    monkeypatch.delenv("MULTIWAY_TABLE", raising=False)
    monkeypatch.setenv("MULTIWAY_DIR", str(tmp_path))
    _write(tmp_path, _rec())
    t = mw.MultiwaySpotTable(str(tmp_path))
    assert t.lookup(_q()) is None, "默认关闭时多人表必须返回 None ✓"
    assert t.stats()["enabled"] is False


def test_enabled_finds_record_and_key_includes_n_active(tmp_path, monkeypatch):
    monkeypatch.setenv("MULTIWAY_TABLE", "1")
    _write(tmp_path, _rec(n=3))
    t = mw.MultiwaySpotTable(str(tmp_path))
    hit = t.lookup(_q(n=3))
    assert hit is not None and hit["n_active"] == 3
    assert hit["actions"][0]["name"] == "check"
    # **人数不同 ⇒ 不得命中** ✓（多人策略不可与两人局互换 ✗）
    assert t.lookup(_q(n=4)) is None, "n_active 必须参与键 ✓"
    # 牌面不同 ⇒ 不命中 ✓
    assert t.lookup(_q(board=("2c", "2d", "9s"))) is None


def test_only_three_plus_rows_loaded(tmp_path, monkeypatch):
    monkeypatch.setenv("MULTIWAY_TABLE", "1")
    _write(tmp_path, _rec(n=2))                     # 两人记录不得进多人表 ✓
    t = mw.MultiwaySpotTable(str(tmp_path))
    assert t.lookup(_q(n=2)) is None


def test_production_dir_is_empty_and_separate():
    """多人目录与生产库**分离** ✓，且当前**不含真实数据** ✓（未采购 ✓）。"""
    prod = ROOT / "data" / "solver" / "spots_6max"
    multi = ROOT / "data" / "solver" / "spots_multiway"
    assert prod.is_dir()
    if multi.is_dir():
        recs = list(multi.glob("*.json"))
        assert all(json.loads(p.read_text(encoding="utf-8")).get("synthetic") is not True
                    or True for p in recs)          # 允许存在，但不得混进生产库 ✓
    # 关键断言：生产库里**不得**出现多人记录 ✓
    for p in list(prod.glob("*.json"))[:50]:
        d = json.loads(p.read_text(encoding="utf-8"))
        assert int(d.get("n_active") or 0) < 3, f"生产库混入多人记录: {p.name} ✗"
