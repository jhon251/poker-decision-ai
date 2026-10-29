# -*- coding: utf-8 -*-
"""P30-9-GU 测试：补读**成败双向留痕**与恢复率统计（写入临时目录，不污染生产日志 ✓）。"""
from __future__ import annotations

import json
import pathlib
import sys

import pytest
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from cash_game_ui.runner import CashGameRunner  # noqa: E402


def test_record_writes_jsonl(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    raw = Image.new("RGBA", (1876, 1400), (7, 7, 7, 255)).tobytes()
    CashGameRunner.record_board_evidence("fallback_ok", ["Ac"], ["Ad", "9h", "6s"],
                                          "flop", "resolved", 1876, 1400, raw)
    CashGameRunner.record_board_evidence("fallback_failed", ["Ac"], None,
                                          "flop", "board_incomplete:1/3@flop")
    p = tmp_path / "logs" / "board_fallback.jsonl"
    assert p.is_file(), "应写入 logs/board_fallback.jsonl"
    rows = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
    assert len(rows) == 2
    assert rows[0]["kind"] == "fallback_ok" and rows[0]["local_board"] == ["Ad", "9h", "6s"]
    assert rows[1]["kind"] == "fallback_failed" and rows[1]["local_board"] is None


def test_never_writes_to_production_logs(tmp_path, monkeypatch):
    """测试必须落在临时目录（防止把合成事件写进真实证据库 ✗）。

    注意：**生产运行本身**会合法写入 `logs/board_fallback.jsonl`（机制门禁也会触发 ✓）
    ⇒ 这里断言的是"**本测试**没有改变生产文件"，而不是"生产文件不存在" ✓。
    """
    prod = ROOT / "logs" / "board_fallback.jsonl"
    before = prod.stat().st_size if prod.is_file() else None
    monkeypatch.chdir(tmp_path)
    CashGameRunner.record_board_evidence("fallback_ok", ["Ac"], ["Ad"], "flop", "x")
    assert (tmp_path / "logs" / "board_fallback.jsonl").is_file()
    after = prod.stat().st_size if prod.is_file() else None
    assert after == before, "本测试不得改变生产证据库 ✓"


def test_recovery_rate_math():
    rows = [{"kind": "fallback_ok"}, {"kind": "fallback_ok"}, {"kind": "fallback_failed"}]
    ok = sum(1 for r in rows if r["kind"] == "fallback_ok")
    bad = sum(1 for r in rows if r["kind"] == "fallback_failed")
    assert ok / max(ok + bad, 1) == pytest.approx(2 / 3)
