"""从 play_history（source=external 会话）重建外部牌谱分析三件套。

用法：.venv/Scripts/python.exe scripts/rebuild_external_report.py [EXE_DIR]
缺省 EXE_DIR = dist/PokerAI-Product，输出到 data/import/analysis_report_dualang.{txt,json,html}
不走 AI：明细行直接取自导入时落盘的 hero_decisions（advisor/provenance）。
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cash_game_ui.weakness_report import build_report  # noqa: E402

STREETS = ("preflop", "flop", "turn", "river")
POS_LABELS = ("BTN", "SB", "BB", "UTG", "MP", "CO")


def hero_position(button: int, table_size: int) -> str:
    return POS_LABELS[(0 - button) % max(1, table_size)]


def hand_id_from_record(rec: str) -> str:
    import re
    m = re.search(r"Hand #(\d+)", rec or "")
    return m.group(1) if m else "?"


def main() -> int:
    exe = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "dist" / "PokerAI-Product"
    hist = exe / "data" / "play_history"
    rows: list[dict] = []
    n_hands = 0
    sessions = 0
    for fp in sorted(hist.glob("play_session_*.jsonl")):
        lines = fp.read_text(encoding="utf-8").splitlines()
        if not lines:
            continue
        try:
            head = json.loads(lines[0])
        except json.JSONDecodeError:
            continue
        if head.get("source") != "external":
            continue
        sessions += 1
        for line in lines[1:]:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            n_hands += 1
            rec = r.get("record") if isinstance(r.get("record"), str) else ""
            hid = hand_id_from_record(rec)
            button = r.get("button") or 0
            ts = r.get("n_players_dealt") or 6
            pos = hero_position(button, min(6, max(2, int(ts))))
            for hd in (r.get("hero_decisions") or []):
                adv = hd.get("advisor") or {}
                if not adv.get("available"):
                    continue
                prov = adv.get("provenance") or {}
                src = prov.get("source") or ""
                rows.append({
                    "hand": hid,
                    "street": hd.get("street") or "?",
                    "hero": hd.get("action") or "?",
                    "amount": hd.get("amount_chips"),
                    "ai": adv.get("action"),
                    "ai_size": adv.get("size_bb"),
                    "conf": None,
                    "source": src,
                    "status": "ok" if src == "preflop_chart" else "degraded",
                    "agree": hd.get("advisor_agree"),
                    "position": hd.get("position") or pos,
                })
    ai_rows = [r for r in rows if r.get("ai")]
    n_agree = sum(1 for r in ai_rows if r.get("agree"))
    by_street = {st: {"n": 0, "agree": 0} for st in STREETS}
    for r in ai_rows:
        st = by_street.setdefault(r["street"], {"n": 0, "agree": 0})
        st["n"] += 1
        st["agree"] += 1 if r.get("agree") else 0
    summary = {
        "sessions": sessions,
        "hands": n_hands,
        "decisions": len(rows),
        "ai_advice": len(ai_rows),
        "agree": n_agree,
        "agree_pct": round(n_agree / max(1, len(ai_rows)) * 100, 1),
        "by_street": by_street,
    }
    out_dir = ROOT / "data" / "import"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "analysis_report_dualang.json").write_text(
        json.dumps({"n_hands": n_hands, "rows": rows, "summary": summary},
                   ensure_ascii=False, indent=1), encoding="utf-8")
    txt = build_report(str(hist), source="external")
    (out_dir / "analysis_report_dualang.txt").write_text(txt, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"json rows: {len(rows)}  -> {out_dir / 'analysis_report_dualang.json'}")
    print(f"txt -> {out_dir / 'analysis_report_dualang.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
