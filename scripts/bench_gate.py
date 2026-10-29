"""P30-9-CO：从基准逐手牌记录算出**回归门**指标并写入 docs/bench_regression_gate.json。"""
from __future__ import annotations

import json
import pathlib
import statistics
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = pathlib.Path(r"<repo>")


def _street_of(row: dict) -> str:
    """spot 形如 `3BP-flop-ace_high` / `FB-3BP-flop-ace_high` ⇒ 取街名。"""
    parts = str(row.get("spot") or "").split("-")
    for p in parts:
        if p in ("preflop", "flop", "turn", "river"):
            return p
    return "?"


def metrics(rows: list) -> dict:
    n = len(rows)
    if not n:
        return {"samples": 0}
    regret = [float(r.get("regret_bb") or r.get("regret") or 0.0) for r in rows]
    support = [r for r in rows if (r.get("in_support_hand")
                                    if "in_support_hand" in r else r.get("in_support"))]
    argmax = [r for r in rows
              if str(r.get("eq_argmax_hand") or "").strip().lower()
              == str(r.get("mapped") or r.get("engine") or "").strip().lower()]
    freq = [float(r.get("eq_freq_hand") or 0.0) for r in rows]
    # fb 套件无 regret/per-hand freq ⇒ 只用可算项，避免伪造 0 拉低/抬高指标
    has_regret = any(("regret_bb" in r or "regret" in r) for r in rows)
    out = {
        "samples": n,
        "support_hit_rate": round(len(support) / n, 4),
    }
    if argmax:
        out["argmax_rate"] = round(len(argmax) / n, 4)
    if has_regret:
        out["regret_mean"] = round(statistics.fmean(regret), 4)
        out["regret_max"] = round(max(regret), 4)
        out["regret_zero_rate"] = round(sum(1 for r in regret if r <= 1e-12) / n, 4)
    if any(f > 0 for f in freq):
        out["eq_freq_mean"] = round(statistics.fmean(freq), 4)
    return out


def by_street(rows: list) -> dict:
    out = {}
    for st in sorted({_street_of(r) for r in rows}):
        sub = [r for r in rows if _street_of(r) == st]
        m = metrics(sub)
        out[st] = {"samples": m.get("samples"),
                    "support_hit_rate": m.get("support_hit_rate"),
                    "regret_mean": m.get("regret_mean"),
                    "regret_max": m.get("regret_max")}
    return out


GATE = ROOT / "docs" / "bench_regression_gate.json"
gate = json.loads(GATE.read_text(encoding="utf-8")) if GATE.exists() else {"runs": []}
for suite, path, cmd in (
        ("r1", ROOT / ".tmp" / "bench_r3_r1.json",
         "bench_vs_solver.py --suite r1 --streets flop,turn,river --spots 8 --hands 30"),
        ("fb", ROOT / ".tmp" / "bench_r3_fb.json",
         "bench_vs_solver.py --suite fb --spots 6 --hands 30")):
    if not path.exists():
        continue
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data if isinstance(data, list) else (data.get("rows") or [])
    if not rows:
        print(suite, "无记录"); continue
    m = metrics(rows)
    gate["runs"].append({"round": "goal-3", "date": "2026-10-11", "suite": suite,
                          "cmd": cmd, "metrics": m, "by_street": by_street(rows)})
    print(f"{suite}: {json.dumps(m, ensure_ascii=False)}")
    print(f"   逐街: {json.dumps(by_street(rows), ensure_ascii=False)}")

gate["gate_rule"] = ("同一命令下：support_hit_rate / argmax_rate 不得下降；"
                      "regret_mean / regret_max 不得上升。超出即视为回退，需定位并复测。")
gate["record_keys"] = ["spot", "board", "hand", "regret_bb",
                        "in_support_hand", "eq_argmax_hand", "eq_freq_hand"]
GATE.write_text(json.dumps(gate, ensure_ascii=False, indent=1), encoding="utf-8")
print("已写入", GATE)
