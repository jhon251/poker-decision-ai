"""P30-0 — Fallback/Default 全项目 grep 分类审计（只读）。"""
from __future__ import annotations
import re
from pathlib import Path

pats = {
    "table_size=6": r"table_size\s*=\s*6",
    "hero_seat=3": r"hero_seat\s*=\s*3\b",
    "hero_seat=S6": r"hero_seat\s*=\s*[\"']S6[\"']",
    "dealer fixed": r"dealer_seat\s*=\s*[\"']S\d[\"']",
    "pot=0": r"pot\s*[:=]\s*0(?:\b|\D)",
    "stack=0": r"stack\s*[:=]\s*0(?:\b|\D)",
    "bet=0": r"bet\s*[:=]\s*0(?:\b|\D)",
    "previous frame": r"previous\s*frame|prev_frame",
    "last known": r"last_known",
    "cache": r"cache",
    "fallback": r"fallback",
}

# 生产代码范围：排除 tests/、docs/、scripts/、data/
prod = Path("poker_decision_ai")
print("=== 生产代码范围 (poker_decision_ai/) ===")
for name, pat in pats.items():
    hits = []
    for p in prod.rglob("*.py"):
        src = p.read_text(encoding="utf-8")
        for m in re.finditer(pat, src, re.IGNORECASE):
            line = src[:m.start()].count("\n") + 1
            hits.append((p, line, m.group(0)[:60]))
    print(f"\n[{name}] {len(hits)} hits")
    for p, line, txt in hits[:4]:
        print(f"  {p.relative_to(prod)}:{line}: {txt}")
