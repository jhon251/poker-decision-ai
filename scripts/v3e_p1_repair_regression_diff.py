"""V3E-P1 ASSET REPAIR — BEFORE/AFTER 视觉回归 diff（逐帧逐字段）。"""
import json
import sys
from pathlib import Path

before = json.loads(Path("logs/v3e_p1_repair_before.json").read_text(encoding="utf-8"))
after = json.loads(Path("logs/v3e_p1_repair_after.json").read_text(encoding="utf-8"))

assert set(before) == set(after), "frame sets differ"
diffs = {}
for f in sorted(before):
    b, a = before[f], after[f]
    if b != a:
        fields = []
        for k in sorted(set(b) | set(a)):
            if b.get(k) != a.get(k):
                fields.append(k)
        diffs[f] = fields
        print("CHANGED", f)
        for k in fields:
            bs = json.dumps(b.get(k), ensure_ascii=False)
            as_ = json.dumps(a.get(k), ensure_ascii=False)
            print("   %-22s %s -> %s" % (k, bs[:160], as_[:160]))
print("frames total:", len(before), "changed:", len(diffs))
if not diffs:
    print("IDENTICAL — 43-frame regression: PASS")
