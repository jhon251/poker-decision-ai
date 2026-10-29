"""GATE-1 战役进度/判决一站式查看（只读）。"""
import json, pathlib, sys
ROOT = pathlib.Path(r"<repo>")
OUT = ROOT / "docs" / "m4_gate" / "gate_gate1"
TOTAL = 50
done = []
for i in range(TOTAL):
    jf, rf = OUT / f"s{i:03d}.json", OUT / f"s{i:03d}.records.jsonl"
    if jf.is_file() and rf.is_file():
        a = json.loads(jf.read_text(encoding="utf-8"))
        done.append((i, a["n_hands_completed"], a["bb_per_100_rakefree"],
                     a["bb_per_100_net"], a["rake_total_bb"],
                     a["latency_ms_p90"]))
print(f"segments complete: {len(done)}/{TOTAL}")
for i, n, rf_, net, rake, p90 in done:
    print(f"  s{i:03d} hands={n} rakefree={rf_} net={net} "
          f"rake_bb={round(rake,1)} p90={round(p90)}ms")
if len(done) == TOTAL:
    print("ALL SEGMENTS COMPLETE — run:")
    print("  .venv/Scripts/python.exe scripts/run_m4_gate_campaign.py "
          "--campaign gate_gate1 --summarize")
else:
    print("missing:", [i for i in range(TOTAL)
                       if i not in {d[0] for d in done}][:10], "...")
