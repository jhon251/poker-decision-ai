"""EXE 冒烟报告校验（UTF-8 权威读取——临时脚本）。"""
import json
from pathlib import Path

r = json.loads(Path("dist_clean/poker-ai-preview/report_smoke.json")
               .read_text(encoding="utf-8"))
print("status      :", r.get("status"))
d = r.get("decision") or {}
print("decision    :", d.get("action"), "size_bb=", d.get("size_bb"),
      "conf=", d.get("confidence"), "source=", d.get("source"))
print("agents ran  :", len(r.get("agents") or []))
ps = r.get("provider_status") or {}
for k in ("llm", "vision", "solver"):
    info = ps.get(k) or {}
    print("provider %-6s: status=%r mock=%s reason=%r"
          % (k, info.get("status"), info.get("is_mock"),
             info.get("reason")))
game = r.get("game") or {}
print("game_mode   :", game.get("mode") or r.get("game_mode"))
