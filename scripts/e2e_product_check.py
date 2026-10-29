"""P30-9-GW：端到端产品链路检查（**真实帧**走 GUI 路径）。

验证四类行为在真实帧上按预期出现：
  ① 牌面残缺 ⇒ 本地补读（board_source=py_reader_fallback / 警告 BOARD_FALLBACK_READ）或诚实阻断；
  ② 多人池（≥3）⇒ multiway_note / MULTIWAY_NO_REFERENCE；
  ③ turn/river 两人池 ⇒ 现场真解（postflop_live）；
  ④ 库内真解 ⇒ 求解器锚定。
"""
from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, r"<repo>")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from PIL import Image                                    # noqa: E402
from cash_game_ui.runner import CashGameRunner           # noqa: E402


def main() -> None:
    # P30-9-HQ：帧回放 ≠ 真实对局 ⇒ 证据事件标来源 ✓（否则被记成 runtime ✗）
    import os as _os
    _os.environ.setdefault("DSH_EVIDENCE_SOURCE", "frames_replay")
    d = pathlib.Path(r"<repo>\dist\logs\frames")
    frames = sorted(d.glob("*.png"))
    print(f"真实帧 {len(frames)} 张")
    runner = CashGameRunner()
    hits = {"fallback": 0, "multiway": 0, "live": 0, "anchored": 0, "blocked": 0}
    for f in frames:
        im = Image.open(f).convert("RGBA")
        try:
            res = runner.analyze(im.tobytes(), im.width, im.height, allow_recapture=False)
            dd = res.to_dict() if hasattr(res, "to_dict") else dict(res)
        except Exception as e:                            # noqa: BLE001
            print(f"{f.name[:36]:<38} 异常 {type(e).__name__}: {e}")
            continue
        txt = json.dumps(dd, ensure_ascii=False)
        fd = dd.get("final_decision") or {}
        br = (dd.get("bridge") or {}) if isinstance(dd.get("bridge"), dict) else {}
        obs = br.get("observation") or {}
        board = obs.get("board") or []
        flags = []
        if "BOARD_FALLBACK_READ" in txt or br.get("board_fallback"):
            flags.append("补读"); hits["fallback"] += 1
        if "MULTIWAY_NO_REFERENCE" in txt or dd.get("multiway_note"):
            flags.append("多人告知"); hits["multiway"] += 1
        if "mode=live_solve" in txt or "postflop_live" in txt:
            flags.append("现场真解"); hits["live"] += 1
        if "求解器锚定" in txt:
            flags.append("库锚定"); hits["anchored"] += 1
        if str(dd.get("status")) == "DECISION_UNAVAILABLE":
            flags.append("诚实阻断"); hits["blocked"] += 1
        print(f"{f.name[:36]:<38} status={str(dd.get('status')):<20} board={''.join(board):<10} "
              f"动作={str(fd.get('action') or '-'):<6} {'/'.join(flags) or '-'}")
    print("\n汇总:", hits)


if __name__ == "__main__":
    main()
