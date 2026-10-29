"""P13-B — 实际识别闭环验证（只读分析工具）。

对 P13-A 分组帧（值→帧 映射，目视/上下文已验证）与全部 282 帧运行
BetRecognizer(window="1280") → 统计：每值正确帧数（含 distinct frame_id）、
未分组帧上的伪阳性、以及全部 UNKNOWN 原因分布。
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from poker_decision_ai.vision2_platform.png_decode import (  # noqa: E402
    decode_png)
from poker_decision_ai.vision2.pixel.realclient.coinpoker_bet import (  # noqa: E402
    BetRecognizer)

# P13-A 已验证映射（hash → (value, zone)）——从 p13a_betlabels.json 的组应用
VALUE_MAP = {
    "A_1d42a42e": (2000, "S_TL"), "A_3bffb39a": (1000, "S_TL"),
    "A_8ec40e7f": (193400, "S_TL"), "A_dc504ef3": (128190, "S_TL"),
    "A_ffad76eb": (14000, "S_TL"), "A_941dc26a": (102000, "S_TL"),
    "A_e411a259": (4000, "S_TL"), "A_e0824efd": (28000, "S_TL"),
    "A_6d35ef1a": (9280, "S_TL"), "B_830ca7ec": (2000, "S_TR"),
    "B_468552f5": (1000, "S_TR"), "B_32b691ab": (199680, "S_TR"),
    "B_687476fc": (128190, "S_TR"), "B_88f1f91e": (10450, "S_TR"),
    "B_53039807": (890401, "S_TR"), "B_1db7f6d5": (4000, "S_TR"),
    "B_51b40e9b": (28000, "S_TR"),
}


def frame_match(fname: str, stem: str) -> bool:
    return fname.rsplit("_", 1)[0] == stem.rsplit("_", 1)[0] and \
        int(re.search(r"_(\d+)_", fname).group(1)) == \
        int(re.search(r"_(\d+)_", stem).group(1))


def main() -> int:
    import datetime as _dt
    since_s = 134324017101490527 / 1e7 - 11644473600
    fulls = sorted(Path("data/vision2/coinpoker_evidence").glob("*_full.png"))
    new = [p for p in fulls if p.stat().st_mtime >= since_s]
    rec = BetRecognizer(window="1280")
    print("templates:", sorted(rec._templates))

    # 1) 全部帧识别 + 按 P13-A 组统计
    groups = json.loads(Path("data/vision2/p13a_betlabels.json")
                        .read_text(encoding="utf-8"))
    group_of_frame = {}
    for name, upper in (("A", "S_TL"), ("B", "S_TR")):
        for hsh, g in groups["labels"][name]["value_groups"].items():
            val, zone = VALUE_MAP.get(f"{name}_{hsh[:8]}", (None, None))
            if val is None:
                continue
            for f in g["frames"]:
                group_of_frame[(f, zone)] = (val, hsh[:8], g["n"],
                                             len(g.get("frame_ids", [])))
    stats = {}
    fp = []          # 未分组帧上的 RECOGNIZED（伪阳性调查）
    unk_reasons = {}
    nframes = 0
    for path in new:
        nframes += 1
        w, h, rgba = decode_png(path.read_bytes())
        arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
        res = rec.recognize(arr)
        for zone, r in res.items():
            key = (path.stem, zone)
            if key in group_of_frame:
                val, hsh, n, dids = group_of_frame[key]
                st = stats.setdefault((val, zone, hsh, n, dids),
                                      {"ok": 0, "bad": 0, "unknown": 0,
                                       "frames": []})
                if r["status"] == "RECOGNIZED" and r["bet"] == val:
                    st["ok"] += 1
                    st["frames"].append(path.stem)
                elif r["status"] == "RECOGNIZED":
                    st["bad"] += 1
                    st["frames"].append(path.stem + "=>" + str(r["bet"]))
                else:
                    st["unknown"] += 1
                    st["frames"].append(path.stem + "=>" + r["reason"])
            else:
                if r["status"] == "RECOGNIZED":
                    fp.append((path.stem, zone, r))
                else:
                    unk_reasons[r["reason"]] = \
                        unk_reasons.get(r["reason"], 0) + 1
    print("frames:", nframes)
    print("=== per-value results (ok/bad/unknown) ===")
    for (val, zone, hsh, n, dids), st in sorted(stats.items(),
                                                key=lambda kv: -kv[0][0]):
        print(f"value={val:7d} zone={zone} group_n={n} distinct_ids={dids} "
              f"ok={st['ok']} bad={st['bad']} unknown={st['unknown']}")
    print("=== false positives on ungrouped frames ===", len(fp))
    for f in fp[:15]:
        print("  ", f[0], f[1], f[2])
    print("=== UNKNOWN reasons (ungrouped) ===",
          dict(sorted(unk_reasons.items(), key=lambda kv: -kv[1])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
