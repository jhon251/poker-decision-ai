"""V3E-P2 — 修复后 HeroCardsRecognizer 真实帧全量覆盖扫描（只读观察，零修改）。

对 data/vision2/coinpoker_evidence 全部帧跑冻结 HeroCardsRecognizer1876.recognize()：
  - 覆盖矩阵：card position × rank × suit 的 KNOWN / UNKNOWN；
  - UNKNOWN 原因分布（no_pair / gate_failed / 低分）；
  - 形近碰撞监测：T/J rank 窄 margin、c/s suit 窄 margin（仅记录，不放宽）；
  - 相邻同时刻帧值翻转（同手牌一致性诊断）。

输出：logs/v3e_p2_coverage_scan.json（全帧明细）+ 汇总打印。
"""
from __future__ import annotations

import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poker_decision_ai.vision2.pixel.realclient.coinpoker_herocards1876 import (
    HeroCardsRecognizer1876)
from poker_decision_ai.vision2_platform.png_decode import decode_png

EV = Path("data/vision2/coinpoker_evidence")
OUT = Path("logs/v3e_p2_coverage_scan.json")

TJ = {"T", "J"}
CS = {"c", "s"}


def main() -> int:
    frames = sorted(p.name for p in EV.glob("*_full.png"))
    rec = HeroCardsRecognizer1876()
    details = []
    t0 = time.time()
    n_pair = n_ok = n_unknown = 0
    reason_counter = Counter()
    rank_known = defaultdict(Counter)   # pos -> char -> count
    rank_unknown_reason = defaultdict(Counter)
    suit_known = defaultdict(Counter)
    collisions = []
    values = []  # (file, ts_tag, value)
    for i, f in enumerate(frames):
        w, h, rgba = decode_png((EV / f).read_bytes())
        arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
        out = rec.recognize(arr)
        cards = out.get("cards") or []
        pair_present = out.get("reason") != "no_card_pair_present" and cards
        if pair_present:
            n_pair += 1
        row = {"frame": f, "status": out.get("status"),
               "reason": out.get("reason"), "value": out.get("value"),
               "cards": []}
        for c in cards:
            ev = c.get("evidence") or {}
            entry = {"status": c.get("status"),
                     "rank": c.get("rank"), "suit": c.get("suit"),
                     "rank_score": ev.get("rank_score"),
                     "rank_margin": ev.get("rank_margin"),
                     "suit_score": ev.get("suit_score"),
                     "suit_margin": ev.get("suit_margin"),
                     "reason": c.get("reason")}
            row["cards"].append(entry)
            pos = 1 if c is cards[0] else 2
            if c.get("status") == "OK":
                n_ok += 1
                rank_known[pos][c["rank"]] += 1
                suit_known[pos][c["suit"]] += 1
                # 形近碰撞监测（仅窄 margin 记录）
                rm, sm = ev.get("rank_margin"), ev.get("suit_margin")
                if c.get("rank") in TJ and rm is not None and rm < 0.12:
                    collisions.append({"frame": f, "pos": pos, "kind": "rank_TJ",
                                       "winner": c["rank"], "margin": rm,
                                       "score": ev.get("rank_score")})
                if c.get("suit") in CS and sm is not None and sm < 0.06:
                    collisions.append({"frame": f, "pos": pos, "kind": "suit_cs",
                                       "winner": c["suit"], "margin": sm,
                                       "score": ev.get("suit_score")})
            else:
                rank_unknown_reason[pos][c.get("reason") or "gate"] += 1
        if out.get("status") != "OK" and out.get("reason"):
            reason_counter[out["reason"]] += 1
            n_unknown += 1
        if out.get("status") == "OK":
            values.append((f, out.get("value")))
        details.append(row)
        if (i + 1) % 2000 == 0:
            print("[%d/%d] %.1fs" % (i + 1, len(frames), time.time() - t0),
                  flush=True)
    # 同手牌值翻转诊断：相邻帧(按文件名时间戳排序)值变化
    flips = []
    for a, b in zip(values, values[1:]):
        if a[1] != b[1]:
            flips.append({"a": a[0], "a_value": a[1], "b": b[0],
                          "b_value": b[1]})
    result = {
        "n_frames": len(frames),
        "n_frames_pair_present": n_pair,
        "n_cards_ok": n_ok,
        "n_frames_unknown": n_unknown,
        "frame_reason_counter": dict(reason_counter),
        "rank_known": {str(k): dict(v) for k, v in sorted(rank_known.items())},
        "suit_known": {str(k): dict(v) for k, v in sorted(suit_known.items())},
        "rank_unknown_reason": {str(k): dict(v)
                                for k, v in sorted(rank_unknown_reason.items())},
        "n_collisions": len(collisions),
        "collisions": collisions[:200],
        "n_value_flips": len(flips),
        "value_flips": flips[:200],
        "elapsed_s": round(time.time() - t0, 1),
        "details": details,
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print("frames:", len(frames), "pair:", n_pair, "cards_ok:", n_ok,
          "unknown_frames:", n_unknown, "collisions:", len(collisions),
          "flips:", len(flips), "elapsed:", result["elapsed_s"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
