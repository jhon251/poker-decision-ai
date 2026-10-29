#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""语料打分：一条命令给出识别覆盖率（一劳永逸的回归门）。

用法：python scripts/vision_score.py [--frames-dir data/vision2/coinpoker_evidence]
输出每个语料帧的字段命中情况与总体覆盖率；**覆盖率不足 100% 时退出码非 0**。

语料期望值写在 `corpus` 内联表（随语料一起维护）：每帧给出
  hero_cards / pot / street / bets（期望的 bet 环键值）。
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

#: 语料期望（键 = 文件名；值 = 期望字段）
CORPUS = {
    "user_2250_preflop_q6s.png": {
        "hero_cards": ["Qh", "6h"], "pot": 0.24, "street": "preflop",
        "bets": {"S1": 0.02, "S2": 0.05, "S3": 0.12, "S4": 0.05, "S6": None},
    },
    "user_2300_preflop_kts.png": {
        "hero_cards": ["Kh", "Th"], "pot": 0.4, "street": "preflop",
        "bets": {"S4": 0.25, "S5": 0.15},
    },
    "20261010-114807-DECISION_UNAVAILABLE.png": {
        "hero_cards": ["Qs", "2s"], "street": "preflop", "bets": {},
    },
    "20261010-114822-DECISION_UNAVAILABLE.png": {
        "hero_cards": ["Qs", "2s"], "street": "flop", "bets": {},
    },
    "20261010-112328-DECISION_UNAVAILABLE.png": {
        "hero_cards": ["Ts", "7c"], "street": "preflop", "bets": {},
    },
    "20261010-113101-DECISION_UNAVAILABLE.png": {
        "hero_cards": ["3h", "2d"], "street": "preflop", "bets": {},
    },
    "8bf82f134651db0add626be0304a2ff1.png": {
        "hero_cards": ["5c", "3c"], "pot": 8920.0, "street": "preflop",
        "bets": {"S1": 1000, "S2": 2000, "S3": 2000, "S4": 2000, "S6": 1920},
    },
}


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:                                       # noqa: BLE001
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-dir", default="data/vision2/coinpoker_evidence")
    a = ap.parse_args()
    bases = [ROOT / a.frames_dir, ROOT / "dist" / "logs" / "frames"]
    total = hit = 0
    for name, expect in CORPUS.items():
        path = next((b / name for b in bases if (b / name).exists()), None)
        if path is None:
            print(f"[skip] 缺帧 {name}")
            continue
        import numpy as np
        from poker_decision_ai.bet_capsule_finder import (
            find_bet_capsules, to_bet_ring_labels)
        from poker_decision_ai.vision2.pixel.realclient import production_bridge as pb
        from poker_decision_ai.vision2.pixel.realclient.hero_seat import (
            SEAT_ANCHORS_1876)
        from poker_decision_ai.vision2_platform.png_decode import decode_png
        w, h, rgba = decode_png(path.read_bytes())
        frame = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
        obs = pb.frame_to_decision_request(rgba, w, h)["bridge"]["observation"] or {}
        stacks = {f"S{s.get('seat')}": s.get("stack")
                  for s in (obs.get("seats") or []) if isinstance(s, dict)}
        found = to_bet_ring_labels(find_bet_capsules(
            frame, SEAT_ANCHORS_1876,
            cross_check={"pot": obs.get("pot"), "stacks": stacks}).get("seats") or {})
        bets_merged = dict(obs.get("extra", {}).get("v3f_bets") or {})
        for k, v in found.items():
            if bets_merged.get(k) is None:
                bets_merged[k] = v
        checks = [("hero_cards", obs.get("hole_cards") == expect["hero_cards"],
                   obs.get("hole_cards"))]
        if "pot" in expect:
            checks.append(("pot", obs.get("pot") == expect["pot"],
                            obs.get("pot")))
        if "street" in expect:
            checks.append(("street", obs.get("street") == expect["street"],
                            obs.get("street")))
        for k, v in (expect.get("bets") or {}).items():
            if v is None:
                continue
            checks.append((f"bet:{k}", bets_merged.get(k) == v, bets_merged.get(k)))
        print(f"\n=== {name} ===")
        for field, ok, got in checks:
            total += 1
            hit += 1 if ok else 0
            print(f"  {'✅' if ok else '❌'} {field:14s} got={got}")
    rate = (100.0 * hit / total) if total else 0.0
    print(f"\n覆盖率: {hit}/{total} = {rate:.1f}%")
    return 0 if total and hit == total else 1


if __name__ == "__main__":
    sys.exit(main())
