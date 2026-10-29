#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P30-9-FS：从**真实会话日志**统计「街 × 在池人数」占比。

目的：用数据回答"该不该为 3-way（$249）与 turn/river 花钱"——
  * 若多人在池的翻后决策占比很低 ⇒ 3-way 优先级低；
  * 若 turn/river 决策占比高 ⇒ turn/river 覆盖优先。

数据源：`dist/logs/events-*.jsonl` 的 `vision_observation`（真实客户端会话）。
另：列出 `data/raw`、`data/hand_history`、`data/normalized` 的规模，判断能否做**更大样本**代理。
"""
from __future__ import annotations

import collections
import json
import pathlib
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = pathlib.Path(r"<repo>")
LOG = ROOT / "dist" / "logs"


def decisions() -> list[dict]:
    """按 request_id 归并出决策点（取每次请求的 vision_observation）。"""
    seen: dict[str, dict] = {}
    for f in sorted(LOG.glob("events-*.jsonl")):
        for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                d = json.loads(line)
            except Exception:                                    # noqa: BLE001
                continue
            if d.get("event") != "agent_invocation" or d.get("agent") != "vision":
                continue
            rid = str(d.get("request_id") or "")
            vo = ((d.get("input") or {}).get("request") or {}).get("vision_observation")
            if not vo:
                continue
            rec = seen.setdefault(rid, {})
            rec.update({"street": vo.get("street"), "board": vo.get("board") or [],
                         "pot": vo.get("pot"), "to_call": vo.get("to_call"),
                         "hero_seat": vo.get("hero_seat"), "button_seat": vo.get("button_seat"),
                         "seats": vo.get("seats") or [], "ts": d.get("ts"),
                         "hole": vo.get("hole_cards")})
    return [v for v in seen.values() if v.get("street")]


def main() -> None:
    ds = decisions()
    print(f"真实决策点（按 request_id 归并）: {len(ds)}")

    by_street = collections.Counter(str(d["street"]) for d in ds)
    print("\n按街:", dict(by_street))

    post = [d for d in ds if str(d["street"]) in ("flop", "turn", "river")]
    print(f"\n翻后决策: {len(post)}/{len(ds)} = {len(post)/max(len(ds),1)*100:.0f}%")

    def n_in(d: dict) -> int:
        return len([s for s in (d.get("seats") or []) if s.get("in_hand")])

    nactive = collections.Counter(n_in(d) for d in post)
    print("翻后「在池人数」分布:", dict(sorted(nactive.items())))
    multi = sum(v for k, v in nactive.items() if k >= 3)
    print(f"  ⇒ **多人（≥3）翻后占比: {multi}/{max(len(post),1)} = {multi/max(len(post),1)*100:.0f}%**")

    streets_post = collections.Counter(str(d["street"]) for d in post)
    print("翻后按街:", dict(streets_post))
    tr = sum(v for k, v in streets_post.items() if k in ("turn", "river"))
    print(f"  ⇒ **turn/river 占翻后: {tr}/{max(len(post),1)} = {tr/max(len(post),1)*100:.0f}%**")

    # 同时交叉：多人 × 街
    cross = collections.Counter((str(d["street"]), n_in(d)) for d in post)
    print("\n街 × 在池人数:", {f"{k[0]}/{k[1]}人": v for k, v in sorted(cross.items())})

    pots = sorted(float(d.get("pot") or 0) for d in post)
    if pots:
        print(f"\n翻后底池（筹码）: 中位 {pots[len(pots)//2]:.0f} | 最大 {pots[-1]:.0f}")

    out = ROOT / "docs" / "spot_mix.json"
    out.write_text(json.dumps({
        "source": "dist/logs/events-*.jsonl（真实客户端会话，vision_observation）",
        "n_decisions": len(ds), "by_street": dict(by_street),
        "postflop": {"n": len(post), "by_street": dict(streets_post),
                      "n_active_hist": {str(k): v for k, v in sorted(nactive.items())},
                      "multiway_share": round(multi / max(len(post), 1), 4),
                      "turn_river_share_of_postflop": round(tr / max(len(post), 1), 4),
                      "pot_median_chips": (pots[len(pots)//2] if pots else None)},
        "caveat": f"样本仅 {len(ds)} 个真实决策 ⇒ 占比估计置信区间宽"},
        ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n已写 {out.relative_to(ROOT)}")

    print("\n=== 其它可用语料（能否做更大样本代理） ===")
    for sub in ("raw", "hand_history", "normalized", "live_t1"):
        d = ROOT / "data" / sub
        if not d.is_dir():
            continue
        files = sorted(d.glob("*"))
        print(f"{sub}: {len(files)} 文件 | 样例 {files[0].name if files else '-'}")
        if files:
            head = files[0].read_text(encoding="utf-8", errors="replace")[:220].replace("\n", " ")
            print(f"    {head}")


if __name__ == "__main__":
    main()
