#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P30-9-HY：**对手档案库报表**（读本地 SQLite；只列证据计数 + 样本量 ✓）。

原则：
  * **只报证据计数**（遭遇数 / 在手数 / 弃牌证据 / 投入事件 ✓），
    **不做** VPIP/PFR/盈利率等推断 ✗（单帧视觉拿不到动作序列 ✓）；
  * 每行都带**样本量**与 `enough_sample` 判定 ✓；样本不足时显式标注"不可作结论" ✓；
  * 打印**来源分布**（runtime / test / frames_replay ✓）⇒ 真实与测试数据可区分 ✓。
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from poker_decision_ai.core import opponents as op  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=None, help="默认 data/opponents.db（EXE 下为 dist/data/…）")
    ap.add_argument("--min-sample", type=int, default=20)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--dist", type=int, default=10,
                     help="显示前 N 位对手的位置/街分布（0 = 不显示）")
    args = ap.parse_args()

    path = pathlib.Path(args.db) if args.db else None
    s = op.stats(path, min_sample=args.min_sample)
    if args.json:
        print(json.dumps(s, ensure_ascii=False, indent=1))
        return
    if not s.get("exists"):
        print(f"对手库尚未建立：{s.get('db')} ✓（下一次点击识别会自动创建 ✓）")
        return
    print(f"对手库: {s['db']}")
    print(f"遭遇(点击) {s['encounters']} | 逐座位观测 {s['observations']} | "
          f"对手 {s['players']} | 迁移推断 {s['transitions']}")
    print(f"来源分布: {s.get('by_source')}  （runtime=真实对局 ✓）")
    print(f"\n{'对手':<16}{'遭遇':>6}{'在手':>6}{'弃牌证据':>8}{'投入事件':>8}  {'样本':<10}位置/筹码")
    for r in (s.get("rows") or [])[:args.top]:
        flag = "足够 ✓" if r["enough_sample"] else f"不足(<{args.min_sample}) ⇒ 不可作结论"
        print(f"{str(r['player'])[:15]:<16}{r['encounters']:>6}{r['in_hand']:>6}"
              f"{r['fold_evidence']:>8}{r['invest_events']:>8}  {flag:<32}"
              f"{r.get('last_position') or '-'}/{r.get('last_stack') or '-'}")
    # P30-9-HZ：**各位置 / 各街分布** ✓（分母 = 该对手遭遇数 ✓；是遭遇分布 ✗ 不是频率推断）
    if args.dist and (s.get("rows") or []):
        print(f"\n各位置 / 各街**遭遇分布**（分母=该对手遭遇数 ✓；不是频率推断 ✗）"
              f"  取前 {args.dist} 位对手：")
        for r in (s.get("rows") or [])[:args.dist]:
            tot = r["encounters"]
            pos = " ".join(f"{k}:{v}" for k, v in list((r.get("by_position") or {}).items())[:6])
            st = " ".join(f"{k}:{v}" for k, v in (r.get("by_street") or {}).items())
            inh = " ".join(f"{k}:{v}" for k, v in
                            list((r.get("in_hand_by_position") or {}).items())[:6])
            print(f"  {str(r['player'])[:15]:<16} n={tot} | 位置[{pos}] | 街[{st}] | 在手(按位置)[{inh}]")
    # P30-9-IA：**动作标签分布**（每条标签都是从可见筹码证据推导 ✓，带依据+置信 ✓）
    if (s.get("rows") or []):
        print(f"\n动作标签分布（**推导**自可见筹码证据 ✓；分母=该对手遭遇数 n ✓；"
              f"**不含过牌** ✗——两帧稀疏序列无法可靠判定 ✓）：")
        for r in (s.get("rows") or [])[:args.dist or 10]:
            mix = " ".join(f"{k}:{v}" for k, v in (r.get("by_action") or {}).items()) or "（暂无）"
            print(f"  {str(r['player'])[:15]:<16} n={r['encounters']} | 动作[{mix}]"
                  f" | 均置信 {r.get('mean_action_confidence')}")
        print("  标签含义：fold 在池→不在池 ✓ | bet 首个下注 ✓ | call 与他人下注持平 ✓ | "
              "raise 超过他人下注 ✓ | allin 筹码≈0 或下注≥全部筹码 ✓ | blind 翻前强制投入 ✓")
    print(f"\n注意：{s['note']}")


if __name__ == "__main__":
    main()
