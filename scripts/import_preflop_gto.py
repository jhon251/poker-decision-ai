#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把外部 GTO 翻前图表转换为本项目格式（P30-9-CA）。

来源与许可（**必须随产物保留**）：
  * 仓库：https://github.com/jensbaagaard/poker-practice
  * 其 README 声明：「License: MIT. The range data in `data/openSourcePokerData/`
    is free to use」⇒ 数据可自由使用；本转换器**仅做格式转换**，不改动数值。
  * 数据内容：Cash 100bb GTO（2.5bb open；另有 2bb/3bb 变体），
    **完整 solver 输出、带混合频率**；6-max 位置 LJ/HJ/CO/BTN/SB/BB。

输出 schema（frequency chart，供 PreflopChartProvider 读取）：
  {
    "chart": {
      "name": ..., "source": "external_gto_json", "source_note": <归属>,
      "open_sizes": {"default": 2.5, "SB": 2.5, "3bet_ratio_ip": 3.0, ...},
      "scenarios": {
        "open:LJ": {"kind": "open", "hero": "LJ", "size_bb": 2.5,
                     "raise": {hand: freq}, "call": {}},
        "vs_raise:BB:BTN": {"kind": "vs_raise", "hero": "BB", "villain": "BTN",
                             "size_bb": 7.5, "raise": {..}, "call": {..}},
        ...
      }
    }
  }
  规则：fold = 1 − raise − call（与来源 README 一致）。
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "data" / "external" / "poker_practice"
OUT = ROOT / "configs" / "solver" / "preflop_chart_6max_gto.json"

#: 来源位置 → 本项目位置（6-max：LJ=UTG、HJ=MP）
POS_MAP = {"LJ": "UTG", "HJ": "MP", "CO": "CO", "BTN": "BTN", "SB": "SB", "BB": "BB"}
SIX_MAX = set(POS_MAP)
ATTRIBUTION = ("来源：github.com/jensbaagaard/poker-practice（README 声明 MIT，"
                "data/openSourcePokerData 可自由使用）；数据集 {set}；"
                "完整 solver 输出、带混合频率；本文件为**格式转换**，未改动数值")
OPEN_SIZE = {"default": 2.5, "SB": 2.5}
THREE_BET_IP, THREE_BET_OOP = 3.0, 5.0     # 来源 sizing profile: cash_default
FOUR_BET_IP, FOUR_BET_OOP = 2.0, 2.5


def _nz(d: dict) -> dict:
    return {str(k): float(v) for k, v in (d or {}).items()
            if isinstance(v, (int, float)) and float(v) > 0}


def build(src: pathlib.Path, set_name: str) -> dict:
    raw = json.loads(src.read_text(encoding="utf-8"))
    scenarios: dict = {}

    def _three_bet_size(hero: str, villain: str) -> float:
        """来源 sizing profile：3-bet = 3x(IP) / 5x(OOP) × open。"""
        ip = ("LJ", "HJ", "CO", "BTN")
        open_bb = OPEN_SIZE.get(villain, OPEN_SIZE["default"])
        mult = THREE_BET_IP if hero in ip else THREE_BET_OOP
        return round(mult * open_bb, 2)

    def _size_for(node: str, hero: str, villain: str) -> float:
        """按来源 sizing profile 给出该节点 hero 的加注尺寸（bb）。"""
        if node == "open":
            return OPEN_SIZE.get(hero, OPEN_SIZE["default"])
        if node == "vs_raise":
            return _three_bet_size(hero, villain)
        if node == "vs_3bet":
            # 4-bet = 2x(IP) / 2.5x(OOP) × 3-bet 尺寸（3-bet 方即 villain）
            base = _three_bet_size(villain, "BTN")     # 近似：对手 3-bet 尺寸
            mult = FOUR_BET_IP if hero in ("LJ", "HJ", "CO", "BTN") else FOUR_BET_OOP
            return round(mult * base, 2)
        if node in ("vs_4bet", "vs_5bet"):
            return 0.0                                  # 5-bet = 全下（由 allin 表达）
        return 0.0

    for key, hands in raw.items():
        if key in ("EmptyRange", "WideRange"):
            continue
        if key.startswith("Open"):
            pos = key[4:]
            if pos not in SIX_MAX:
                continue
            scenarios[f"open:{pos}"] = {
                "kind": "open", "hero": pos, "villain": None,
                "size_bb": OPEN_SIZE.get(pos, OPEN_SIZE["default"]),
                "raise": _nz(hands), "call": {}}
            continue
        for prefix, kind in (("3Bet", "vs_raise"), ("4Bet", "vs_3bet"), ("5Bet", "vs_4bet"),
                              ("Call 3Bet", "vs_3bet"), ("Call 4Bet", "vs_4bet"),
                              ("Call 5Bet", "vs_5bet"), ("Call", "vs_raise")):
            if not key.startswith(prefix):
                continue
            rest = key[len(prefix):]
            if "vs" not in rest:
                continue
            hero, villain = rest.split("vs", 1)
            if hero not in SIX_MAX or villain not in SIX_MAX:
                break
            sc = scenarios.setdefault(f"{kind}:{hero}:{villain}",
                                       {"kind": kind, "hero": hero, "villain": villain,
                                        "size_bb": _size_for(kind, hero, villain),
                                        "raise": {}, "call": {}})
            # 3Bet/4Bet/5Bet → raise 列；Call* → call 列
            bucket = "raise" if prefix in ("3Bet", "4Bet", "5Bet") else "call"
            for h, f in _nz(hands).items():
                sc[bucket][h] = max(sc[bucket].get(h, 0.0), f)
            break

    return {"chart": {
        "name": f"preflop_6max_cash100_gto_{set_name}",
        "version": "p30-9-ca-preflop-gto-v1",
        "source": "external_gto_json",
        "source_note": ATTRIBUTION.format(set=set_name),
        "equilibrium": True,          # 来源自述为 solver 输出（非启发式）
        "positions": {k: POS_MAP[k] for k in ("LJ", "HJ", "CO", "BTN", "SB", "BB")},
        "open_sizes": OPEN_SIZE,
        "fold_rule": "fold = 1 - raise - call",
        "n_scenarios": len(scenarios),
        "scenarios": scenarios,
    }}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="Cash_100_GTO",
                    choices=["Cash_100_GTO", "Cash_100_GTO_2bb", "Cash_100_GTO_3bb",
                             "ALL"])
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    if args.set == "ALL":
        # P30-9-DB：三套开局尺寸合并进同一图表（场景键带尺寸标签 @2 / @2.5 / @3），
        # 默认（无标签）键 = 2.5bb 那套（向后兼容）。运行时按**观测到的开局筹码**
        # 选标签；缺该标签时回退默认键。
        variants = [("Cash_100_GTO_2bb", "2", 2.0),
                    ("Cash_100_GTO", "2.5", 2.5),
                    ("Cash_100_GTO_3bb", "3", 3.0)]
        merged: dict = {}
        note_bits = []
        for set_name, tag, size in variants:
            s = SRC_DIR / f"{set_name}.json"
            if not s.exists():
                print("缺源文件:", s)
                continue
            chart = build(s, set_name)["chart"]
            note_bits.append(f"{set_name}({size}bb)")
            for key, sc in (chart.get("scenarios") or {}).items():
                sc2 = dict(sc)
                sc2["size_bb"] = size
                merged[f"{key}@{tag}"] = sc2
                if tag == "2.5":
                    merged[key] = dict(sc2)          # 默认键（无标签）
        out = {"chart": {
            "name": "preflop_6max_cash100_gto_sizematch",
            "version": "p30-9-db-preflop-gto-size-v1",
            "source": "external_gto_json",
            "source_note": (ATTRIBUTION.format(set="+".join(note_bits))
                             + "；**三套开局尺寸合并**，场景键带 @2/@2.5/@3 标签，"
                               "无标签键=2.5bb 默认"),
            "equilibrium": True,
            "positions": {k: POS_MAP[k] for k in ("LJ", "HJ", "CO", "BTN", "SB", "BB")},
            "open_size_tags": {"2": 2.0, "2.5": 2.5, "3": 3.0},
            "fold_rule": "fold = 1 - raise - call",
            "n_scenarios": len(merged),
            "scenarios": merged,
        }}
        pathlib.Path(args.out).write_text(
            json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"已写出合并图表: {args.out}（场景 {len(merged)} 个，含尺寸标签）")
        return
    src = SRC_DIR / f"{args.set}.json"
    if not src.exists():
        print("缺源文件:", src)
        sys.exit(1)
    chart = build(src, args.set)
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(chart, ensure_ascii=False, indent=1), encoding="utf-8")
    sc = chart["chart"]["scenarios"]
    kinds: dict = {}
    for v in sc.values():
        kinds[v["kind"]] = kinds.get(v["kind"], 0) + 1
    print(f"已生成 {out.name}：{len(sc)} 个场景 | 类型分布 {kinds}")
    for k in sorted(sc)[:6]:
        v = sc[k]
        print(f"  {k:<22} raise={len(v['raise'])} 手 call={len(v['call'])} 手 "
              f"size={v['size_bb']}bb")
    print("  归属声明:", chart["chart"]["source_note"][:120], "…")


if __name__ == "__main__":
    main()
