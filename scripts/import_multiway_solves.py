#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P30-9-HT：**多人（3-way+）解导入器**（管道就位 ✓；数据源采购后再用 ✓）。

用法（将来买到 3-way 求解器导出后）：
    python scripts/import_multiway_solves.py --src <导出目录> --dry-run
    python scripts/import_multiway_solves.py --src <导出目录>

期望的**输入**（我们定义的通用 schema；导入端做严格校验 ✓）：
    {"spot_id": "...", "street": "flop|turn|river", "board": ["As","Kd","7h"],
     "pot_bb": 12.0, "effective_stack_bb": 90.0, "n_active": 3,
     "positions": {"oop": "BB", "ip": "BTN", "third": "CO"},
     "actor_position": "BB", "node": "first_in|facing_bet",
     "actions": [{"name": "check", "size_bb": null, "frequency": 0.42}, ...],
     "ranges": {...},  # 若数据源提供范围，原样保留（如实标注 ✓）
     "source": "simple3way"}

写出到 `data/solver/spots_multiway/`（**不参与 dist 打包** ✓，与生产库隔离 ✓）；
`synthetic` 一律置 `false`（导入器不伪造 ✓）；`--dry-run` 只校验不写盘 ✓。
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_DST = ROOT / "data" / "solver" / "spots_multiway"
VALID_ACTIONS = {"check", "bet", "call", "fold", "raise", "allin"}


def validate(rec: dict) -> tuple[bool, str]:
    if int(rec.get("n_active") or 0) < 3:
        return False, "n_active < 3（多人表只收 3 人以上 ✓）"
    if str(rec.get("street") or "") not in ("flop", "turn", "river"):
        return False, "street 非法"
    board = [str(c) for c in (rec.get("board") or [])]
    need = {"flop": 3, "turn": 4, "river": 5}[str(rec.get("street"))]
    if len(board) != need:
        return False, f"牌面张数 {len(board)} ≠ {need}"
    if not rec.get("actions"):
        return False, "缺 actions"
    for a in rec["actions"]:
        if str(a.get("name") or "").lower() not in VALID_ACTIONS:
            return False, f"非法动作 {a.get('name')!r}"
    if not rec.get("actor_position"):
        return False, "缺 actor_position（多人节点必须明确行动者 ✓）"
    return True, "ok"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", default=str(DEFAULT_DST))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    src = pathlib.Path(args.src)
    if not src.is_dir():
        print(f"源目录不存在: {src} ✗")
        sys.exit(2)
    dst = pathlib.Path(args.dst)
    written, skipped, errors = 0, 0, []
    for f in sorted(src.glob("*.json")):
        try:
            rec = json.loads(f.read_text(encoding="utf-8"))
        except Exception as e:                                 # noqa: BLE001
            errors.append((f.name, f"解析失败 {e}"))
            continue
        ok, why = validate(rec)
        if not ok:
            errors.append((f.name, why))
            continue
        rec = {**rec, "solved": True, "synthetic": False,
                "source": rec.get("source") or "unknown_multiway_source",
                "source_note": (rec.get("source_note")
                                 or "多人解（采购数据源导入；范围/来源见本记录 ✓）")}
        sid = str(rec.get("spot_id") or f.stem)
        if not args.dry_run:
            dst.mkdir(parents=True, exist_ok=True)
            (dst / f"{sid}.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1),
                                              encoding="utf-8")
        written += 1
    print(f"校验通过 {written} 条，跳过 {skipped} 条，错误 {len(errors)} 条"
          + ("（dry-run，未写盘 ✓）" if args.dry_run else f" ⇒ 已写入 {dst} ✓"))
    for name, why in errors[:10]:
        print(f"  ✗ {name}: {why}")


if __name__ == "__main__":
    main()
