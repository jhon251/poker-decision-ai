"""Dataset Qualification Tool（V2 Phase 4 §四·五·六·七）。

流程（不修改 Raw 数据）：
    data/hand_history/ 扫描
      ↓ 解析（手牌文本 / Canonical JSON）
      ↓ 标准化（Canonical Hand；缺失 → null，不猜）
      ↓ 质量检查（10 类，valid/invalid/warning + 原因）
      ↓ 分层落盘 data/{raw,normalized,validated}/
      ↓ Dataset Manifest（data/manifest.json + docs/DATASET_MANIFEST.md）
      ↓ 统计（按模式 / 玩家画像字段 / 缺失字段 / 日期范围）

用法: .\\.venv\\Scripts\\python.exe -m poker_decision_ai.dataset_tool qualify
      --dir data/hand_history --source "synthetic" --license "internal-synthetic"
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .core.orchestration.runtime import build_runtime
from .domain.poker.canonical import CanonicalHand, parsed_hand_to_canonical
from .domain.poker.hand_history import parse_hand_history
from .domain.poker.stats import compute_all
from .domain.poker.validator import validate_hand

LAYERS = ("raw", "normalized", "validated", "replay", "experiments")


@dataclass
class QualifyReport:
    input_dir: str = ""
    files_scanned: int = 0
    files_parsed: int = 0
    files_failed: int = 0
    hands: int = 0
    valid: int = 0
    invalid: int = 0
    warning: int = 0
    replayable: int = 0
    by_mode: dict = field(default_factory=dict)
    missing_fields: dict = field(default_factory=dict)
    date_range: dict = field(default_factory=dict)
    player_stats: dict = field(default_factory=dict)
    failures: list = field(default_factory=list)
    manifest_path: str = ""

    def to_dict(self) -> dict:
        return {
            "input_dir": self.input_dir, "files_scanned": self.files_scanned,
            "files_parsed": self.files_parsed, "files_failed": self.files_failed,
            "hands": self.hands, "valid": self.valid, "invalid": self.invalid,
            "warning": self.warning, "replayable": self.replayable,
            "by_mode": self.by_mode, "missing_fields": self.missing_fields,
            "date_range": self.date_range, "player_stats": self.player_stats,
            "failures": self.failures[:20], "manifest_path": self.manifest_path,
        }


class DatasetQualifier:
    def __init__(self, base_dir: str = "data", docs_dir: str = "docs") -> None:
        self.base = Path(base_dir)
        self.docs = Path(docs_dir)
        for layer in LAYERS:
            (self.base / layer).mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- 主流程
    def qualify(self, input_dir: str, *, source: str = "", license_info: str = "",
                copy_raw: bool = True) -> QualifyReport:
        report = QualifyReport(input_dir=input_dir)
        # 每次 qualify 重置校验报告文件（避免跨运行累积）
        vpath = self.base / "validated" / "validation_report.jsonl"
        if vpath.exists():
            vpath.unlink()
        in_dir = Path(input_dir)
        if not in_dir.exists():
            report.failures.append(f"input dir not found: {input_dir}")
            return report

        files = sorted(p for p in in_dir.rglob("*")
                       if p.suffix.lower() in (".txt", ".json") and p.is_file())
        report.files_scanned = len(files)
        canonical_hands: list[CanonicalHand] = []
        for fp in files:
            try:
                hand = self._load_hand(fp, source, license_info)
                if hand is None:
                    report.files_failed += 1
                    report.failures.append(f"{fp}: 无法解析（跳过的格式或空文件）")
                    continue
                report.files_parsed += 1
                canonical_hands.append(hand)
                if copy_raw:
                    self._copy_raw(fp, hand.hand_id)
                self._write_normalized(hand)
                self._write_validated(hand, report)
            except Exception as e:
                report.files_failed += 1
                report.failures.append(f"{fp}: {type(e).__name__}: {e}")

        report.hands = len(canonical_hands)
        self._aggregate(canonical_hands, report)
        self._write_manifest(report, source, license_info)
        return report

    # ---------------------------------------------------------------- 加载
    def _load_hand(self, fp: Path, source: str, license_info: str) -> CanonicalHand | None:
        rel = str(fp.relative_to(self.base.parent)) if self.base.parent.name in str(fp) else str(fp)
        if fp.suffix.lower() == ".txt":
            text = fp.read_text(encoding="utf-8", errors="replace")
            if not text.strip():
                return None
            parsed = parse_hand_history(text)
            if not parsed.hand_id:                 # 无头部 = 不是手牌历史 → 拒绝
                return None
            return parsed_hand_to_canonical(parsed, source=rel, license_info=license_info)
        if fp.suffix.lower() == ".json":
            data = json.loads(fp.read_text(encoding="utf-8", errors="replace"))
            if isinstance(data, dict) and data.get("hand_id"):
                hand = CanonicalHand.from_dict(data)
                hand.source = hand.source or rel
                hand.meta["license"] = hand.meta.get("license") or license_info
                return hand
            return None
        return None

    # ---------------------------------------------------------------- 分层落盘
    def _copy_raw(self, fp: Path, hand_id: str) -> None:
        """Raw 不可修改：内容相同则跳过；已存在同名不同内容 → 加 hash 后缀新存。"""
        raw_bytes = fp.read_bytes()
        digest = hashlib.sha1(raw_bytes).hexdigest()[:12]
        dest = self.base / "raw" / f"{fp.stem}_{digest}{fp.suffix}"
        if not dest.exists():
            dest.write_bytes(raw_bytes)

    def _write_normalized(self, hand: CanonicalHand) -> None:
        p = self.base / "normalized" / f"{hand.hand_id}.json"
        p.write_text(json.dumps(hand.to_dict(), ensure_ascii=False, indent=1),
                     encoding="utf-8")

    def _write_validated(self, hand: CanonicalHand, report: QualifyReport) -> None:
        v = validate_hand(hand)
        if v.status == "invalid":
            report.invalid += 1
            dest = self.base / "validated" / "invalid" / f"{hand.hand_id}.json"
        else:
            dest = self.base / "validated" / f"{hand.hand_id}.json"
            if v.status == "warning":
                report.warning += 1
            else:
                report.valid += 1
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(hand.to_dict(), ensure_ascii=False, indent=1),
                        encoding="utf-8")
        with open(self.base / "validated" / "validation_report.jsonl", "a",
                  encoding="utf-8") as fh:
            fh.write(json.dumps(v.to_dict(), ensure_ascii=False) + "\n")

    # ---------------------------------------------------------------- 统计
    def _aggregate(self, hands: list[CanonicalHand], report: QualifyReport) -> None:
        records: list[dict] = []
        ts_list: list[str] = []
        for hand in hands:
            mode = hand.game_mode or "unknown"
            report.by_mode[mode] = report.by_mode.get(mode, 0) + 1
            if hand.game_mode == "bounty":
                key = f"bounty:{hand.bounty_type or 'unknown'}"
                report.by_mode[key] = report.by_mode.get(key, 0) + 1
            if hand.started_at:
                ts_list.append(hand.started_at)
            for field in _missing_of(hand):
                report.missing_fields[field] = report.missing_fields.get(field, 0) + 1
            if _replayable(hand):
                report.replayable += 1
            records.extend(canonical_to_player_records(hand))
        if ts_list:
            report.date_range = {"min": min(ts_list), "max": max(ts_list),
                                 "n_with_timestamp": len(ts_list)}
        report.player_stats = _player_summary(records)

    def _write_manifest(self, report: QualifyReport, source: str, license_info: str) -> None:
        manifest = {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "source": source, "license": license_info,
            **report.to_dict(),
        }
        (self.base / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        report.manifest_path = str(self.base / "manifest.json")
        md = _manifest_markdown(manifest)
        self.docs.mkdir(parents=True, exist_ok=True)
        (self.docs / "DATASET_MANIFEST.md").write_text(md, encoding="utf-8")


# ---------------------------------------------------------------- 助手
def _missing_of(hand: CanonicalHand) -> list[str]:
    missing = []
    for field, value in (("started_at", hand.started_at), ("hero_id", hand.hero_id),
                         ("hole_cards", hand.hole_cards or None),
                         ("stack_end", any(p.stack_end is not None for p in hand.players)),
                         ("bounty", any(p.bounty is not None for p in hand.players)),
                         ("total_pot", hand.total_pot), ("winners", hand.winners or None),
                         ("showdown", hand.showdown)):
        if value in (None, "", [], False):
            missing.append(field)
    return missing


def _replayable(hand: CanonicalHand) -> bool:
    return bool(hand.hand_id and hand.hero_id and hand.hole_cards
                and any(p.stack_start is not None for p in hand.players))


def canonical_to_player_records(hand: CanonicalHand, now: float | None = None) -> list[dict]:
    """Canonical → 每玩家归一化统计记录（喂 Player/Population 模型）。"""
    now = time.time() if now is None else now
    stakes = hand.stakes or ""
    game = hand.game_type or "cash"
    records = []
    actions_by_player: dict[str, list] = {}
    for a in (hand.preflop_actions + hand.flop_actions + hand.turn_actions
              + hand.river_actions):
        actions_by_player.setdefault(a.player_id, []).append(a)
    winner_names = {w.get("player_id") for w in hand.winners}
    first_raiser = None
    for a in hand.preflop_actions:
        if a.action == "raise":
            first_raiser = a.player_id
            break
    for p in hand.players:
        acts = actions_by_player.get(p.player_id, [])
        pf = [a for a in acts if a.street == "preflop"]
        fl = [a for a in acts if a.street == "flop"]
        vpip = any(a.action in ("call", "raise") for a in pf) and not all(
            a.action == "post" for a in pf)
        pfr = any(a.action == "raise" for a in pf)
        faced_raise = any(a.action == "raise" and a.player_id != p.player_id for a in pf)
        folded_to_raise = faced_raise and any(a.action == "fold" for a in pf)
        cbet_opp = p.player_id == first_raiser and any(a.action in ("bet", "raise") for a in fl)
        faced_flop_bet = any(a.action in ("bet", "raise") and a.player_id != p.player_id
                             for a in hand.flop_actions)
        folded_to_flop_bet = faced_flop_bet and any(a.action == "fold" for a in fl)
        went_to_sd = bool(hand.showdown and hand.showdown.get("happened")
                          and any(s.get("player_id") == p.player_id
                                  for s in hand.showdown.get("shows") or []))
        won_sd = p.player_id in winner_names if went_to_sd else None
        won = p.player_id in winner_names
        invested = sum(float(a.amount or 0.0) for a in acts if a.action in ("call", "raise", "bet", "post"))
        won_amount = sum(float(w.get("amount") or 0.0) for w in hand.winners
                         if w.get("player_id") == p.player_id)
        bb = hand.bb or 1.0
        records.append({
            "hand_id": hand.hand_id, "player_id": p.player_id, "game": game,
            "stakes": stakes, "n_players": len(hand.players),
            "vpip": vpip, "pfr": pfr,
            "three_bet": pfr and p.player_id != first_raiser,
            "faced_3bet": faced_raise, "folded_to_3bet": folded_to_raise,
            "four_bet": False,
            "cbet_opportunity": cbet_opp, "cbet": cbet_opp,
            "faced_cbet": faced_flop_bet, "folded_to_cbet": folded_to_flop_bet,
            "saw_flop": bool(fl) or bool(hand.flop),
            "went_to_showdown": went_to_sd, "won_at_showdown": won_sd,
            "won_hand": won, "net_bb": round((won_amount - invested) / bb, 4),
            "bets": sum(1 for a in acts if a.action in ("bet", "raise")),
            "calls": sum(1 for a in acts if a.action == "call"),
            "folds": sum(1 for a in acts if a.action == "fold"),
            "ts": now,
        })
    return records


def _player_summary(records: list[dict], top_n: int = 12) -> dict:
    by_player: dict[str, list] = {}
    for r in records:
        by_player.setdefault(r["player_id"], []).append(r)
    out = {}
    for pid, recs in sorted(by_player.items(), key=lambda kv: -len(kv[1]))[:top_n]:
        stats = compute_all(recs).to_dict()
        out[pid] = {
            "hands": len(recs),
            "vpip": round(stats.get("vpip", {}).get("value", 0.0), 3),
            "pfr": round(stats.get("pfr", {}).get("value", 0.0), 3),
            "three_bet": round(stats.get("three_bet", {}).get("value", 0.0), 3),
            "fold_to_3bet": round(stats.get("fold_to_3bet", {}).get("value", 0.0), 3),
            "cbet": round(stats.get("cbet", {}).get("value", 0.0), 3),
            "fold_to_cbet": round(stats.get("fold_to_cbet", {}).get("value", 0.0), 3),
            "aggression_frequency": round(stats.get("aggression_frequency", {}).get("value", 0.0), 3),
            "wtsd": round(stats.get("wtsd", {}).get("value", 0.0), 3),
            "vpip_samples": stats.get("vpip", {}).get("samples", 0),
        }
    return out


def _manifest_markdown(manifest: dict) -> str:
    by_mode = json.dumps(manifest.get("by_mode", {}), ensure_ascii=False, indent=1)
    players = json.dumps(manifest.get("player_stats", {}), ensure_ascii=False, indent=1)
    missing = json.dumps(manifest.get("missing_fields", {}), ensure_ascii=False, indent=1)
    return f"""# Dataset Manifest（自动生成，勿手改）

- 生成时间: {manifest.get('generated_at')}
- Source: {manifest.get('source')}
- License / 使用许可: {manifest.get('license')}
- 输入目录: {manifest.get('input_dir')}

## 规模

- 文件: scanned={manifest.get('files_scanned')} parsed={manifest.get('files_parsed')} failed={manifest.get('files_failed')}
- 手牌: {manifest.get('hands')}（valid={manifest.get('valid')} invalid={manifest.get('invalid')} warning={manifest.get('warning')}）
- Replayable: {manifest.get('replayable')}
- 日期范围: {json.dumps(manifest.get('date_range'), ensure_ascii=False)}

## 按游戏模式

```json
{by_mode}
```

## 缺失字段（保持 null，不猜测）

```json
{missing}
```

## 玩家画像统计（样本量不足时系统自动使用 Population Model）

```json
{players}
```

> 数据分层：`data/raw/`（不可修改）/ `data/normalized/` / `data/validated/` /
> `data/replay/` / `data/experiments/`。100 手只能验证 Pipeline，不能证明策略有效。
"""


def main() -> int:
    parser = argparse.ArgumentParser(prog="poker-ai-dataset",
                                     description="Dataset Qualification Tool（V2 Phase 4）")
    parser.add_argument("--dir", default="data/hand_history", help="原始手牌输入目录")
    parser.add_argument("--source", default="manual", help="来源标识")
    parser.add_argument("--license", default="unspecified", help="许可/使用授权说明")
    parser.add_argument("--no-copy-raw", action="store_true", help="不复制 raw 层")
    args = parser.parse_args()
    runtime = build_runtime()
    qualifier = DatasetQualifier(base_dir=str(Path("data")))
    report = qualifier.qualify(args.dir, source=args.source, license_info=args.license,
                               copy_raw=not args.no_copy_raw)
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
