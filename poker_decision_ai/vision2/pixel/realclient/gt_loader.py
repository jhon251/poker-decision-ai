"""V3F-P17-A — 1876 Ground-Truth loader + schema 校验器（纯数据层）。

与任何 Recognizer/Observer 解耦：只负责加载/校验用户手工标注的 1876 GT
三元组 `(render_class=1876, frame_id, seat/field, value)`，**绝不自动生成 GT、
绝不跨字段推导、绝不把 Action 标注转成 Status、绝不把 null 解释为 0**。

校验失败 → 返回错误列表（不抛异常、不改写 GT、不回填默认值）。
"""
from __future__ import annotations

import json
from pathlib import Path

#: 允许的字段名（P17 schema E 约定）
ALLOWED_FIELDS = ("bet", "position", "stack", "pot", "hero_cards", "status")

#: 合法 status 值域（仅用户目视确认的 PlayerStatus 徽章语义；不含 Action 词）
#: 注：disconnected（断开）为 P17-B 用户标注新增的第四个徽章状态。
STATUS_VALUES = ("folded", "away", "allin", "disconnected")

#: position 字段允许的键
POSITION_KEYS = ("dealer", "occupied", "hero")

#: 合法 render class（本 GT 数据集仅 1876）
GT_RENDER_CLASS = "1876"

#: 合法座位标识
SEATS = ("S1", "S2", "S3", "S4", "S5", "S6")


def load_gt(path: str | Path) -> dict:
    """加载 GT JSON（纯数据读取；不做校验）。"""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_gt(gt: dict, evidence_dir: str | Path) -> list[str]:
    """校验 GT 数据集，返回错误列表（空列表 = 通过）。

    规则（P17 硬门槛）：
    - render_class 必须 == "1876"；
    - frame_id 可追溯到 evidence_dir 下的真实帧文件；
    - geometry 与实际 PNG geometry 一致；
    - fields 仅允许 {bet, position, stack, pot, hero_cards, status}；
    - null = 该字段该帧不可见（绝不解释为 0，本校验不转换 null）；
    - 禁止把 Action 词（fold/call/check/raise）当作 status；
    - 本 loader 绝不自动生成/回填任何值。
    """
    ev_dir = Path(evidence_dir)
    errors: list[str] = []
    if not isinstance(gt, dict):
        return ["gt 顶层必须是 object"]
    if gt.get("render_class") != GT_RENDER_CLASS:
        errors.append(f"render_class 必须为 '{GT_RENDER_CLASS}'，"
                      f"实际 {gt.get('render_class')!r}")
    anns = gt.get("annotations")
    if not isinstance(anns, list) or not anns:
        errors.append("annotations 必须为非空 list")
        return errors
    for i, ann in enumerate(anns):
        prefix = f"annotations[{i}]"
        if not isinstance(ann, dict):
            errors.append(f"{prefix} 必须是 object")
            continue
        fid = ann.get("frame_id")
        if not isinstance(fid, str) or not fid:
            errors.append(f"{prefix}.frame_id 必须为非空 str")
            continue
        # frame_id 可追溯：真实帧文件存在
        fp = ev_dir / fid
        if not fp.exists():
            errors.append(f"{prefix}.frame_id {fid!r} 在 {ev_dir} 不存在")
            continue
        # geometry 与实际 PNG 一致
        geo = ann.get("geometry")
        if not (isinstance(geo, list) and len(geo) == 2
                and all(isinstance(v, int) for v in geo)):
            errors.append(f"{prefix}.geometry 必须为 [width,height] int 列表")
        else:
            try:
                from poker_decision_ai.vision2_platform.png_decode import \
                    decode_png
                w, h, _ = decode_png(fp.read_bytes())
                if [w, h] != geo:
                    errors.append(f"{prefix}.geometry {geo} 与实际 PNG "
                                  f"[{w},{h}] 不一致")
            except Exception as e:  # noqa: BLE001
                errors.append(f"{prefix} 无法解码 PNG: {e}")
        # fields 校验
        fields = ann.get("fields")
        if not isinstance(fields, dict):
            errors.append(f"{prefix}.fields 必须是 object")
            continue
        for field, value in fields.items():
            if field not in ALLOWED_FIELDS:
                errors.append(f"{prefix}.fields.{field} 不是允许字段 "
                              f"{ALLOWED_FIELDS}")
                continue
            errors.extend(_validate_field(f"{prefix}.fields.{field}",
                                          field, value))
    return errors


def _validate_field(prefix: str, field: str, value) -> list[str]:
    errors: list[str] = []
    if value is None:
        return errors                     # null = 不可见（合法，不转换）
    if field in ("bet", "stack"):
        if not isinstance(value, dict):
            return [f"{prefix} 必须是 {seat:金额|None} dict"]
        for seat, v in value.items():
            if seat not in SEATS:
                errors.append(f"{prefix}.{seat} 不是合法座位 {SEATS}")
            if v is not None and (not isinstance(v, int) or v < 0):
                errors.append(f"{prefix}.{seat} 必须为非负 int 或 null")
        return errors
    if field == "pot":
        if not isinstance(value, int) or value < 0:
            return [f"{prefix} 必须为非负 int 或 null"]
        return errors
    if field == "hero_cards":
        if not (isinstance(value, list) and len(value) == 2
                and all(isinstance(c, str) for c in value)):
            return [f"{prefix} 必须为 [card1,card2] 双牌 str 或 null"]
        return errors
    if field == "status":
        if not isinstance(value, dict):
            return [f"{prefix} 必须是 {seat:状态|None} dict"]
        for seat, v in value.items():
            if seat not in SEATS:
                errors.append(f"{prefix}.{seat} 不是合法座位 {SEATS}")
            if v is not None and v not in STATUS_VALUES:
                errors.append(f"{prefix}.{seat}={v!r} 不是合法 status "
                              f"{STATUS_VALUES}（Action 词 fold/call/check/"
                              f"raise 禁止作 status）")
        return errors
    if field == "position":
        if not isinstance(value, dict):
            return [f"{prefix} 必须是 {{dealer,occupied,hero}} 或 null"]
        for k in value:
            if k not in POSITION_KEYS:
                errors.append(f"{prefix}.{k} 不是允许 position 键 "
                              f"{POSITION_KEYS}")
        dealer = value.get("dealer")
        if dealer is not None and isinstance(dealer, str):
            if dealer not in SEATS:
                errors.append(f"{prefix}.dealer={dealer!r} 不是合法座位")
        elif dealer is not None and isinstance(dealer, dict):
            # P21 扩展：dealer = {seat, button_bbox, button_desc}（视觉 GT）
            dseat = dealer.get("seat")
            if dseat not in SEATS:
                errors.append(f"{prefix}.dealer.seat={dseat!r} 不是合法座位")
            bb = dealer.get("button_bbox")
            if bb is not None and not (isinstance(bb, list) and len(bb) == 4
                                       and all(isinstance(v, int) for v in bb)):
                errors.append(f"{prefix}.dealer.button_bbox 必须为 "
                              f"[x,y,w,h] int 列表")
        else:
            errors.append(f"{prefix}.dealer 必须为座位 str 或 "
                          f"{{seat,button_bbox}} dict")
        hero = value.get("hero")
        if hero is not None and hero not in SEATS:
            errors.append(f"{prefix}.hero={hero!r} 不是合法座位")
        occ = value.get("occupied")
        if occ is not None:
            if not isinstance(occ, dict):
                errors.append(f"{prefix}.occupied 必须是 dict")
            else:
                for s, v in occ.items():
                    if s not in SEATS:
                        errors.append(f"{prefix}.occupied.{s} 不是合法座位")
                    if v not in ("occupied", "empty", "UNKNOWN"):
                        errors.append(f"{prefix}.occupied.{s}={v!r} 必须为 "
                                      f"occupied/empty/UNKNOWN")
        return errors
    return errors


def gt_summary(gt: dict) -> dict:
    """GT 字段覆盖汇总（null 计为不可见，非 0）。

    - dict 字段（bet/stack/status）：按座位计 null（invisible）vs 非空（annotated）；
    - 标量/列表字段（pot/hero_cards/position）：按帧计。
    """
    out = {f: {"annotated": 0, "invisible": 0} for f in ALLOWED_FIELDS}
    for ann in gt.get("annotations", []):
        for field, value in (ann.get("fields") or {}).items():
            if field not in out:
                continue
            if value is None:
                out[field]["invisible"] += 1
            elif field in ("bet", "stack", "status"):
                out[field]["annotated"] += sum(
                    1 for v in value.values() if v is not None)
                out[field]["invisible"] += sum(
                    1 for v in value.values() if v is None)
            else:
                out[field]["annotated"] += 1
    return out
