"""Freezeout 插件：与普通 MTT 同一套 Pipeline（V1 不支持 Rebuy/Add-on，
故结构复用 MttPlugin；独立插件便于未来加入 Rebuy 专属逻辑）。"""
from __future__ import annotations

from typing import ClassVar

from .mtt import MttPlugin


class FreezeoutPlugin(MttPlugin):
    type_id: ClassVar[str | None] = "freezeout"
    display_name: ClassVar[str] = "Freezeout（无重购 MTT）"
    supported_modes: ClassVar[list[str]] = ["tournament:freezeout"]
