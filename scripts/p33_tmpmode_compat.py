"""P33 测试环境兼容插件（仅 pytest 进程内生效——不修改项目代码）。

背景：Windows + CPython 3.13 + DSH 沙箱下，`Path.mkdir(mode=0o700)`
创建的目录对创建进程自身也不可枚举（WinError 5），导致 pytest 9.1.1
tmpdir 全链路（getbasetemp / make_numbered_dir 均为 mode=0o700）大面积
setup error。此插件在 pytest 进程内把目录创建 mode 强制为 0o777
（默认宽松——与 probe_a 行为一致），仅影响临时目录创建，不触碰任何
测试逻辑。项目文件零修改。
"""
from __future__ import annotations

import os
import pathlib

_PERMISSIVE = 0o777

_orig_pathlib_mkdir = pathlib.Path.mkdir
_orig_os_mkdir = os.mkdir


def _patched_pathlib_mkdir(self, mode=0o777, parents=False, exist_ok=False):
    return _orig_pathlib_mkdir(self, mode=_PERMISSIVE, parents=parents,
                               exist_ok=exist_ok)


def _patched_os_mkdir(path, mode=0o777, *, dir_fd=None):
    return _orig_os_mkdir(path, _PERMISSIVE, dir_fd=dir_fd)


pathlib.Path.mkdir = _patched_pathlib_mkdir
os.mkdir = _patched_os_mkdir
