# license.py → license.pyd 编译脚本（防逆向层；发行前构建用）。
# 用法：python packaging/build_license_pyd.py
# 产出：cash_game_ui/license.<tag>.pyd；源 license.py 保留在仓库
# （Python 导入优先扩展模块；PyInstaller 也优先打包 .pyd）。
import os
import shutil
import sys

from Cython.Build import cythonize
from setuptools import Distribution, Extension

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "cash_game_ui", "license.py")
OUT_DIR = os.path.join(ROOT, "cash_game_ui")

os.chdir(OUT_DIR)
ext = Extension("license", ["license.py"])
setup_args = dict(ext_modules=cythonize([ext], language_level=3),
                   script_args=["build_ext", "--inplace", "--build-lib", "_pyd_build"])
dist = Distribution(setup_args)
dist.parse_config_files()
try:
    dist.run_command("build_ext")
except Exception as exc:                          # noqa: BLE001
    print(f"FAILED: {exc}")
    sys.exit(1)

# build_ext --inplace 已把 .pyd 放到本目录
built = [f for f in os.listdir(OUT_DIR)
         if f.startswith("license.") and f.endswith(".pyd")]
print("built:", built)
print("import check:", end=" ")
import importlib
import importlib.util
spec = importlib.util.find_spec("license")
print("ok" if spec else "MISSING")
