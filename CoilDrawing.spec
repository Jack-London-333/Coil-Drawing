# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller 打包配置：CoilDrawing → 免安装的 Windows 文件夹应用。

构建命令（依赖由 uv 管理）：
    uv run pyinstaller CoilDrawing.spec --noconfirm --clean

产物：dist/CoilDrawing/CoilDrawing.exe（双击即用，Win10/11 x64，无需安装 Python）
"""

import sysconfig
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

site_packages = Path(sysconfig.get_paths()["purelib"])

# ---- 额外二进制 ----
binaries = []

# OCCT 内核 DLL（cadquery-ocp-novtk 用 delvewheel 打包，运行时通过
# os.add_dll_directory 加载，PyInstaller 静态分析不一定能全部带上，手工收集）
ocp_libs = site_packages / "cadquery_ocp_novtk.libs"
if ocp_libs.is_dir():
    binaries += [(str(p), "cadquery_ocp_novtk.libs") for p in ocp_libs.glob("*.dll")]

# lib3mf 的 DLL 位于包目录内，经 ctypes 加载
binaries += collect_dynamic_libs("lib3mf")

# ---- 数据文件 ----
datas = []
datas += collect_data_files("build123d")
datas += collect_data_files("ocpsvg")
datas += collect_data_files("ocp_gordon")
datas += collect_data_files("lib3mf")

a = Analysis(
    ["main.py"],
    pathex=["src"],
    binaries=binaries,
    datas=datas,
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "tkinter",
        "PyQt5",
        "PyQt6",
        "pymupdf",
        "fitz",
        "pytest",
        # 未使用的 Qt 重型模块（保险起见显式排除）
        "PySide6.QtWebEngineCore",
        "PySide6.QtWebEngineWidgets",
        "PySide6.QtQuick",
        "PySide6.QtQml",
        "PySide6.Qt3DCore",
        "PySide6.Qt3DRender",
        "PySide6.QtMultimedia",
        "PySide6.QtCharts",
        "PySide6.QtDataVisualization",
        "PySide6.QtPdf",
        "PySide6.QtDesigner",
    ],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="CoilDrawing",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,               # 图形界面程序，双击不弹黑窗
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="CoilDrawing",
)
