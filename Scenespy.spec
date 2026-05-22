# -*- mode: python ; coding: utf-8 -*-

import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_submodules


root = Path.cwd()
if sys.platform == "win32":
    dist_name = "Scenespy-windows-x64"
elif sys.platform == "darwin":
    dist_name = "Scenespy-macos"
else:
    dist_name = "Scenespy-linux-x64"

datas = [
    (str(root / "scenespy" / "assets"), "scenespy/assets"),
    (str(root / "models"), "models"),
]
binaries = []
hiddenimports = [
    # External AI packs import torch at runtime. Because torch is excluded from
    # the app bundle, PyInstaller does not see all stdlib modules torch imports.
    "cProfile",
    "cmath",
    "modulefinder",
    "pickletools",
    "profile",
    "pstats",
    "timeit",
    "unittest",
    "unittest.mock",
]
hiddenimports += collect_submodules("PIL")
hiddenimports += collect_submodules("html")

for package in (
    "customtkinter",
    "scenedetect",
):
    package_datas, package_binaries, package_hiddenimports = collect_all(package)
    datas += package_datas
    binaries += package_binaries
    hiddenimports += package_hiddenimports


a = Analysis(
    ["Scenespy.pyw" if sys.platform == "win32" else "Scenespy.py"],
    pathex=[str(root)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "torch",
        "torchvision",
        "torchaudio",
        "ultralytics",
        "mediapipe",
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
    name="Scenespy",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(root / "scenespy" / "assets" / "images" / ("exe-icon.ico" if sys.platform == "win32" else "x.icns")),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name=dist_name,
)

if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="Scenespy.app",
        icon=str(root / "scenespy" / "assets" / "images" / "x.icns"),
        bundle_identifier="com.scenespy.app",
    )
