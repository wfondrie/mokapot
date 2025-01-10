import importlib.metadata
import os
import re

from PyInstaller.building.build_main import COLLECT, EXE, PYZ, Analysis
from PyInstaller.utils.hooks import collect_all

from mokapot import __version__

# Package info
exe_name = "mokapot"
script_name = "mokapot/mokapot.py"
location = os.getcwd()
project = "mokapot"
bundle_name = "mokapot"
bundle_identifier = f"{bundle_name}.{__version__}"
block_cipher = None

# Requirements config
skip_requirements_regex = r"^(?:.*\..*)"

# Collect hidden imports and data for all requirements
requirements = importlib.metadata.requires(project)
requirements = {
    re.match(r"^[\w\-]+", req)[0]  # Remove version specifiers
    for req in requirements
    if "; extra ==" not in req  # Exclude optional dependencies
}
requirements.update([project])
hidden_imports = set()
datas = []
binaries = []
checked = set()
while requirements:
    requirement = requirements.pop()
    if re.match(skip_requirements_regex, requirement):
        continue
    checked.add(requirement)
    module_version = importlib.metadata.version(re.match(r"^[\w\-]+", requirement)[0])
    try:
        datas_, binaries_, hidden_imports_ = collect_all(requirement, include_py_files=True)
    except ImportError:
        continue
    datas += datas_
    hidden_imports_ = set(hidden_imports_)
    if "" in hidden_imports_:
        hidden_imports_.remove("")
    if None in hidden_imports_:
        hidden_imports_.remove(None)
    requirements |= hidden_imports_ - checked
    hidden_imports |= hidden_imports_

hidden_imports = sorted([h for h in hidden_imports if "tests" not in h.split(".")])
hidden_imports = [h for h in hidden_imports if "__pycache__" not in h]
datas = [
    d
    for d in datas
    if ("__pycache__" not in d[0]) and (d[1] not in [".", "build", "dist", "Output"])
]

# Build package
a = Analysis(
    [script_name],
    pathex=[location],
    binaries=binaries,
    datas=datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=exe_name,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    # Windows specific
    console=True,
    windowed=False,
    disable_windowed_traceback=False,
)

coll = COLLECT(
    exe, a.binaries, a.zipfiles, a.datas, strip=False, upx=True, upx_exclude=[], name=exe_name
)
