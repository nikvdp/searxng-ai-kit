# -*- mode: python ; coding: utf-8 -*-

import os
import sys
import glob
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, os.getcwd())

block_cipher = None

# Cross-platform binary name
binary_name = 'searxng-cli'
if sys.platform.startswith('win'):
    binary_name += '.exe'

# Data files that need to be included
datas = [
    ('searx/settings.yml', 'searx/'),
    ('searx/limiter.toml', 'searx/'),
    ('searx/searxng.msg', 'searx/'),
    ('searx/data', 'searx/data'),
    ('searx/favicons', 'searx/favicons'),
    ('searx/infopage', 'searx/infopage'),
    ('searx/static', 'searx/static'),
    ('searx/templates', 'searx/templates'),
    ('searx/translations', 'searx/translations'),
    ('searx/search/checker/scheduler.lua', 'searx/search/checker/'),
    ('searx/answerers', 'searx/answerers'),
    ('searx/plugins', 'searx/plugins'),
    ('searx/engines', 'searx/engines'),
]

# Hidden imports that PyInstaller might miss
hiddenimports = [
    'searx',
    'searx.engines',
    'searx.search',
    'searx.search.processors',
    'searx.search.processors.online',
    'searx.search.processors.offline',
    'searx.search.processors.online_currency',
    'searx.search.processors.online_dictionary',
    'searx.search.processors.online_url_search',
    'searx.results',
    'searx.preferences',
    'searx.webadapter',
    'searx.network',
    'searx.botdetection',
    'searx.metrics',
    'searx.plugins',
    'searx.answerers',
    'babel.core',
    'babel.dates',
    'babel.numbers',
    'babel.util',
    'lxml.etree',
    'lxml.html',
    'httpx',
    'httpx_socks',
    'uvloop',
    'setproctitle',
    'redis',
    'tomli',
    'msgspec',
    'typer',
    'rich',
    'mcp',
    'mcp.server',
    'mcp.server.stdio',
    'mcp.types',
    'asyncio',
]

# Add all engine modules
engine_files = glob.glob('searx/engines/*.py')
for engine_file in engine_files:
    if '__' not in os.path.basename(engine_file):  # Skip __init__.py and __pycache__
        module_name = f"searx.engines.{os.path.basename(engine_file)[:-3]}"
        hiddenimports.append(module_name)

a = Analysis(
    ['searxng_cli.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
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
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name=binary_name,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,  # Disabled for cross-platform compatibility
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity='Developer ID Application',  # Use your Apple Developer cert
    entitlements_file='entitlements.plist',  # Optional: for specific permissions
    icon=None,  # Cross-platform compatibility
)