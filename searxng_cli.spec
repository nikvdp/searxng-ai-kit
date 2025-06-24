# -*- mode: python ; coding: utf-8 -*-

import os
import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, os.getcwd())

block_cipher = None

# Data files that need to be included
datas = [
    ('searx/settings.yml', 'searx/'),
    ('searx/limiter.toml', 'searx/'),
    ('searx/searxng.msg', 'searx/'),
    ('searx/data/*.json', 'searx/data/'),
    ('searx/data/*.txt', 'searx/data/'),
    ('searx/data/*.ftz', 'searx/data/'),
    ('searx/favicons/*.toml', 'searx/favicons/'),
    ('searx/infopage/*/*', 'searx/infopage/'),
    ('searx/static/themes/simple/css/*', 'searx/static/themes/simple/css/'),
    ('searx/static/themes/simple/css/*/*', 'searx/static/themes/simple/css/'),
    ('searx/static/themes/simple/img/*', 'searx/static/themes/simple/img/'),
    ('searx/static/themes/simple/js/*', 'searx/static/themes/simple/js/'),
    ('searx/templates/*/*', 'searx/templates/'),
    ('searx/templates/*/*/*', 'searx/templates/'),
    ('searx/translations/*', 'searx/translations/'),
    ('searx/translations/*/*', 'searx/translations/'),
    ('searx/translations/*/*/*', 'searx/translations/'),
    ('searx/search/checker/scheduler.lua', 'searx/search/checker/'),
    ('searx/answerers/*.py', 'searx/answerers/'),
    ('searx/plugins/*.py', 'searx/plugins/'),
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
    'fasttext_predict',
    'redis',
    'tomli',
    'msgspec',
    'typer',
    'rich',
]

# Add all engine modules
import glob
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
    name='searxng',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)