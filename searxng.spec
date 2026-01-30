# -*- mode: python ; coding: utf-8 -*-

import os
import sys
import glob
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files

# Add the current directory to Python path
sys.path.insert(0, os.getcwd())

block_cipher = None

# Cross-platform binary name
binary_name = 'searxng'
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

# Add tiktoken data files for LiteLLM support
datas.extend(collect_data_files("tiktoken"))

# Add LiteLLM data files
try:
    datas.extend(collect_data_files("litellm"))
except Exception:
    pass

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
    'litellm',
    'litellm.completion',
    'litellm.router',
    'litellm.proxy',
    'litellm.exceptions',
    'litellm.utils',
    'openai',
    'anthropic',
    'google',
    'google.generativeai',
    'tokenizers',
    'tiktoken',
    'tiktoken_ext',
    'tiktoken_ext.openai_public',
    'requests',
    'aiohttp',
    'pydantic',
    'pydantic_core',
]

# Add uvloop only on non-Windows platforms
if not sys.platform.startswith('win'):
    hiddenimports.append('uvloop')

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

# Code signing configuration (disabled by default, enable via env vars)
# Only enable code signing if explicitly requested via environment variables
# This prevents auto-detection issues and makes builds work by default
codesign_identity = os.environ.get('PYINSTALLER_CODESIGN_IDENTITY')
entitlements_file = os.environ.get('PYINSTALLER_ENTITLEMENTS_FILE')

# Explicitly disable code signing if no identity is provided
# This prevents PyInstaller from auto-detecting signing identities
if not codesign_identity:
    codesign_identity = False

exe_kwargs = {
    'pyz': pyz,
    'a_scripts': a.scripts,
    'a_binaries': a.binaries,
    'a_zipfiles': a.zipfiles,
    'a_datas': a.datas,
    'strip': False,
    'upx': False,  # Disabled for cross-platform compatibility
    'upx_exclude': [],
    'name': binary_name,
    'debug': False,
    'bootloader_ignore_signals': False,
    'runtime_tmpdir': None,
    'console': True,
    'disable_windowed_traceback': False,
    'argv_emulation': False,
    'target_arch': None,
    'icon': None,  # Cross-platform compatibility
}

# Only add code signing if explicitly requested via environment variables
# codesign_identity will be False if not set, which disables signing
if codesign_identity and codesign_identity is not False:
    exe_kwargs['codesign_identity'] = codesign_identity
if entitlements_file:
    exe_kwargs['entitlements_file'] = entitlements_file

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    **{k: v for k, v in exe_kwargs.items() if k not in ['pyz', 'a_scripts', 'a_binaries', 'a_zipfiles', 'a_datas']},
)