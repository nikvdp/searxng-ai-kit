#!/usr/bin/env python3
"""
Setup script for SearXNG AI Kit with bundled SearXNG wheel.

This setup.py includes the pre-built SearXNG wheel and installs it
automatically during package installation.
"""

import os
import subprocess
import sys
from pathlib import Path

from setuptools import setup, find_packages
from setuptools.command.install import install

# Package metadata
PACKAGE_NAME = "searxng-ai-kit"
PACKAGE_VERSION = "0.1.0"

# Path to bundled SearXNG wheel
WHEELS_DIR = Path(__file__).parent / "wheels"
SEARXNG_WHEEL = next(WHEELS_DIR.glob("searxng-*.whl"), None)

class PostInstallCommand(install):
    """Post-installation for installation mode."""
    
    def run(self):
        # Run the standard installation
        install.run(self)
        
        # Install the bundled SearXNG wheel
        self._install_searxng_wheel()
    
    def _install_searxng_wheel(self):
        """Install the bundled SearXNG wheel."""
        if not SEARXNG_WHEEL or not SEARXNG_WHEEL.exists():
            print("WARNING: SearXNG wheel not found. Run build_searxng_wheel.py first.")
            return
        
        print(f"Installing bundled SearXNG wheel: {SEARXNG_WHEEL}")
        
        # Get the python executable used for installation
        python_exe = sys.executable
        
        # Install the wheel
        try:
            subprocess.run([
                python_exe, "-m", "pip", "install", 
                "--no-deps",  # Don't install dependencies again
                str(SEARXNG_WHEEL)
            ], check=True)
            print("Successfully installed bundled SearXNG wheel")
        except subprocess.CalledProcessError as e:
            print(f"ERROR: Failed to install SearXNG wheel: {e}")
            sys.exit(1)

# Load long description from README
README_PATH = Path(__file__).parent / "README.md"
if README_PATH.exists():
    with open(README_PATH, "r", encoding="utf-8") as f:
        long_description = f.read()
else:
    long_description = "AI-powered search toolkit: CLI, Python library, and MCP server for intelligent research"

# Define dependencies (without searxng since it's bundled)
dependencies = [
    "fastapi>=0.104.0",
    "litellm>=1.0.0",
    "mcp>=1.0.0",
    "rich>=14.0.0",
    "sse-starlette>=1.6.0",
    "typer-slim==0.16.0",
    "uvicorn>=0.24.0",
    # All SearXNG dependencies for compatibility
    "babel==2.17.0",
    "brotli==1.1.0",
    "certifi==2025.6.15",
    "fasttext-predict==0.9.2.4",
    "flask-babel==4.0.0",
    "flask==3.1.1",
    "httpx-socks[asyncio]==0.10.0",
    "httpx[http2]==0.28.1",
    "isodate==0.7.2",
    "jinja2==3.1.6",
    "lxml==5.4.0",
    "markdown-it-py==3.0.0",
    "msgspec==0.19.0",
    "pygments==2.19.1",
    "python-dateutil==2.9.0.post0",
    "pyyaml==6.0.2",
    "redis==5.2.1",
    "setproctitle==1.3.6",
    "toml>=0.10.2",
    "tomli==2.2.1 ; python_full_version < '3.11'",
    "uvloop==0.21.0",
]

# Package data to include
package_data = {
    "": [
        "wheels/*.whl",  # Include all wheel files
        "README.md",
        "LICENSE",
        "searxng.spec",
        "entitlements.plist",
        "clean.sh",
        "test-macos-build.sh",
    ]
}

if __name__ == "__main__":
    # Ensure the SearXNG wheel exists
    if not SEARXNG_WHEEL or not SEARXNG_WHEEL.exists():
        print("ERROR: SearXNG wheel not found.")
        print("Please run: python build_searxng_wheel.py")
        sys.exit(1)
    
    print(f"Found SearXNG wheel: {SEARXNG_WHEEL}")
    
    setup(
        name=PACKAGE_NAME,
        version=PACKAGE_VERSION,
        description="AI-powered search toolkit: CLI, Python library, and MCP server for intelligent research",
        long_description=long_description,
        long_description_content_type="text/markdown",
        author="SearXNG AI Kit Contributors",
        license="AGPL-3.0",
        url="https://github.com/nik-ai/searxng-ai-kit",
        packages=find_packages(include=["searxng", "searxng.*"]),
        py_modules=["searxng_cli"],
        install_requires=dependencies,
        python_requires=">=3.11",
        entry_points={
            "console_scripts": [
                "searxng=searxng_cli:main",
            ],
        },
        package_data=package_data,
        include_package_data=True,
        cmdclass={
            "install": PostInstallCommand,
        },
        classifiers=[
            "Development Status :: 4 - Beta",
            "Environment :: Console",
            "Intended Audience :: Developers",
            "Intended Audience :: End Users/Desktop",
            "License :: OSI Approved :: GNU Affero General Public License v3",
            "Operating System :: OS Independent",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.11",
            "Programming Language :: Python :: 3.12",
            "Topic :: Internet :: WWW/HTTP :: Indexing/Search",
            "Topic :: Software Development :: Libraries :: Python Modules",
            "Topic :: Utilities",
        ],
        keywords=["search", "privacy", "metasearch", "searxng", "ai", "llm", "cli", "library", "mcp", "toolkit"],
    )