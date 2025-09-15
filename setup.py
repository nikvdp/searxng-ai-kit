#!/usr/bin/env python3
"""
Setup script with build hooks to automatically build SearXNG wheel.
"""

import os
import subprocess
import sys
from pathlib import Path
from setuptools import setup
from setuptools.command.build_py import build_py
from setuptools.command.egg_info import egg_info


class BuildSearXNGWheel:
    """Mixin to build SearXNG wheel during setup."""

    def run_searxng_build(self):
        """Build SearXNG wheel if needed."""
        script_dir = Path(__file__).parent
        wheels_dir = script_dir / "wheels"

        # Check if we already have a wheel
        existing_wheels = list(wheels_dir.glob("searxng-*.whl"))
        if existing_wheels:
            print(f"✓ Using existing SearXNG wheel: {existing_wheels[0].name}")
            wheel_file = existing_wheels[0]
        else:
            print("🔧 Building SearXNG wheel...")

            # Run the build script
            build_script = script_dir / "build_searxng_wheel.py"
            if not build_script.exists():
                print("Error: build_searxng_wheel.py not found")
                return

            try:
                result = subprocess.run([sys.executable, str(build_script)],
                                      cwd=script_dir,
                                      capture_output=True,
                                      text=True,
                                      check=True)
                print("✓ SearXNG wheel built successfully")

                # Find the generated wheel
                new_wheels = list(wheels_dir.glob("searxng-*.whl"))
                if not new_wheels:
                    print("Warning: No wheel file found after build")
                    return
                wheel_file = new_wheels[0]

            except subprocess.CalledProcessError as e:
                print(f"Error building SearXNG wheel:")
                print(f"STDOUT: {e.stdout}")
                print(f"STDERR: {e.stderr}")
                return

        # Extract and vendor SearXNG modules into our package
        self.extract_and_vendor_searxng(wheel_file)

    def install_searxng_wheel(self, wheel_file):
        """Install SearXNG wheel directly into the current environment."""
        print(f"Installing SearXNG wheel: {wheel_file.name}")

        try:
            # Install the wheel directly
            subprocess.run([
                sys.executable, "-m", "pip", "install", str(wheel_file)
            ], check=True, capture_output=True, text=True)
            print("✓ SearXNG wheel installed successfully")

            # Remove the placeholder dependency from pyproject.toml
            self.remove_placeholder_dependency()

        except subprocess.CalledProcessError as e:
            print(f"Error installing SearXNG wheel: {e}")
            print(f"STDOUT: {e.stdout}")
            print(f"STDERR: {e.stderr}")

    def extract_and_vendor_searxng(self, wheel_file):
        """Extract SearXNG wheel and vendor its modules into our package."""
        import shutil
        import tempfile
        import zipfile

        script_dir = Path(__file__).parent

        print(f"Extracting SearXNG wheel: {wheel_file.name}")

        # Create temporary extraction directory
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Extract the wheel
            with zipfile.ZipFile(wheel_file, 'r') as wheel_zip:
                wheel_zip.extractall(temp_path)

            # Find the searx directory in the extracted content
            searx_source = None
            for item in temp_path.iterdir():
                if item.is_dir() and item.name == "searx":
                    searx_source = item
                    break

            if not searx_source:
                print("Error: searx directory not found in wheel")
                return False

            # Copy searx directory to our package root
            searx_dest = script_dir / "searx"
            if searx_dest.exists():
                shutil.rmtree(searx_dest)

            shutil.copytree(searx_source, searx_dest)
            print(f"✓ Vendored SearXNG modules to: {searx_dest}")

            # Also copy any .dist-info for licensing/metadata
            for item in temp_path.iterdir():
                if item.is_dir() and item.name.endswith(".dist-info"):
                    dist_info_dest = script_dir / item.name
                    if dist_info_dest.exists():
                        shutil.rmtree(dist_info_dest)
                    shutil.copytree(item, dist_info_dest)
                    print(f"✓ Copied metadata: {item.name}")

        # Also update pyproject.toml dependencies if needed
        self.ensure_searxng_deps_in_pyproject()

        return True

    def ensure_searxng_deps_in_pyproject(self):
        """Ensure SearXNG dependencies are in pyproject.toml (informational only)."""
        script_dir = Path(__file__).parent
        wheels_dir = script_dir / "wheels"

        # Get SearXNG dependencies for reference
        existing_wheels = list(wheels_dir.glob("searxng-*.whl"))
        if existing_wheels:
            wheel_file = existing_wheels[0]

            # Extract dependency info from wheel metadata
            import tempfile
            import zipfile

            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)

                with zipfile.ZipFile(wheel_file, 'r') as wheel_zip:
                    wheel_zip.extractall(temp_path)

                # Find METADATA file
                for item in temp_path.rglob("METADATA"):
                    with open(item, 'r') as f:
                        metadata = f.read()

                    # Extract Requires-Dist lines
                    requires = []
                    for line in metadata.split('\n'):
                        if line.startswith('Requires-Dist:'):
                            dep = line.replace('Requires-Dist: ', '').strip()
                            # Skip extra deps like [dev]
                            if ';' not in dep:
                                requires.append(dep)

                    if requires:
                        print(f"✓ Found {len(requires)} SearXNG dependencies")
                        # Note: Dependencies are already in pyproject.toml
                    break


class CustomEggInfo(BuildSearXNGWheel, egg_info):
    """Custom egg_info command that builds SearXNG wheel first."""

    def run(self):
        self.run_searxng_build()
        super().run()


class CustomBuildPy(BuildSearXNGWheel, build_py):
    """Custom build_py command that ensures SearXNG wheel is built."""

    def run(self):
        self.run_searxng_build()
        super().run()




if __name__ == "__main__":
    setup(
        cmdclass={
            'egg_info': CustomEggInfo,
            'build_py': CustomBuildPy,
        }
    )