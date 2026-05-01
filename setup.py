#!/usr/bin/env python3
"""
Setup script with build hooks to automatically build SearXNG wheel.
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from setuptools import find_packages, setup
from setuptools.command.build_py import build_py
from setuptools.command.egg_info import egg_info

SEARXNG_COMMIT = "cba0cffa8fd56bd691e319e3069fb02b4212a4df"


def resolve_searxng_commit():
    """Resolve the SearXNG commit used for reproducible wheel builds."""
    override = os.environ.get("SEARXNG_COMMIT")
    if override:
        print(f"Using SearXNG commit from SEARXNG_COMMIT: {override}")
        return override

    print(f"Using pinned SearXNG commit: {SEARXNG_COMMIT}")
    return SEARXNG_COMMIT


class BuildSearXNGWheel:
    """Mixin to build SearXNG wheel during setup."""

    def run_searxng_build(self):
        """Build SearXNG wheel if needed."""
        script_dir = Path(__file__).parent
        wheels_dir = script_dir / "wheels"
        expected_commit = resolve_searxng_commit()

        # Check if we already have a wheel for the pinned commit.
        wheel_file = self.find_matching_searxng_wheel(wheels_dir, expected_commit)
        if wheel_file:
            print(
                f"✓ Using existing SearXNG wheel for {expected_commit}: {wheel_file.name}"
            )
        else:
            print("🔧 Building SearXNG wheel...")

            # Run the build script
            build_script = script_dir / "build_searxng_wheel.py"
            if not build_script.exists():
                print("Error: build_searxng_wheel.py not found")
                return

            try:
                env = os.environ.copy()
                env["SEARXNG_COMMIT"] = expected_commit
                result = subprocess.run(
                    [sys.executable, str(build_script)],
                    cwd=script_dir,
                    capture_output=True,
                    text=True,
                    check=True,
                    env=env,
                )
                print("✓ SearXNG wheel built successfully")

                # Find the generated wheel
                wheel_file = self.find_matching_searxng_wheel(
                    wheels_dir, expected_commit
                )
                if not wheel_file:
                    print("Warning: No wheel file found after build")
                    return

            except subprocess.CalledProcessError as e:
                print(f"Error building SearXNG wheel:")
                print(f"STDOUT: {e.stdout}")
                print(f"STDERR: {e.stderr}")
                return

        # Extract and vendor SearXNG modules into our package
        if self.extract_and_vendor_searxng(wheel_file):
            self.refresh_package_discovery()

    def find_matching_searxng_wheel(self, wheels_dir, expected_commit):
        """Return an existing wheel only when metadata matches the pinned commit."""
        metadata_file = wheels_dir / "build_metadata.json"
        if not metadata_file.exists():
            stale_wheels = list(wheels_dir.glob("searxng-*.whl"))
            if stale_wheels:
                print("Ignoring local SearXNG wheels without build metadata")
            return None

        try:
            metadata = json.loads(metadata_file.read_text())
        except json.JSONDecodeError:
            print("Ignoring local SearXNG wheel metadata that is not valid JSON")
            return None

        metadata_commit = metadata.get("searxng_commit")
        wheel_name = metadata.get("wheel_file")
        if metadata_commit != expected_commit:
            print(
                "Ignoring local SearXNG wheel for "
                f"{metadata_commit}; pinned commit is {expected_commit}"
            )
            return None

        if not wheel_name:
            print("Ignoring local SearXNG wheel metadata without wheel_file")
            return None

        wheel_file = wheels_dir / wheel_name
        if not wheel_file.exists():
            print(f"Ignoring missing SearXNG wheel from metadata: {wheel_name}")
            return None

        return wheel_file

    def refresh_package_discovery(self):
        """Refresh setuptools package discovery after vendoring SearXNG."""
        script_dir = Path(__file__).parent
        packages = find_packages(
            where=str(script_dir),
            include=["searxng*", "searx*"],
            exclude=["searx.static*", "searx.templates*", "searx.translations*"],
        )
        self.distribution.packages = packages
        if hasattr(self, "packages"):
            self.packages = packages

    def install_searxng_wheel(self, wheel_file):
        """Install SearXNG wheel directly into the current environment."""
        print(f"Installing SearXNG wheel: {wheel_file.name}")

        try:
            # Install the wheel directly
            subprocess.run(
                [sys.executable, "-m", "pip", "install", str(wheel_file)],
                check=True,
                capture_output=True,
                text=True,
            )
            print("✓ SearXNG wheel installed successfully")

            # Remove the placeholder dependency from pyproject.toml
            self.remove_placeholder_dependency()

        except subprocess.CalledProcessError as e:
            print(f"Error installing SearXNG wheel: {e}")
            print(f"STDOUT: {e.stdout}")
            print(f"STDERR: {e.stderr}")

    def is_searx_up_to_date(self, wheel_file):
        """Check if the vendored searx directory is up-to-date with the wheel."""
        script_dir = Path(__file__).parent
        searx_dest = script_dir / "searx"

        # If searx directory doesn't exist, needs extraction
        if not searx_dest.exists():
            return False

        # Check if __init__.py exists (basic sanity check)
        init_file = searx_dest / "__init__.py"
        if not init_file.exists():
            return False

        # Compare wheel mtime with searx directory mtime
        # If wheel is newer than searx, we need to re-extract
        wheel_mtime = wheel_file.stat().st_mtime
        searx_mtime = searx_dest.stat().st_mtime

        if wheel_mtime > searx_mtime:
            return False

        # Also check pyproject.toml exists and is newer than wheel
        pyproject_file = script_dir / "pyproject.toml"
        if not pyproject_file.exists():
            return False

        pyproject_mtime = pyproject_file.stat().st_mtime
        if wheel_mtime > pyproject_mtime:
            return False

        return True

    def extract_and_vendor_searxng(self, wheel_file):
        """Extract SearXNG wheel and vendor its modules into our package."""
        import shutil
        import tempfile
        import zipfile

        script_dir = Path(__file__).parent

        # Check if extraction is needed
        if self.is_searx_up_to_date(wheel_file):
            print(f"✓ Vendored searx is up-to-date with wheel")
            return True

        print(f"Extracting SearXNG wheel: {wheel_file.name}")

        # Create temporary extraction directory
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Extract the wheel
            with zipfile.ZipFile(wheel_file, "r") as wheel_zip:
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

        # Generate pyproject.toml with dynamic dependencies
        self.generate_pyproject_with_dynamic_deps(wheel_file)

        return True

    def extract_searxng_dependencies(self, wheel_file):
        """Extract dependencies from SearXNG wheel METADATA."""
        import tempfile
        import zipfile

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Extract the wheel
            with zipfile.ZipFile(wheel_file, "r") as wheel_zip:
                wheel_zip.extractall(temp_path)

            # Find METADATA file
            for metadata_file in temp_path.rglob("METADATA"):
                with open(metadata_file, "r") as f:
                    metadata = f.read()

                # Extract Requires-Dist lines
                requires = []
                for line in metadata.split("\n"):
                    if line.startswith("Requires-Dist:"):
                        dep = line.replace("Requires-Dist: ", "").strip()
                        # Skip test/dev dependencies (those with extras)
                        if "; extra ==" not in dep:
                            # Handle conditional dependencies like "tomli>=2.2.1; python_version < \"3.11\""
                            if ";" in dep and "python_version" in dep:
                                # Keep conditional dependencies as-is
                                requires.append(dep)
                            elif ";" not in dep:
                                # Keep regular dependencies
                                requires.append(dep)

                print(
                    f"✓ Extracted {len(requires)} runtime dependencies from wheel METADATA"
                )
                return requires

        print("Warning: Could not find METADATA in wheel")
        return []

    def generate_pyproject_with_dynamic_deps(self, wheel_file):
        """Generate pyproject.toml from template with dynamic SearXNG dependencies."""
        script_dir = Path(__file__).parent
        template_file = script_dir / "pyproject.toml.template"
        output_file = script_dir / "pyproject.toml"

        if not template_file.exists():
            print("Warning: pyproject.toml.template not found")
            return

        # Check if pyproject.toml already exists and is newer than both template and wheel
        if output_file.exists():
            output_mtime = output_file.stat().st_mtime
            template_mtime = template_file.stat().st_mtime
            wheel_mtime = wheel_file.stat().st_mtime
            if output_mtime > template_mtime and output_mtime > wheel_mtime:
                # pyproject.toml is up-to-date, skip regeneration
                return

        # Extract SearXNG dependencies
        searxng_deps = self.extract_searxng_dependencies(wheel_file)

        # Format dependencies for pyproject.toml
        deps_lines = []
        for dep in searxng_deps:
            # Handle quotes in conditional dependencies
            if '"' in dep:
                # Use single quotes for the TOML string to avoid escaping issues
                deps_lines.append(f"    '{dep}',")
            else:
                # Use double quotes for normal dependencies
                deps_lines.append(f'    "{dep}",')

        deps_section = "\n".join(deps_lines)

        # Read template and replace placeholder
        with open(template_file, "r") as f:
            template_content = f.read()

        final_content = template_content.replace(
            "{{SEARXNG_DEPENDENCIES}}", deps_section
        )

        # Write final pyproject.toml
        with open(output_file, "w") as f:
            f.write(final_content)

        print(
            f"✓ Generated pyproject.toml with {len(searxng_deps)} dynamic SearXNG dependencies"
        )


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
            "egg_info": CustomEggInfo,
            "build_py": CustomBuildPy,
        }
    )
