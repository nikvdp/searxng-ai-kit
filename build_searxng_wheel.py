#!/usr/bin/env python3
"""
Build script to generate SearXNG wheel using --no-build-isolation approach.

This script solves SearXNG's circular import issue by pre-installing all
dependencies before building the wheel with --no-build-isolation.
"""

import os
import shutil
import subprocess
import sys
import tempfile
import hashlib
import json
from datetime import datetime
from pathlib import Path

# SearXNG git repository and commit hash
SEARXNG_REPO = "https://github.com/searxng/searxng.git"
# SEARXNG_COMMIT will be fetched dynamically as latest commit
SEARXNG_COMMIT = None  # Will be set to latest commit


def run_command(cmd, cwd=None, check=True):
    """Run a command and return the result."""
    print(f"Running: {' '.join(cmd)}")
    if cwd:
        print(f"  in directory: {cwd}")

    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)

    if check and result.returncode != 0:
        print(f"Command failed with return code {result.returncode}")
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        sys.exit(1)

    return result


def fetch_latest_commit():
    """Fetch the latest commit hash from SearXNG repository."""
    print("Fetching latest commit from SearXNG repository...")
    result = run_command(["git", "ls-remote", "--heads", SEARXNG_REPO, "master"])

    # Parse the output: "commit_hash\trefs/heads/master"
    commit_hash = result.stdout.strip().split("\t")[0]
    print(f"Latest SearXNG commit: {commit_hash}")
    return commit_hash


def calculate_wheel_hash(wheel_path):
    """Calculate SHA256 hash of the wheel file."""
    print(f"Calculating SHA256 hash for {wheel_path}...")
    sha256_hash = hashlib.sha256()

    with open(wheel_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256_hash.update(chunk)

    hash_value = sha256_hash.hexdigest()
    print(f"SHA256 hash: {hash_value}")
    return hash_value


def save_build_metadata(wheel_file, commit_hash, output_dir):
    """Save build metadata to JSON file."""
    metadata = {
        "searxng_commit": commit_hash,
        "build_date": datetime.now().isoformat(),
        "wheel_file": str(wheel_file.name),
        "wheel_size": wheel_file.stat().st_size,
        "wheel_hash": calculate_wheel_hash(wheel_file),
    }

    metadata_file = Path(output_dir) / "build_metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Build metadata saved to: {metadata_file}")
    return metadata


def create_build_env():
    """Create a temporary build environment."""
    print("Creating temporary build environment...")
    build_env = tempfile.mkdtemp(prefix="searxng_build_")
    print(f"Build environment: {build_env}")

    # Create virtual environment
    run_command([sys.executable, "-m", "venv", "venv"], cwd=build_env)

    # Determine python executable in venv
    if os.name == "nt":
        python_exe = os.path.join(build_env, "venv", "Scripts", "python.exe")
        pip_exe = os.path.join(build_env, "venv", "Scripts", "pip.exe")
    else:
        python_exe = os.path.join(build_env, "venv", "bin", "python")
        pip_exe = os.path.join(build_env, "venv", "bin", "pip")

    return build_env, python_exe, pip_exe


def get_searxng_dependencies(searxng_dir):
    """Extract dependencies from SearXNG's requirements.txt."""
    requirements_file = Path(searxng_dir) / "requirements.txt"
    
    if not requirements_file.exists():
        print("ERROR: requirements.txt not found in SearXNG repository")
        sys.exit(1)
    
    deps = []
    with open(requirements_file, "r") as f:
        for line in f:
            line = line.strip()
            # Skip comments and empty lines
            if line and not line.startswith("#"):
                deps.append(line)
    
    print(f"Found {len(deps)} dependencies in requirements.txt")
    return deps


def install_dependencies(pip_exe, searxng_dir):
    """Install all SearXNG dependencies."""
    print("Installing SearXNG dependencies...")

    # Upgrade pip first
    run_command([pip_exe, "install", "--upgrade", "pip", "setuptools", "wheel"])

    # Get dependencies from SearXNG's requirements.txt
    deps = get_searxng_dependencies(searxng_dir)
    
    # Install all dependencies
    for dep in deps:
        print(f"Installing {dep}...")
        run_command([pip_exe, "install", dep])


def clone_searxng(build_env, commit_hash):
    """Clone SearXNG repository at specific commit."""
    print(f"Cloning SearXNG repository...")
    searxng_dir = os.path.join(build_env, "searxng")

    # Clone repository
    run_command(["git", "clone", SEARXNG_REPO, searxng_dir])

    # Checkout specific commit
    print(f"Checking out commit: {commit_hash}")
    run_command(["git", "checkout", commit_hash], cwd=searxng_dir)

    # Get commit info for metadata
    result = run_command(["git", "log", "-1", "--format=%H %s"], cwd=searxng_dir)
    commit_info = result.stdout.strip()
    print(f"Commit info: {commit_info}")

    return searxng_dir


def build_wheel(pip_exe, searxng_dir, output_dir):
    """Build SearXNG wheel using --no-build-isolation."""
    print("Building SearXNG wheel...")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Build wheel with --no-build-isolation
    run_command(
        [
            pip_exe,
            "wheel",
            "--no-build-isolation",
            "--wheel-dir",
            str(output_dir),
            searxng_dir,
        ]
    )

    # Find the generated wheel file
    wheel_files = list(Path(output_dir).glob("searxng-*.whl"))
    if not wheel_files:
        print("ERROR: No wheel file generated!")
        sys.exit(1)

    wheel_file = wheel_files[0]
    print(f"Generated wheel: {wheel_file}")
    return wheel_file


def main():
    """Main build process."""
    print("SearXNG Wheel Builder")
    print("=" * 50)

    # Determine output directory
    script_dir = Path(__file__).parent
    output_dir = script_dir / "wheels"

    try:
        # Fetch latest commit hash
        commit_hash = fetch_latest_commit()

        # Create build environment
        build_env, python_exe, pip_exe = create_build_env()

        # Clone SearXNG at latest commit first (need it to get dependencies)
        searxng_dir = clone_searxng(build_env, commit_hash)
        
        # Install dependencies from SearXNG's requirements.txt
        install_dependencies(pip_exe, searxng_dir)

        # Build wheel
        wheel_file = build_wheel(pip_exe, searxng_dir, output_dir)

        # Save build metadata
        metadata = save_build_metadata(wheel_file, commit_hash, output_dir)

        print("\n" + "=" * 50)
        print("SUCCESS: SearXNG wheel built successfully!")
        print(f"Wheel file: {wheel_file}")
        print(f"Size: {wheel_file.stat().st_size / (1024 * 1024):.1f} MB")
        print(f"Commit: {commit_hash}")
        print(f"SHA256: {metadata['wheel_hash']}")

        return str(wheel_file), metadata

    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    finally:
        # Clean up build environment
        if "build_env" in locals():
            print(f"Cleaning up build environment: {build_env}")
            shutil.rmtree(build_env, ignore_errors=True)


if __name__ == "__main__":
    main()
