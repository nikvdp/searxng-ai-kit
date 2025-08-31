#!/usr/bin/env python3
"""
Development setup script for local testing.

This script helps developers set up their local environment by:
1. Building a local SearXNG wheel from latest commit
2. Generating config files that reference the local wheel
3. Installing the package for testing

This is useful for development and testing before CI runs.
"""

import json
import os
import subprocess
import sys
from pathlib import Path


def run_command(cmd, cwd=None):
    """Run a command and return the result."""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Command failed with return code {result.returncode}")
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        sys.exit(1)

    return result


def main():
    """Main development setup process."""
    print("SearXNG AI Kit - Development Setup")
    print("=" * 40)

    script_dir = Path(__file__).parent

    print("\n1. Building SearXNG wheel from latest commit...")
    result = run_command([sys.executable, "build_searxng_wheel.py"], cwd=script_dir)
    print("SearXNG wheel built successfully!")

    print("\n2. Generating local configuration files...")
    # For local development, use a file:// URL to the local wheel
    wheels_dir = script_dir / "wheels"
    wheel_files = list(wheels_dir.glob("searxng-*.whl"))

    if not wheel_files:
        print("ERROR: No SearXNG wheel found!")
        sys.exit(1)

    wheel_file = wheel_files[0]
    local_wheel_url = f"file://{wheel_file.absolute()}"

    # For local development, directly generate config from template with local file URL
    # Read the template
    template_file = script_dir / "pyproject.toml.template"
    with open(template_file, "r") as f:
        template_content = f.read()
    
    # Replace the placeholder with absolute file URL
    absolute_wheel_path = wheel_file.absolute()
    local_wheel_url = f"file://{absolute_wheel_path}"
    
    # Replace template variables
    final_content = template_content.replace("{SEARXNG_WHEEL_URL}", local_wheel_url)
    
    # Write the final pyproject.toml
    pyproject_file = script_dir / "pyproject.toml"
    with open(pyproject_file, "w") as f:
        f.write(final_content)
    
    # Also generate requirements.txt for consistency
    req_template_file = script_dir / "requirements.txt.template"
    if req_template_file.exists():
        with open(req_template_file, "r") as f:
            req_template = f.read()
        
        req_content = req_template.replace("{SEARXNG_WHEEL_URL}", local_wheel_url)
        
        req_file = script_dir / "requirements.txt"
        with open(req_file, "w") as f:
            f.write(req_content)

    print(f"Generated config files with local wheel: {local_wheel_url}")

    print("\n3. Development setup complete!")
    print("\nNext steps:")
    print("  1. Install in development mode: uv pip install -e .")
    print("  2. Test functionality: uv run searxng search 'test' --engines duckduckgo")
    print(
        "  3. Run tests: uv run python -c 'import searxng; print(\"Import successful\")'"
    )
    print("\nNote: This uses a local file:// URL for development.")
    print("CI will generate proper GitHub release URLs for production builds.")


if __name__ == "__main__":
    main()
