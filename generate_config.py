#!/usr/bin/env python3
"""
Generate configuration files from templates with SearXNG wheel metadata.

This script is used by CI to generate the final pyproject.toml and requirements.txt
files with the correct GitHub release URLs and hashes after building the SearXNG wheel.
"""

import argparse
import json
import sys
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path


def load_build_metadata(metadata_file):
    """Load build metadata from JSON file."""
    if not Path(metadata_file).exists():
        print(f"ERROR: Metadata file not found: {metadata_file}")
        sys.exit(1)

    with open(metadata_file, "r") as f:
        metadata = json.load(f)

    return metadata


def extract_searxng_dependencies(wheel_file):
    """Extract dependencies from SearXNG wheel METADATA."""
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

            print(f"Extracted {len(requires)} runtime dependencies from wheel METADATA")
            return requires

    print("Warning: Could not find METADATA in wheel")
    return []


def format_dependencies_for_toml(deps):
    """Format dependencies as TOML array entries."""
    lines = []
    for dep in deps:
        # Add platform guard for uvloop (doesn't support Windows)
        if dep.startswith("uvloop"):
            dep_with_marker = f'{dep}; sys_platform != "win32"'
            lines.append(f"    '{dep_with_marker}',")
        # Handle quotes in conditional dependencies
        elif '"' in dep:
            # Use single quotes for the TOML string to avoid escaping issues
            lines.append(f"    '{dep}',")
        else:
            # Use double quotes for normal dependencies
            lines.append(f'    "{dep}",')
    return "\n".join(lines)


def generate_github_release_url(repo_owner, repo_name, wheel_filename, tag=None):
    """Generate GitHub release URL for the wheel."""
    if tag is None:
        # Use date-based tag format
        tag = f"searxng-wheels-{datetime.now().strftime('%Y-%m-%d')}"

    url = f"https://github.com/{repo_owner}/{repo_name}/releases/download/{tag}/{wheel_filename}"
    return url, tag


def generate_from_template(template_file, output_file, variables, deps_section=None):
    """Generate file from template by substituting variables."""
    if not Path(template_file).exists():
        print(f"ERROR: Template file not found: {template_file}")
        sys.exit(1)

    with open(template_file, "r") as f:
        content = f.read()

    # Substitute variables (single braces)
    for key, value in variables.items():
        placeholder = "{" + key + "}"
        content = content.replace(placeholder, str(value))

    # Substitute dependencies section (double braces)
    if deps_section is not None:
        content = content.replace("{{SEARXNG_DEPENDENCIES}}", deps_section)

    # Write output file
    with open(output_file, "w") as f:
        f.write(content)

    print(f"Generated {output_file} from {template_file}")


def main():
    """Main generation process."""
    parser = argparse.ArgumentParser(description="Generate config files from templates")
    parser.add_argument(
        "--metadata", required=True, help="Path to build metadata JSON file"
    )
    parser.add_argument("--repo-owner", required=True, help="GitHub repository owner")
    parser.add_argument("--repo-name", required=True, help="GitHub repository name")
    parser.add_argument(
        "--tag", help="GitHub release tag (auto-generated if not provided)"
    )
    parser.add_argument(
        "--wheel-filename",
        help="Wheel filename (extracted from metadata if not provided)",
    )

    args = parser.parse_args()

    # Load build metadata
    metadata = load_build_metadata(args.metadata)

    # Determine wheel filename and path
    wheel_filename = args.wheel_filename or metadata["wheel_file"]

    # Find the wheel file in the wheels directory
    script_dir = Path(__file__).parent
    wheels_dir = script_dir / "wheels"
    wheel_files = list(wheels_dir.glob("searxng-*.whl"))

    if not wheel_files:
        print(f"ERROR: No SearXNG wheel found in {wheels_dir}")
        sys.exit(1)

    wheel_file = wheel_files[0]
    print(f"Using wheel: {wheel_file}")

    # Extract SearXNG dependencies from wheel
    searxng_deps = extract_searxng_dependencies(wheel_file)
    deps_section = format_dependencies_for_toml(searxng_deps)

    # Generate GitHub release URL
    github_url, release_tag = generate_github_release_url(
        args.repo_owner, args.repo_name, wheel_filename, args.tag
    )

    print(f"GitHub release URL: {github_url}")
    print(f"Release tag: {release_tag}")

    # Prepare template variables
    variables = {
        "SEARXNG_WHEEL_URL": github_url,
        "SEARXNG_COMMIT": metadata["searxng_commit"],
        "BUILD_DATE": metadata["build_date"],
        "RELEASE_TAG": release_tag,
    }

    print("\nTemplate variables:")
    for key, value in variables.items():
        print(f"  {key}: {value}")

    # Generate configuration files
    # Generate pyproject.toml
    generate_from_template(
        script_dir / "pyproject.toml.template",
        script_dir / "pyproject.toml",
        variables,
        deps_section=deps_section,
    )

    # Generate requirements.txt
    generate_from_template(
        script_dir / "requirements.txt.template",
        script_dir / "requirements.txt",
        variables,
    )

    print("\nConfiguration files generated successfully!")
    print("Ready to build final package with GitHub-hosted SearXNG wheel.")


if __name__ == "__main__":
    main()
