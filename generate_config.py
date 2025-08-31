#!/usr/bin/env python3
"""
Generate configuration files from templates with SearXNG wheel metadata.

This script is used by CI to generate the final pyproject.toml and requirements.txt
files with the correct GitHub release URLs and hashes after building the SearXNG wheel.
"""

import argparse
import json
import sys
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


def generate_github_release_url(repo_owner, repo_name, wheel_filename, tag=None):
    """Generate GitHub release URL for the wheel."""
    if tag is None:
        # Use date-based tag format
        tag = f"searxng-wheels-{datetime.now().strftime('%Y-%m-%d')}"

    url = f"https://github.com/{repo_owner}/{repo_name}/releases/download/{tag}/{wheel_filename}"
    return url, tag


def generate_from_template(template_file, output_file, variables):
    """Generate file from template by substituting variables."""
    if not Path(template_file).exists():
        print(f"ERROR: Template file not found: {template_file}")
        sys.exit(1)

    with open(template_file, "r") as f:
        content = f.read()

    # Substitute variables
    for key, value in variables.items():
        placeholder = "{" + key + "}"
        content = content.replace(placeholder, str(value))

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

    # Determine wheel filename
    wheel_filename = args.wheel_filename or metadata["wheel_file"]

    # Generate GitHub release URL
    github_url, release_tag = generate_github_release_url(
        args.repo_owner, args.repo_name, wheel_filename, args.tag
    )

    print(f"GitHub release URL: {github_url}")
    print(f"Release tag: {release_tag}")

    # Prepare template variables
    variables = {
        "SEARXNG_WHEEL_URL": github_url,
        "SEARXNG_WHEEL_HASH": metadata["wheel_hash"],
        "SEARXNG_COMMIT": metadata["searxng_commit"],
        "BUILD_DATE": metadata["build_date"],
        "RELEASE_TAG": release_tag,
    }

    print("\nTemplate variables:")
    for key, value in variables.items():
        print(f"  {key}: {value}")

    # Generate configuration files
    script_dir = Path(__file__).parent

    # Generate pyproject.toml
    generate_from_template(
        script_dir / "pyproject.toml.template", script_dir / "pyproject.toml", variables
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
