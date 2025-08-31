# Build System Documentation

This document explains the new automated build system for SearXNG AI Kit that dynamically builds and hosts SearXNG wheels from the latest upstream commits.

## Overview

The build system solves the challenge of packaging SearXNG AI Kit with the latest SearXNG version while maintaining standard Python packaging compatibility. It uses a two-stage CI pipeline:

1. **Stage 1**: Build SearXNG wheel from latest upstream commit
2. **Stage 2**: Generate configuration files and build final package

## Architecture

### Template-Based Configuration

The system uses template files that get filled in during the build process:

- `pyproject.toml.template` - Package configuration template
- `requirements.txt.template` - Requirements with hash verification template

Template variables:
- `{SEARXNG_WHEEL_URL}` - GitHub release URL for the SearXNG wheel
- `{SEARXNG_WHEEL_HASH}` - SHA256 hash for security verification
- `{SEARXNG_COMMIT}` - Upstream SearXNG commit hash
- `{BUILD_DATE}` - Build timestamp

### Dynamic SearXNG Building

The `build_searxng_wheel.py` script:
- Fetches the latest commit from SearXNG repository
- Builds a wheel using `--no-build-isolation` to solve circular imports
- Calculates SHA256 hash for security verification
- Saves metadata for configuration generation

### GitHub Releases Integration

The CI workflow:
- Uploads SearXNG wheels to GitHub releases with predictable URLs
- Uses tag format: `searxng-wheels-YYYY-MM-DD-{commit8}`
- URL format: `https://github.com/{owner}/{repo}/releases/download/{tag}/{wheel}`

## Files

### Core Scripts

- `build_searxng_wheel.py` - Builds SearXNG wheel from latest commit
- `generate_config.py` - Generates final config files from templates
- `dev-setup.py` - Local development setup script

### Templates

- `pyproject.toml.template` - Package configuration template
- `requirements.txt.template` - Requirements template with hash verification

### CI/CD

- `.github/workflows/build-and-release.yml` - Two-stage build pipeline

## Usage

### For CI/CD

The GitHub Actions workflow automatically:

1. **Builds SearXNG wheel**:
   ```bash
   python build_searxng_wheel.py
   ```

2. **Generates configuration**:
   ```bash
   python generate_config.py \
     --metadata wheels/build_metadata.json \
     --repo-owner $OWNER \
     --repo-name $REPO \
     --tag $RELEASE_TAG
   ```

3. **Builds final package**:
   ```bash
   python -m build --wheel
   ```

4. **Creates GitHub release** with both wheels

### For Local Development

Run the development setup script:

```bash
python dev-setup.py
```

This will:
- Build SearXNG wheel from latest commit
- Generate local configuration files with `file://` URLs
- Prepare the package for local testing

Then install in development mode:
```bash
uv pip install -e .
```

### For End Users

Users install normally and get the latest release:

```bash
# Standard installation
pip install searxng-ai-kit

# With hash verification
pip install -r requirements.txt
```

## Sequencing Solution

The original challenge was that `requirements.txt` needed to reference a GitHub URL that didn't exist until CI built it. This was solved by:

1. **Template files** stored in repository instead of final configs
2. **Dynamic generation** during CI after wheel is built and uploaded
3. **Predictable URLs** using date and commit-based release tags

## Security Features

- **SHA256 hash verification** prevents supply chain attacks
- **Pinned commit references** in build metadata for reproducibility
- **GitHub releases hosting** leverages GitHub's infrastructure security

## Local Development Workflow

1. **Make changes** to AI toolkit code
2. **Run dev setup**: `python dev-setup.py`
3. **Test locally**: `uv run searxng search "test"`
4. **Commit changes** to trigger CI build
5. **CI automatically** builds latest SearXNG and creates release

## Production Workflow

1. **Scheduled builds** (weekly) automatically get latest SearXNG
2. **Manual triggers** available for immediate updates
3. **Release creation** with both SearXNG wheel and final package
4. **Automatic testing** verifies installation works

## Benefits

- **Always up-to-date** with latest SearXNG improvements
- **Standard packaging** compatible with pip, uv, and other tools
- **Secure distribution** with hash verification
- **Transparent building** with full commit tracking
- **Minimal maintenance** through automation

## Troubleshooting

### Build Failures

If SearXNG build fails:
1. Check if SearXNG repository structure changed
2. Verify dependency versions in `build_searxng_wheel.py`
3. Test with pinned commit to isolate issues

### Local Development Issues

If local setup fails:
1. Ensure git is available for remote commit fetching
2. Check network connectivity to GitHub
3. Verify Python and uv are properly installed

### Hash/Caching Issues After Rebuilding Wheels

**Problem**: After running `dev-setup.py`, you might encounter:
```
× Failed to read searxng @ file:///.../searxng-wheel.whl
╰─▶ Hash mismatch for searxng wheel
```

**Root Cause**: 
- `uv.lock` contains hash of previous wheel build
- New wheel has different hash even with same version number
- uv cache may contain stale wheel data

**Solution**:
```bash
# Method 1: Remove lock file (simplest)
rm uv.lock && uv sync

# Method 2: If cache issues persist
uv cache clean
rm uv.lock && uv sync
```

### uv Tool Installation Issues

**Problem**: After rebuilding wheels, `uv tool install --reinstall .` may use cached old wheel instead of new one.

**Symptoms**:
- Tool installs successfully but imports fail
- Getting `ModuleNotFoundError` for modules that should exist
- Old code behavior despite rebuilding wheel

**Root Cause**: uv aggressively caches wheels even with `--reinstall` flag.

**Solution** (The uv Tool Dance™):
```bash
# Complete uninstall/reinstall cycle
uv tool uninstall searxng-ai-kit
uv tool install .

# If still having issues, clear cache first
uv cache clean
uv tool uninstall searxng-ai-kit  # (if still installed)
uv tool install .
```

**Note**: `--reinstall` alone is often insufficient for local wheel development due to caching behavior.

### CI Pipeline Issues

If CI fails:
1. Check GitHub Actions logs for specific errors
2. Verify GitHub token permissions for releases
3. Ensure artifact upload/download is working

## Future Improvements

- **Dependency caching** to speed up builds
- **Multi-platform wheels** for broader compatibility  
- **Automated testing** against multiple Python versions
- **Release notifications** for downstream users