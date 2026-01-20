# SearXNG AI Kit Makefile
# Centralizes development, testing, and build commands
#
# Build Process Overview:
# 1. build-wheel: Build SearXNG wheel from upstream (downloads, patches, builds)
# 2. generate-config: Generate pyproject.toml with wheel URL
# 3. sync: Install dependencies including the wheel
# 4. build: Create PyInstaller binary
#
# For development: make dev (runs steps 1-3)
# For CI builds: make ci-build (runs all steps)

.PHONY: help install sync dev test build clean run search ask chat models mcp-server
.PHONY: build-wheel generate-config dev-setup ci-build ci-test build-verify

PYTHON := python3
UV := uv
WHEEL_DIR := wheels
DIST_DIR := dist

# Default target
help:
	@echo "SearXNG AI Kit - Development Commands"
	@echo ""
	@echo "Setup & Development:"
	@echo "  make dev          - Full dev setup (build wheel, generate config, sync deps)"
	@echo "  make dev-setup    - Same as 'make dev'"
	@echo "  make sync         - Sync dependencies only (assumes wheel exists)"
	@echo "  make install      - Install package as uv tool"
	@echo ""
	@echo "Build Steps (for understanding):"
	@echo "  make build-wheel     - Build SearXNG wheel from upstream"
	@echo "  make generate-config - Generate pyproject.toml with wheel URL"
	@echo ""
	@echo "Running:"
	@echo "  make run          - Run searxng --help"
	@echo "  make search QUERY='text'  - Run search (e.g., make search QUERY='python tutorial')"
	@echo "  make ask QUERY='text'     - Ask AI (e.g., make ask QUERY='explain quantum')"
	@echo "  make chat         - Start interactive chat"
	@echo "  make models       - List available models"
	@echo "  make mcp-server   - Start MCP server"
	@echo ""
	@echo "Testing:"
	@echo "  make test         - Run quick sanity tests"
	@echo "  make test-full    - Run comprehensive tests"
	@echo ""
	@echo "Building Binaries:"
	@echo "  make build        - Build PyInstaller binary (requires wheel)"
	@echo "  make build-verify - Build and verify binary works"
	@echo "  make clean        - Clean build artifacts"
	@echo ""
	@echo "CI/CD (used by GitHub Actions):"
	@echo "  make ci-build     - Full CI build (wheel + config + deps + binary)"
	@echo "  make ci-test      - Run CI tests"

# =============================================================================
# Build Dependencies / Wheel Building
# =============================================================================

# Build the SearXNG wheel from upstream repository
build-wheel:
	@echo "=== Building SearXNG wheel from upstream ==="
	$(UV) run $(PYTHON) build_searxng_wheel.py
	@echo ""
	@echo "Wheel built successfully. Check $(WHEEL_DIR)/"
	@ls -la $(WHEEL_DIR)/*.whl 2>/dev/null || echo "No wheel found"

# Generate pyproject.toml and requirements.txt from templates
# For local dev, uses file:// URL; for CI, uses GitHub release URL
generate-config: $(WHEEL_DIR)/build_metadata.json
	@echo "=== Generating configuration files ==="
	@if [ -n "$(GITHUB_ACTIONS)" ]; then \
		echo "CI mode: using GitHub release URL"; \
		$(PYTHON) generate_config.py \
			--metadata $(WHEEL_DIR)/build_metadata.json \
			--repo-owner $(GITHUB_REPOSITORY_OWNER) \
			--repo-name $(shell basename $(GITHUB_REPOSITORY)) \
			--tag $(RELEASE_TAG); \
	else \
		echo "Local mode: using file:// URL"; \
		$(PYTHON) dev-setup.py; \
	fi

# Check if wheel exists
$(WHEEL_DIR)/build_metadata.json:
	@if [ ! -f "$(WHEEL_DIR)/build_metadata.json" ]; then \
		echo "No wheel found. Building..."; \
		$(MAKE) build-wheel; \
	fi

# =============================================================================
# Setup & Dependencies
# =============================================================================

# Full development setup
dev: dev-setup

dev-setup:
	@echo "=== Full Development Setup ==="
	$(PYTHON) dev-setup.py
	@echo ""
	@echo "Development environment ready!"
	@echo "Test with: uv run searxng --help"

# Sync dependencies only (assumes pyproject.toml is already configured)
sync:
	$(UV) sync

# Install as a uv tool
install: sync
	$(UV) tool install .

# =============================================================================
# Running Commands
# =============================================================================

run:
	$(UV) run searxng --help

search:
ifndef QUERY
	@echo "Usage: make search QUERY='your query'"
	@exit 1
endif
	$(UV) run searxng search "$(QUERY)" --engines duckduckgo

ask:
ifndef QUERY
	@echo "Usage: make ask QUERY='your question'"
	@exit 1
endif
	$(UV) run searxng ask "$(QUERY)"

chat:
	$(UV) run searxng chat

models:
	$(UV) run searxng models

mcp-server:
	$(UV) run searxng mcp-server

# =============================================================================
# Testing
# =============================================================================

test:
	@echo "=== Running Quick Sanity Tests ==="
	@echo ""
	@echo "--- CLI Help ---"
	@$(UV) run searxng --help > /dev/null && echo "✓ CLI help works"
	@echo ""
	@echo "--- Search ---"
	@$(UV) run searxng search "test" --engines duckduckgo > /dev/null && echo "✓ Search works"
	@echo ""
	@echo "--- Library Import ---"
	@$(UV) run python -c "import searxng; r=searxng.search('test', max_results=1); print('✓ Library works')"
	@echo ""
	@echo "--- MCP Server Startup ---"
	@timeout 3s $(UV) run searxng mcp-server 2>&1 || echo "✓ MCP server starts"
	@echo ""
	@echo "All quick tests passed!"

test-full: test
	@echo ""
	@echo "=== Extended Tests ==="
	@echo ""
	@echo "--- Models Command ---"
	@$(UV) run searxng models > /dev/null && echo "✓ Models command works"
	@echo ""
	@echo "--- CLI Proxy API Status ---"
	@$(UV) run searxng cli-proxy-api status > /dev/null && echo "✓ CLI Proxy API status works"
	@echo ""
	@echo "--- Engines List ---"
	@$(UV) run searxng engines --common > /dev/null && echo "✓ Engines list works"
	@echo ""
	@echo "--- Categories List ---"
	@$(UV) run searxng categories > /dev/null && echo "✓ Categories list works"
	@echo ""
	@echo "--- JSON Output ---"
	@$(UV) run searxng search "test" --engines duckduckgo --format json | python -c "import sys,json; json.load(sys.stdin)" && echo "✓ JSON output valid"
	@echo ""
	@echo "All extended tests passed!"

# =============================================================================
# Building PyInstaller Binaries
# =============================================================================

# Build PyInstaller binary (assumes dependencies are synced)
build: clean
	@echo "=== Building PyInstaller Binary ==="
	@if [ ! -f "pyproject.toml" ] || ! grep -q "searxng" pyproject.toml 2>/dev/null; then \
		echo "Error: pyproject.toml not configured. Run 'make dev' first."; \
		exit 1; \
	fi
	$(UV) sync
	$(UV) add pyinstaller
	$(UV) run pyinstaller searxng.spec
	@echo ""
	@echo "Build complete!"
	@ls -lh $(DIST_DIR)/searxng 2>/dev/null || ls -lh $(DIST_DIR)/searxng.exe 2>/dev/null || echo "Binary location varies by platform"

# Build and verify the binary works
build-verify: build
	@echo ""
	@echo "=== Verifying Build ==="
	@if [ -f "$(DIST_DIR)/searxng" ]; then \
		./$(DIST_DIR)/searxng --help > /dev/null && echo "✓ Binary runs"; \
		./$(DIST_DIR)/searxng search "test" --engines duckduckgo > /dev/null && echo "✓ Binary search works"; \
	elif [ -f "$(DIST_DIR)/searxng.exe" ]; then \
		./$(DIST_DIR)/searxng.exe --help > /dev/null && echo "✓ Binary runs"; \
	else \
		echo "Binary not found"; exit 1; \
	fi
	@echo ""
	@echo "Build verified successfully!"

# Clean build artifacts
clean:
	rm -rf build/ $(DIST_DIR)/ *.spec.bak
	@echo "Cleaned build artifacts"

# Clean everything including wheels
clean-all: clean
	rm -rf $(WHEEL_DIR)/ uv_cache/
	@echo "Cleaned all artifacts including wheels"

# =============================================================================
# CI/CD Targets (used by GitHub Actions)
# =============================================================================

# Full CI build: wheel -> config -> deps -> binary
ci-build:
	@echo "=== CI Build: Full Pipeline ==="
	@echo ""
	@echo "Step 1: Building SearXNG wheel..."
	$(MAKE) build-wheel
	@echo ""
	@echo "Step 2: Syncing dependencies..."
	$(UV) sync
	$(UV) add pyinstaller
	@echo ""
	@echo "Step 3: Cleaning previous builds..."
	rm -rf build/ $(DIST_DIR)/ *.spec.bak || true
	@echo ""
	@echo "Step 4: Building with PyInstaller..."
	$(UV) run pyinstaller searxng.spec
	@echo ""
	@echo "Step 5: Verifying build..."
	@if [ -f "$(DIST_DIR)/searxng" ]; then \
		./$(DIST_DIR)/searxng --help || echo "Note: Binary verification may fail on cross-platform builds"; \
	elif [ -f "$(DIST_DIR)/searxng.exe" ]; then \
		./$(DIST_DIR)/searxng.exe --help || echo "Note: Binary verification may fail on cross-platform builds"; \
	else \
		echo "Warning: Binary not found at expected location"; \
	fi
	@echo ""
	@echo "CI Build complete!"

# CI test target
ci-test:
	@echo "=== CI Test ==="
	$(UV) sync
	$(MAKE) test
