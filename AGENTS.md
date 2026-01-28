# SearXNG AI Kit Development Context and Reminders

## Critical Reminders

### Testing and Running Commands
- **ALWAYS use `uv sync` for dependency management** - not `pip install` or `uv pip install`
- **ALWAYS use `uv run` for Python execution** - not just `python`
- Example: `uv run python script.py` not `python script.py`
- This ensures dependencies are properly resolved in the uv environment
- When recreating venv: `rm -rf .venv && uv sync`

### Project Context
- This is **SearXNG AI Kit** - a fork of SearXNG focused on AI-powered search toolkit functionality
- **Main documentation is in `README.cli.md`** - NOT README.md
- Don't create separate README files for new features
- Update the existing toolkit-focused documentation

### Package Structure
- This is `searxng-ai-kit` package - CLI + library + MCP server + AI chat toolkit
- Main CLI entry point: `searxng_cli.py` with `searxng` command
- Library interface: `searxng/` package for programmatic use
- MCP server: Both stdio and HTTP modes for AI assistant integration
- AI chat: LiteLLM integration with function calling for research
- All interfaces should work together seamlessly

### Git and Commit Guidelines
- **NEVER include Claude Code attribution in commits** - this is explicitly forbidden
- **Use atomic commits that tell the development story**
  - Each commit should contain ONE logical change (single feature, single bug fix, single refactor)
  - NEVER use `git add .` indiscriminately - always review what you're committing
  - Use `git status` and `git diff --staged` to verify staged changes before committing
  - Stage only the files related to the specific change you're committing
  - Example: Don't mix "add new feature" with "fix unrelated bug" in same commit
- Only commit working code, no partial implementations
- Use `master` branch (not `main`)

### Development Workflow
- Read existing code before implementing new features
- Leverage existing infrastructure (like `searx.network.multi_requests()`)
- Maintain consistency between CLI, MCP server, and library interfaces
- Test all interfaces (CLI, library, MCP stdio, MCP HTTP)
- Use `act` to test GitHub Actions workflows locally (when Docker available)
- Position as a comprehensive AI-powered search toolkit, not just a CLI tool

### Code Architecture Patterns
- **Async as Primary Implementation**: All core logic should be implemented in async functions
- **Sync as Convenience Wrappers**: Sync functions should just use `asyncio.run()` to wrap async versions
- **Shared Functions**: Use single implementations across CLI, library, and MCP interfaces (KISS/DRY principle)
- **Atomic Functions**: Each function should do one thing well and be reusable across modalities

## Common Mistakes to Avoid

1. **Creating unnecessary files**: Don't create new READMEs when updating existing docs
2. **Wrong test commands**: Remember `uv run` prefix for all Python execution
3. **Scope issues**: Define shared functions globally when needed by multiple modules
4. **Import errors**: Check actual function names in source before importing
5. **Documentation scattered**: Keep toolkit docs in README.cli.md, not separate files
6. **Context loss**: This is SearXNG AI Kit, a comprehensive AI-powered search toolkit, not the main SearXNG project

## Architecture Notes

### MCP Server Implementation
- Supports both stdio (default) and HTTP (`--remote`) modes
- Shared `get_mcp_tools()` and `handle_tool_call()` functions
- JSON-RPC 2.0 over HTTP with SSE support
- OAuth 2.0 dynamic client registration for modern MCP clients

### Library Interface
- Clean Python API: `searxng.search()`, `searxng.fetch_url()`, etc.
- Both sync and async variants of all functions
- `SearXNGClient` class for reusable configurations
- Proper error handling and result formatting

### URL Processing
- Use `urllib.parse.quote(url, safe='')` for Jina.ai requests
- Parallel processing via existing `searx.network.multi_requests()`
- Order preservation in parallel URL fetching

## Comprehensive Testing Checklist

Before committing, verify ALL interfaces work properly:

### Installation Test
- [ ] Package installs: `uv pip install -e .`
- [ ] Command available: `uv run searxng --help`

### CLI Interface Testing
- [ ] Help works: `uv run searxng --help`
- [ ] Engines list: `uv run searxng engines --common`
- [ ] Categories list: `uv run searxng categories`
- [ ] Basic search: `uv run searxng search "test query" --engines duckduckgo`
- [ ] JSON output: `uv run searxng search "test" --engines duckduckgo --format json`
- [ ] URL fetching: `uv run searxng fetch-urls "https://httpbin.org/json"`
- [ ] Multi-search: `uv run searxng multi-search "python" "javascript" --max-results 2`
- [ ] Multi-search JSON: `uv run searxng multi-search "test1" "test2" --format json`
- [ ] Ask command help: `uv run searxng ask --help`
- [ ] Ask without API key: `uv run searxng ask "test"` (should show error message)
- [ ] Ask with custom base URL: `uv run searxng ask "test" --base-url "https://openrouter.ai/api/v1"`
- [ ] Ask with OpenRouter: `OPENROUTER_API_KEY=xxx uv run searxng ask "test" --model "openrouter/openai/gpt-4o-mini"`
- [ ] Chat command help: `uv run searxng chat --help`
- [ ] Chat without API key: `uv run searxng chat` then type `exit` (should show error message for API key)

### Library Interface Testing
- [ ] Import works: `uv run python -c "import searxng; print('Library imported successfully')"`
- [ ] Basic search: `uv run python -c "import searxng; r=searxng.search('test', max_results=3); print(f'Found {r[\"total_results\"]} results')"`
- [ ] URL fetch: `uv run python -c "import searxng; r=searxng.fetch_url('https://httpbin.org/json'); print('URL fetch:', 'success' if r['success'] else 'failed')"`
- [ ] Ask AI without API key: `uv run python -c "import searxng; r=searxng.ask('test'); print('Ask result:', 'success' if r['success'] else r['error'])"`
- [ ] Client class: `uv run python -c "import searxng; c=searxng.SearXNGClient(); r=c.search('test', max_results=2); print('Client works')"`
- [ ] Client ask: `uv run python -c "import searxng; c=searxng.SearXNGClient(); r=c.ask('test'); print('Client ask:', 'success' if r['success'] else 'error')"`
- [ ] Run example: `uv run python examples/library_usage.py`

### MCP Server Testing (stdio mode)
- [ ] Server starts: `uv run searxng mcp-server` (should start without errors)
- [ ] Test with timeout: `timeout 10s uv run searxng mcp-server || echo "Server started and stopped correctly"`

### MCP Server Testing (HTTP mode)  
- [ ] HTTP server starts: `uv run searxng mcp-server --remote --port 9999` (should start on port 9999)
- [ ] Test with timeout: `timeout 10s uv run searxng mcp-server --remote --port 9999 || echo "HTTP server started and stopped correctly"`

### Documentation Testing
- [ ] README.rst has correct SearXNG AI Kit branding
- [ ] README.cli.md has correct examples and installation instructions
- [ ] All repo URLs point to searxng-ai-kit (not searxng-cli)

### PyInstaller Build Testing
- [ ] Add PyInstaller: `uv add pyinstaller`
- [ ] Clean builds: `rm -rf build/ dist/ *.spec.bak || true`
- [ ] Build succeeds: `uv run pyinstaller searxng.spec`
- [ ] Binary works: `./dist/searxng --help`
- [ ] Binary search: `./dist/searxng search "test" --engines duckduckgo`

### GitHub Actions Testing (with act)
**Note:** Only run on Darwin (macOS) systems where `act` is available. Skip on Linux or when running Claude Code inside Docker (Docker-in-Docker limitation).

- [ ] Check system: Only proceed if `uname` returns "Darwin" and `which act` succeeds
- [ ] Test Linux build: `act -P ubuntu-latest=catthehacker/ubuntu:act-latest -j build --matrix platform:linux --matrix arch:x64 --container-architecture linux/amd64`
- [ ] Verify build success: Look for "Success - Main Verify build" in output
- [ ] Expected failure: Artifact upload fails (missing ACTIONS_RUNTIME_TOKEN) - this is normal in local testing
- [ ] Build artifacts: Should see `./dist/searxng --help` output in verification step

**Common Issues:**
- Architecture mismatch warnings on Apple Silicon Macs (use `--container-architecture linux/amd64`)
- Missing spec file errors indicate workflow needs updating to use correct `searxng.spec`
- Binary name mismatches show in verification step failures

## Package Distribution

- PyPI package name: `searxng-ai-kit`
- CLI command: `searxng`
- Library import: `import searxng`
- MCP server: `searxng mcp-server` (stdio) or `searxng mcp-server --remote` (HTTP)
- Supports Python 3.11+
- Dependencies managed in pyproject.toml

## Pre-Commit Sanity Check Procedure

**CRITICAL:** Run this complete sanity check before ANY commit to ensure all modalities work properly.

### Quick Sanity Check (Minimum Required)
Run these essential tests to verify core functionality:

```bash
# Installation and basic CLI
uv sync
uv run searxng --help
uv run searxng search "test" --engines duckduckgo

# Library interface
uv run python -c "import searxng; r=searxng.search('test', max_results=2); print(f'Found {r[\"total_results\"]} results')"

# MCP server startup test
timeout 5s uv run searxng mcp-server || echo "MCP stdio server test completed"
timeout 5s uv run searxng mcp-server --remote --port 9998 || echo "MCP HTTP server test completed"
```

### Full Comprehensive Sanity Check
For major changes or releases, run the complete testing checklist above.

**Never commit without running at least the Quick Sanity Check first.**

---

## CRITICAL: Task Tracking with `lb`

**DO NOT use built-in todo/task tracking tools. Use `lb` instead.**

This repo uses `lb` for issue tracking. All tasks live in Linear. The `lb` CLI is your todo list - not your built-in task tools.

### Quick Start

```bash
lb sync                    # Pull latest from Linear
lb ready                   # See unblocked work (issues with no blockers)
lb show LIN-XXX            # Read full description before starting
lb update LIN-XXX --status in_progress   # Claim it
```

### Dependencies & Blocking

`lb` tracks relationships between issues. `lb ready` only shows unblocked issues.

```bash
# This issue blocks another (other can't start until this is done)
lb create "Must do first" --blocks LIN-123

# This issue is blocked by another (can't start until other is done)
lb create "Depends on auth" --blocked-by LIN-100

# Found a bug while working on LIN-50? Link it
lb create "Found: race condition" --discovered-from LIN-50 -d "Details..."

# General relation (doesn't block)
lb create "Related work" --related LIN-200

# Manage deps after creation
lb dep add LIN-A --blocks LIN-B
lb dep remove LIN-A LIN-B
lb dep tree LIN-A          # Visualize dependency tree
```

**Dependency types:**
- `--blocks ID` - This issue must finish before ID can start
- `--blocked-by ID` - This issue can't start until ID finishes
- `--related ID` - Soft link, doesn't block progress
- `--discovered-from ID` - Found while working on ID (creates relation)

### Planning Work

Break down tasks into subtasks:

```bash
lb create "Step 1: Do X" --parent LIN-XXX -d "Details..."
lb create "Step 2: Do Y" --parent LIN-XXX -d "Details..."
```

### Workflow

1. `lb ready` - Find unblocked work
2. `lb update ID --status in_progress` - Claim it
3. Work on it
4. Found new issue? `lb create "Found: X" --discovered-from ID`
5. `lb close ID --reason "Done"`

### Viewing Issues

```bash
lb list                    # All issues
lb list --status open      # Filter by status
lb ready                   # Unblocked issues ready to work
lb blocked                 # Blocked issues (shows what's blocking them)
lb show LIN-XXX            # Full details with all relationships
```

### Key Commands

| Command | Purpose |
|---------|---------|
| `lb sync` | Sync with Linear |
| `lb ready` | Show unblocked issues |
| `lb blocked` | Show blocked issues with blockers |
| `lb show ID` | Full issue details + relationships |
| `lb create "Title" -d "..."` | Create issue |
| `lb create "Title" --parent ID` | Create subtask |
| `lb create "Title" --blocked-by ID` | Create blocked issue |
| `lb update ID --status in_progress` | Claim work |
| `lb close ID --reason "why"` | Complete work |
| `lb dep add ID --blocks OTHER` | Add blocking dependency |
| `lb dep tree ID` | Show dependency tree |

### Rules

1. **NEVER use built-in task tools** - use `lb create` for subtasks
2. **Always `lb ready`** before asking what to work on
3. **Always `lb show`** to read the full description before starting
4. **Link discovered work** with `--discovered-from` to maintain context graph
5. **Include descriptions** with enough context for handoff
6. **Close with reasons** explaining what was done
