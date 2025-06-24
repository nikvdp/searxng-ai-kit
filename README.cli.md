# SearXNG CLI

A command-line interface for the SearXNG privacy-respecting metasearch engine. Search the web from your terminal with support for 180+ search engines.

## Quick Start

### Prerequisites
- Python 3.11+
- UV package manager

### Installation & Setup

1. **Install dependencies:**
   ```bash
   uv sync
   ```

2. **Test the CLI:**
   ```bash
   uv run python searxng_cli.py --help
   ```

## Usage Examples

### Basic Search
```bash
# Simple search with default engines
uv run python searxng_cli.py search "python tutorial"

# Search with specific engines
uv run python searxng_cli.py search "machine learning" --engines duckduckgo,startpage

# Search in specific category
uv run python searxng_cli.py search "neural networks" --category science
```

### Engine Management
```bash
# List all available engines (compact overview)
uv run python searxng_cli.py engines

# List common engines only (clean 4-column layout)
uv run python searxng_cli.py engines --common

# List engines in specific category
uv run python searxng_cli.py engines --category general

# List only common engines in a category
uv run python searxng_cli.py engines --category general --common

# Search while disabling specific engines
uv run python searxng_cli.py search "web development" --disable google,bing
```

### Output Formats
```bash
# Human-readable output (default)
uv run python searxng_cli.py search "AI research"

# JSON output for scripting
uv run python searxng_cli.py search "data science" --format json

# JSON output with specific engines
uv run python searxng_cli.py search "rust programming" --engines duckduckgo --format json > results.json
```

### Advanced Options
```bash
# Search with language preference
uv run python searxng_cli.py search "tutorial python" --lang fr

# Search with safe search enabled
uv run python searxng_cli.py search "programming" --safe 1

# Search recent results only
uv run python searxng_cli.py search "tech news" --time week --category news

# Search specific page
uv run python searxng_cli.py search "javascript" --page 2
```

### URL Content Retrieval
```bash
# Fetch content from a single URL
uv run python searxng_cli.py fetch-urls "https://example.com"

# Fetch content from multiple URLs in parallel
uv run python searxng_cli.py fetch-urls "https://site1.com" "https://site2.com" "https://site3.com"

# Human-readable output format
uv run python searxng_cli.py fetch-urls "https://example.com" --format human

# Control concurrency for rate limiting
uv run python searxng_cli.py fetch-urls url1 url2 url3 --concurrent 2

# JSON output for scripting
uv run python searxng_cli.py fetch-urls "https://docs.python.org" --format json > content.json
```

### Browse Available Options
```bash
# List all search categories
uv run python searxng_cli.py categories

# Get help for any command
uv run python searxng_cli.py search --help
uv run python searxng_cli.py engines --help
```

## Available Categories

- **general** - General web search (51 engines)
- **images** - Image search (34 engines)  
- **videos** - Video search (29 engines)
- **news** - News search (18 engines)
- **science** - Scientific publications (9 engines)
- **it** - IT/Programming resources (44 engines)
- **music** - Music search (10 engines)
- **files** - File search (17 engines)
- **maps** - Map search (3 engines)
- And many more...

## Common Engines

⭐ **Most Popular:**
- `duckduckgo` - Privacy-focused search
- `startpage` - Google results without tracking
- `brave` - Independent search index
- `google` - Google search
- `bing` - Microsoft search
- `qwant` - European search engine

## Testing the CLI

### Test Basic Functionality
```bash
# Test categories listing
uv run python searxng_cli.py categories

# Test clean engine listing (shows popular engines in 4 columns)
uv run python searxng_cli.py engines --common

# Test engine overview (compact category summary)
uv run python searxng_cli.py engines

# Test simple search
uv run python searxng_cli.py search "test query" --engines duckduckgo
```

### Test Different Output Formats
```bash
# Test human-readable output
uv run python searxng_cli.py search "python" --engines duckduckgo | head -20

# Test JSON output
uv run python searxng_cli.py search "programming" --engines duckduckgo --format json | jq .
```

### Test Multiple Engines
```bash
# Test with multiple engines
uv run python searxng_cli.py search "web development" --engines duckduckgo,startpage

# Test engine filtering
uv run python searxng_cli.py search "javascript" --category general --disable google,bing
```

### Test Categories
```bash
# Test different categories
uv run python searxng_cli.py search "machine learning" --category science --engines arxiv
uv run python searxng_cli.py search "react tutorial" --category it --engines github
uv run python searxng_cli.py search "funny cats" --category images --engines duckduckgo_images
```

## Building Executable (Optional)

To create a standalone executable:

```bash
# Build with PyInstaller
uv run pyinstaller searxng_cli.spec

# Test the executable
./dist/searxng-cli --help
./dist/searxng-cli search "test" --engines duckduckgo
```

**Note:** The executable may have some engine loading issues. The Python version is recommended for full functionality.

## MCP Server Integration

SearXNG CLI includes a Model Context Protocol (MCP) server for AI application integration. This allows AI assistants like Claude to search the web and fetch URL content directly.

### Starting the MCP Server
```bash
# Start the MCP server (runs until Ctrl+C)
uv run python searxng_cli.py mcp-server
```

### Available MCP Tools

1. **web_search**: Search using SearXNG's 180+ search engines
   - Supports all search parameters (query, category, engines, language, etc.)
   - Returns structured JSON results

2. **fetch_url**: Retrieve and extract content from URL(s) using Jina.ai
   - **Single URL**: `{"url": "https://example.com"}`
   - **Multiple URLs (parallel)**: `{"url": ["https://site1.com", "https://site2.com"]}`
   - Maintains order of results when using arrays
   - Handles errors gracefully while preserving partial results

### Claude Desktop Configuration
Add this to your `claude_desktop_config.json`:
```json
{
  "mcpServers": {
    "searxng": {
      "command": "/path/to/searxng-cli",
      "args": ["mcp-server"]
    }
  }
}
```

### Environment Variables
- `JINA_API_KEY`: Optional API key for enhanced Jina.ai URL reading features

## Troubleshooting

### Common Issues

1. **No results returned:**
   - Try different engines: `--engines duckduckgo,startpage`
   - Check if engines are available: `uv run python searxng_cli.py engines --common`

2. **Engine errors in output:**
   - Normal behavior - some engines may fail to initialize
   - Errors are printed to stderr, results to stdout

3. **Slow searches:**
   - Normal for first run (engine initialization)
   - Use fewer engines: `--engines duckduckgo` instead of defaults

### Performance Tips

- Use `--engines` to specify fast, reliable engines
- Redirect stderr to hide engine errors: `2>/dev/null`
- Use JSON format for programmatic usage

## Example Scripts

### Search and save results
```bash
#!/bin/bash
query="$1"
uv run python searxng_cli.py search "$query" --format json --engines duckduckgo,startpage > "results_$(date +%Y%m%d_%H%M%S).json"
```

### Quick search function for .bashrc
```bash
search() {
    uv run python /path/to/searxng_cli.py search "$*" --engines duckduckgo 2>/dev/null
}
```

## License

This CLI tool uses SearXNG, which is licensed under AGPL-3.0. See the main repository for full license details.