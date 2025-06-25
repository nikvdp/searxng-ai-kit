# SearXNG Kit

**Privacy-respecting search for your terminal, code, and AI assistants**

A command-line interface, Python library, and MCP server for the SearXNG privacy-respecting metasearch engine. Search the web from your terminal, integrate into Python projects, or connect AI assistants with support for 180+ search engines.

## Features

🔍 **CLI tool** - Search from your terminal using SearXNG's 180+ search engines  
🐍 **Python library** - Programmatic search and URL content retrieval  
🔌 **MCP server** - AI assistant integration with stdio and HTTP modes  
📦 **Standalone binaries** - Pre-built executables for Linux, Windows, and macOS  

## Installation

### Prerequisites
- Python 3.11+

### Install from Source

1. **Clone the repository:**
   ```bash
   git clone https://github.com/nikvdp/searxng-kit
   cd searxng-kit
   ```

2. **Install the package:**
   ```bash
   # With pip
   pip install .
   
   # Or with uv
   uv pip install .
   ```

3. **Verify installation:**
   ```bash
   searxng --help
   ```

## Usage Examples

### Basic Search
```bash
# Simple search with default engines
searxng search "python tutorial"

# Search with specific engines
searxng search "machine learning" --engines duckduckgo,startpage

# Search in specific category
searxng search "neural networks" --category science
```

### Engine Management
```bash
# List all available engines (compact overview)
searxng engines

# List common engines only (clean 4-column layout)
searxng engines --common

# List engines in specific category
searxng engines --category general

# List only common engines in a category
searxng engines --category general --common

# Search while disabling specific engines
searxng search "web development" --disable google,bing
```

### Output Formats
```bash
# Human-readable output (default)
searxng search "AI research"

# JSON output for scripting
searxng search "data science" --format json

# JSON output with specific engines
searxng search "rust programming" --engines duckduckgo --format json > results.json
```

### Advanced Options
```bash
# Search with language preference
searxng search "tutorial python" --lang fr

# Search with safe search enabled
searxng search "programming" --safe 1

# Search recent results only
searxng search "tech news" --time week --category news

# Search specific page
searxng search "javascript" --page 2
```

### URL Content Retrieval
```bash
# Fetch content from a single URL
searxng fetch-urls "https://example.com"

# Fetch content from multiple URLs in parallel
searxng fetch-urls "https://site1.com" "https://site2.com" "https://site3.com"

# Human-readable output (default)
searxng fetch-urls "https://example.com"

# JSON output for scripting
searxng fetch-urls "https://docs.python.org" --format json > content.json

# Control concurrency for rate limiting
searxng fetch-urls url1 url2 url3 --concurrent 2
```

### Browse Available Options
```bash
# List all search categories
searxng categories

# Get help for any command
searxng search --help
searxng engines --help
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
searxng categories

# Test clean engine listing (shows popular engines in 4 columns)
searxng engines --common

# Test engine overview (compact category summary)
searxng engines

# Test simple search
searxng search "test query" --engines duckduckgo
```

### Test Different Output Formats
```bash
# Test human-readable output
searxng search "python" --engines duckduckgo | head -20

# Test JSON output
searxng search "programming" --engines duckduckgo --format json | jq .
```

### Test Multiple Engines
```bash
# Test with multiple engines
searxng search "web development" --engines duckduckgo,startpage

# Test engine filtering
searxng search "javascript" --category general --disable google,bing
```

### Test Categories
```bash
# Test different categories
searxng search "machine learning" --category science --engines arxiv
searxng search "react tutorial" --category it --engines github
searxng search "funny cats" --category images --engines duckduckgo_images
```

## Building Executable (Optional)

To create a standalone executable:

```bash
# Install PyInstaller in development environment
pip install pyinstaller

# Build with PyInstaller
pyinstaller searxng.spec

# Test the executable
./dist/searxng --help
./dist/searxng search "test" --engines duckduckgo
```

**Note:** The executable may have some engine loading issues. The Python version is recommended for full functionality.

## MCP Server Integration

SearXNG Kit includes a Model Context Protocol (MCP) server for AI application integration. This allows AI assistants like Claude to search the web and fetch URL content directly.

### Starting the MCP Server
```bash
# Start the MCP server (runs until Ctrl+C)
searxng mcp-server
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
      "command": "searxng",
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
   - Check if engines are available: `searxng engines --common`

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
searxng search "$query" --format json --engines duckduckgo,startpage > "results_$(date +%Y%m%d_%H%M%S).json"
```

### Quick search function for .bashrc
```bash
search() {
    searxng search "$*" --engines duckduckgo 2>/dev/null
}
```

## Python Library Usage

SearXNG Kit can also be used as a Python library in your projects:

```python
import searxng

# Basic search
results = searxng.search("python tutorial", max_results=10)
print(f"Found {results['total_results']} results")

# Fetch URL content
content = searxng.fetch_url("https://example.com")
if content["success"]:
    print(f"Title: {content['title']}")

# Parallel URL fetching
urls = ["https://site1.com", "https://site2.com"]
contents = searxng.fetch_urls(urls, max_concurrent=3)

# Using client with defaults
client = searxng.SearXNGClient(
    default_engines=["duckduckgo", "startpage"],
    default_max_results=15
)
results = client.search("machine learning")
```

### Library API

- `searxng.search(query, **kwargs)` - Web search with customizable parameters
- `searxng.fetch_url(url)` - Extract content from single URL  
- `searxng.fetch_urls(urls, max_concurrent=5)` - Parallel URL content extraction
- `searxng.get_available_engines()` - Get engine information
- `searxng.get_categories()` - Get search categories
- `searxng.SearXNGClient(**defaults)` - Client class with reusable settings

All functions have async variants (`search_async`, `fetch_url_async`, etc.).

## License

This CLI tool uses SearXNG, which is licensed under AGPL-3.0. See the main repository for full license details.