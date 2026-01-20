# SearXNG AI Kit

**AI-powered search for your terminal, code, and AI assistants**

An AI-enhanced command-line interface, Python library, and MCP server for the SearXNG privacy-respecting metasearch engine. Research topics with AI assistance, search the web from your terminal, integrate into Python projects, or connect AI assistants with support for 180+ search engines.

## Features

🔍 **CLI tool** - Search from your terminal with AI-enhanced research capabilities  
🤖 **AI chat** - Ask questions and get comprehensive research using 180+ search engines  
🐍 **Python library** - Programmatic search and AI-powered content retrieval  
🔌 **MCP server** - AI assistant integration with advanced research tools  
📦 **Standalone binaries** - Pre-built executables for Linux, Windows, and macOS  

## Installation

### Prerequisites
- Python 3.11+
- [uv](https://docs.astral.sh/uv/getting-started/installation/) (recommended)

### Install with uv (Recommended)

1. **Install uv:**
   ```bash
   # On macOS and Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh
   # On Windows: powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

2. **Build and install:**
   ```bash
   git clone https://github.com/nikvdp/searxng-ai-kit
   cd searxng-ai-kit

   # Build wheel and set up development environment
   python dev-setup.py

   # Install as tool
   uv tool install .
   ```

3. **Verify:**
   ```bash
   searxng --help
   ```

### Development Setup

```bash
# Clone and set up for development
git clone https://github.com/nikvdp/searxng-ai-kit
cd searxng-ai-kit

# Build SearXNG wheel, generate configs, sync dependencies
python dev-setup.py

# Test
uv run searxng search "test" --engines duckduckgo
```

The build system automatically:
- Installs Python 3.11+ via uv
- Builds SearXNG wheel with proper dependency isolation
- Handles modern type annotations (`str | Exception`)
- Generates local configuration files

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
uv run searxng categories
# Or: searxng categories

# Test clean engine listing (shows popular engines in 4 columns)
uv run searxng engines --common
# Or: searxng engines --common

# Test engine overview (compact category summary)
uv run searxng engines
# Or: searxng engines

# Test simple search
uv run searxng search "test query" --engines duckduckgo
# Or: searxng search "test query" --engines duckduckgo
```

### Test Different Output Formats
```bash
# Test human-readable output
uv run searxng search "python" --engines duckduckgo | head -20
# Or: searxng search "python" --engines duckduckgo | head -20

# Test JSON output
uv run searxng search "programming" --engines duckduckgo --format json | jq .
# Or: searxng search "programming" --engines duckduckgo --format json | jq .
```

### Test Multiple Engines
```bash
# Test with multiple engines
uv run searxng search "web development" --engines duckduckgo,startpage
# Or: searxng search "web development" --engines duckduckgo,startpage

# Test engine filtering
uv run searxng search "javascript" --category general --disable google,bing
# Or: searxng search "javascript" --category general --disable google,bing
```

### Test Categories
```bash
# Test different categories
uv run searxng search "machine learning" --category science --engines arxiv
# Or: searxng search "machine learning" --category science --engines arxiv

uv run searxng search "react tutorial" --category it --engines github
# Or: searxng search "react tutorial" --category it --engines github

uv run searxng search "funny cats" --category images --engines duckduckgo_images
# Or: searxng search "funny cats" --category images --engines duckduckgo_images
```

## Building Executable (Optional)

To create a standalone executable:

```bash
# Sync development dependencies (includes PyInstaller)
uv sync

# Build with PyInstaller (no code signing)
uv run pyinstaller searxng.spec

# Test the executable
./dist/searxng --help
./dist/searxng search "test" --engines duckduckgo
```

### Code Signing (macOS only)

For distribution on macOS, you can optionally enable code signing if you have a valid Developer ID certificate:

```bash
# Set environment variables for code signing
export PYINSTALLER_CODESIGN_IDENTITY="Developer ID Application: Your Name"
export PYINSTALLER_ENTITLEMENTS_FILE="entitlements.plist"  # optional

# Build with code signing
uv run pyinstaller searxng.spec
```

**Note:** The executable may have some engine loading issues. The Python version is recommended for full functionality.

## AI Assistant Integration

SearXNG AI Kit includes an AI assistant feature that combines web search capabilities with powerful language models. The assistant can perform parallel searches and fetch content from multiple URLs to provide comprehensive research and answers.

### Ask Command (One-Shot Q&A)

```bash
# Ask a question using the default o3 model
searxng ask "What are the latest developments in quantum computing?"

# Use a specific model
searxng ask "Research renewable energy trends" --model "openai/gpt-4o-mini"

# Get JSON output for programmatic use
searxng ask "Compare Python vs JavaScript" --format json

# Read prompt from stdin for longer questions
echo "Analyze the current state of renewable energy adoption globally, including recent policy changes and technological breakthroughs" | searxng ask

# Interactive stdin mode
searxng ask
# (then type your question and press Ctrl+D)

# Pipe from file
searxng ask < my_research_question.txt
```

### Chat Command (Interactive Conversations)

```bash
# Start an interactive chat session
searxng chat

# Use a specific model for the conversation
searxng chat --model "openai/gpt-4o-mini"

# Use OpenRouter models
searxng chat --model "openrouter/anthropic/claude-3.5-sonnet"

# Use custom API endpoint
searxng chat --base-url "https://api.openai.com/v1"
```

The chat command maintains conversation history and allows follow-up questions:

```
You: What's the weather like in Tokyo?
Assistant: [searches for current Tokyo weather information]

You: How about tomorrow?
Assistant: [remembers we're discussing Tokyo weather and searches for tomorrow's forecast]

You: What should I wear?
Assistant: [provides clothing recommendations based on the weather information from previous searches]
```

**Chat Features:**
- **Conversation Memory**: Remembers all previous messages in the session
- **Context Awareness**: References earlier parts of the conversation
- **Chat History**: Automatically saves conversations to markdown files
- **Model Display**: Shows model name (e.g., `o3`) instead of generic "Assistant"
- **Progress Indicator**: Animated spinner while waiting for AI responses
- **Exit Commands**: Type `exit`, `quit`, `bye`, or press Ctrl+C to end
- **Web Search**: Full access to search tools during conversations
- **Tool Visibility**: Shows search queries and URL fetches in real-time

**Chat History Location:**
- Saved to: `~/.local/share/searxng-ai-kit/chats/` (XDG Base Directory spec)
- Format: `chat-YYYY-MM-DD_HH-MM-SS-modelname.md`
- Example: `chat-2025-06-29_14-30-15-o3.md`

### API Key Configuration

Set one of these environment variables:

```bash
# OpenAI (recommended)
export OPENAI_API_KEY="your-openai-key"

# OpenRouter (access to many models)
export OPENROUTER_API_KEY="your-openrouter-key"

# Anthropic
export ANTHROPIC_API_KEY="your-anthropic-key"

# Google
export GOOGLE_API_KEY="your-google-key"
```

### OpenRouter Integration

OpenRouter provides access to many AI models through a single API. SearXNG Kit has native OpenRouter support:

```bash
# Set OpenRouter API key
export OPENROUTER_API_KEY="your-openrouter-key"

# Use OpenRouter models with native prefix
searxng ask "Analyze market trends" --model "openrouter/openai/gpt-4o-mini"
searxng ask "Technical research" --model "openrouter/anthropic/claude-3.5-sonnet"

# Use custom base URL (alternative method)
searxng ask "Question" --base-url "https://openrouter.ai/api/v1" --model "openai/gpt-4o-mini"
```

### CLI Proxy API Integration

SearXNG AI Kit integrates with [CLI Proxy API](https://github.com/router-for-me/CLIProxyAPI) to provide access to multiple AI providers through their OAuth and subscription plans - **no API keys needed!** Use your existing Claude Pro, ChatGPT Plus, or Gemini Advanced subscriptions.

#### Step 1: Install CLI Proxy API

Download the latest release from GitHub:

```bash
# macOS (Apple Silicon)
curl -L https://github.com/router-for-me/CLIProxyAPI/releases/latest/download/cli-proxy-api-darwin-arm64 -o cli-proxy-api
chmod +x cli-proxy-api
sudo mv cli-proxy-api /usr/local/bin/

# macOS (Intel)
curl -L https://github.com/router-for-me/CLIProxyAPI/releases/latest/download/cli-proxy-api-darwin-amd64 -o cli-proxy-api
chmod +x cli-proxy-api
sudo mv cli-proxy-api /usr/local/bin/

# Linux (x64)
curl -L https://github.com/router-for-me/CLIProxyAPI/releases/latest/download/cli-proxy-api-linux-amd64 -o cli-proxy-api
chmod +x cli-proxy-api
sudo mv cli-proxy-api /usr/local/bin/
```

Or build from source: https://github.com/router-for-me/CLIProxyAPI

#### Step 2: Create Configuration File

Create `~/.cli-proxy-api/config.yaml`:

```yaml
host: "127.0.0.1"
port: 8317
auth-dir: "~/.cli-proxy-api"
```

#### Step 3: Authenticate with AI Providers

Log in to the providers you want to use:

```bash
# Claude (Anthropic) - opens browser for OAuth
cli-proxy-api -claude-login

# Google Gemini - opens browser for OAuth  
cli-proxy-api -login

# OpenAI Codex - opens browser for OAuth
cli-proxy-api -codex-login

# Qwen - opens browser for OAuth
cli-proxy-api -qwen-login
```

Each login opens your browser for OAuth authentication. After authenticating, tokens are stored in `~/.cli-proxy-api/`.

#### Step 4: Configure SearXNG

Tell SearXNG where your CLI Proxy API config is:

```bash
# Point to your config file
searxng cli-proxy-api set-config ~/.cli-proxy-api/config.yaml

# Enable the integration
searxng cli-proxy-api enable

# Verify everything is working
searxng cli-proxy-api status
```

#### Step 5: Set a Default Model (Optional)

```bash
# Set a default model so you don't need --model every time
searxng cli-proxy-api set-default-model claude-sonnet-4-5-20250929

# Now just use ask/chat without --model
searxng ask "Explain quantum computing"
searxng chat
```

#### Step 6: Use It!

```bash
# List all available models
searxng models

# Use CLI Proxy API models (with explicit model)
searxng ask --model cli-proxy-api/claude-sonnet-4-5-20250929 "Explain quantum computing"

# Or just use the default (if set)
searxng ask "What's new in AI?"

# Start interactive chat
searxng chat
```

#### Available Commands

```bash
# List all available models (shows default)
searxng models

# Check status
searxng cli-proxy-api status

# Set/clear default model
searxng cli-proxy-api set-default-model claude-sonnet-4-5-20250929
searxng cli-proxy-api clear-default-model

# Enable/disable integration
searxng cli-proxy-api enable
searxng cli-proxy-api disable

# Set config path (if not in default location)
searxng cli-proxy-api set-config /path/to/config.yaml

# Clear explicit config (use auto-detection)
searxng cli-proxy-api clear-config
```

#### How It Works

When you use a `cli-proxy-api/*` model:

1. SearXNG automatically starts a managed cli-proxy-api subprocess on a random port
2. Your OAuth tokens (stored by cli-proxy-api) handle authentication
3. Requests route through the local proxy to AI providers
4. The proxy handles token refresh and provider-specific quirks

**Supported Providers:**
- **Anthropic Claude** (via OAuth) - claude-sonnet-4-5, claude-opus-4, etc.
- **Google Gemini** (via OAuth) - gemini-2.5-pro, gemini-2.5-flash, etc.
- **OpenAI** (via OAuth) - gpt-4o, gpt-4o-mini, etc.
- **Alibaba Qwen** (via OAuth)
- Any OpenAI-compatible provider (via config)

### Custom API Endpoints

Use custom OpenAI-compatible endpoints:

```bash
# Via environment variable
export OPENAI_BASE_URL="https://api.openai.com/v1"
searxng ask "Question"

# Via command line flag (overrides environment)
searxng ask "Question" --base-url "https://custom-endpoint.com/v1"
```

### Available AI Tools

The assistant has access to these tools for research:
- **web_search**: Search using SearXNG's 180+ engines  
- **multi_web_search**: Run multiple searches in parallel
- **fetch_url**: Extract content from a single URL
- **fetch_urls**: Fetch content from multiple URLs simultaneously

## MCP Server Integration

SearXNG AI Kit includes a Model Context Protocol (MCP) server for AI application integration. This allows AI assistants like Claude to search the web and fetch URL content directly.

### Starting the MCP Server
```bash
# Start the MCP server (runs until Ctrl+C)
searxng mcp-server
```

### Available MCP Tools

1. **web_search**: Search using SearXNG's 180+ search engines
   - Supports all search parameters (query, category, engines, language, etc.)
   - Returns structured JSON results

2. **multi_web_search**: Run multiple search queries in parallel
   - Execute multiple searches simultaneously for comprehensive coverage
   - Faster than sequential searches for research tasks

3. **fetch_url**: Retrieve and extract content from a single URL using Jina.ai
   - Clean text extraction and content processing

4. **fetch_urls**: Retrieve and extract content from multiple URLs in parallel
   - Maintains order of results when using arrays
   - Handles errors gracefully while preserving partial results

5. **ask**: Ask an AI assistant with access to all the above tools
   - Uses OpenAI o3 model by default
   - Can recursively use search and URL fetching tools
   - Supports custom models and base URLs via parameters

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

### CLI Proxy API Issues

**"cli-proxy-api not found"**
- Install cli-proxy-api: https://github.com/router-for-me/CLIProxyAPI
- Ensure it's in your PATH

**"No cli-proxy-api config found"**
- Create config at `~/.cli-proxy-api/config.yaml`
- Or set path: `searxng cli-proxy-api set-config /path/to/config.yaml`

**"Failed to start cli-proxy-api"**
- Check config syntax: `cli-proxy-api -config ~/.cli-proxy-api/config.yaml`
- Check for port conflicts

**"No models available"**
- Run OAuth login: `cli-proxy-api -login`
- Check that config has valid credentials
- Try `searxng cli-proxy-api status` to see detailed status

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

SearXNG AI Kit can also be used as a Python library in your projects:

```python
import searxng

# Basic search
results = searxng.search("python tutorial", max_results=10)
print(f"Found {results['total_results']} results")

# Fetch URL content
content = searxng.fetch_url("https://example.com")
if content["success"]:
    print(f"Title: {content['title']}")

# Ask AI assistant with web search access
response = searxng.ask("What are the latest developments in quantum computing?")
if response["success"]:
    print(response["response"])

# Ask with custom model
response = searxng.ask(
    "Research renewable energy trends",
    model="openrouter/openai/gpt-4o-mini"
)

# Parallel URL fetching
urls = ["https://site1.com", "https://site2.com"]
contents = searxng.fetch_urls(urls, max_concurrent=3)

# Using client with defaults
client = searxng.SearXNGClient(
    default_engines=["duckduckgo", "startpage"],
    default_max_results=15
)
results = client.search("machine learning")
ai_response = client.ask("Explain machine learning basics")
```

### Library API

- `searxng.search(query, **kwargs)` - Web search with customizable parameters
- `searxng.fetch_url(url)` - Extract content from single URL  
- `searxng.fetch_urls(urls, max_concurrent=5)` - Parallel URL content extraction
- `searxng.ask(prompt, model="openai/o3", base_url=None)` - AI assistant with web search tools
- `searxng.get_available_engines()` - Get engine information
- `searxng.get_categories()` - Get search categories
- `searxng.SearXNGClient(**defaults)` - Client class with reusable settings

All functions have async variants (`search_async`, `fetch_url_async`, `ask_async`, etc.).

## License

This CLI tool uses SearXNG, which is licensed under AGPL-3.0. See the main repository for full license details.