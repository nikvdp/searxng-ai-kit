#!/usr/bin/env python3
"""SearXNG CLI - Command line interface for SearXNG search engine."""

import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import typer
from rich.console import Console
from rich.table import Table
from rich.text import Text

# Add the current directory to Python path so we can import searx
sys.path.insert(0, str(Path(__file__).parent))

import searx
import searx.engines
import searx.preferences
import searx.search
import searx.webadapter
from searx.search.models import EngineRef, SearchQuery
from searx.search.processors import PROCESSORS
from searx.results import ResultContainer
import threading
from timeit import default_timer
from uuid import uuid4

app = typer.Typer(
    name="searxng-cli",
    help="Privacy-respecting metasearch engine CLI with MCP server support",
    no_args_is_help=True,
)
console = Console()

# Default engines for different categories
DEFAULT_ENGINES = {
    "general": ["duckduckgo", "startpage", "brave"],
    "images": ["duckduckgo_images", "bing_images"],
    "videos": ["youtube_noapi", "vimeo"],
    "news": ["duckduckgo", "reuters"],
    "science": ["arxiv", "google_scholar"],
    "it": ["github", "stackoverflow"],
}

# Popular engines that users might want to enable/disable
COMMON_ENGINES = [
    "google", "duckduckgo", "bing", "startpage", "brave", "qwant",
    "google_images", "bing_images", "duckduckgo_images",
    "youtube_noapi", "vimeo", "dailymotion",
    "google_news", "bing_news", "reuters",
    "github", "gitlab", "stackoverflow",
    "arxiv", "google_scholar", "pubmed",
]


class CLISearch:
    """Custom search class that works without Flask context."""
    
    def __init__(self, search_query: SearchQuery):
        self.search_query = search_query
        self.result_container = ResultContainer()
        self.start_time = None
        self.actual_timeout = None
    
    def _get_requests(self):
        """Get search requests for all selected engines."""
        requests = []
        default_timeout = 0
        
        for engineref in self.search_query.engineref_list:
            if engineref.name not in PROCESSORS:
                continue
                
            processor = PROCESSORS[engineref.name]
            
            # Skip suspended engines
            if processor.extend_container_if_suspended(self.result_container):
                continue
            
            # Get request parameters
            request_params = processor.get_params(self.search_query, engineref.category)
            if request_params is None:
                continue
            
            requests.append((engineref.name, self.search_query.query, request_params))
            default_timeout = max(default_timeout, processor.engine.timeout)
        
        # Set timeout
        max_request_timeout = searx.settings['outgoing']['max_request_timeout']
        actual_timeout = default_timeout
        query_timeout = self.search_query.timeout_limit
        
        if max_request_timeout is None and query_timeout is None:
            pass
        elif max_request_timeout is None and query_timeout is not None:
            actual_timeout = min(default_timeout, query_timeout)
        elif max_request_timeout is not None and query_timeout is None:
            actual_timeout = min(default_timeout, max_request_timeout)
        elif max_request_timeout is not None and query_timeout is not None:
            actual_timeout = min(query_timeout, max_request_timeout)
        
        return requests, actual_timeout
    
    def search_multiple_requests(self, requests):
        """Execute multiple search requests in parallel."""
        search_id = str(uuid4())
        
        for engine_name, query, request_params in requests:
            processor = PROCESSORS[engine_name]
            
            def _search_wrapper():
                try:
                    processor.search(query, request_params, self.result_container, self.start_time, self.actual_timeout)
                except Exception as e:
                    self.result_container.add_unresponsive_engine(engine_name, str(e))
            
            th = threading.Thread(target=_search_wrapper, name=search_id)
            th._timeout = False
            th._engine_name = engine_name
            th.start()
        
        # Wait for all threads to complete
        for th in threading.enumerate():
            if th.name == search_id:
                remaining_time = max(0.0, self.actual_timeout - (default_timer() - self.start_time))
                th.join(remaining_time)
                if th.is_alive():
                    th._timeout = True
                    self.result_container.add_unresponsive_engine(th._engine_name, 'timeout')
    
    def search(self) -> ResultContainer:
        """Perform the search."""
        self.start_time = default_timer()
        
        requests, self.actual_timeout = self._get_requests()
        
        if requests:
            self.search_multiple_requests(requests)
        
        return self.result_container


def json_serial(obj: Any) -> Any:
    """JSON serializer for objects not serializable by default json code."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, bytes):
        return obj.decode('utf8')
    if isinstance(obj, set):
        return list(obj)
    raise TypeError(f"Type ({type(obj)}) not serializable")


def initialize_searx():
    """Initialize SearXNG search system."""
    try:
        settings_engines = searx.settings['engines']
        searx.search.load_engines(settings_engines)
        searx.search.initialize_network(settings_engines, searx.settings['outgoing'])
        searx.search.initialize_metrics([engine['name'] for engine in settings_engines])
        searx.search.initialize_processors(settings_engines)
        return True
    except Exception as e:
        console.print(f"[red]Error initializing SearXNG: {e}[/red]")
        return False


def get_available_engines() -> Dict[str, List[str]]:
    """Get available engines organized by category."""
    engines_by_category = {}
    for name, engine in searx.engines.engines.items():
        for category in engine.categories:
            if category not in engines_by_category:
                engines_by_category[category] = []
            engines_by_category[category].append(name)
    return engines_by_category


def create_search_query(
    query: str,
    category: str = "general",
    engines: Optional[List[str]] = None,
    lang: str = "all",
    safe_search: int = 0,
    page: int = 1,
    time_range: Optional[str] = None,
) -> SearchQuery:
    """Create a SearchQuery object."""
    
    # Get available engines for the category
    available_engines = get_available_engines()
    category_engines = available_engines.get(category, [])
    
    # Use specified engines or default engines for the category
    if engines:
        # Filter to only include engines that exist and are in the category
        target_engines = [e for e in engines if e in category_engines]
        if not target_engines:
            # Fallback to default engines if none of the specified engines are available
            target_engines = [e for e in DEFAULT_ENGINES.get(category, category_engines[:3]) if e in category_engines]
    else:
        # Use default engines for the category
        target_engines = [e for e in DEFAULT_ENGINES.get(category, category_engines[:3]) if e in category_engines]
    
    # If still no engines, use first 3 available engines in category
    if not target_engines and category_engines:
        target_engines = category_engines[:3]
    
    # Create EngineRef objects
    engine_refs = [EngineRef(name, category) for name in target_engines]
    
    return SearchQuery(
        query=query,
        engineref_list=engine_refs,
        lang=lang,
        safesearch=safe_search,
        pageno=page,
        time_range=time_range,
    )


def format_results_human(results_dict: Dict[str, Any]) -> None:
    """Format and display search results in human-readable format."""
    search_info = results_dict["search"]
    results = results_dict["results"]
    
    # Display search info
    console.print(f"\n[bold blue]Search:[/bold blue] {search_info['q']}")
    
    # Show which engines were used
    engines_used = set()
    for result in results:
        if result.get('engine'):
            engines_used.add(result['engine'])
    
    if engines_used:
        engines_str = ", ".join(sorted(engines_used))
        console.print(f"[green]Engines:[/green] {engines_str}")
    
    console.print(f"[dim]Language: {search_info['lang']}, Page: {search_info['pageno']}, Results: {len(results)}[/dim]\n")
    
    if not results:
        console.print("[yellow]No results found.[/yellow]")
        return
    
    # Display results
    for i, result in enumerate(results[:10], 1):  # Show top 10 results
        console.print(f"[bold cyan]{i}.[/bold cyan] [bold]{result.get('title', 'No title')}[/bold]")
        if result.get('content'):
            # Truncate content to reasonable length
            content = result['content'][:200] + "..." if len(result['content']) > 200 else result['content']
            console.print(f"   {content}")
        console.print(f"   [link]{result.get('url', 'No URL')}[/link]")
        if result.get('engine'):
            console.print(f"   [dim]Source: {result['engine']}[/dim]")
        console.print()
    
    # Display suggestions if any
    if results_dict.get("suggestions"):
        console.print("[bold yellow]Suggestions:[/bold yellow]")
        for suggestion in results_dict["suggestions"][:5]:
            console.print(f"  • {suggestion}")
        console.print()


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    category: str = typer.Option("general", "--category", "-c", help="Search category"),
    output_format: str = typer.Option("human", "--format", "-f", help="Output format: human or json"),
    engines: Optional[str] = typer.Option(None, "--engines", "-e", help="Comma-separated list of engines to use"),
    disable_engines: Optional[str] = typer.Option(None, "--disable", "-d", help="Comma-separated list of engines to disable"),
    language: str = typer.Option("all", "--lang", "-l", help="Search language"),
    safe_search: int = typer.Option(0, "--safe", "-s", help="Safe search level (0=off, 1=moderate, 2=strict)"),
    page: int = typer.Option(1, "--page", "-p", help="Page number"),
    time_range: Optional[str] = typer.Option(None, "--time", "-t", help="Time range: day, week, month, year"),
):
    """Search using SearXNG engines."""
    
    if not initialize_searx():
        raise typer.Exit(1)
    
    # Parse engines
    target_engines = None
    if engines:
        target_engines = [e.strip() for e in engines.split(",")]
    elif disable_engines:
        # Get all engines for category and remove disabled ones
        available_engines = get_available_engines()
        category_engines = available_engines.get(category, [])
        disabled = [e.strip() for e in disable_engines.split(",")]
        target_engines = [e for e in category_engines if e not in disabled]
    
    try:
        # Create search query
        search_query = create_search_query(
            query=query,
            category=category,
            engines=target_engines,
            lang=language,
            safe_search=safe_search,
            page=page,
            time_range=time_range,
        )
        
        # Show which engines will be used (for human-readable output)
        if output_format.lower() != "json":
            engine_names = [ref.name for ref in search_query.engineref_list]
            if engine_names:
                console.print(f"[dim]Using engines: {', '.join(sorted(engine_names))}[/dim]")
        
        # Perform search
        result_container = CLISearch(search_query).search()
        
        # Prepare results
        results_dict = {
            "search": {
                "q": search_query.query,
                "pageno": search_query.pageno,
                "lang": search_query.lang,
                "safesearch": search_query.safesearch,
                "timerange": search_query.time_range,
                "category": category,
            },
            "results": result_container.get_ordered_results(),
            "infoboxes": result_container.infoboxes,
            "suggestions": list(result_container.suggestions),
            "answers": list(result_container.answers),
            "number_of_results": result_container.number_of_results,
        }
        
        # Remove parsed_url from results for cleaner output
        for result in results_dict["results"]:
            if "parsed_url" in result:
                del result["parsed_url"]
        
        # Output results
        if output_format.lower() == "json":
            print(json.dumps(results_dict, indent=2, ensure_ascii=False, default=json_serial))
        else:
            format_results_human(results_dict)
            
    except Exception as e:
        console.print(f"[red]Error performing search: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def engines(
    category: Optional[str] = typer.Option(None, "--category", "-c", help="Filter by category"),
    show_common: bool = typer.Option(False, "--common", help="Show only common engines"),
):
    """List available search engines."""
    
    if not initialize_searx():
        raise typer.Exit(1)
    
    engines_by_category = get_available_engines()
    
    if category:
        if category not in engines_by_category:
            console.print(f"[red]Category '{category}' not found.[/red]")
            console.print(f"Available categories: {', '.join(sorted(engines_by_category.keys()))}")
            raise typer.Exit(1)
        
        console.print(f"\n[bold blue]Engines in '{category}' category:[/bold blue]")
        engines_list = engines_by_category[category]
        if show_common:
            engines_list = [e for e in engines_list if e in COMMON_ENGINES]
        
        # Display engines in columns for better readability
        engines_list = sorted(engines_list)
        for i in range(0, len(engines_list), 3):
            row_engines = engines_list[i:i+3]
            formatted_engines = []
            for engine in row_engines:
                marker = "⭐" if engine in COMMON_ENGINES else " "
                formatted_engines.append(f"{marker} {engine}")
            console.print("  ".join(f"{engine:<25}" for engine in formatted_engines))
        console.print()
    else:
        # Show all engines in a compact format
        if show_common:
            console.print("\n[bold blue]Common Search Engines:[/bold blue]")
            all_engines = []
            for engines_list in engines_by_category.values():
                all_engines.extend([e for e in engines_list if e in COMMON_ENGINES])
            all_engines = sorted(set(all_engines))
            
            # Display in 4 columns
            for i in range(0, len(all_engines), 4):
                row_engines = all_engines[i:i+4]
                console.print("  ".join(f"⭐ {engine:<20}" for engine in row_engines))
        else:
            console.print("\n[bold blue]Search Engines by Category:[/bold blue]")
            for cat in sorted(engines_by_category.keys()):
                engines_list = sorted(engines_by_category[cat])
                common_count = len([e for e in engines_list if e in COMMON_ENGINES])
                total_count = len(engines_list)
                
                if common_count > 0:
                    console.print(f"[cyan]{cat:<18}[/cyan] {total_count} engines ({common_count} common)")
                else:
                    console.print(f"[dim]{cat:<18}[/dim] {total_count} engines")
        
        console.print(f"\n[dim]Use --category <name> to see engines in a specific category[/dim]")
        console.print(f"[dim]Use --common to show only popular engines[/dim]")


@app.command()
def categories():
    """List available search categories."""
    
    if not initialize_searx():
        raise typer.Exit(1)
    
    engines_by_category = get_available_engines()
    
    console.print("\n[bold blue]Available Categories:[/bold blue]")
    for category in sorted(engines_by_category.keys()):
        engine_count = len(engines_by_category[category])
        console.print(f"  • [cyan]{category}[/cyan] ({engine_count} engines)")
    console.print()


# MCP Server Implementation
async def fetch_url_content(url: str) -> Dict[str, Any]:
    """Fetch URL content using Jina.ai's reader service."""
    jina_url = f"https://r.jina.ai/{url}"
    
    # Use JINA_API_KEY environment variable if available
    headers = {}
    if os.environ.get("JINA_API_KEY"):
        headers["Authorization"] = f"Bearer {os.environ['JINA_API_KEY']}"
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(jina_url, headers=headers)
            response.raise_for_status()
            
            # Try to parse as JSON first
            try:
                content_data = response.json()
                return {
                    "success": True,
                    "title": content_data.get("title", ""),
                    "content": content_data.get("content", ""),
                    "url": content_data.get("url", url),
                    "timestamp": content_data.get("timestamp", ""),
                }
            except json.JSONDecodeError:
                # If not JSON, return as plain text
                return {
                    "success": True,
                    "title": "",
                    "content": response.text,
                    "url": url,
                    "timestamp": "",
                }
    except httpx.HTTPError as e:
        return {
            "success": False,
            "error": f"HTTP error: {str(e)}",
            "url": url,
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Error fetching URL: {str(e)}",
            "url": url,
        }


async def fetch_multiple_urls_async(urls: List[str]) -> List[Dict[str, Any]]:
    """Fetch multiple URLs in parallel using existing network infrastructure, preserving order."""
    if not urls:
        return []
    
    # Import here to avoid circular imports
    from searx.network import Request, multi_requests
    
    # Use JINA_API_KEY environment variable if available
    headers = {}
    if os.environ.get("JINA_API_KEY"):
        headers["Authorization"] = f"Bearer {os.environ['JINA_API_KEY']}"
    
    # Create requests for all URLs
    requests = []
    for url in urls:
        jina_url = f"https://r.jina.ai/{url}"
        requests.append(Request.get(jina_url, headers=headers, timeout=30.0))
    
    # Execute requests in parallel
    responses = multi_requests(requests)
    
    # Process responses while preserving order
    results = []
    for i, (url, response) in enumerate(zip(urls, responses)):
        if isinstance(response, Exception):
            # Handle errors
            results.append({
                "success": False,
                "error": f"Error fetching URL: {str(response)}",
                "url": url,
                "index": i,
            })
        else:
            try:
                # Check if request was successful
                response.raise_for_status()
                
                # Try to parse as JSON first
                try:
                    content_data = response.json()
                    results.append({
                        "success": True,
                        "title": content_data.get("title", ""),
                        "content": content_data.get("content", ""),
                        "url": content_data.get("url", url),
                        "timestamp": content_data.get("timestamp", ""),
                        "index": i,
                    })
                except json.JSONDecodeError:
                    # If not JSON, return as plain text
                    results.append({
                        "success": True,
                        "title": "",
                        "content": response.text,
                        "url": url,
                        "timestamp": "",
                        "index": i,
                    })
            except Exception as e:
                results.append({
                    "success": False,
                    "error": f"HTTP error: {str(e)}",
                    "url": url,
                    "index": i,
                })
    
    return results


async def perform_search_async(
    query: str,
    category: str = "general",
    engines: Optional[List[str]] = None,
    language: str = "all",
    safe_search: int = 0,
    max_results: int = 10,
) -> Dict[str, Any]:
    """Async wrapper for search functionality."""
    # Initialize SearXNG if not already done
    if not initialize_searx():
        return {"error": "Failed to initialize SearXNG"}
    
    try:
        # Create search query
        search_query = create_search_query(
            query=query,
            category=category,
            engines=engines,
            lang=language,
            safe_search=safe_search,
            page=1,
        )
        
        # Perform search
        result_container = CLISearch(search_query).search()
        
        # Get results
        results = result_container.get_ordered_results()[:max_results]
        
        # Clean up results for JSON serialization
        clean_results = []
        for result in results:
            clean_result = {
                "title": result.get("title", ""),
                "url": result.get("url", ""),
                "content": result.get("content", ""),
                "engine": result.get("engine", ""),
            }
            clean_results.append(clean_result)
        
        return {
            "success": True,
            "query": query,
            "category": category,
            "engines_used": [ref.name for ref in search_query.engineref_list],
            "results": clean_results,
            "suggestions": list(result_container.suggestions),
            "number_of_results": result_container.number_of_results,
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Search error: {str(e)}",
            "query": query,
        }


@app.command("fetch-urls")
def fetch_urls_command(
    urls: List[str] = typer.Argument(..., help="URLs to fetch content from"),
    output_format: str = typer.Option("json", "--format", "-f", help="Output format: json or human"),
    max_concurrent: int = typer.Option(5, "--concurrent", "-c", help="Maximum number of concurrent requests"),
):
    """Fetch content from multiple URLs in parallel using Jina.ai's reader service.
    
    This command fetches content from one or more URLs concurrently, maintaining the 
    order of results to match the input order. Each URL is processed through Jina.ai's
    reader service to extract clean, readable content.
    
    Examples:
    
    # Fetch single URL
    searxng-cli fetch-urls "https://example.com"
    
    # Fetch multiple URLs in parallel  
    searxng-cli fetch-urls "https://site1.com" "https://site2.com" "https://site3.com"
    
    # Output in human-readable format
    searxng-cli fetch-urls "https://example.com" --format human
    
    # Control concurrency
    searxng-cli fetch-urls url1 url2 url3 --concurrent 2
    
    Environment variables:
    - JINA_API_KEY: Optional API key for enhanced Jina.ai features
    """
    if not urls:
        console.print("[red]Error: At least one URL is required[/red]")
        raise typer.Exit(1)
    
    # Limit concurrent requests to reasonable number
    if max_concurrent > 20:
        console.print("[yellow]Warning: Limiting concurrent requests to 20 for stability[/yellow]")
        max_concurrent = 20
    
    try:
        if len(urls) == 1:
            # Single URL - use existing async function but run it
            import asyncio
            result = asyncio.run(fetch_url_content(urls[0]))
            results = [result]
        else:
            # Multiple URLs - use parallel fetching
            import asyncio
            results = asyncio.run(fetch_multiple_urls_async(urls))
        
        if output_format.lower() == "json":
            # JSON output
            output = json.dumps(results, indent=2, ensure_ascii=False, default=json_serial)
            console.print(output)
        else:
            # Human-readable output
            for i, result in enumerate(results):
                if result.get("success"):
                    console.print(f"\n[bold green]✓ URL {i+1}:[/bold green] {result['url']}")
                    if result.get("title"):
                        console.print(f"[bold]Title:[/bold] {result['title']}")
                    if result.get("content"):
                        # Truncate very long content for readability
                        content = result['content']
                        if len(content) > 1000:
                            content = content[:1000] + "... [truncated]"
                        console.print(f"[bold]Content:[/bold] {content}")
                    if result.get("timestamp"):
                        console.print(f"[dim]Timestamp: {result['timestamp']}[/dim]")
                else:
                    console.print(f"\n[bold red]✗ URL {i+1}:[/bold red] {result['url']}")
                    console.print(f"[red]Error: {result.get('error', 'Unknown error')}[/red]")
                
                if i < len(results) - 1:  # Add separator between results
                    console.print("[dim]" + "─" * 50 + "[/dim]")
                    
    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled by user[/yellow]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error fetching URLs: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def mcp_server():
    """Start the MCP server for Model Context Protocol integration.
    
    This starts an MCP server that communicates over stdio, providing web search
    and URL content retrieval tools for AI applications like Claude.
    
    Available tools:
    - web_search: Search using SearXNG's 180+ search engines
    - fetch_url: Retrieve and extract content from URL(s) using Jina.ai (supports arrays)
    
    Usage with Claude Desktop:
    Add this to your claude_desktop_config.json:
    {
      "mcpServers": {
        "searxng": {
          "command": "/path/to/searxng-cli",
          "args": ["mcp-server"]
        }
      }
    }
    
    Environment variables:
    - JINA_API_KEY: Optional API key for enhanced Jina.ai features
    """
    try:
        from mcp.server import Server
        from mcp.server.stdio import stdio_server
        from mcp.types import Tool, TextContent
        import mcp.server.stdio
        import mcp.types
    except ImportError:
        console.print("[red]MCP library not found. Please install with: pip install mcp[/red]")
        raise typer.Exit(1)
    
    # Create MCP server
    server = Server("searxng-cli")
    
    @server.list_tools()
    async def list_tools() -> list[Tool]:
        """List available MCP tools."""
        return [
            Tool(
                name="web_search",
                description="Search the web using SearXNG's search engines",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query",
                        },
                        "category": {
                            "type": "string",
                            "description": "Search category (general, images, videos, news, science, it)",
                            "default": "general",
                        },
                        "engines": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of search engines to use",
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum number of results to return",
                            "default": 10,
                        },
                        "language": {
                            "type": "string",
                            "description": "Search language",
                            "default": "all",
                        },
                    },
                    "required": ["query"],
                },
            ),
            Tool(
                name="fetch_url",
                description="Fetch and extract content from URL(s) using Jina.ai's reader service. Supports single URL or array of URLs for parallel fetching.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "url": {
                            "oneOf": [
                                {
                                    "type": "string",
                                    "description": "Single URL to fetch content from"
                                },
                                {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Array of URLs to fetch content from in parallel"
                                }
                            ],
                            "description": "URL or array of URLs to fetch content from",
                        },
                    },
                    "required": ["url"],
                },
            ),
        ]
    
    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[TextContent]:
        """Handle tool calls."""
        if name == "web_search":
            query = arguments.get("query")
            category = arguments.get("category", "general")
            engines = arguments.get("engines")
            max_results = arguments.get("max_results", 10)
            language = arguments.get("language", "all")
            
            result = await perform_search_async(
                query=query,
                category=category,
                engines=engines,
                language=language,
                max_results=max_results,
            )
            
            return [TextContent(
                type="text",
                text=json.dumps(result, indent=2, ensure_ascii=False, default=json_serial)
            )]
        
        elif name == "fetch_url":
            url_input = arguments.get("url")
            if not url_input:
                return [TextContent(
                    type="text",
                    text=json.dumps({"error": "URL is required"})
                )]
            
            # Handle both single URL and array of URLs
            if isinstance(url_input, list):
                # Multiple URLs - use parallel fetching
                if not url_input:
                    return [TextContent(
                        type="text",
                        text=json.dumps({"error": "URL array cannot be empty"})
                    )]
                
                result = await fetch_multiple_urls_async(url_input)
                return [TextContent(
                    type="text",
                    text=json.dumps(result, indent=2, ensure_ascii=False, default=json_serial)
                )]
            else:
                # Single URL - use existing function
                result = await fetch_url_content(url_input)
                return [TextContent(
                    type="text",
                    text=json.dumps(result, indent=2, ensure_ascii=False, default=json_serial)
                )]
        
        else:
            return [TextContent(
                type="text",
                text=json.dumps({"error": f"Unknown tool: {name}"})
            )]
    
    async def run_server():
        """Run the MCP server."""
        async with stdio_server() as (read_stream, write_stream):
            await server.run(read_stream, write_stream, server.create_initialization_options())
    
    # Run the server
    console.print("[blue]Starting MCP server...[/blue]")
    print("Server is running on stdio. Use Ctrl+C to stop.", file=sys.stderr)
    
    try:
        asyncio.run(run_server())
    except KeyboardInterrupt:
        print("\nMCP server stopped.", file=sys.stderr)
    except Exception as e:
        print(f"Error running MCP server: {e}", file=sys.stderr)
        raise typer.Exit(1)


if __name__ == "__main__":
    app()