#!/usr/bin/env python3
"""SearXNG CLI - Command line interface for SearXNG search engine."""

import asyncio
import json
import logging
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import quote

import httpx
import typer
from rich.console import Console
from rich.table import Table
from rich.text import Text

# Add the current directory to Python path so we can import searx
sys.path.insert(0, str(Path(__file__).parent))

# Configure logging to suppress noisy errors from SearXNG engines
def configure_logging():
    """Configure logging to suppress engine errors that don't affect overall functionality."""
    # Set SearXNG loggers to WARNING level to suppress INFO/ERROR for individual engine failures
    logging.getLogger('searx').setLevel(logging.CRITICAL)
    logging.getLogger('searx.engines').setLevel(logging.CRITICAL) 
    logging.getLogger('searx.search').setLevel(logging.CRITICAL)
    logging.getLogger('searx.network').setLevel(logging.CRITICAL)
    logging.getLogger('httpx').setLevel(logging.CRITICAL)
    logging.getLogger('httpcore').setLevel(logging.CRITICAL)
    
# Configure logging early
configure_logging()

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

# Shared AI system prompt for consistent tool usage across all modes
BASE_AI_SYSTEM_PROMPT = """You have access to powerful web search and URL fetching tools. When researching topics, you should:

**PRIORITIZE PARALLEL OPERATIONS FOR SPEED:**
- ALWAYS prefer multi_web_search over single web_search when you need multiple searches
- ALWAYS prefer fetch_urls over single fetch_url when you need multiple URLs  
- Use parallel tools aggressively - users want comprehensive data delivered quickly
- Run multiple related searches simultaneously rather than sequentially
- Fetch all relevant URLs at once rather than one-by-one

**TOOL USAGE GUIDELINES:**
- multi_web_search: Use for related queries like ["topic overview", "recent developments", "expert opinions"]
- fetch_urls: Use when you find multiple relevant URLs in search results
- Be thorough but efficient - parallel execution lets you gather more data faster
- Don't hesitate to use 3-5 searches or 5-10 URL fetches if it improves your response"""

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
    # Properly encode the URL to handle special characters and parameters
    encoded_url = quote(url, safe='')
    jina_url = f"https://r.jina.ai/{encoded_url}"
    
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
        # Properly encode the URL to handle special characters and parameters
        encoded_url = quote(url, safe='')
        jina_url = f"https://r.jina.ai/{encoded_url}"
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


async def perform_multi_search_async(
    queries: List[str],
    category: str = "general",
    engines: Optional[List[str]] = None,
    language: str = "all",
    safe_search: int = 0,
    max_results: int = 10,
) -> Dict[str, Any]:
    """Perform multiple search queries in parallel."""
    if not queries:
        return {"error": "No queries provided"}
    
    # Create tasks for parallel execution
    tasks = []
    for query in queries:
        task = perform_search_async(
            query=query,
            category=category,
            engines=engines,
            language=language,
            safe_search=safe_search,
            max_results=max_results,
        )
        tasks.append(task)
    
    try:
        # Execute all searches in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results and handle exceptions
        search_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                search_results.append({
                    "success": False,
                    "query": queries[i],
                    "error": f"Search failed: {str(result)}",
                })
            else:
                search_results.append(result)
        
        # Create a more structured response that clearly maps queries to results
        structured_results = []
        for i, (query, result) in enumerate(zip(queries, search_results)):
            structured_result = {
                "query_index": i + 1,
                "query": query,
                "search_result": result
            }
            structured_results.append(structured_result)
        
        return {
            "success": True,
            "summary": {
                "total_queries": len(queries),
                "successful_queries": sum(1 for r in search_results if r.get("success", False)),
                "queries_executed": queries
            },
            "results": structured_results,
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Multi-search error: {str(e)}",
            "queries": queries,
        }


@app.command("fetch-urls")
def fetch_urls_command(
    urls: List[str] = typer.Argument(..., help="URLs to fetch content from"),
    output_format: str = typer.Option("human", "--format", "-f", help="Output format: human or json"),
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
                        console.print(f"[bold]Content:[/bold] {result['content']}")
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


def get_mcp_tools():
    """Get the list of MCP tools available."""
    from mcp.types import Tool
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
            name="multi_web_search",
            description="Search the web with multiple queries in parallel using SearXNG",
            inputSchema={
                "type": "object",
                "properties": {
                    "queries": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Array of search queries to execute in parallel",
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
                        "description": "Maximum number of results to return per query",
                        "default": 10,
                    },
                    "language": {
                        "type": "string",
                        "description": "Search language",
                        "default": "all",
                    },
                },
                "required": ["queries"],
            },
        ),
        Tool(
            name="fetch_url",
            description="Fetch and extract content from a single URL using Jina.ai's reader service.",
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "URL to fetch content from"
                    },
                },
                "required": ["url"],
            },
        ),
        Tool(
            name="fetch_urls",
            description="Fetch and extract content from multiple URLs in parallel using Jina.ai's reader service.",
            inputSchema={
                "type": "object",
                "properties": {
                    "urls": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Array of URLs to fetch content from in parallel"
                    },
                },
                "required": ["urls"],
            },
        ),
        Tool(
            name="ask",
            description="Ask an AI assistant with access to web search and URL fetching tools. The assistant can run parallel searches and fetch content from multiple URLs to provide comprehensive research and answers.",
            inputSchema={
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "Question or research request to ask the AI assistant"
                    },
                    "model": {
                        "type": "string", 
                        "description": "Model to use (default: openai/o3)",
                        "default": "openai/o3"
                    },
                    "base_url": {
                        "type": "string",
                        "description": "Custom API base URL (optional, overrides OPENAI_BASE_URL env var)"
                    }
                },
                "required": ["prompt"],
            },
        ),
    ]


async def ask_ai_async(
    prompt: str,
    model: str = "openai/o3",
    base_url: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Core async function for asking AI with web search tools.
    This is used by both the CLI command and the library interface.
    """
    import litellm
    import os
    import sys
    
    # Check for API keys
    api_keys = {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY"), 
        "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY"),
        "OPENROUTER_API_KEY": os.getenv("OPENROUTER_API_KEY"),
    }
    
    # Check if we have at least one API key
    if not any(api_keys.values()):
        return {
            "success": False,
            "error": "No API keys found. Please set OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY, or OPENROUTER_API_KEY",
            "prompt": prompt,
            "model": model
        }
    
    # Define tools for the LLM
    tools = [
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Search the web using SearXNG's search engines",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "category": {"type": "string", "description": "Search category", "default": "general"},
                        "max_results": {"type": "integer", "description": "Maximum results", "default": 10}
                    },
                    "required": ["query"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "multi_web_search",
                "description": "Search the web with multiple queries in parallel",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "queries": {"type": "array", "items": {"type": "string"}, "description": "Array of search queries"},
                        "category": {"type": "string", "description": "Search category", "default": "general"},
                        "max_results": {"type": "integer", "description": "Maximum results per query", "default": 10}
                    },
                    "required": ["queries"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "fetch_url",
                "description": "Fetch and extract content from a single URL",
                "parameters": {
                    "type": "object",
                    "properties": {"url": {"type": "string", "description": "URL to fetch"}},
                    "required": ["url"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "fetch_urls",
                "description": "Fetch and extract content from multiple URLs in parallel",
                "parameters": {
                    "type": "object",
                    "properties": {"urls": {"type": "array", "items": {"type": "string"}, "description": "URLs to fetch"}},
                    "required": ["urls"]
                }
            }
        }
    ]
    
    try:
        # Use base system prompt with ask-specific additions
        system_content = BASE_AI_SYSTEM_PROMPT + """\n\n**ASK MODE - COMPREHENSIVE RESEARCH:**
- Provide thorough, well-researched responses with comprehensive coverage
- Use extensive parallel searches to gather complete information
- Include relevant data, statistics, examples, and expert perspectives
- Cite sources and provide context for your findings
- Aim for depth and completeness in your analysis"""
        
        user_prompt = f"User request: {prompt}"
        
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_prompt}
        ]
        
        # Prepare completion arguments
        completion_args = {
            "model": model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto"
        }
        
        # Add base_url if provided (overrides environment variable)
        if base_url:
            completion_args["base_url"] = base_url
        
        # Make initial request to the LLM
        response = litellm.completion(**completion_args)
        
        # Handle tool calls iteratively
        while response.choices[0].message.tool_calls:
            # Add the assistant's message with tool calls
            messages.append(response.choices[0].message.model_dump())
            
            # Execute each tool call
            for tool_call in response.choices[0].message.tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                
                # Log tool usage to stderr for shell piping friendliness
                stderr_console = Console(file=sys.stderr, force_terminal=True)
                if function_name == "web_search":
                    stderr_console.print(f"🔍 [cyan]Searching:[/cyan] {function_args.get('query', 'N/A')}")
                elif function_name == "multi_web_search":
                    queries = function_args.get('queries', [])
                    stderr_console.print(f"🔍 [cyan]Multi-search:[/cyan] {', '.join(queries[:3])}{'...' if len(queries) > 3 else ''}")
                elif function_name == "fetch_url":
                    stderr_console.print(f"📄 [blue]Fetching:[/blue] {function_args.get('url', 'N/A')}")
                elif function_name == "fetch_urls":
                    urls = function_args.get('urls', [])
                    stderr_console.print(f"📄 [blue]Fetching {len(urls)} URLs:[/blue] {urls[0] if urls else 'N/A'}{'...' if len(urls) > 1 else ''}")
                else:
                    stderr_console.print(f"🔧 [magenta]Tool:[/magenta] {function_name}")
                
                # Execute the tool using our existing handler
                tool_result = await handle_tool_call(function_name, function_args)
                
                # Add tool result to messages
                messages.append({
                    "role": "tool",
                    "content": tool_result,
                    "tool_call_id": tool_call.id
                })
            
            # Get next response from LLM
            response = litellm.completion(**completion_args)
        
        # Return the final response
        final_response = response.choices[0].message.content
        return {
            "success": True,
            "model": model,
            "prompt": prompt,
            "response": final_response
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Error calling {model}: {str(e)}",
            "prompt": prompt,
            "model": model
        }


async def handle_tool_call(name: str, arguments: dict):
    """Shared tool call handler for both stdio and HTTP versions."""
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
        
        return json.dumps(result, indent=2, ensure_ascii=False, default=json_serial)
    
    elif name == "multi_web_search":
        queries = arguments.get("queries")
        if not queries:
            return json.dumps({"error": "Queries array is required"})
        
        if not isinstance(queries, list):
            return json.dumps({"error": "Queries must be an array"})
        
        if not queries:
            return json.dumps({"error": "Queries array cannot be empty"})
        
        category = arguments.get("category", "general")
        engines = arguments.get("engines")
        max_results = arguments.get("max_results", 10)
        language = arguments.get("language", "all")
        
        result = await perform_multi_search_async(
            queries=queries,
            category=category,
            engines=engines,
            language=language,
            max_results=max_results,
        )
        
        return json.dumps(result, indent=2, ensure_ascii=False, default=json_serial)
    
    elif name == "fetch_url":
        url = arguments.get("url")
        if not url:
            return json.dumps({"error": "URL is required"})
        
        # Single URL
        result = await fetch_url_content(url)
        return json.dumps(result, indent=2, ensure_ascii=False, default=json_serial)
    
    elif name == "fetch_urls":
        urls = arguments.get("urls")
        if not urls:
            return json.dumps({"error": "URLs array is required"})
        
        if not isinstance(urls, list):
            return json.dumps({"error": "URLs must be an array"})
        
        if not urls:
            return json.dumps({"error": "URLs array cannot be empty"})
        
        # Multiple URLs - use parallel fetching
        result = await fetch_multiple_urls_async(urls)
        return json.dumps(result, indent=2, ensure_ascii=False, default=json_serial)
    
    elif name == "ask":
        prompt = arguments.get("prompt")
        if not prompt:
            return json.dumps({"error": "Prompt is required"})
        
        model = arguments.get("model", "openai/o3")
        base_url = arguments.get("base_url")  # Optional custom base URL
        
        # Use the shared ask_ai_async function instead of duplicating logic
        result = await ask_ai_async(prompt, model, base_url)
        return json.dumps(result, indent=2, ensure_ascii=False, default=json_serial)
    
    else:
        return json.dumps({"error": f"Unknown tool: {name}"})


@app.command()
def mcp_server(
    remote: bool = typer.Option(False, "--remote", help="Start as remote HTTP server instead of stdio"),
    host: str = typer.Option("0.0.0.0", "--host", help="Host to bind to (default: 0.0.0.0)"),
    port: int = typer.Option(8000, "--port", help="Port to bind to (default: 8000)"),
):
    """Start the MCP server for Model Context Protocol integration.
    
    This starts an MCP server that communicates over stdio by default, or over HTTP
    when using the --remote flag. Provides web search and URL content retrieval 
    tools for AI applications like Claude.
    
    Available tools:
    - web_search: Search using SearXNG's 180+ search engines
    - fetch_url: Retrieve and extract content from URL(s) using Jina.ai (supports arrays)
    
    Usage with Claude Desktop (stdio):
    Add this to your claude_desktop_config.json:
    {
      "mcpServers": {
        "searxng": {
          "command": "searxng",
          "args": ["mcp-server"]
        }
      }
    }
    
    Usage as remote HTTP server:
    searxng mcp-server --remote --host 0.0.0.0 --port 8000
    
    Then connect MCP clients to: http://your-server:8000
    
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
        return get_mcp_tools()

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[TextContent]:
        """Handle tool calls for stdio version."""
        content_text = await handle_tool_call(name, arguments)
        return [TextContent(type="text", text=content_text)]
    
    async def run_server():
        """Run the MCP server."""
        async with stdio_server() as (read_stream, write_stream):
            await server.run(read_stream, write_stream, server.create_initialization_options())
    
    # Choose server type based on remote flag
    if remote:
        console.print(f"[blue]Starting MCP remote HTTP server on {host}:{port}...[/blue]")
        asyncio.run(run_http_server(server, host, port))
    else:
        console.print("[blue]Starting MCP server on stdio...[/blue]")
        print("Server is running on stdio. Use Ctrl+C to stop.", file=sys.stderr)
        
        try:
            asyncio.run(run_server())
        except KeyboardInterrupt:
            print("\nMCP server stopped.", file=sys.stderr)
        except Exception as e:
            print(f"Error running MCP server: {e}", file=sys.stderr)
            raise typer.Exit(1)


async def run_http_server(mcp_server, host: str, port: int):
    """Run the MCP server over HTTP with JSON-RPC 2.0 and SSE support."""
    try:
        from fastapi import FastAPI, Request, HTTPException
        from fastapi.responses import Response, StreamingResponse
        from fastapi.middleware.cors import CORSMiddleware
        from sse_starlette.sse import EventSourceResponse
        import uvicorn
    except ImportError:
        console.print("[red]FastAPI and dependencies not found. Please install with: pip install fastapi uvicorn sse-starlette[/red]")
        raise typer.Exit(1)
    
    # Store active sessions
    sessions = {}
    
    app = FastAPI(
        title="SearXNG MCP Server",
        description="Model Context Protocol server for SearXNG search engine",
        version="0.1.0"
    )
    
    # Add CORS middleware for web clients
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    def validate_origin(request: Request):
        """Validate Origin header to prevent DNS rebinding attacks."""
        origin = request.headers.get("origin")
        if origin:
            # For localhost/0.0.0.0, allow common variations
            if host in ["127.0.0.1", "localhost", "0.0.0.0"]:
                allowed_origins = [
                    "http://localhost",
                    "http://127.0.0.1",
                    f"http://localhost:{port}",
                    f"http://127.0.0.1:{port}"
                ]
                if not any(origin.startswith(allowed) for allowed in allowed_origins):
                    raise HTTPException(status_code=403, detail="Forbidden origin")
            elif not origin.startswith(f"http://{host}"):
                raise HTTPException(status_code=403, detail="Forbidden origin")
    
    def create_jsonrpc_response(id: Any, result: Any = None, error: Any = None):
        """Create a JSON-RPC 2.0 response."""
        response = {"jsonrpc": "2.0", "id": id}
        if error:
            response["error"] = error
        else:
            response["result"] = result
        return response
    
    def create_jsonrpc_error(code: int, message: str, data: Any = None):
        """Create a JSON-RPC 2.0 error object."""
        error = {"code": code, "message": message}
        if data:
            error["data"] = data
        return error
    
    @app.get("/")
    async def get_endpoint(request: Request):
        """Handle GET requests for SSE stream or capabilities."""
        validate_origin(request)
        
        accept = request.headers.get("accept", "")
        if "text/event-stream" in accept:
            # Start SSE stream
            session_id = str(uuid.uuid4())
            sessions[session_id] = {"created": datetime.now()}
            
            async def event_stream():
                # Send initial endpoint event
                yield {
                    "event": "endpoint",
                    "data": json.dumps({
                        "method": "notifications/initialized",
                        "params": {
                            "protocolVersion": "2024-11-05",
                            "capabilities": {
                                "tools": {
                                    "listChanged": True
                                }
                            },
                            "serverInfo": {
                                "name": "searxng-cli",
                                "version": "0.1.0"
                            }
                        }
                    })
                }
                
                # Keep connection alive
                try:
                    while True:
                        await asyncio.sleep(30)
                        yield {"event": "ping", "data": ""}
                except asyncio.CancelledError:
                    sessions.pop(session_id, None)
                    return
            
            return EventSourceResponse(
                event_stream(),
                headers={"Mcp-Session-Id": session_id}
            )
        else:
            # Return server capabilities
            return {
                "name": "searxng-cli",
                "version": "0.1.0",
                "capabilities": {
                    "tools": {},
                    "resources": {}
                },
                "protocolVersion": "2024-11-05"
            }
    
    @app.post("/register")
    async def register_client(request: Request):
        """Handle OAuth 2.0 dynamic client registration."""
        validate_origin(request)
        
        try:
            body = await request.json()
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid JSON")
        
        # Simple client registration response
        # In production, you'd validate and store client info
        client_id = str(uuid.uuid4())
        
        return {
            "client_id": client_id,
            "client_id_issued_at": int(datetime.now().timestamp()),
            "registration_access_token": str(uuid.uuid4()),
            "registration_client_uri": f"http://{host}:{port}/register/{client_id}",
            "grant_types": ["authorization_code"],
            "response_types": ["code"],
            "token_endpoint_auth_method": "none"  # No auth required for this demo
        }
    
    @app.post("/")
    async def post_endpoint(request: Request):
        """Handle POST requests with JSON-RPC 2.0."""
        validate_origin(request)
        
        try:
            body = await request.json()
        except Exception:
            return Response(
                content=json.dumps(create_jsonrpc_response(
                    None, 
                    error=create_jsonrpc_error(-32700, "Parse error")
                )),
                media_type="application/json",
                status_code=400
            )
        
        # Validate JSON-RPC 2.0 format
        if not isinstance(body, dict) or body.get("jsonrpc") != "2.0":
            return Response(
                content=json.dumps(create_jsonrpc_response(
                    body.get("id"),
                    error=create_jsonrpc_error(-32600, "Invalid Request")
                )),
                media_type="application/json",
                status_code=400
            )
        
        method = body.get("method")
        params = body.get("params", {})
        request_id = body.get("id")
        
        # Generate session ID if not present
        session_id = request.headers.get("mcp-session-id")
        if not session_id:
            session_id = str(uuid.uuid4())
            sessions[session_id] = {"created": datetime.now()}
        
        response_headers = {"Mcp-Session-Id": session_id}
        
        try:
            if method == "initialize":
                result = {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "tools": {
                            "listChanged": True
                        }
                    },
                    "serverInfo": {
                        "name": "searxng-cli",
                        "version": "0.1.0"
                    }
                }
                
            elif method == "tools/list":
                # Use shared tool definitions
                tools = get_mcp_tools()
                result = {
                    "tools": [
                        {
                            "name": tool.name,
                            "description": tool.description,
                            "inputSchema": tool.inputSchema
                        }
                        for tool in tools
                    ]
                }
                
            elif method == "notifications/initialized":
                # Client is notifying that it has initialized
                # For notifications (id is null), we should return 204 No Content
                if request_id is None:
                    return Response(status_code=204, headers=response_headers)
                else:
                    result = {"status": "initialized"}
                
            elif method == "tools/call":
                tool_name = params.get("name")
                arguments = params.get("arguments", {})
                
                if not tool_name:
                    return Response(
                        content=json.dumps(create_jsonrpc_response(
                            request_id,
                            error=create_jsonrpc_error(-32602, "Invalid params: missing tool name")
                        )),
                        media_type="application/json",
                        status_code=400,
                        headers=response_headers
                    )
                
                # Use shared tool handler
                content_text = await handle_tool_call(tool_name, arguments)
                
                # Check for error responses
                try:
                    result_data = json.loads(content_text)
                    if isinstance(result_data, dict) and "error" in result_data and result_data.get("error", "").startswith("Unknown tool:"):
                        return Response(
                            content=json.dumps(create_jsonrpc_response(
                                request_id,
                                error=create_jsonrpc_error(-32601, result_data["error"])
                            )),
                            media_type="application/json",
                            status_code=404,
                            headers=response_headers
                        )
                except json.JSONDecodeError:
                    pass  # Content is not JSON, treat as regular response
                
                result = {
                    "content": [
                        {
                            "type": "text",
                            "text": content_text
                        }
                    ]
                }
                
            else:
                return Response(
                    content=json.dumps(create_jsonrpc_response(
                        request_id,
                        error=create_jsonrpc_error(-32601, f"Method not found: {method}")
                    )),
                    media_type="application/json",
                    status_code=404,
                    headers=response_headers
                )
            
            return Response(
                content=json.dumps(create_jsonrpc_response(request_id, result), 
                                 ensure_ascii=False, default=json_serial),
                media_type="application/json",
                headers=response_headers
            )
            
        except Exception as e:
            console.print(f"[red]Error handling request: {e}[/red]")
            return Response(
                content=json.dumps(create_jsonrpc_response(
                    request_id,
                    error=create_jsonrpc_error(-32603, f"Internal error: {str(e)}")
                )),
                media_type="application/json",
                status_code=500,
                headers=response_headers
            )
    
    # Start server
    print(f"MCP HTTP server starting on {host}:{port}", file=sys.stderr)
    try:
        config = uvicorn.Config(
            app=app,
            host=host,
            port=port,
            log_level="info",
            access_log=False
        )
        server = uvicorn.Server(config)
        await server.serve()
    except KeyboardInterrupt:
        print("\nMCP HTTP server stopped.", file=sys.stderr)
    except Exception as e:
        print(f"Error running MCP HTTP server: {e}", file=sys.stderr)
        raise typer.Exit(1)


@app.command("multi-search")
def multi_search_command(
    queries: List[str] = typer.Argument(..., help="Search queries to execute in parallel"),
    output_format: str = typer.Option("human", "--format", "-f", help="Output format: human or json"),
    category: str = typer.Option("general", "--category", "-c", help="Search category"),
    engines: Optional[List[str]] = typer.Option(None, "--engines", "-e", help="Search engines to use"),
    max_results: int = typer.Option(10, "--max-results", "-n", help="Maximum results per query"),
    language: str = typer.Option("all", "--language", "-l", help="Search language"),
):
    """Execute multiple search queries in parallel."""
    if not queries:
        console.print("[red]Error: At least one query is required[/red]")
        raise typer.Exit(1)
    
    try:
        import asyncio
        result = asyncio.run(perform_multi_search_async(
            queries=queries,
            category=category,
            engines=engines,
            language=language,
            max_results=max_results,
        ))
        
        if output_format.lower() == "json":
            # JSON output
            output = json.dumps(result, indent=2, ensure_ascii=False, default=json_serial)
            console.print(output)
        else:
            # Human-readable output
            if result.get("success"):
                summary = result.get("summary", {})
                console.print(f"[bold green]✓ Multi-Search Results[/bold green]")
                console.print(f"Executed {summary.get('total_queries', 0)} queries, {summary.get('successful_queries', 0)} successful")
                console.print()
                
                for i, query_result in enumerate(result.get("results", [])):
                    query_info = query_result.get("search_result", {})
                    query_text = query_result.get("query", "")
                    query_index = query_result.get("query_index", i + 1)
                    
                    if query_info.get("success"):
                        console.print(f"[bold blue]Query {query_index}:[/bold blue] {query_text}")
                        console.print(f"[dim]Found {len(query_info.get('results', []))} results[/dim]")
                        
                        for j, search_result in enumerate(query_info.get("results", [])[:3]):  # Show top 3
                            console.print(f"  {j+1}. [bold]{search_result.get('title', 'No title')}[/bold]")
                            console.print(f"     [link]{search_result.get('url', '')}[/link]")
                            if search_result.get("content"):
                                content = search_result["content"][:100] + "..." if len(search_result["content"]) > 100 else search_result["content"]
                                console.print(f"     {content}")
                        
                        if len(query_info.get("results", [])) > 3:
                            console.print(f"     [dim]... and {len(query_info.get('results', [])) - 3} more results[/dim]")
                    else:
                        console.print(f"[bold red]Query {query_index} (FAILED):[/bold red] {query_text}")
                        console.print(f"[red]  Error: {query_info.get('error', 'Unknown error')}[/red]")
                    
                    if i < len(result.get("results", [])) - 1:  # Add separator between queries
                        console.print("[dim]" + "─" * 60 + "[/dim]")
            else:
                console.print(f"[red]Multi-search failed: {result.get('error', 'Unknown error')}[/red]")
                
    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled by user[/yellow]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error executing multi-search: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def ask(
    prompt: Optional[str] = typer.Argument(None, help="Question or research request (use '-' or omit to read from stdin)"),
    model: str = typer.Option("openai/o3", "--model", "-m", help="Model to use (format: provider/model)"),
    format_output: str = typer.Option("human", "--format", "-f", help="Output format: human or json"),
    base_url: Optional[str] = typer.Option(None, "--base-url", help="Custom API base URL (overrides OPENAI_BASE_URL env var)"),
):
    """Ask an AI assistant with web search tools (one-shot Q&A). For interactive conversations, use 'searxng chat'."""
    import litellm
    import os
    import sys
    
    # Handle stdin input for prompt
    if prompt is None or prompt == "-":
        if sys.stdin.isatty():
            # Interactive mode - show prompt on stderr so it doesn't interfere with piping
            stderr_console = Console(file=sys.stderr, force_terminal=True)
            stderr_console.print("[yellow]Reading prompt from stdin (press Ctrl+D when done):[/yellow]")
        
        try:
            # Read all available input from stdin
            prompt_lines = []
            for line in sys.stdin:
                prompt_lines.append(line.rstrip('\n\r'))
            prompt = '\n'.join(prompt_lines).strip()
        except KeyboardInterrupt:
            stderr_console = Console(file=sys.stderr, force_terminal=True)
            stderr_console.print("\n[red]Cancelled.[/red]")
            raise typer.Exit(1)
        
        if not prompt:
            stderr_console = Console(file=sys.stderr, force_terminal=True)
            stderr_console.print("[red]Error: No prompt provided.[/red]")
            raise typer.Exit(1)
    
    # Check for API keys
    api_keys = {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY"), 
        "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY"),
        "OPENROUTER_API_KEY": os.getenv("OPENROUTER_API_KEY"),
    }
    
    # Check if we have at least one API key
    if not any(api_keys.values()):
        stderr_console = Console(file=sys.stderr, force_terminal=True)
        stderr_console.print("[red]Error: No API keys found. Please set one of:[/red]")
        stderr_console.print("  - OPENAI_API_KEY")
        stderr_console.print("  - ANTHROPIC_API_KEY") 
        stderr_console.print("  - GOOGLE_API_KEY")
        stderr_console.print("  - OPENROUTER_API_KEY")
        raise typer.Exit(1)
    
    async def run_chat():
        # Log model info to stderr
        stderr_console = Console(file=sys.stderr, force_terminal=True)
        stderr_console.print(f"[dim]Using model: [blue]{model}[/blue][/dim]")
        
        # Use the shared ask function
        result = await ask_ai_async(prompt=prompt, model=model, base_url=base_url)
        
        if format_output.lower() == "json":
            # JSON output goes to stdout for piping
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            if result["success"]:
                # Just the response content goes to stdout for piping
                print(result['response'])
            else:
                # Errors go to stderr
                stderr_console.print(f"[red]Error: {result['error']}[/red]")
                raise typer.Exit(1)
    
    # Run the async chat function
    asyncio.run(run_chat())


async def ask_ai_conversational_async(
    messages: List[Dict[str, str]],
    model: str = "openai/o3",
    base_url: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Core async function for conversational AI chat with web search tools.
    Takes a conversation history as input and returns the response with updated history.
    """
    import litellm
    import os
    
    # Check for API keys
    api_keys = {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY"), 
        "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY"),
        "OPENROUTER_API_KEY": os.getenv("OPENROUTER_API_KEY"),
    }
    
    # Check if we have at least one API key
    if not any(api_keys.values()):
        return {
            "success": False,
            "error": "No API keys found. Please set OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY, or OPENROUTER_API_KEY",
            "model": model,
            "messages": messages
        }
    
    # Define tools for the LLM (same as ask function)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Search the web using SearXNG's search engines",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "category": {"type": "string", "description": "Search category", "default": "general"},
                        "max_results": {"type": "integer", "description": "Maximum results", "default": 10}
                    },
                    "required": ["query"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "multi_web_search",
                "description": "Search the web with multiple queries in parallel",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "queries": {"type": "array", "items": {"type": "string"}, "description": "Array of search queries"},
                        "category": {"type": "string", "description": "Search category", "default": "general"},
                        "max_results": {"type": "integer", "description": "Maximum results per query", "default": 10}
                    },
                    "required": ["queries"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "fetch_url",
                "description": "Fetch and extract content from a single URL",
                "parameters": {
                    "type": "object",
                    "properties": {"url": {"type": "string", "description": "URL to fetch"}},
                    "required": ["url"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "fetch_urls",
                "description": "Fetch and extract content from multiple URLs in parallel",
                "parameters": {
                    "type": "object",
                    "properties": {"urls": {"type": "array", "items": {"type": "string"}, "description": "URLs to fetch"}},
                    "required": ["urls"]
                }
            }
        }
    ]
    
    try:
        # Add system message if not present
        if not messages or messages[0].get("role") != "system":
            system_content = BASE_AI_SYSTEM_PROMPT + """\n\n**CHAT MODE - CONVERSATIONAL RESEARCH:**
- Engage naturally in conversation while maintaining research capabilities
- Reference previous messages and build on earlier searches when relevant
- Balance thoroughness with conversational flow - be comprehensive but not overwhelming
- Use context from conversation history to make more targeted searches
- Provide concise but informative responses that invite further questions"""
            
            system_message = {
                "role": "system", 
                "content": system_content
            }
            messages = [system_message] + messages
        
        # Prepare completion arguments
        completion_args = {
            "model": model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto"
        }
        
        # Add base_url if provided (overrides environment variable)
        if base_url:
            completion_args["base_url"] = base_url
        
        # Make initial request to the LLM
        response = litellm.completion(**completion_args)
        
        # Handle tool calls iteratively
        while response.choices[0].message.tool_calls:
            # Add the assistant's message with tool calls
            messages.append(response.choices[0].message.model_dump())
            
            # Execute each tool call
            for tool_call in response.choices[0].message.tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                
                # Log tool usage to stderr for shell piping friendliness
                stderr_console = Console(file=sys.stderr, force_terminal=True)
                if function_name == "web_search":
                    stderr_console.print(f"🔍 [cyan]Searching:[/cyan] {function_args.get('query', 'N/A')}")
                elif function_name == "multi_web_search":
                    queries = function_args.get('queries', [])
                    stderr_console.print(f"🔍 [cyan]Multi-search:[/cyan] {', '.join(queries[:3])}{'...' if len(queries) > 3 else ''}")
                elif function_name == "fetch_url":
                    stderr_console.print(f"📄 [blue]Fetching:[/blue] {function_args.get('url', 'N/A')}")
                elif function_name == "fetch_urls":
                    urls = function_args.get('urls', [])
                    stderr_console.print(f"📄 [blue]Fetching {len(urls)} URLs:[/blue] {urls[0] if urls else 'N/A'}{'...' if len(urls) > 1 else ''}")
                else:
                    stderr_console.print(f"🔧 [magenta]Tool:[/magenta] {function_name}")
                
                # Execute the tool using our existing handler
                tool_result = await handle_tool_call(function_name, function_args)
                
                # Add tool result to messages
                messages.append({
                    "role": "tool",
                    "content": tool_result,
                    "tool_call_id": tool_call.id
                })
            
            # Update completion args with new messages
            completion_args["messages"] = messages
            
            # Get next response from LLM
            response = litellm.completion(**completion_args)
        
        # Add the final assistant response to messages
        final_response = response.choices[0].message.content
        messages.append({"role": "assistant", "content": final_response})
        
        # Return the response and updated conversation history
        return {
            "success": True,
            "model": model,
            "response": final_response,
            "messages": messages
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Error calling {model}: {str(e)}",
            "model": model,
            "messages": messages
        }


@app.command()
def chat(
    initial_message: Optional[str] = typer.Argument(None, help="Initial message to send (use '-' to read from stdin)"),
    model: str = typer.Option("openai/o3", "--model", "-m", help="Model to use (format: provider/model)"),
    base_url: Optional[str] = typer.Option(None, "--base-url", help="Custom API base URL (overrides OPENAI_BASE_URL env var)"),
):
    """Start an interactive chat session with AI assistant that has web search capabilities."""
    import litellm
    import os
    import sys
    
    # Check for API keys
    api_keys = {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY"), 
        "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY"),
        "OPENROUTER_API_KEY": os.getenv("OPENROUTER_API_KEY"),
    }
    
    # Check if we have at least one API key
    if not any(api_keys.values()):
        stderr_console = Console(file=sys.stderr, force_terminal=True)
        stderr_console.print("[red]Error: No API keys found. Please set one of:[/red]")
        stderr_console.print("  - OPENAI_API_KEY")
        stderr_console.print("  - ANTHROPIC_API_KEY") 
        stderr_console.print("  - GOOGLE_API_KEY")
        stderr_console.print("  - OPENROUTER_API_KEY")
        raise typer.Exit(1)
    
    async def run_interactive_chat():
        import signal
        from datetime import datetime
        from pathlib import Path
        
        # Initialize conversation history
        messages = []
        
        # Handle stdin input if requested
        first_message = None
        if initial_message == "-":
            import sys
            if sys.stdin.isatty():
                stderr_console = Console(file=sys.stderr, force_terminal=True)
                stderr_console.print("[red]Error: stdin is not available (no pipe detected)[/red]")
                raise typer.Exit(1)
            first_message = sys.stdin.read().strip()
        elif initial_message:
            first_message = initial_message
        
        # Setup chat history directory (XDG Base Directory spec)
        data_home = os.getenv("XDG_DATA_HOME", os.path.expanduser("~/.local/share"))
        chat_dir = Path(data_home) / "searxng-ai-kit" / "chats"
        chat_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate chat session filename with timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        model_name = model.split('/')[-1] if '/' in model else model
        chat_file = chat_dir / f"chat-{timestamp}-{model_name}.md"
        
        # Initialize markdown file
        with open(chat_file, 'w', encoding='utf-8') as f:
            f.write(f"# SearXNG AI Kit Chat Session\n\n")
            f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n")
            f.write(f"**Model:** {model}  \n")
            f.write(f"**Session:** {timestamp}  \n\n")
            f.write("---\n\n")
        
        # Setup console for input/output
        stderr_console = Console(file=sys.stderr, force_terminal=True)
        stderr_console.print(f"[dim]SearXNG AI Kit - Interactive Chat[/dim]")
        stderr_console.print(f"[dim]Using model: [blue]{model}[/blue][/dim]")
        stderr_console.print(f"[dim]Chat history: {chat_file}[/dim]")
        stderr_console.print(f"[dim]Type 'exit', 'quit', or press Ctrl+C to end the conversation[/dim]")
        stderr_console.print()
        
        # Setup signal handling for graceful shutdown
        shutdown_requested = False
        def signal_handler(signum, frame):
            nonlocal shutdown_requested
            shutdown_requested = True
            
        # Install signal handlers
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Enable bracketed paste mode for proper multi-line input handling
        import sys
        import termios
        import tty
        
        def enable_bracketed_paste():
            if sys.stdin.isatty():
                sys.stdout.write("\x1b[?2004h")  # Enable bracketed paste
                sys.stdout.flush()
        
        def disable_bracketed_paste():
            if sys.stdin.isatty():
                sys.stdout.write("\x1b[?2004l")  # Disable bracketed paste
                sys.stdout.flush()
        
        def get_multiline_input(prompt=""):
            """Get input that properly handles bracketed paste and multi-line content."""
            stderr_console.print(prompt, end="")
            
            lines = []
            in_paste = False
            
            try:
                while True:
                    line = input()
                    
                    # Check for bracketed paste start/end sequences
                    if line.startswith("\x1b[200~"):
                        in_paste = True
                        line = line[6:]  # Remove paste start sequence
                    
                    if line.endswith("\x1b[201~"):
                        in_paste = False
                        line = line[:-6]  # Remove paste end sequence
                        lines.append(line)
                        break
                    
                    lines.append(line)
                    
                    # If not in paste mode and we have content, break
                    if not in_paste and lines and lines[-1].strip():
                        break
                    
                    # Show continuation prompt for multi-line input
                    if in_paste:
                        continue
                    else:
                        stderr_console.print("... ", end="")
                        
            except (EOFError, KeyboardInterrupt):
                return None
            
            return "\n".join(lines).strip()
        
        enable_bracketed_paste()
        
        # Process initial message if provided
        if first_message:
            messages.append({"role": "user", "content": first_message})
            
            # Save initial message to markdown file
            with open(chat_file, 'a', encoding='utf-8') as f:
                f.write(f"## You\n\n{first_message}\n\n")
            
            # Display the initial message
            stderr_console.print(f"[bold green]You:[/bold green] {first_message}")
            stderr_console.print()
            
            # Show spinner while getting AI response
            from rich.spinner import Spinner
            from rich.live import Live
            
            spinner = Spinner("dots", text="[dim]Thinking... [/dim]")
            
            try:
                with Live(spinner, console=stderr_console, refresh_per_second=10):
                    # Check for shutdown request during AI processing
                    if shutdown_requested:
                        stderr_console.print("\n[yellow]Interrupted. Goodbye![/yellow]")
                        return
                    
                    # Get AI response
                    result = await ask_ai_conversational_async(
                        messages=messages,
                        model=model,
                        base_url=base_url
                    )
            except KeyboardInterrupt:
                stderr_console.print("\n[yellow]Interrupted. Goodbye![/yellow]")
                return
            
            if result["success"]:
                # Update conversation history with the response
                messages = result["messages"]
                
                # Display the response with model name instead of "Assistant"
                stderr_console.print(f"[bold blue]{model_name}:[/bold blue] ", end="")
                print(result['response'])
                stderr_console.print()
                
                # Save assistant response to markdown file
                with open(chat_file, 'a', encoding='utf-8') as f:
                    f.write(f"## {model_name}\n\n{result['response']}\n\n")
            else:
                # Display error
                stderr_console.print(f"[red]Error: {result['error']}[/red]")
                stderr_console.print()
                
                # Save error to markdown file
                with open(chat_file, 'a', encoding='utf-8') as f:
                    f.write(f"## Error\n\n{result['error']}\n\n")
        
        try:
            while True:
                # Check for shutdown request
                if shutdown_requested:
                    stderr_console.print("\n[yellow]Interrupted. Goodbye![/yellow]")
                    break
                
                # Get user input with multi-line support
                try:
                    user_input = get_multiline_input("[bold green]You:[/bold green] ")
                    if user_input is None:  # Handle Ctrl-C/Ctrl-D
                        stderr_console.print("\n[yellow]Goodbye![/yellow]")
                        break
                    user_input = user_input.strip()
                except (EOFError, KeyboardInterrupt):
                    stderr_console.print("\n[yellow]Goodbye![/yellow]")
                    break
                
                # Handle exit commands
                if user_input.lower() in ['exit', 'quit', 'bye']:
                    stderr_console.print("[yellow]Goodbye![/yellow]")
                    break
                
                # Skip empty input
                if not user_input:
                    continue
                
                # Add user message to history
                messages.append({"role": "user", "content": user_input})
                
                # Save user message to markdown file
                with open(chat_file, 'a', encoding='utf-8') as f:
                    f.write(f"## You\n\n{user_input}\n\n")
                
                # Show spinner while getting AI response
                from rich.spinner import Spinner
                from rich.live import Live
                
                spinner = Spinner("dots", text="[dim]Thinking... [/dim]")
                
                try:
                    with Live(spinner, console=stderr_console, refresh_per_second=10):
                        # Check for shutdown request during AI processing
                        if shutdown_requested:
                            stderr_console.print("\n[yellow]Interrupted. Goodbye![/yellow]")
                            break
                        
                        # Get AI response
                        result = await ask_ai_conversational_async(
                            messages=messages,
                            model=model,
                            base_url=base_url
                        )
                except KeyboardInterrupt:
                    stderr_console.print("\n[yellow]Interrupted. Goodbye![/yellow]")
                    break
                
                if result["success"]:
                    # Update conversation history with the response
                    messages = result["messages"]
                    
                    # Display the response with model name instead of "Assistant"
                    stderr_console.print(f"[bold blue]{model_name}:[/bold blue] ", end="")
                    print(result['response'])
                    stderr_console.print()
                    
                    # Save assistant response to markdown file
                    with open(chat_file, 'a', encoding='utf-8') as f:
                        f.write(f"## {model_name}\n\n{result['response']}\n\n")
                else:
                    # Display error
                    stderr_console.print(f"[red]Error: {result['error']}[/red]")
                    stderr_console.print()
                    
                    # Save error to markdown file
                    with open(chat_file, 'a', encoding='utf-8') as f:
                        f.write(f"## Error\n\n{result['error']}\n\n")
        
        except Exception as e:
            stderr_console.print(f"\n[red]Unexpected error: {str(e)}[/red]")
        finally:
            # Clean up bracketed paste mode
            disable_bracketed_paste()
    
    # Run the interactive chat
    asyncio.run(run_interactive_chat())


def main():
    """Main entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()