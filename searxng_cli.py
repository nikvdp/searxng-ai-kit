#!/usr/bin/env python3
"""SearXNG CLI - Command line interface for SearXNG search engine."""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

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
    help="Privacy-respecting metasearch engine CLI",
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


if __name__ == "__main__":
    app()