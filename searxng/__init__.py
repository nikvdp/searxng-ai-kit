"""
SearXNG CLI Library

A Python library for privacy-respecting web search and URL content extraction.
Provides programmatic access to SearXNG's 180+ search engines and Jina.ai's
content extraction service.

Basic Usage:
    >>> import searxng
    >>> results = searxng.search("python tutorial")
    >>> content = searxng.fetch_url("https://example.com")
    >>> contents = searxng.fetch_urls(["https://site1.com", "https://site2.com"])
    >>> response = searxng.ask("What are the latest developments in AI?")

Advanced Usage:
    >>> # Customize search engines and parameters
    >>> results = searxng.search(
    ...     query="machine learning",
    ...     engines=["duckduckgo", "startpage"],
    ...     category="science",
    ...     max_results=20,
    ...     language="en"
    ... )
    
    >>> # Parallel URL fetching with concurrency control
    >>> urls = ["https://site1.com", "https://site2.com", "https://site3.com"]
    >>> contents = searxng.fetch_urls(urls, max_concurrent=2)
    
    >>> # AI assistant with custom model and endpoint
    >>> response = searxng.ask(
    ...     "Research renewable energy trends",
    ...     model="openrouter/openai/gpt-4o-mini",
    ...     base_url="https://openrouter.ai/api/v1"
    ... )
"""

from .client import (
    search,
    search_async,
    fetch_url,
    fetch_url_async,
    fetch_urls,
    fetch_urls_async,
    ask,
    ask_async,
    get_available_engines,
    get_categories,
    SearXNGClient,
)

__version__ = "0.1.0"
__author__ = "SearXNG CLI Contributors"
__description__ = "Privacy-respecting web search and URL content extraction library"

# Main public API
__all__ = [
    "search",
    "search_async", 
    "fetch_url",
    "fetch_url_async",
    "fetch_urls", 
    "fetch_urls_async",
    "ask",
    "ask_async",
    "get_available_engines",
    "get_categories",
    "SearXNGClient",
]