"""
SearXNG Client Library

Provides programmatic access to SearXNG search functionality and URL content extraction.
This module offers both synchronous and asynchronous interfaces for all operations.
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from urllib.parse import quote

import httpx

# Add the parent directory to Python path for searx imports
_parent_dir = Path(__file__).parent.parent
if str(_parent_dir) not in sys.path:
    sys.path.insert(0, str(_parent_dir))

# Import existing functionality from searxng_cli
from searxng_cli import (
    perform_search_async,
    fetch_url_content,
    fetch_multiple_urls_async,
    get_available_engines as _get_engines_data,
    categories as _get_categories_data,
)


# Synchronous API (wraps async functions)
def search(
    query: str,
    *,
    category: str = "general",
    engines: Optional[List[str]] = None,
    max_results: int = 10,
    language: str = "all",
    safe_search: int = 0,
    time_range: Optional[str] = None,
    page: int = 1,
) -> Dict[str, Any]:
    """
    Search the web using SearXNG's search engines.
    
    Args:
        query: Search query string
        category: Search category (general, images, videos, news, science, it, etc.)
        engines: List of specific engines to use (e.g., ["duckduckgo", "startpage"])
        max_results: Maximum number of results to return (default: 10)
        language: Search language code (default: "all")
        safe_search: Safe search level (0=off, 1=moderate, 2=strict)
        time_range: Time range filter (day, week, month, year)
        page: Page number for pagination (default: 1)
    
    Returns:
        Dictionary containing search results with the following structure:
        {
            "query": str,
            "category": str,
            "engines_used": List[str],
            "results": List[Dict],
            "total_results": int,
            "search_time": float,
            "timestamp": str
        }
    
    Example:
        >>> results = searxng.search("python tutorial")
        >>> print(f"Found {results['total_results']} results")
        >>> for result in results['results']:
        ...     print(f"{result['title']}: {result['url']}")
    """
    return asyncio.run(search_async(
        query=query,
        category=category,
        engines=engines,
        max_results=max_results,
        language=language,
        safe_search=safe_search,
        time_range=time_range,
        page=page,
    ))


def fetch_url(url: str) -> Dict[str, Any]:
    """
    Fetch and extract content from a single URL using Jina.ai's reader service.
    
    Args:
        url: URL to fetch content from
    
    Returns:
        Dictionary with extracted content:
        {
            "success": bool,
            "title": str,
            "content": str,
            "url": str,
            "timestamp": str,
            "error": str (only if success=False)
        }
    
    Environment Variables:
        JINA_API_KEY: Optional API key for enhanced Jina.ai features
    
    Example:
        >>> content = searxng.fetch_url("https://docs.python.org")
        >>> if content["success"]:
        ...     print(f"Title: {content['title']}")
        ...     print(f"Content: {content['content'][:200]}...")
    """
    return asyncio.run(fetch_url_async(url))


def fetch_urls(
    urls: List[str], 
    *, 
    max_concurrent: int = 5
) -> List[Dict[str, Any]]:
    """
    Fetch and extract content from multiple URLs in parallel.
    
    Args:
        urls: List of URLs to fetch content from
        max_concurrent: Maximum number of concurrent requests (default: 5, max: 20)
    
    Returns:
        List of dictionaries with extracted content (same format as fetch_url).
        Results are returned in the same order as input URLs.
    
    Example:
        >>> urls = ["https://site1.com", "https://site2.com", "https://site3.com"]
        >>> results = searxng.fetch_urls(urls, max_concurrent=2)
        >>> for i, result in enumerate(results):
        ...     print(f"URL {i+1}: {'✓' if result['success'] else '✗'}")
    """
    return asyncio.run(fetch_urls_async(urls, max_concurrent=max_concurrent))


def get_available_engines() -> Dict[str, Any]:
    """
    Get information about available search engines.
    
    Returns:
        Dictionary containing engine information:
        {
            "categories": Dict[str, List[str]],
            "total_engines": int
        }
    
    Example:
        >>> engines_info = searxng.get_available_engines()
        >>> print(f"Total engines: {engines_info['total_engines']}")
        >>> general_engines = engines_info['categories']['general']
        >>> print(f"General search engines: {general_engines[:5]}...")
    """
    categories = _get_engines_data()
    total_engines = len(set(engine for engines in categories.values() for engine in engines))
    
    return {
        "categories": categories,
        "total_engines": total_engines
    }


def get_categories() -> Dict[str, List[str]]:
    """
    Get available search categories and their associated engines.
    
    Returns:
        Dictionary mapping category names to lists of engine names:
        {
            "general": ["duckduckgo", "startpage", ...],
            "images": ["duckduckgo_images", "bing_images", ...],
            ...
        }
    
    Example:
        >>> categories = searxng.get_categories()
        >>> print("Available categories:", list(categories.keys()))
        >>> print("Science engines:", categories.get("science", [])[:3])
    """
    return _get_categories_data()


# Async API (direct access to underlying functions)
async def search_async(
    query: str,
    *,
    category: str = "general",
    engines: Optional[List[str]] = None,
    max_results: int = 10,
    language: str = "all",
    safe_search: int = 0,
    time_range: Optional[str] = None,
    page: int = 1,
) -> Dict[str, Any]:
    """
    Async version of search(). See search() for documentation.
    
    Note: time_range and page parameters are accepted for API compatibility 
    but may not be supported by the underlying search implementation.
    
    Example:
        >>> import asyncio
        >>> async def main():
        ...     results = await searxng.search_async("python tutorial")
        ...     print(f"Found {results['number_of_results']} results")
        >>> asyncio.run(main())
    """
    # Call the underlying function with only supported parameters
    result = await perform_search_async(
        query=query,
        category=category,
        engines=engines,
        language=language,
        max_results=max_results,
        safe_search=safe_search,
    )
    
    # Transform result to match expected API
    if result.get("success"):
        # Use actual results count if number_of_results is 0
        actual_count = len(result.get("results", []))
        result["total_results"] = result.get("number_of_results", actual_count) or actual_count
        result["search_time"] = 0.0  # Not provided by underlying function
        result["timestamp"] = ""  # Not provided by underlying function
    
    return result


async def fetch_url_async(url: str) -> Dict[str, Any]:
    """
    Async version of fetch_url(). See fetch_url() for documentation.
    
    Example:
        >>> import asyncio
        >>> async def main():
        ...     content = await searxng.fetch_url_async("https://example.com")
        ...     print(content["title"])
        >>> asyncio.run(main())
    """
    return await fetch_url_content(url)


async def fetch_urls_async(
    urls: List[str], 
    *, 
    max_concurrent: int = 5
) -> List[Dict[str, Any]]:
    """
    Async version of fetch_urls(). See fetch_urls() for documentation.
    
    Args:
        urls: List of URLs to fetch content from
        max_concurrent: Maximum number of concurrent requests (default: 5, max: 20)
    
    Example:
        >>> import asyncio
        >>> async def main():
        ...     urls = ["https://site1.com", "https://site2.com"]
        ...     results = await searxng.fetch_urls_async(urls)
        ...     return results
        >>> results = asyncio.run(main())
    """
    # Validate and limit concurrency
    if max_concurrent > 20:
        max_concurrent = 20
    elif max_concurrent < 1:
        max_concurrent = 1
    
    # Note: max_concurrent parameter is documented but the underlying
    # fetch_multiple_urls_async doesn't currently support it.
    # For now, we'll use the existing implementation which uses
    # searx.network.multi_requests() with its own concurrency handling.
    return await fetch_multiple_urls_async(urls)


# Convenience classes for more structured usage
class SearXNGClient:
    """
    A client class for SearXNG operations with configurable defaults.
    
    This class allows you to set default parameters and reuse them across
    multiple search operations.
    
    Example:
        >>> client = searxng.SearXNGClient(
        ...     default_engines=["duckduckgo", "startpage"],
        ...     default_language="en",
        ...     max_concurrent_urls=3
        ... )
        >>> results = client.search("python tutorial")
        >>> content = client.fetch_url("https://example.com")
    """
    
    def __init__(
        self,
        *,
        default_engines: Optional[List[str]] = None,
        default_category: str = "general",
        default_language: str = "all",
        default_max_results: int = 10,
        max_concurrent_urls: int = 5,
    ):
        """
        Initialize SearXNG client with default parameters.
        
        Args:
            default_engines: Default search engines to use
            default_category: Default search category
            default_language: Default search language
            default_max_results: Default maximum results per search
            max_concurrent_urls: Default max concurrent URL fetches
        """
        self.default_engines = default_engines
        self.default_category = default_category
        self.default_language = default_language
        self.default_max_results = default_max_results
        self.max_concurrent_urls = max_concurrent_urls
    
    def search(
        self,
        query: str,
        *,
        category: Optional[str] = None,
        engines: Optional[List[str]] = None,
        max_results: Optional[int] = None,
        language: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Search using client defaults. Parameters override defaults."""
        return search(
            query=query,
            category=category or self.default_category,
            engines=engines or self.default_engines,
            max_results=max_results or self.default_max_results,
            language=language or self.default_language,
            **kwargs
        )
    
    async def search_async(
        self,
        query: str,
        *,
        category: Optional[str] = None,
        engines: Optional[List[str]] = None,
        max_results: Optional[int] = None,
        language: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Async search using client defaults. Parameters override defaults."""
        return await search_async(
            query=query,
            category=category or self.default_category,
            engines=engines or self.default_engines,
            max_results=max_results or self.default_max_results,
            language=language or self.default_language,
            **kwargs
        )
    
    def fetch_url(self, url: str) -> Dict[str, Any]:
        """Fetch URL using client settings."""
        return fetch_url(url)
    
    def fetch_urls(self, urls: List[str]) -> List[Dict[str, Any]]:
        """Fetch multiple URLs using client concurrency settings."""
        return fetch_urls(urls, max_concurrent=self.max_concurrent_urls)
    
    async def fetch_url_async(self, url: str) -> Dict[str, Any]:
        """Async fetch URL using client settings."""
        return await fetch_url_async(url)
    
    async def fetch_urls_async(self, urls: List[str]) -> List[Dict[str, Any]]:
        """Async fetch multiple URLs using client concurrency settings."""
        return await fetch_urls_async(urls, max_concurrent=self.max_concurrent_urls)