#!/usr/bin/env python3
"""
SearXNG Kit Library Usage Examples

This script demonstrates how to use the SearXNG Kit library for programmatic
web search and URL content extraction.
"""

import asyncio
import json
from typing import List, Dict, Any

# Import the SearXNG library
import searxng


def basic_search_example():
    """Basic web search example."""
    print("=== Basic Search Example ===")
    
    # Simple search
    results = searxng.search("python tutorial")
    print(f"Found {results['total_results']} results for 'python tutorial'")
    
    # Display first 3 results
    for i, result in enumerate(results['results'][:3]):
        print(f"{i+1}. {result['title']}")
        print(f"   URL: {result['url']}")
        print(f"   Snippet: {result.get('content', 'No snippet')[:100]}...")
        print()


def advanced_search_example():
    """Advanced search with engine and parameter customization."""
    print("=== Advanced Search Example ===")
    
    # Search with specific engines and parameters
    results = searxng.search(
        query="machine learning research",
        engines=["duckduckgo", "startpage", "brave"],
        category="science",
        max_results=15,
        language="en"
    )
    
    print(f"Search query: {results['query']}")
    print(f"Category: {results['category']}")
    print(f"Engines used: {', '.join(results['engines_used'])}")
    print(f"Results found: {results['total_results']}")
    print(f"Search time: {results['search_time']:.2f}s")
    print()


def url_fetching_example():
    """URL content fetching examples."""
    print("=== URL Content Fetching Example ===")
    
    # Single URL fetch
    url = "https://httpbin.org/json"
    content = searxng.fetch_url(url)
    
    if content["success"]:
        print(f"✓ Successfully fetched: {url}")
        print(f"  Title: {content.get('title', 'No title')}")
        print(f"  Content length: {len(content['content'])} characters")
    else:
        print(f"✗ Failed to fetch: {url}")
        print(f"  Error: {content.get('error', 'Unknown error')}")
    print()


def parallel_url_fetching_example():
    """Parallel URL fetching example."""
    print("=== Parallel URL Fetching Example ===")
    
    # Multiple URLs to fetch in parallel
    urls = [
        "https://httpbin.org/json",
        "https://httpbin.org/uuid",
        "https://httpbin.org/ip",
        "https://httpbin.org/user-agent"
    ]
    
    print(f"Fetching {len(urls)} URLs in parallel...")
    contents = searxng.fetch_urls(urls, max_concurrent=2)
    
    for i, (url, content) in enumerate(zip(urls, contents)):
        status = "✓" if content["success"] else "✗"
        print(f"{status} URL {i+1}: {url}")
        if content["success"]:
            print(f"    Content: {len(content['content'])} characters")
        else:
            print(f"    Error: {content.get('error', 'Unknown error')}")
    print()


def client_class_example():
    """SearXNGClient class usage example."""
    print("=== Client Class Example ===")
    
    # Create a client with default settings
    client = searxng.SearXNGClient(
        default_engines=["duckduckgo", "startpage"],
        default_language="en",
        default_max_results=5,
        max_concurrent_urls=2
    )
    
    # Use client for searches (uses defaults)
    results = client.search("python web scraping")
    print(f"Client search found {results['total_results']} results")
    print(f"Used engines: {', '.join(results['engines_used'])}")
    
    # Override defaults for specific search
    science_results = client.search(
        "quantum computing",
        category="science",
        max_results=10
    )
    print(f"Science search found {science_results['total_results']} results")
    print()


def engines_and_categories_example():
    """Available engines and categories example."""
    print("=== Available Engines and Categories ===")
    
    # Get available engines
    engines_info = searxng.get_available_engines()
    print(f"Total engines available: {engines_info['total_engines']}")
    
    # Get categories
    categories = searxng.get_categories()
    print(f"Available categories: {list(categories.keys())}")
    
    # Show engines for specific categories
    for category in ["general", "science", "news"][:3]:
        engines = categories.get(category, [])
        print(f"  {category}: {len(engines)} engines ({', '.join(engines[:3])}...)")
    print()


async def async_example():
    """Async API usage example."""
    print("=== Async API Example ===")
    
    # Async search
    search_task = searxng.search_async("async python programming")
    
    # Async URL fetching
    url_task = searxng.fetch_url_async("https://httpbin.org/json")
    
    # Wait for both operations
    search_results, url_content = await asyncio.gather(search_task, url_task)
    
    print(f"Async search found {search_results['total_results']} results")
    print(f"Async URL fetch: {'✓' if url_content['success'] else '✗'}")
    print()


def json_output_example():
    """Example of working with JSON output for integration."""
    print("=== JSON Output Example ===")
    
    # Search and get results as structured data
    results = searxng.search(
        "FastAPI tutorial",
        engines=["duckduckgo"],
        max_results=3
    )
    
    # Convert to JSON for storage/transmission
    json_output = json.dumps(results, indent=2, default=str)
    print("Search results as JSON:")
    print(json_output[:500] + "..." if len(json_output) > 500 else json_output)
    print()


def main():
    """Run all examples."""
    print("SearXNG Kit Library Usage Examples")
    print("=" * 50)
    print()
    
    try:
        # Synchronous examples
        basic_search_example()
        advanced_search_example()
        url_fetching_example()
        parallel_url_fetching_example()
        client_class_example()
        engines_and_categories_example()
        json_output_example()
        
        # Async example
        print("Running async example...")
        asyncio.run(async_example())
        
        print("All examples completed successfully!")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        print("\nNote: Make sure SearXNG Kit engines are properly initialized.")
        print("Some examples may fail if search engines are not available.")


if __name__ == "__main__":
    main()