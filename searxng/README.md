# SearXNG Python Library

A Python library providing programmatic access to SearXNG's privacy-respecting search capabilities and URL content extraction.

## Installation

```bash
pip install searxng-cli
```

## Quick Start

```python
import searxng

# Basic web search
results = searxng.search("python tutorial")
print(f"Found {results['total_results']} results")

# Fetch content from a URL
content = searxng.fetch_url("https://example.com")
if content["success"]:
    print(f"Title: {content['title']}")

# Fetch multiple URLs in parallel
urls = ["https://site1.com", "https://site2.com", "https://site3.com"]
contents = searxng.fetch_urls(urls)
```

## Features

- **Privacy-focused search**: Access 180+ search engines through SearXNG
- **URL content extraction**: Clean content extraction using Jina.ai
- **Parallel processing**: Fetch multiple URLs concurrently
- **Engine customization**: Choose specific search engines
- **Async support**: Full async/await API
- **No tracking**: Privacy-respecting search without tracking

## API Reference

### Search Functions

#### `search(query, **kwargs)`
Perform a web search with customizable parameters.

```python
results = searxng.search(
    query="machine learning",
    engines=["duckduckgo", "startpage"],
    category="science",
    max_results=20,
    language="en"
)
```

**Parameters:**
- `query` (str): Search query
- `category` (str): Search category (general, images, videos, news, science, it)
- `engines` (List[str]): Specific engines to use
- `max_results` (int): Maximum results to return (default: 10)
- `language` (str): Language code (default: "all")
- `safe_search` (int): Safe search level (0=off, 1=moderate, 2=strict)

**Returns:** Dictionary with search results, engines used, and metadata.

#### `search_async(query, **kwargs)`
Async version of `search()`.

```python
import asyncio

async def main():
    results = await searxng.search_async("python tutorial")
    print(f"Found {results['total_results']} results")

asyncio.run(main())
```

### URL Content Functions

#### `fetch_url(url)`
Extract content from a single URL.

```python
content = searxng.fetch_url("https://docs.python.org")
if content["success"]:
    print(f"Title: {content['title']}")
    print(f"Content: {content['content'][:200]}...")
```

**Returns:** Dictionary with extracted content, title, and success status.

#### `fetch_urls(urls, max_concurrent=5)`
Extract content from multiple URLs in parallel.

```python
urls = ["https://site1.com", "https://site2.com", "https://site3.com"]
contents = searxng.fetch_urls(urls, max_concurrent=2)

for i, content in enumerate(contents):
    status = "✓" if content["success"] else "✗"
    print(f"{status} URL {i+1}: {urls[i]}")
```

**Parameters:**
- `urls` (List[str]): List of URLs to fetch
- `max_concurrent` (int): Maximum concurrent requests (default: 5, max: 20)

**Returns:** List of content dictionaries in the same order as input URLs.

#### `fetch_url_async(url)` and `fetch_urls_async(urls, max_concurrent=5)`
Async versions of the URL fetching functions.

### Information Functions

#### `get_available_engines()`
Get information about available search engines.

```python
engines_info = searxng.get_available_engines()
print(f"Total engines: {engines_info['total_engines']}")
print(f"Categories: {list(engines_info['categories'].keys())}")
```

#### `get_categories()`
Get search categories and their associated engines.

```python
categories = searxng.get_categories()
print("Science engines:", categories["science"][:5])
```

### Client Class

#### `SearXNGClient`
A client class for reusable configurations.

```python
client = searxng.SearXNGClient(
    default_engines=["duckduckgo", "startpage"],
    default_language="en",
    default_max_results=15,
    max_concurrent_urls=3
)

# Use defaults
results = client.search("python tutorial")

# Override defaults
science_results = client.search(
    "quantum computing",
    category="science",
    max_results=25
)
```

## Environment Variables

- `JINA_API_KEY`: Optional API key for enhanced Jina.ai URL content extraction

## Error Handling

All functions return structured results with success indicators:

```python
# Search results
results = searxng.search("query")
if results.get("success", True):  # Search success is implicit
    print(f"Found {results['total_results']} results")
else:
    print(f"Search failed: {results.get('error', 'Unknown error')}")

# URL fetching results
content = searxng.fetch_url("https://example.com")
if content["success"]:
    print(f"Content: {content['content']}")
else:
    print(f"Failed: {content['error']}")
```

## Examples

### Custom Search Engine Selection

```python
# Search only privacy-focused engines
privacy_engines = ["duckduckgo", "startpage", "brave"]
results = searxng.search(
    "sensitive topic",
    engines=privacy_engines,
    safe_search=1
)
```

### Parallel Content Extraction

```python
# Research workflow: search then extract content
search_results = searxng.search("AI research papers", max_results=10)
urls = [result["url"] for result in search_results["results"]]

# Extract content from all found URLs
contents = searxng.fetch_urls(urls, max_concurrent=3)
for url, content in zip(urls, contents):
    if content["success"]:
        print(f"📄 {content['title']}")
        print(f"   {content['content'][:100]}...")
```

### Async Batch Processing

```python
import asyncio

async def research_topic(topic):
    # Parallel search and URL fetching
    search_task = searxng.search_async(f"{topic} tutorial")
    info_task = searxng.fetch_url_async(f"https://en.wikipedia.org/wiki/{topic}")
    
    search_results, wiki_content = await asyncio.gather(search_task, info_task)
    
    return {
        "search": search_results,
        "wiki": wiki_content
    }

# Research multiple topics
topics = ["Python", "JavaScript", "Rust"]
results = asyncio.run(asyncio.gather(*[research_topic(topic) for topic in topics]))
```

## Integration Examples

### Data Pipeline

```python
def create_knowledge_base(queries):
    """Create a knowledge base from search queries."""
    knowledge_base = []
    
    for query in queries:
        # Search for relevant content
        results = searxng.search(query, max_results=5)
        
        # Extract content from top results
        urls = [r["url"] for r in results["results"]]
        contents = searxng.fetch_urls(urls)
        
        # Build knowledge entries
        for result, content in zip(results["results"], contents):
            if content["success"]:
                knowledge_base.append({
                    "query": query,
                    "title": result["title"],
                    "url": result["url"],
                    "content": content["content"],
                    "extracted_at": content.get("timestamp", "")
                })
    
    return knowledge_base
```

### Research Assistant

```python
class ResearchAssistant:
    def __init__(self):
        self.client = searxng.SearXNGClient(
            default_engines=["duckduckgo", "startpage"],
            default_max_results=10
        )
    
    def research_topic(self, topic, categories=None):
        """Research a topic across multiple categories."""
        results = {}
        categories = categories or ["general", "science", "news"]
        
        for category in categories:
            search_results = self.client.search(
                topic,
                category=category,
                max_results=5
            )
            results[category] = search_results
        
        return results
    
    def get_source_content(self, urls):
        """Extract content from source URLs."""
        return self.client.fetch_urls(urls)

# Usage
assistant = ResearchAssistant()
research = assistant.research_topic("climate change")
```

## License

This library is part of the searxng-cli project and is licensed under AGPL-3.0.