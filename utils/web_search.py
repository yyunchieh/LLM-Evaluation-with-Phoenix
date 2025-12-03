# utils/web_search.py

import os
from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()

def get_tavily_client():
    """Get Tavily client instance"""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise ValueError("Missing TAVILY_API_KEY in .env file")
    return TavilyClient(api_key=api_key)

def web_search(query: str, max_results: int = 5) -> str:
    """
    Perform web search using Tavily API

    Args:
        query: Search query
        max_results: Maximum number of results to return

    Returns:
        Formatted search results as a string
    """
    try:
        client = get_tavily_client()

        # Perform search
        response = client.search(
            query=query,
            max_results=max_results,
            search_depth="advanced"  # Use advanced search for better quality
        )

        # Format results
        results = []
        for i, result in enumerate(response.get('results', []), 1):
            title = result.get('title', 'No title')
            url = result.get('url', '')
            content = result.get('content', 'No content')

            results.append(f"[{i}] {title}\nURL: {url}\n{content}\n")

        if not results:
            return "No search results found."

        return "\n".join(results)

    except Exception as e:
        return f"Error performing web search: {str(e)}"

def web_search_for_research(topic: str) -> str:
    """
    Perform web search specifically for research topics

    Args:
        topic: Research topic to search for

    Returns:
        Formatted search results
    """
    # Create research-focused query
    query = f"latest research papers and developments in {topic}"

    return web_search(query, max_results=5)
