# import requests
from scraper import scrape_website
from typing import List, Dict, Optional
from newsapi import NewsApiClient
from config import get_config

# Initialize the News API client
newsapi = NewsApiClient(api_key=get_config()["NEWS_API_KEY"])


def get_outlet_news(
    source_id: Optional[str], query: Optional[str] = None
) -> Optional[List[Dict[str, str]]]:
    """
    Fetch news articles from a specific outlet and scrape their content.

    Args:
        source_id (Optional[str]): The unique ID of the news source (e.g., 'bbc-news'). Required.
        query (Optional[str]): The search keyword for filtering articles. Defaults to None.

    Returns:
        Optional[List[Dict[str, str]]]: A list of dictionaries containing article titles and scraped content from url.
        Returns None if source_id is not provided or no articles are found.

    Raises:
        Exception: If an error occurs while scraping an article's content.
    """
    # Validate the input
    if not source_id:
        print("Error: Source ID must be provided.")
        return None

    try:
        # Fetch articles from the specified source
        everything = newsapi.get_everything(q=query, sources=source_id)

        # Validate API response
        if everything.get("status") != "ok":
            print(f"Error fetching news: {everything.get('message', 'Unknown error')}")
            return None

        # Extract articles
        all_articles = everything.get("articles", [])
        if not all_articles:
            print(f"No articles found for source: {source_id}")
            return None

        # Process articles and scrape content
        outlet_news_details = []
        for count, article in enumerate(all_articles):
            if count >= 5:  # Limit to the first 5 articles
                break

            title = article.get("title", "No Title")
            url = article.get("url")
            if not url:
                continue

            try:
                # Scrape website content
                content = scrape_website(url)
            except Exception as e:
                print(f"Error scraping content from {url}: {e}")
                continue

            # Append article details to the list
            outlet_news_details.append({"title": title, "content": content})

        return outlet_news_details if outlet_news_details else None

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None
