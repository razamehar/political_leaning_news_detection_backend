import requests
from bs4 import BeautifulSoup
from loguru import logger

# Scrape Web Content
def scrape_website(url):
    """
    Fetch and parse webpage content.
    """
    try:
        logger.info("Fetching webpage content from URL: {}", url)
        response = requests.get(url)
        if response.status_code != 200:
            logger.warning("Non-success status code received: {}", response.status_code)
            raise Exception(f"Failed to fetch webpage: {response.status_code}")

        logger.debug("Parsing HTML content.")
        soup = BeautifulSoup(response.content, "html.parser")
        paragraphs = soup.find_all("p")
        content = "\n".join([para.get_text() for para in paragraphs])

        logger.info("Successfully scraped content from URL: {}", url)
        return content
    except requests.RequestException as e:
        logger.exception("HTTP request failed: {}", e)
        raise
    except Exception as e:
        logger.exception("An error occurred while scraping: {}", e)
        raise

# Main Script
if __name__ == "__main__":
    # Example URLs to scrape
    url1 = "https://edition.cnn.com/2025/01/05/europe/ukraine-kursk-counteroffensive-russia-intl/index.html"
    url2 = "https://www.bbc.com/sport/football/articles/c4gj10jmmx"
    url = url1

    try:
        logger.info("Starting the web scraping script.")
        # Step 1: Scrape content
        logger.info("Scraping content from URL: {}", url)
        scraped_content = scrape_website(url)
        print(scraped_content)
        logger.info("Scraping completed successfully.")
    except Exception as e:
        logger.error("Error occurred during script execution: {}", e)
        print(f"Error: {e}")
