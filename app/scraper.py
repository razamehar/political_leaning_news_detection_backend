import requests
from bs4 import BeautifulSoup
from loguru import logger


# Scrape Web Content
def scrape_website(url):
    # Fetch webpage content
    response = requests.get(url)
    if response.status_code != 200:
        logger.info(f"Failed to fetch webpage from : {url}")
        logger.error(f"Failed to fetch webpage: {response.status_code}")
        raise Exception(f"Failed to fetch webpage: {response.status_code}")

    # Parse HTML with BeautifulSoup
    soup = BeautifulSoup(response.content, "html.parser")
    paragraphs = soup.find_all("p")
    content = "\n".join([para.get_text() for para in paragraphs])

    return content


# Main Script
if __name__ == "__main__":
    # Example URL to scrape
    url1 = "https://edition.cnn.com/2025/01/05/europe/ukraine-kursk-counteroffensive-russia-intl/index.html"
    url2 = "https://www.bbc.com/sport/football/articles/c4gj10jmmx"
    url = url1

    try:
        # Step 1: Scrape content
        print("Scraping website...")
        scraped_content = scrape_website(url)
        print(scraped_content)

    except Exception as e:
        print(f"Error: {e}")
