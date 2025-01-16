import pytest
import requests
from requests.exceptions import RequestException
from bs4 import BeautifulSoup
from unittest.mock import patch
from your_script import scrape_website  # Replace "your_script" with the filename of your main script

# Test URL for scraping
TEST_URL = "https://example.com"

def test_scrape_website_success(requests_mock):
    """
    Test that the scrape_website function successfully fetches and parses HTML content.
    """
    # Mock the response from the URL
    html_content = """
    <html>
        <body>
            <p>First paragraph</p>
            <p>Second paragraph</p>
        </body>
    </html>
    """
    requests_mock.get(TEST_URL, text=html_content, status_code=200)

    # Call the scrape_website function
    content = scrape_website(TEST_URL)

    # Validate the output
    assert content == "First paragraph\nSecond paragraph"

def test_scrape_website_non_200_status(requests_mock):
    """
    Test that scrape_website raises an exception for non-200 status codes.
    """
    requests_mock.get(TEST_URL, status_code=404)  # Mock a 404 response

    with pytest.raises(Exception, match="Failed to fetch webpage: 404"):
        scrape_website(TEST_URL)

def test_scrape_website_request_exception():
    """
    Test that scrape_website raises an exception when a RequestException occurs.
    """
    with patch("requests.get", side_effect=RequestException("Request failed")):
        with pytest.raises(RequestException, match="Request failed"):
            scrape_website(TEST_URL)

def test_scrape_website_empty_content(requests_mock):
    """
    Test that scrape_website handles cases with no <p> tags gracefully.
    """
    requests_mock.get(TEST_URL, text="<html><body></body></html>", status_code=200)

    # Call the function and check that the result is an empty string
    content = scrape_website(TEST_URL)
    assert content == ""

def test_scrape_website_invalid_html(requests_mock):
    """
    Test that scrape_website handles invalid HTML content.
    """
    invalid_html = "<html><body><p>Valid paragraph<p>Unclosed tag</body></html>"
    requests_mock.get(TEST_URL, text=invalid_html, status_code=200)

    # Ensure that only valid paragraphs are returned
    content = scrape_website(TEST_URL)
    assert content == "Valid paragraph"

