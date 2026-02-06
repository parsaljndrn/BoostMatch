import requests
from bs4 import BeautifulSoup

def get_article_content(url: str) -> str:
    """
    Fetches the main textual content of an article from a given URL.
    
    Args:
        url (str): The URL of the article.
    
    Returns:
        str: Cleaned text content of the article.
    """
    if not url:
        return ""

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/144.0.0.0 Safari/537.36"
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Failed to fetch article: {e}")
        return ""

    # Parse HTML
    soup = BeautifulSoup(response.text, "html.parser")

    # Extract paragraphs
    paragraphs = soup.find_all("p")
    text_content = " ".join([p.get_text(strip=True) for p in paragraphs])

    # Clean extra whitespace
    text_content = " ".join(text_content.split())

    return text_content