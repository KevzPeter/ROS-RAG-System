from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import requests


def is_valid_url(url, base_domain):
    """Validate if URL belongs to base domain"""
    try:
        parsed = urlparse(url)
        return base_domain in parsed.netloc
    except:
        return False


def scrape_page(url, visited_urls, base_domain):
    """Scrape single page and return content + new URLs"""
    if url in visited_urls:
        return None, set()

    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')

        # Remove script and style elements
        for element in soup(['script', 'style']):
            element.decompose()

        # Extract text
        text_content = soup.get_text(separator='\n', strip=True)

        # Extract code snippets
        code_snippets = []
        for code in soup.find_all(['code', 'pre']):
            snippet = code.get_text(strip=True)
            if snippet:
                code_snippets.append(snippet)

        # Find new URLs to crawl
        new_urls = set()
        for link in soup.find_all('a'):
            href = link.get('href')
            if href:
                full_url = urljoin(url, href)
                if is_valid_url(full_url, base_domain):
                    new_urls.add(full_url)

        return {
            'url': url,
            'text_content': text_content,
            'code_snippets': code_snippets
        }, new_urls

    except Exception as e:
        print(f"Error scraping {url}: {str(e)}")
        return None, set()


def crawl_website(start_url, max_pages=10):
    """Crawl website starting from URL"""
    base_domain = urlparse(start_url).netloc

    visited_urls = set()
    to_visit = {start_url}
    collected_data = []

    while to_visit and len(visited_urls) < max_pages:
        current_url = to_visit.pop()
        if current_url in visited_urls:
            continue
        data, new_urls = scrape_page(
            current_url, visited_urls, base_domain)
        if data:
            collected_data.append(data)
            visited_urls.add(current_url)
            to_visit.update(new_urls - visited_urls)
            print(f"Scraped {current_url}")

    return collected_data
