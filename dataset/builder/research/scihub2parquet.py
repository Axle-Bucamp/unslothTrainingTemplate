import argparse
import requests
import time
import random
from bs4 import BeautifulSoup
import pandas as pd
from stem import Signal
from stem.control import Controller
from urllib.parse import quote_plus

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept-Language": "en-US,en;q=0.9",
}
PROXIES = {
    "http": "socks5h://127.0.0.1:9050",
    "https": "socks5h://127.0.0.1:9050"
}


def renew_tor_ip(password="your_password"):
    try:
        with Controller.from_port(port=9051) as c:
            c.authenticate(password=password)
            c.signal(Signal.NEWNYM)
            print("[*] Requested new Tor identity.")
    except Exception as e:
        print(f"[!] Failed to renew Tor IP: {e}")


def scrape_scholar_articles(query, num_pages, rotate_ip=False, tor_password="your_password"):
    articles = []
    for page in range(num_pages):
        if rotate_ip:
            renew_tor_ip(tor_password)
            time.sleep(random.uniform(5, 10))

        url = f"https://scholar.google.com/scholar?start={page*10}&q={quote_plus(query)}&hl=en&as_sdt=0,5"

        try:
            response = requests.get(url, headers=HEADERS, proxies=PROXIES, timeout=20)

            if response.status_code == 429:
                print(f"[!] Rate limited (429) at page {page + 1}. Stopping.")
                break
            elif response.status_code != 200:
                print(f"[!] Failed to fetch page {page + 1} (status {response.status_code})")
                continue

            soup = BeautifulSoup(response.text, "html.parser")
            results = soup.find_all("div", class_="gs_ri")

            if not results:
                print(f"[!] No results found on page {page + 1}")
                break

            for result in results:
                title_tag = result.find("h3", class_="gs_rt")
                title = title_tag.text if title_tag else "No title"

                authors_tag = result.find("div", class_="gs_a")
                authors = authors_tag.text if authors_tag else "No authors"

                link_tag = title_tag.find("a") if title_tag else None
                link = link_tag["href"] if link_tag else "No link"

                snippet_tag = result.find("div", class_="gs_rs")
                snippet = snippet_tag.text if snippet_tag else "No snippet"

                articles.append({
                    "Title": title,
                    "Authors": authors,
                    "Link": link,
                    "Snippet": snippet,
                    "Page": page + 1
                })

            print(f"[✔] Page {page + 1}: {len(results)} results.")
            time.sleep(random.uniform(8, 15))

        except Exception as e:
            print(f"[!] Exception while fetching page {page + 1}: {e}")
            continue

    return articles


def save_to_parquet(articles, filename):
    df = pd.DataFrame(articles)
    df.to_parquet(filename, index=False)
    print(f"[✔] Saved {len(df)} articles to {filename}")


def main():
    parser = argparse.ArgumentParser(description="Google Scholar Scraper (via Tor, Parquet output)")
    parser.add_argument("query", help="Search query (e.g. 'artificial intelligence')")
    parser.add_argument("-n", "--num_pages", type=int, default=1, help="Number of result pages to scrape")
    parser.add_argument("-o", "--output", default="scholar.parquet", help="Output Parquet filename")
    parser.add_argument("--rotate", action="store_true", help="Rotate Tor IP between page fetches")
    parser.add_argument("--tor-password", default="your_password", help="Tor control password (ControlPort must be open)")

    args = parser.parse_args()

    print(f"[*] Query: '{args.query}', Pages: {args.num_pages}, Output: {args.output}")
    if args.rotate:
        print("[*] Tor IP rotation enabled.")

    articles = scrape_scholar_articles(
        query=args.query,
        num_pages=args.num_pages,
        rotate_ip=args.rotate,
        tor_password=args.tor_password
    )

    if articles:
        save_to_parquet(articles, args.output)
    else:
        print("[!] No articles found.")


if __name__ == "__main__":
    main()
