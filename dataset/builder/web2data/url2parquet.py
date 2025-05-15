import requests
from bs4 import BeautifulSoup, Comment
import pandas as pd
from urllib.parse import urljoin, urlparse
import argparse
import os
import re

# common file extensions to skip
SKIP_EXT = re.compile(r".*\.(?:pdf|zip|png|jpe?g|gif|svg|mp4|mp3|exe|docx?|xlsx?|pptx?)$", re.IGNORECASE)

def is_html_link(href: str) -> bool:
    """Skip empty, anchors, mailto, JS, or known file links."""
    if not href:
        return False
    href = href.strip()
    if href.startswith("#"):
        return False
    if href.startswith("mailto:") or href.startswith("javascript:"):
        return False
    if SKIP_EXT.match(href):
        return False
    return True

def clean_html(html: str) -> str:
    """Remove header/script/style/nav/footer tags & comments, then prettify."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup.find_all(["header", "script", "style", "nav", "footer"]):
        tag.decompose()
    # remove comments
    for c in soup.find_all(text=lambda t: isinstance(t, Comment)):
        c.extract()
    # return pretty HTML
    return soup.prettify()

def scrape_and_clean(url: str, timeout: int = 10) -> str:
    """Fetch and clean a URL's HTML, or return empty on failure."""
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
    except Exception as e:
        print(f"⚠️ Failed to fetch content from {url}: {e}")
        return ""
    return clean_html(r.text)

def read_urls(file_path: str) -> list[str]:
    with open(file_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

def scrape_urls_to_parquet(url_file: str, out_path: str):
    all_records = []
    seeds = read_urls(url_file)

    for seed in seeds:
        print(f"\n🔍 Processing seed URL: {seed}")
        try:
            resp = requests.get(seed, timeout=10)
            resp.raise_for_status()
        except Exception as e:
            print(f"⚠️ Cannot load {seed}: {e}")
            continue

        base = resp.text
        soup = BeautifulSoup(base, "html.parser")
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if not is_html_link(href):
                continue

            full_url = urljoin(seed, href)
            text = a.get_text(strip=True) or ""
            print(f"  ↳ Scraping link_text='{text[:30]}…' → {full_url}")

            content = scrape_and_clean(full_url)
            all_records.append({
                "source_url": seed,
                "link_text": text,
                "link_url": full_url,
                "content": content
            })

    if not all_records:
        print("⚠️ No records scraped; exiting.")
        return

    df = pd.DataFrame(all_records)
    df.to_parquet(out_path, engine="pyarrow", index=False)
    print(f"\n✅ Saved {len(df)} rows to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Scrape nested links from seed URLs, clean HTML, and save to Parquet"
    )
    parser.add_argument(
        "--file", "-f", required=True,
        help="Path to newline-delimited text file of seed URLs"
    )
    parser.add_argument(
        "--out", "-o", default="scraped_nested.parquet",
        help="Output Parquet file path"
    )

    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    scrape_urls_to_parquet(args.file, args.out)
