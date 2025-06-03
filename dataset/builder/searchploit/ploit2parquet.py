import os
import argparse
import pandas as pd
import requests
from bs4 import BeautifulSoup

BASE_URL = "https://www.exploit-db.com"

SECTIONS = {
    "exploits": 50000,
    "ghdb": 8000,
    "docs": 100,
    "papers": 500,
    "shellcodes": 500
}

DEFAULT_OUTPUT_DIR = "exploitdb_datasets"
os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)


def fetch_page(url):
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            return resp.text
        else:
            print(f"[!] Failed ({resp.status_code}): {url}")
    except Exception as e:
        print(f"[!] Error fetching {url}: {e}")
    return None


def parse_exploit(content, section, id_):
    soup = BeautifulSoup(content, "html.parser")
    page_data = {"id": id_, "section": section}

    title_tag = soup.find("h1")
    page_data["title"] = title_tag.text.strip() if title_tag else None

    desc_div = soup.find("div", class_="description")
    page_data["description"] = desc_div.text.strip() if desc_div else None

    code_div = soup.find("code") or soup.find("pre")
    page_data["code"] = code_div.text if code_div else None

    tags = soup.find_all("span", class_="tag")
    page_data["tags"] = [tag.text.strip() for tag in tags]

    return page_data


def generate_dataset(section, start_id, end_id):
    print(f"\n[+] Generating dataset for section: {section}, IDs {start_id} to {end_id}")
    records = []
    for i in range(start_id, end_id + 1):
        url = f"{BASE_URL}/{section}/{i}"
        html = fetch_page(url)
        if html:
            record = parse_exploit(html, section, i)
            if record.get("title") or record.get("code"):
                records.append(record)
        if i % 50 == 0:
            print(f"    > Processed ID {i}")
    return records


def save_parquet(data, filename):
    df = pd.DataFrame(data)
    filepath = os.path.abspath(filename)
    df.to_parquet(filepath, engine="pyarrow", index=False)
    print(f"[✔] Saved {len(df)} records to {filepath}")


def main():
    parser = argparse.ArgumentParser(description="Exploit-DB Dataset Generator")
    parser.add_argument("--section", required=True, choices=SECTIONS.keys(), help="Section to scrape")
    parser.add_argument("--start", type=int, required=True, help="Start ID")
    parser.add_argument("--end", type=int, required=True, help="End ID")
    parser.add_argument("--output", required=False, help="Output Parquet filename")

    args = parser.parse_args()

    if args.start < 1 or args.end > SECTIONS[args.section] or args.start > args.end:
        print(f"[!] Invalid ID range: {args.start}-{args.end}. Max ID for {args.section} is {SECTIONS[args.section]}")
        return

    output_file = args.output or os.path.join(DEFAULT_OUTPUT_DIR, f"{args.section}_{args.start}_{args.end}.parquet")

    dataset = generate_dataset(args.section, args.start, args.end)
    save_parquet(dataset, output_file)


if __name__ == "__main__":
    main()
