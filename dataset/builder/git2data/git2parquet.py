#!/usr/bin/env python3
import os
import re
import time
import argparse
import requests
import pandas as pd
from urllib.parse import urlparse

# ──────────────────────────────────────────────────────────────────────────────
#  CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

GITHUB_API_REPO = "https://api.github.com/repos/{owner}/{repo}"
GITHUB_API_TREE = "https://api.github.com/repos/{owner}/{repo}/git/trees/{branch}?recursive=1"
RAW_FILE_URL = "https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}"

# fetch only “text” files under this size (in bytes) to avoid huge binaries
MAX_FILE_BYTES = 200_000  
TEXT_EXTENSIONS = (".py", ".js", ".java", ".md", ".txt", ".json", ".yaml", ".yml", ".cpp", ".c", ".ts")

def parse_github_url(url: str):
    m = re.match(r"https?://github\.com/([^/]+)/([^/]+)/?", url)
    if not m:
        raise ValueError(f"Invalid GitHub URL: {url}")
    return m.group(1), m.group(2)

def fetch_repo_metadata(owner: str, repo: str, token: str = None):
    headers = {"Accept": "application/vnd.github.v3+json"}
    if token:
        headers["Authorization"] = f"token {token}"
    r = requests.get(GITHUB_API_REPO.format(owner=owner, repo=repo), headers=headers)
    r.raise_for_status()
    data = r.json()
    return {
        "repo_full_name": data["full_name"],
        "description": data.get("description", ""),
        "html_url": data["html_url"],
        "default_branch": data.get("default_branch", "main")
    }

def fetch_file_tree(owner: str, repo: str, branch: str, token: str = None):
    """List all blobs in the repo under the given branch."""
    headers = {"Accept": "application/vnd.github.v3+json"}
    if token:
        headers["Authorization"] = f"token {token}"
    url = GITHUB_API_TREE.format(owner=owner, repo=repo, branch=branch)
    r = requests.get(url, headers=headers)
    r.raise_for_status()
    tree = r.json().get("tree", [])
    # only blobs, only desired extensions
    return [
        item["path"]
        for item in tree
        if item["type"] == "blob"
        and item.get("size", 0) <= MAX_FILE_BYTES
        and item["path"].lower().endswith(TEXT_EXTENSIONS)
    ]

def fetch_file_content(owner: str, repo: str, branch: str, path: str, token: str = None):
    """Fetch raw file content from raw.githubusercontent.com."""
    url = RAW_FILE_URL.format(owner=owner, repo=repo, branch=branch, path=path)
    headers = {}
    if token:
        headers["Authorization"] = f"token {token}"
    r = requests.get(url, headers=headers, timeout=15)
    if r.status_code == 200:
        return r.text
    else:
        print(f"⚠️ Failed to fetch {url} ({r.status_code})")
        return ""

def read_repo_list(file_path: str):
    with open(file_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

def github_to_parquet(url_file: str, output_path: str, token: str = None, pause: float = 1.0):
    repos = read_repo_list(url_file)
    records = []

    for repo_url in repos:
        try:
            owner, repo = parse_github_url(repo_url)
        except ValueError as e:
            print(f"⚠️ Skipping invalid URL {repo_url}: {e}")
            continue

        print(f"\n🔍 Processing repository: {owner}/{repo}")
        meta = fetch_repo_metadata(owner, repo, token)
        branch = meta["default_branch"]

        # 1) README
        readme = fetch_file_content(owner, repo, branch, "README.md", token)
        records.append({
            **meta,
            "file_path": "README.md",
            "content": readme
        })

        # 2) All other text/code files
        try:
            file_paths = fetch_file_tree(owner, repo, branch, token)
        except Exception as e:
            print(f"⚠️ Failed to list tree for {owner}/{repo}: {e}")
            file_paths = []

        for path in file_paths:
            if path.lower() == "readme.md":
                continue
            content = fetch_file_content(owner, repo, branch, path, token)
            records.append({
                **meta,
                "file_path": path,
                "content": content
            })

        time.sleep(pause)

    if not records:
        print("⚠️ No files fetched; exiting.")
        return

    df = pd.DataFrame(records)
    df.to_parquet(output_path, engine="pyarrow", index=False)
    print(f"\n✅ Saved {len(df)} rows to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fetch GitHub repo metadata + code files, save to Parquet"
    )
    parser.add_argument(
        "--file", "-f", required=True,
        help="Text file with GitHub repo URLs (one per line)"
    )
    parser.add_argument(
        "--out", "-o", default="github_code.parquet",
        help="Output Parquet file path"
    )
    parser.add_argument(
        "--token", "-t", default=None,
        help="GitHub personal access token (for higher rate limits)"
    )
    parser.add_argument(
        "--pause", type=float, default=1.0,
        help="Seconds to wait between API calls"
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    github_to_parquet(args.file, args.out, token=args.token, pause=args.pause)
