#!/usr/bin/env python3
import os
import re
import time
import argparse
import requests
import pandas as pd
from urllib.parse import quote

# ──────────────────────────────────────────────────────────────────────────────
#  CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

GITLAB_API_REPO = "https://gitlab.com/api/v4/projects/{project_id}"
GITLAB_API_TREE = "https://gitlab.com/api/v4/projects/{project_id}/repository/tree?ref={branch}&recursive=true&per_page=100"
GITLAB_API_RAW = "https://gitlab.com/api/v4/projects/{project_id}/repository/files/{file_path}/raw?ref={branch}"

# Fetch only text files under this size (in bytes) to avoid huge binaries
MAX_FILE_BYTES = 200_000
TEXT_EXTENSIONS = (".py", ".js", ".java", ".md", ".txt", ".json", ".yaml", ".yml", ".cpp", ".c", ".ts")

def parse_gitlab_url(url: str):
    m = re.match(r"https?://gitlab\.com/([^/]+/[^/]+(?:/[^/]+)*)", url)
    if not m:
        raise ValueError(f"Invalid GitLab URL: {url}")
    return quote(m.group(1), safe="")  # URL-encoded project ID

def fetch_repo_metadata(project_id: str, token: str = None):
    headers = {"Accept": "application/json"}
    if token:
        headers["PRIVATE-TOKEN"] = token
    r = requests.get(GITLAB_API_REPO.format(project_id=project_id), headers=headers)
    r.raise_for_status()
    data = r.json()
    return {
        "repo_full_name": data["path_with_namespace"],
        "description": data.get("description", ""),
        "html_url": data["web_url"],
        "default_branch": data.get("default_branch", "main")
    }

def fetch_file_tree(project_id: str, branch: str, token: str = None):
    headers = {"Accept": "application/json"}
    if token:
        headers["PRIVATE-TOKEN"] = token

    url = GITLAB_API_TREE.format(project_id=project_id, branch=branch)
    r = requests.get(url, headers=headers)
    r.raise_for_status()
    tree = r.json()
    return [
        item["path"]
        for item in tree
        if item["type"] == "blob"
        and item["path"].lower().endswith(TEXT_EXTENSIONS)
    ]

def fetch_file_content(project_id: str, branch: str, path: str, token: str = None):
    headers = {}
    if token:
        headers["PRIVATE-TOKEN"] = token
    url = GITLAB_API_RAW.format(
        project_id=project_id,
        file_path=quote(path, safe=""),
        branch=branch
    )
    r = requests.get(url, headers=headers, timeout=15)
    if r.status_code == 200:
        return r.text
    else:
        print(f"⚠️ Failed to fetch {url} ({r.status_code})")
        return ""

def read_repo_list(file_path: str):
    with open(file_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

def gitlab_to_parquet(url_file: str, output_path: str, token: str = None, pause: float = 1.0):
    repos = read_repo_list(url_file)
    records = []

    for repo_url in repos:
        try:
            project_id = parse_gitlab_url(repo_url)
        except ValueError as e:
            print(f"⚠️ Skipping invalid URL {repo_url}: {e}")
            continue

        print(f"\n🔍 Processing repository: {repo_url}")
        try:
            meta = fetch_repo_metadata(project_id, token)
        except Exception as e:
            print(f"⚠️ Failed to fetch metadata: {e}")
            continue

        branch = meta["default_branch"]

        # 1) README
        readme = fetch_file_content(project_id, branch, "README.md", token)
        records.append({
            **meta,
            "file_path": "README.md",
            "content": readme
        })

        # 2) All other text/code files
        try:
            file_paths = fetch_file_tree(project_id, branch, token)
        except Exception as e:
            print(f"⚠️ Failed to list tree for {project_id}: {e}")
            file_paths = []

        for path in file_paths:
            if path.lower() == "readme.md":
                continue
            content = fetch_file_content(project_id, branch, path, token)
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
        description="Fetch GitLab repo metadata + code files, save to Parquet"
    )
    parser.add_argument("--file", "-f", required=True, help="Text file with GitLab repo URLs (one per line)")
    parser.add_argument("--out", "-o", default="gitlab_code.parquet", help="Output Parquet file path")
    parser.add_argument("--token", "-t", default=None, help="GitLab personal access token (for private repos or higher rate limits)")
    parser.add_argument("--pause", type=float, default=1.0, help="Seconds to wait between API calls")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    gitlab_to_parquet(args.file, args.out, token=args.token, pause=args.pause)
