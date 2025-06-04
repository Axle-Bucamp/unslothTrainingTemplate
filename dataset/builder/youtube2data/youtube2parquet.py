import argparse
import os
import pandas as pd
import time
from tqdm import tqdm
from stem import Signal
from stem.control import Controller
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound, VideoUnavailable

# Renew Tor IP
def renew_tor_ip(password="tor_password"):
    try:
        with Controller.from_port(port=9051) as c:
            c.authenticate(password=password)
            c.signal(Signal.NEWNYM)
            print("[*] Tor IP renewed.")
    except Exception as e:
        print(f"[!] Failed to renew Tor IP: {e}")

# Fetch subtitles for a given video ID
def fetch_subtitle(video_id):
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        return " ".join([segment['text'] for segment in transcript_list])
    except (TranscriptsDisabled, NoTranscriptFound, VideoUnavailable) as e:
        print(f"[!] Transcript issue for {video_id}: {e}")
    except Exception as e:
        print(f"[!] Error fetching subtitle for {video_id}: {e}")
    return None

# Use Selenium to fetch all video URLs from the channel's /videos page
def fetch_videos_from_videos_page(handle_url):
    videos = set()
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    driver = webdriver.Chrome(options=options)
    try:
        url = f"{handle_url}/videos"
        driver.get(url)
        time.sleep(3)
        last_height = driver.execute_script("return document.documentElement.scrollHeight")
        while True:
            driver.execute_script("window.scrollTo(0, document.documentElement.scrollHeight);")
            time.sleep(2)
            new_height = driver.execute_script("return document.documentElement.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height
        elements = driver.find_elements(By.XPATH, '//a[@id="video-title"]')
        for elem in elements:
            href = elem.get_attribute('href')
            if href and 'watch?v=' in href:
                videos.add(href)
    except Exception as e:
        print(f"[!] Selenium failed for {handle_url}/videos: {e}")
    finally:
        driver.quit()
    return list(videos)

# Fetch video metadata via yt-dlp
def fetch_metadata(video_url):
    try:
        import yt_dlp
        with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
            info = ydl.extract_info(video_url, download=False)
            subtitle = fetch_subtitle(info.get('id'))
            return {
                'url': video_url,
                'id': info.get('id'),
                'title': info.get('title'),
                'description': info.get('description'),
                'upload_date': info.get('upload_date'),
                'uploader': info.get('uploader'),
                'duration': info.get('duration'),
                'subtitle': subtitle
            }
    except Exception as e:
        print(f"[!] Failed to fetch metadata for {video_url}: {e}")
        return None

# Main function
def main():
    parser = argparse.ArgumentParser(description="Fetch YouTube video metadata and subtitles")
    parser.add_argument('--input', type=str, help='Path to video URL list (txt)')
    parser.add_argument('--channels', type=str, help='Path to channel handle list (txt)')
    parser.add_argument('--output', type=str, default='dataset/youtube.parquet')
    parser.add_argument('--agree-terms', action='store_true', help='You must agree to terms of use')
    parser.add_argument('--use-tor', action='store_true', help='Use Tor for IP rotation')
    parser.add_argument('--tor-password', default='tor_password', help='Password for Tor control port')
    args = parser.parse_args()

    if not args.agree_terms:
        print("[!] You must agree to the terms of use with --agree-terms.")
        return

    video_urls = []
    if args.input and os.path.exists(args.input):
        with open(args.input, 'r') as f:
            video_urls.extend([line.strip() for line in f if line.strip()])

    if args.channels and os.path.exists(args.channels):
        with open(args.channels, 'r') as f:
            handles = [line.strip() for line in f if line.strip()]
        for handle in tqdm(handles, desc="Fetching videos from channels"):
            urls = fetch_videos_from_videos_page(handle)
            video_urls.extend(urls)

    video_urls = list(set(video_urls))
    print(f"[✔] Total videos to process: {len(video_urls)}")

    records = []
    for url in tqdm(video_urls, desc="Fetching metadata and subtitles"):
        if args.use_tor:
            renew_tor_ip(password=args.tor_password)
            time.sleep(5)
        metadata = fetch_metadata(url)
        if metadata:
            records.append(metadata)

    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_parquet(args.output, index=False)
    print(f"[✔] Dataset saved to {args.output}")

if __name__ == '__main__':
    main()
