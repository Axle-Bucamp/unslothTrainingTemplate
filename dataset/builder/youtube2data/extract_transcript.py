import argparse
import os
import sys
import time
import pandas as pd
from tqdm import tqdm
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound, VideoUnavailable
from stem import Signal
from stem.control import Controller

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
        # Concatenate all text segments into a single string
        return " ".join([segment['text'] for segment in transcript_list])
    except TranscriptsDisabled:
        print(f"[!] Transcripts disabled for {video_id}.")
    except NoTranscriptFound:
        print(f"[!] No transcript found for {video_id}.")
    except VideoUnavailable:
        print(f"[!] Video unavailable: {video_id}.")
    except Exception as e:
        print(f"[!] Error fetching subtitle for {video_id}: {e}")
    return None

# Main function
def main():
    parser = argparse.ArgumentParser(description="Fetch YouTube subtitles from existing parquet metadata")
    parser.add_argument("--input", required=True, help="Path to input Parquet file with video metadata")
    parser.add_argument("--output", default="subtitles.parquet", help="Path to output Parquet file for subtitles")
    parser.add_argument("--use-tor", action="store_true", help="Use Tor for IP rotation between requests")
    parser.add_argument("--tor-password", default="tor_password", help="Password for Tor control port")
    parser.add_argument("--agree-terms", action="store_true", help="Agree to YouTube Terms of Service")
    args = parser.parse_args()

    if not args.agree_terms:
        print("[!] You must agree to YouTube's Terms of Service using --agree-terms.")
        sys.exit(1)

    # Read input parquet
    if not os.path.exists(args.input):
        print(f"[!] Input file not found: {args.input}")
        sys.exit(1)

    df = pd.read_parquet(args.input)
    if 'id' not in df.columns:
        print("[!] Input Parquet must contain 'id' column with video IDs.")
        sys.exit(1)

    video_ids = df['id'].dropna().unique().tolist()
    print(f"[*] Fetching subtitles for {len(video_ids)} videos.")

    records = []
    for vid in tqdm(video_ids, desc="Fetching subtitles"):
        if args.use_tor:
            renew_tor_ip(password=args.tor_password)
            time.sleep(5)
        subtitle_text = fetch_subtitle(vid)
        if subtitle_text:
            records.append({'id': vid, 'subtitle': subtitle_text})

    if records:
        out_df = pd.DataFrame(records)
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        out_df.to_parquet(args.output, index=False)
        print(f"[✔] Saved subtitles for {len(records)} videos to {args.output}")
    else:
        print("[!] No subtitles fetched.")

if __name__ == '__main__':
    main()
