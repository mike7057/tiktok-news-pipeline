"""
Orchestrates the full daily pipeline:
  1. Fetch top 5 headlines (Google News RSS - free)
  2. Summarize each into a short voiceover line (Claude API)
  3. Render narration audio for each line (edge-tts - free)
  4. Assemble everything into one vertical video (moviepy + ffmpeg)

Usage:
    python main.py                  # top 5 general news
    python main.py --count 5 --topic technology
Output:
    output/top5_YYYY-MM-DD.mp4
    output/top5_YYYY-MM-DD_script.txt   (so you can read what was said before posting)
"""

import argparse
import datetime
import os
import shutil
import sys

from assemble_video import assemble_video
from fetch_news import (
    MULTI_FEED_TOPICS,
    build_search_feed_url,
    fetch_from_multiple_feeds,
    fetch_top_stories,
)
from generate_audio import generate_audio
from summarize import select_and_summarize, summarize_stories

TOPIC_FEEDS = {
    "world": "https://news.google.com/rss/headlines/section/topic/WORLD?hl=en-US&gl=US&ceid=US:en",
    "business": "https://news.google.com/rss/headlines/section/topic/BUSINESS?hl=en-US&gl=US&ceid=US:en",
    "technology": "https://news.google.com/rss/headlines/section/topic/TECHNOLOGY?hl=en-US&gl=US&ceid=US:en",
    "sports": "https://news.google.com/rss/headlines/section/topic/SPORTS?hl=en-US&gl=US&ceid=US:en",
    "entertainment": "https://news.google.com/rss/headlines/section/topic/ENTERTAINMENT?hl=en-US&gl=US&ceid=US:en",
    "science": "https://news.google.com/rss/headlines/section/topic/SCIENCE?hl=en-US&gl=US&ceid=US:en",
    "health": "https://news.google.com/rss/headlines/section/topic/HEALTH?hl=en-US&gl=US&ceid=US:en",
}

SEARCH_TOPIC_QUERIES = {
    "crypto": "crypto OR cryptocurrency OR bitcoin",
    "ai": '"artificial intelligence" OR AI',
}


def resolve_feed_url(topic: str | None, query: str | None) -> str | None:
    """Priority: explicit --query > fixed topic sections > search-based niche
    topics > None (general feed). Note: MULTI_FEED_TOPICS (e.g. gaming) is
    handled separately in run() since it aggregates several feeds rather than
    resolving to one URL."""
    if query:
        return build_search_feed_url(query)
    if topic and topic in TOPIC_FEEDS:
        return TOPIC_FEEDS[topic]
    if topic and topic in SEARCH_TOPIC_QUERIES:
        return build_search_feed_url(SEARCH_TOPIC_QUERIES[topic])
    return None


def run(count: int, topic: str | None, query: str | None, output_dir: str, tmp_dir: str):
    today = datetime.date.today().isoformat()
    label = query or topic or "general"
    use_significance_ranking = bool(topic and topic in MULTI_FEED_TOPICS and not query)

    if use_significance_ranking:
        pool_size = max(count * 4, 20)
        print(f"[1/4] Fetching a pool of {pool_size} candidate stories ({label})...")
        candidates = fetch_from_multiple_feeds(MULTI_FEED_TOPICS[topic], pool_size=pool_size)

        if not candidates:
            print("No stories returned - aborting.", file=sys.stderr)
            sys.exit(1)

        for i, c in enumerate(candidates, 1):
            print(f"    {i}. [{c['coverage_count']}x] {c['title']} ({c['source']})")

        print(f"[2/4] Selecting the top {count} most significant + summarizing with Claude...")
        selections = select_and_summarize(candidates, desired_count=count)

        if not selections:
            print("Claude returned no selections - aborting.", file=sys.stderr)
            sys.exit(1)

        stories = [sel["story"] for sel in selections]
        script_lines = [sel["script"] for sel in selections]

        print("    Selected:")
        for i, s in enumerate(stories, 1):
            print(f"    {i}. {s['title']} ({s['source']})")

    else:
        print(f"[1/4] Fetching top {count} stories ({label})...")
        feed_url = resolve_feed_url(topic, query)
        stories = fetch_top_stories(count, feed_url) if feed_url else fetch_top_stories(count)

        if not stories:
            print("No stories returned - aborting.", file=sys.stderr)
            sys.exit(1)

        for i, s in enumerate(stories, 1):
            print(f"    {i}. {s['title']} ({s['source']})")

        print("[2/4] Summarizing with Claude...")
        script_lines = summarize_stories(stories)

    print("[3/4] Generating narration audio...")
    os.makedirs(tmp_dir, exist_ok=True)
    audio_paths = []
    for i, line in enumerate(script_lines):
        audio_path = os.path.join(tmp_dir, f"narration_{i}.mp3")
        generate_audio(line, audio_path)
        audio_paths.append(audio_path)
        print(f"    Rendered audio {i + 1}/{len(script_lines)}")

    print("[4/4] Assembling video...")
    os.makedirs(output_dir, exist_ok=True)
    video_path = os.path.join(output_dir, f"top{count}_{today}.mp4")
    assemble_video(
        stories,
        script_lines,
        audio_paths,
        output_path=video_path,
        tmp_dir=tmp_dir,
        video_title=f"Top {count} News Today",
        video_subtitle=datetime.date.today().strftime("%B %d, %Y"),
    )

    # Save the script + sources alongside the video so you can sanity-check
    # or add a caption before posting
    script_path = os.path.join(output_dir, f"top{count}_{today}_script.txt")
    with open(script_path, "w") as f:
        for i, (s, line) in enumerate(zip(stories, script_lines), 1):
            f.write(f"{i}. {s['title']}\n   Source: {s['source']}\n   Link: {s['link']}\n")
            f.write(f"   Script: {line}\n\n")

    print(f"\nDone.\n  Video:  {video_path}\n  Script: {script_path}")

    # Clean up intermediate audio/background files
    shutil.rmtree(tmp_dir, ignore_errors=True)

    return video_path, script_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a daily Top-N news recap video.")
    parser.add_argument("--count", type=int, default=5, help="Number of stories (default 5)")
    parser.add_argument(
        "--topic",
        choices=list(TOPIC_FEEDS.keys()) + list(SEARCH_TOPIC_QUERIES.keys()) + list(MULTI_FEED_TOPICS.keys()),
        default=None,
        help="Optional topic filter (default: general/world mix)",
    )
    parser.add_argument(
        "--query",
        default=None,
        help="Custom Google News search query, e.g. --query 'Nintendo OR PlayStation'. "
        "Overrides --topic if both are given.",
    )
    parser.add_argument("--output-dir", default="../output", help="Where to save final files")
    parser.add_argument("--tmp-dir", default="../tmp", help="Scratch space for intermediate files")
    args = parser.parse_args()

    run(args.count, args.topic, args.query, args.output_dir, args.tmp_dir)
