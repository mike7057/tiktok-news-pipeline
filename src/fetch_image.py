"""
Finds and downloads a real, story-specific image for a news item - either
from the RSS feed's own media enclosure/thumbnail (if fetch_news.py already
found one), or by fetching the article page directly and reading its
og:image meta tag (the image the publisher itself set for social-sharing
previews of that specific article).

This pulls the publisher's own actual image for that article, not a generic
or AI-generated one - every image sourced this way should get a human glance
before posting to confirm it's genuinely the right image for the story,
which is why main.py also saves each one as a standalone file for easy
review, not just inside the assembled video.
"""
import os
import sys

import requests
from bs4 import BeautifulSoup

USER_AGENT = "Mozilla/5.0 (compatible; NewsRecapBot/1.0)"


def find_image_url(media_url: str | None, article_url: str) -> str | None:
    """media_url: whatever fetch_news.py's _extract_media_url already found
    from the RSS entry, or None if the feed didn't have one. article_url:
    the story's actual article link, used as a fallback to fetch the page
    directly and read its og:image tag."""
    if media_url:
        return media_url

    try:
        resp = requests.get(article_url, headers={"User-Agent": USER_AGENT}, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        tag = soup.find("meta", property="og:image") or soup.find("meta", attrs={"name": "og:image"})
        if tag and tag.get("content"):
            return tag["content"]
    except Exception as e:
        print(f"    (could not read og:image from {article_url}: {e})", file=sys.stderr)

    return None


def download_image(image_url: str, output_path: str) -> str | None:
    try:
        resp = requests.get(image_url, headers={"User-Agent": USER_AGENT}, timeout=20)
        resp.raise_for_status()
        with open(output_path, "wb") as f:
            f.write(resp.content)
        return output_path
    except Exception as e:
        print(f"    (failed to download image {image_url}: {e})", file=sys.stderr)
        return None


def fetch_story_image(media_url: str | None, article_url: str, output_path: str) -> str | None:
    """Convenience wrapper: find + download in one call. Returns output_path
    on success, None on any failure - caller should treat None as "no image
    for this story," not a fatal error."""
    image_url = find_image_url(media_url, article_url)
    if not image_url:
        return None
    return download_image(image_url, output_path)
