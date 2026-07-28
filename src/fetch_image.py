"""
Finds and downloads up to 2 real, story-specific images for a news item:
  1. The primary image - from the RSS feed's own media enclosure/thumbnail
     if fetch_news.py found one, otherwise the article page's og:image tag
     (the image the publisher itself set for social-sharing previews).
  2. A second, genuinely different image - found by parsing the article
     page itself for additional content images, filtering out logos, icons,
     avatars, ads, tracking pixels, and share buttons. This is a best-effort
     heuristic, not a guarantee: unlike og:image (a single well-defined,
     universally-supported tag), there's no equivalently reliable convention
     for "the second most relevant image" - site HTML structures vary
     widely. Every candidate found on the page is tried in document order
     (not just the first) until one downloads successfully and isn't a
     near-duplicate of the primary (via perceptual hash,
     images_are_near_duplicates) - a site serving the same underlying photo
     at two different sizes/URLs is a confirmed real case, and one
     duplicate candidate shouldn't rule out every other image on the page.
     If every candidate is exhausted without finding a distinct one, only
     the primary is returned (the second slot is None) rather than pasting
     the same photo into both slots - showing one image once reads as
     deliberate; showing it twice reads as an obvious, lazy-looking repeat.

Every image sourced this way should get a quick human glance before a video
posts (confirm it's genuinely the right image for the story) - this is even
more true for the second image than the first, given the heuristic involved.
"""
import os
import re
import sys
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from PIL import Image

USER_AGENT = "Mozilla/5.0 (compatible; NewsRecapBot/1.0)"

# Filters out common non-content image patterns by URL, class, alt text, or
# id - logos, icons, avatars, ads, tracking pixels, share buttons. Checked
# as a single combined pattern for simplicity; not exhaustive, just enough
# to catch the common cases across typical news-site markup.
_NON_CONTENT_PATTERN = re.compile(
    r"logo|icon|avatar|sprite|pixel|tracking|badge|share-|social-|gravatar|"
    r"author|byline|-ad-|/ad/|advert",
    re.IGNORECASE,
)


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


def find_second_image_candidates(html: str, base_url: str, exclude_url: str | None) -> list[str]:
    """
    Looks for candidate second, genuinely different content images within
    the article page's own HTML - scoped to <article> or <main> if present
    (to avoid header/footer/sidebar noise), skipping anything matching
    exclude_url (the primary image already chosen), SVGs (almost always
    icons), anything matching _NON_CONTENT_PATTERN, and anything with
    width/height attributes indicating a small (<200px) image. Returns
    EVERY remaining candidate's absolute URL, in document order - not just
    the first - so fetch_story_images can try each one in turn until it
    finds a genuinely distinct image, rather than giving up entirely the
    moment the very first candidate turns out to be a near-duplicate of
    the primary or fails to download.
    """
    soup = BeautifulSoup(html, "html.parser")
    scope = soup.find("article") or soup.find("main") or soup

    exclude_key = exclude_url.split("?")[0] if exclude_url else None
    candidates = []

    for img in scope.find_all("img"):
        src = img.get("src") or img.get("data-src")
        if not src:
            continue

        absolute_url = urljoin(base_url, src)

        if exclude_key and absolute_url.split("?")[0] == exclude_key:
            continue
        if absolute_url.lower().endswith(".svg"):
            continue

        class_attr = img.get("class")
        haystack = " ".join(
            [
                absolute_url,
                " ".join(class_attr) if class_attr else "",
                img.get("alt", ""),
                img.get("id", ""),
            ]
        )
        if _NON_CONTENT_PATTERN.search(haystack):
            continue

        width, height = img.get("width"), img.get("height")
        try:
            if width and int(width) < 200:
                continue
            if height and int(height) < 200:
                continue
        except ValueError:
            pass  # non-numeric width/height (e.g. "100%") - don't filter on it

        if absolute_url not in candidates:  # the same <img> URL can appear more than once
            candidates.append(absolute_url)

    return candidates


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


MIN_IMAGE_WIDTH = 600
MIN_IMAGE_HEIGHT = 400
# A floor for "looks reasonable on a phone screen" - low enough to accept
# ordinary web content photos (which typically run 600-1200px+), high
# enough to reject the tiny inline thumbnails/icons/related-article
# previews that occasionally slip through _NON_CONTENT_PATTERN and the
# <200px width/height ATTRIBUTE filter in find_second_image_candidates
# (which only catches images that declare their small size in the HTML
# itself - many don't, and a real example downloaded fine at only 6KB).
# assemble_video's _oversized_crop upscales whatever it's given via LANCZOS
# to fill its target panel regardless of source size, so an
# under-resolution source doesn't fail loudly - it just renders soft/
# blurry, which is what this floor exists to catch before that happens.


def _meets_min_resolution(image_path: str) -> bool:
    """True if the downloaded image's ACTUAL decoded pixel dimensions clear
    MIN_IMAGE_WIDTH/MIN_IMAGE_HEIGHT - not any HTML width/height attribute
    (which can be missing, wrong, or a percentage), the same principle as
    images_are_near_duplicates comparing real pixel content rather than
    trusting metadata. A file that can't even be opened as an image fails
    this check too - never let a corrupt download pass the quality bar."""
    try:
        with Image.open(image_path) as img:
            width, height = img.size
    except Exception:
        return False
    return width >= MIN_IMAGE_WIDTH and height >= MIN_IMAGE_HEIGHT


def _average_hash(image_path: str, hash_size: int = 8) -> int:
    """
    Simple perceptual hash (aHash): grayscale + shrink to hash_size x
    hash_size, then each bit is 1 if that pixel is brighter than the
    image's own mean brightness, 0 otherwise. Returns the hash_size**2-bit
    hash as a Python int. Two renditions of the same underlying photo
    (different crop/resize/recompression) hash to nearly the same value;
    two genuinely different photos don't - unlike an exact byte or file-size
    comparison, this survives a site serving the same image at two
    different sizes/quality levels under different URLs.
    """
    img = Image.open(image_path).convert("L").resize((hash_size, hash_size), Image.LANCZOS)
    pixels = list(img.getdata())
    avg = sum(pixels) / len(pixels)
    bits = 0
    for i, p in enumerate(pixels):
        if p > avg:
            bits |= 1 << i
    return bits


def _hamming_distance(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def images_are_near_duplicates(path1: str, path2: str, hash_size: int = 8, max_distance: int = 5) -> bool:
    """
    True if the two images are close enough to be considered "the same
    photo" rather than genuinely different content - catches a confirmed
    real failure mode: a site serving the primary image (og:image) and a
    second in-article <img> that point at two different URLs but the exact
    same underlying photo (e.g. a full-size and a thumbnail rendition of
    one screenshot). If comparison fails for any reason (corrupt/unreadable
    image), treat them as NOT duplicates - never let a broken image quietly
    cause the real second image to be discarded.

    max_distance=5 out of hash_size**2=64 bits is a fairly strict "very
    likely the same image" threshold: genuinely different photos of the
    same subject (e.g. two different screenshots from the same game)
    typically differ by well more than that.
    """
    try:
        hash1 = _average_hash(path1, hash_size)
        hash2 = _average_hash(path2, hash_size)
    except Exception:
        return False
    return _hamming_distance(hash1, hash2) <= max_distance


def fetch_story_image(media_url: str | None, article_url: str, output_path: str) -> str | None:
    """Convenience wrapper: find + download the PRIMARY image in one call.
    Returns output_path on success, None on any failure - caller should
    treat None as "no image for this story," not a fatal error. Kept as a
    separate function (rather than folded into fetch_story_images) for
    backward compatibility with any single-image call sites."""
    image_url = find_image_url(media_url, article_url)
    if not image_url:
        return None
    return download_image(image_url, output_path)


def fetch_story_images(
    media_url: str | None, article_url: str, output_path_1: str, output_path_2: str
) -> tuple[str | None, str | None]:
    """
    Fetches up to 2 images for a story: the primary (media_url or og:image),
    and a second, distinct image found by re-parsing the article page's own
    content images. If no distinct second image is found, the second slot
    is returned as None (NOT the primary reused) - the caller/renderer is
    expected to show a single available image at full size rather than the
    same photo pasted into both slots, which reads as an obvious repeat.
    Returns (path_1_or_None, path_2_or_None) - both None only if no usable
    image could be found for this story at all.

    Every downloaded candidate (primary AND second-image candidates) must
    clear MIN_IMAGE_WIDTH/MIN_IMAGE_HEIGHT (see _meets_min_resolution) -
    real content photos on these sites are generally well above that, but
    an occasional inline thumbnail/icon slips through the HTML-level
    filters in find_second_image_candidates, and og:image itself isn't
    always a genuine content photo either. If the primary fails that bar,
    it's not just discarded outright - the second-image search still runs
    (excluding that URL) and the first candidate that both downloads and
    meets the resolution floor is promoted to fill the primary slot
    instead, since one good image beats zero.

    Tries EVERY second-image candidate found on the page (in document
    order), not just the first - the first candidate turning out to be a
    near-duplicate of the primary, too small, or failing to download
    doesn't mean no genuinely distinct/usable image exists; it may just
    mean the first one wasn't it.
    """
    primary_url = find_image_url(media_url, article_url)
    primary_path = None
    if primary_url:
        downloaded = download_image(primary_url, output_path_1)
        if downloaded and _meets_min_resolution(downloaded):
            primary_path = downloaded
        elif downloaded:
            print(
                f"    (primary image {primary_url} is below the "
                f"{MIN_IMAGE_WIDTH}x{MIN_IMAGE_HEIGHT} quality floor - looking for an "
                "alternative)",
                file=sys.stderr,
            )

    candidates = []
    try:
        resp = requests.get(article_url, headers={"User-Agent": USER_AGENT}, timeout=15)
        resp.raise_for_status()
        candidates = find_second_image_candidates(resp.text, article_url, exclude_url=primary_url)
    except Exception as e:
        print(f"    (could not search for a second image on {article_url}: {e})", file=sys.stderr)

    if primary_path:
        for candidate_url in candidates:
            second_path = download_image(candidate_url, output_path_2)
            if not second_path:
                continue
            if not _meets_min_resolution(second_path):
                print(
                    f"    (second image candidate {candidate_url} is below the "
                    f"{MIN_IMAGE_WIDTH}x{MIN_IMAGE_HEIGHT} quality floor - trying the next "
                    "candidate)",
                    file=sys.stderr,
                )
                continue
            if images_are_near_duplicates(primary_path, second_path):
                print(
                    f"    (second image candidate {candidate_url} looks like a duplicate of "
                    "the primary - trying the next candidate)",
                    file=sys.stderr,
                )
                continue
            return primary_path, second_path

        # No distinct, sufficiently large second image found - return just
        # the primary. The renderer shows a single available image at full
        # size rather than pasting the same photo into both slots.
        return primary_path, None

    # No usable primary (missing entirely, or below the quality floor) -
    # fall back to the first second-image candidate that both downloads and
    # meets the resolution floor, promoted to fill the sole image slot
    # instead of leaving the story with no image at all.
    for candidate_url in candidates:
        fallback_path = download_image(candidate_url, output_path_1)
        if not fallback_path or not _meets_min_resolution(fallback_path):
            continue
        return fallback_path, None

    return None, None


if __name__ == "__main__":
    import sys as _sys

    if len(_sys.argv) < 2:
        print("Usage: python fetch_image.py <article_url>")
    else:
        p1, p2 = fetch_story_images(None, _sys.argv[1], "test_image_1.jpg", "test_image_2.jpg")
        print(f"image1: {p1}\nimage2: {p2}")
