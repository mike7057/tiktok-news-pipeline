"""
Pulls today's top headlines from Google News RSS.

No API key needed and no request limits - it's a public RSS feed.
Swap GOOGLE_NEWS_RSS for a topic-specific feed if you want a niche
(e.g. tech, sports) instead of general world news. Topic feed URLs:
https://news.google.com/rss/headlines/section/topic/<TOPIC>?hl=en-US&gl=US&ceid=US:en
where <TOPIC> is one of: WORLD, NATION, BUSINESS, TECHNOLOGY, ENTERTAINMENT,
SPORTS, SCIENCE, HEALTH
"""

import json
import math
import re
import sys
import time
import urllib.parse
from collections import Counter
from html import unescape

import feedparser

GOOGLE_NEWS_RSS = "https://news.google.com/rss?hl=en-US&gl=US&ceid=US:en"

# Direct publisher RSS feeds, used for niches (like gaming) where publisher
# feeds are fresher and more reliable than routing through Google News search.
# These update the moment an article publishes - no aggregator lag.
MULTI_FEED_TOPICS = {
    # GameSpot deliberately excluded: confirmed its RSS entries only ever
    # carry one small (~300x169) thumbnail and its article pages sit behind
    # active Cloudflare bot-challenge protection (a real "Just a moment..."
    # JS-interstitial, not just a User-Agent check - a plain HTTP request
    # can never get past it), so a second story-specific image can never be
    # found for a GameSpot story. It was still winning the significance
    # ranking often enough that most videos ended up single-image.
    "gaming": [
        "https://www.ign.com/rss",
        "https://kotaku.com/rss",
        "https://www.polygon.com/rss/index.xml",
        "https://www.pcgamer.com/rss/",
        "https://www.eurogamer.net/feed",
    ],
}

_TOKEN_RE = re.compile(r"[a-z0-9']+")
_GENERIC_HEADLINE_WORDS = {
    "the", "a", "an", "is", "are", "to", "for", "of", "in", "on", "at", "with",
    "and", "or", "new", "after", "as", "its", "this", "that", "says", "say",
    "said", "will", "how", "why", "what", "before", "now", "today", "confirmed",
    "confirms", "confirm", "reveal", "reveals", "revealed", "announce",
    "announces", "announced", "report", "reports", "reported", "online",
    "high", "record", "game", "title", "date", "out",
}


def build_search_feed_url(query: str, hl: str = "en-US", gl: str = "US", ceid: str = "US:en") -> str:
    """
    Build a Google News RSS feed URL for a keyword search, e.g. build_search_feed_url("gaming").
    Use this for niches that aren't in Google's fixed topic list (WORLD, BUSINESS,
    TECHNOLOGY, SPORTS, ENTERTAINMENT, SCIENCE, HEALTH) - "gaming" is one of these,
    since Google News has no dedicated Gaming topic section.
    """
    encoded_query = urllib.parse.quote(query)
    return f"https://news.google.com/rss/search?q={encoded_query}&hl={hl}&gl={gl}&ceid={ceid}"


def clean_html(raw_html: str) -> str:
    """Strip HTML tags out of the RSS <summary> field."""
    text = re.sub(r"<[^>]+>", " ", raw_html)
    text = re.sub(r"\s+", " ", text)
    return unescape(text).strip()


def _stem(word: str) -> str:
    """Crude suffix-stripping so 'delay'/'delayed'/'delays' etc. count as the
    same token. Not linguistically rigorous, just enough to stop simple verb
    conjugation from hiding a true duplicate story."""
    return word if word.isdigit() else word[:5]


def _tokenize(title: str) -> list[str]:
    """Lowercase, drop generic headline filler words, stem lightly, but keep
    short numbers (e.g. "6" in "GTA 6") since those are often the one thing
    that distinguishes two stories."""
    words = _TOKEN_RE.findall(title.lower())
    tokens = []
    for w in words:
        if w in _GENERIC_HEADLINE_WORDS:
            continue
        if w.isdigit():
            tokens.append(w)
        elif len(w) > 1:
            tokens.append(_stem(w))
    return tokens


def _extract_media_url(entry) -> str | None:
    """
    Pull a representative image URL directly from the RSS entry itself, if
    the feed provides one - checks Media RSS thumbnail/content tags first
    (media_thumbnail, media_content - each a list of dicts with a 'url' key),
    then falls back to a standard enclosure (a list of dicts with 'href' and
    'type' keys, checking type starts with 'image/'). Returns None if the
    feed entry has neither - the caller then falls back to fetching the
    article page's og:image tag instead.
    """
    for attr in ("media_thumbnail", "media_content"):
        media_list = getattr(entry, attr, None)
        if media_list:
            url = media_list[0].get("url")
            if url:
                return url
    for enc in getattr(entry, "enclosures", []):
        if enc.get("type", "").startswith("image/") and enc.get("href"):
            return enc["href"]
    return None


def _cluster_entries(entries: list[dict], threshold: float = 0.5) -> list[list[int]]:
    """
    Groups entry indices that are likely about the same underlying story,
    using IDF-weighted token overlap: words that recur across many of
    today's headlines (e.g. "reveals", "new") count for little, while
    distinctive words (a specific game/platform/number) count for a lot.

    This is more reliable than plain string-similarity (e.g. difflib),
    which gets confused by headlines that share a template - "X Reveals
    New Y Game" - but are actually about two different, unrelated games.
    Assumes `entries` is already sorted so index 0 is the freshest.
    """
    token_lists = [_tokenize(e["title"]) for e in entries]

    doc_freq = Counter()
    for tokens in token_lists:
        doc_freq.update(set(tokens))
    n_docs = max(len(entries), 1)
    idf = {tok: math.log((n_docs + 1) / (count + 1)) + 1 for tok, count in doc_freq.items()}

    def weighted_overlap(tokens_a: list[str], tokens_b: list[str]) -> float:
        set_a, set_b = set(tokens_a), set(tokens_b)
        if not set_a or not set_b:
            return 0.0
        inter_weight = sum(idf[t] for t in (set_a & set_b))
        union_weight = sum(idf[t] for t in (set_a | set_b))
        return inter_weight / union_weight if union_weight else 0.0

    clusters: list[dict] = []  # {"indices": [...], "tokens": set}
    for i, tokens in enumerate(token_lists):
        best_cluster = None
        best_score = 0.0
        for cluster in clusters:
            score = weighted_overlap(tokens, cluster["tokens"])
            if score > best_score:
                best_score = score
                best_cluster = cluster
        if best_cluster is not None and best_score >= threshold:
            best_cluster["indices"].append(i)
            best_cluster["tokens"] |= set(tokens)
        else:
            clusters.append({"indices": [i], "tokens": set(tokens)})

    return [c["indices"] for c in clusters]


def fetch_from_multiple_feeds(feed_urls: list[str], pool_size: int = 20) -> list[dict]:
    """
    Pull entries from several RSS feeds and cluster same-story headlines
    across outlets. Returns up to `pool_size` story clusters, each with a
    coverage_count (how many distinct outlets ran essentially this story) and
    a sources list - this is a free, non-AI proxy for "how big a deal is
    this", since stories multiple outlets independently cover the same day
    tend to be the ones with real reach (big franchises, major platforms,
    breaking news) rather than single-outlet niche pieces.

    Sorted freshest-first, then by coverage_count as a tiebreak. Intended to
    be a *candidate pool* for further ranking (e.g. by an LLM), not the
    final top-N - fetch a bigger pool than you actually need.
    """
    all_entries = []

    for url in feed_urls:
        feed = feedparser.parse(url)
        if getattr(feed, "bozo", False) and not feed.entries:
            print(f"    (warning: could not read feed {url}: {feed.bozo_exception})", file=sys.stderr)
            continue

        source_name = feed.feed.get("title", url)

        for entry in feed.entries:
            published_parsed = entry.get("published_parsed") or entry.get("updated_parsed")
            all_entries.append(
                {
                    "title": entry.title.strip(),
                    "source": source_name,
                    "summary": clean_html(entry.get("summary", "")),
                    "link": entry.link,
                    "published": entry.get("published", ""),
                    "media_url": _extract_media_url(entry),
                    "_sort_key": published_parsed or time.gmtime(0),
                }
            )

    # Freshest first, so each cluster's representative (first index in its
    # group) ends up being the most recently published version of the story
    all_entries.sort(key=lambda e: e["_sort_key"], reverse=True)
    for entry in all_entries:
        entry.pop("_sort_key", None)

    clusters = []
    for indices in _cluster_entries(all_entries):
        rep = all_entries[indices[0]]

        # Keep one snippet per distinct outlet in this cluster (skip exact
        # repeats from the same outlet) - this is what lets the summarizer
        # actually draw on every source's specific details instead of only
        # ever seeing the single freshest outlet's text.
        seen_sources = set()
        snippets = []
        for idx in indices:
            entry = all_entries[idx]
            if entry["source"] in seen_sources:
                continue
            seen_sources.add(entry["source"])
            snippets.append(
                {"source": entry["source"], "summary": entry["summary"], "link": entry["link"]}
            )

        clusters.append(
            {
                "title": rep["title"],
                "source": rep["source"],
                "sources": sorted(seen_sources),
                "coverage_count": len(seen_sources),
                "summary": rep["summary"],  # kept for simple/back-compat display
                "snippets": snippets,  # full multi-outlet detail for summarization
                "link": rep["link"],
                "published": rep["published"],
                "media_url": rep.get("media_url"),
            }
        )

    # Freshest first, but let heavily-covered stories bubble up within that
    clusters.sort(key=lambda c: c["coverage_count"], reverse=True)

    return clusters[:pool_size]


def fetch_from_multiple_feeds_simple(feed_urls: list[str], n: int = 5) -> list[dict]:
    """Back-compat: no clustering, just freshest-first, deduped by exact-ish title."""
    clusters = fetch_from_multiple_feeds(feed_urls, pool_size=n * 4)
    clusters.sort(key=lambda c: c.get("published", ""), reverse=True)
    return clusters[:n]


def fetch_top_stories(n: int = 5, feed_url: str = GOOGLE_NEWS_RSS) -> list[dict]:
    """Return the top n stories as a list of dicts with title/source/summary/link."""
    feed = feedparser.parse(feed_url)

    if getattr(feed, "bozo", False) and not feed.entries:
        raise RuntimeError(f"Could not parse feed at {feed_url}: {feed.bozo_exception}")

    stories = []
    seen_titles = set()

    for entry in feed.entries:
        title = entry.title
        source = ""

        # Google News titles are usually formatted "Headline - Source Name"
        if " - " in title:
            title, source = title.rsplit(" - ", 1)
        title = title.strip()

        # Skip near-duplicate headlines (Google News often lists the same
        # story from multiple angles back to back)
        dedup_key = title.lower()[:40]
        if dedup_key in seen_titles:
            continue
        seen_titles.add(dedup_key)

        summary = clean_html(entry.get("summary", ""))

        stories.append(
            {
                "title": title,
                "source": source.strip(),
                "summary": summary,
                "link": entry.link,
                "published": entry.get("published", ""),
            }
        )

        if len(stories) >= n:
            break

    return stories


if __name__ == "__main__":
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    print(json.dumps(fetch_top_stories(count), indent=2))
