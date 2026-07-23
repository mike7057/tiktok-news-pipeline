"""
Turns raw headlines + RSS snippets into short, original voiceover lines
using the Claude API. Uses Haiku since this is a simple, short task and
keeps per-run cost tiny (a handful of cents at most for 5 stories/day).
"""

import json
import os
import sys

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

MODEL = "claude-haiku-4-5-20251001"

SYSTEM_PROMPT = """You write short voiceover scripts for a daily 60-second TikTok \
news recap video called "Top 5 Today".

For EACH story you're given, write ONE 2-sentence script line that:
- Explains what happened and why it matters, in plain, conversational spoken language
- Is written entirely in your own words - never copy phrasing from the source snippet
- Is short enough to read aloud in about 8-10 seconds (roughly 25-35 words)
- Has no headline restated verbatim, no "Story #1:", no preamble - just the line itself

Return ONLY a JSON array of strings, one per story, in the same order given.
No markdown formatting, no code fences, no extra commentary - just the raw JSON array."""


def summarize_stories(stories: list[dict], api_key: str | None = None) -> list[str]:
    client = Anthropic(api_key=api_key or os.environ["ANTHROPIC_API_KEY"])

    stories_text = "\n\n".join(
        f"{i + 1}. Headline: {s['title']}\n"
        f"   Source: {s['source']}\n"
        f"   Snippet: {s['summary']}"
        for i, s in enumerate(stories)
    )

    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": stories_text}],
    )

    text = response.content[0].text.strip()

    # Strip accidental code-fence wrapping, just in case
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()

    summaries = json.loads(text)

    if len(summaries) != len(stories):
        raise ValueError(
            f"Expected {len(stories)} summaries back, got {len(summaries)}. "
            f"Raw response: {text}"
        )

    return summaries


SELECT_SYSTEM_PROMPT = """You are selecting stories for a daily Top-{desired_count} gaming news \
recap video and writing its voiceover script.

You'll get a pool of candidate gaming news stories. Each includes a headline, a snippet, \
the list of outlets covering it, and a coverage_count (how many separate publications are \
running essentially this same story - a higher count is one signal of real-world reach, \
since stories about major franchises/platforms or big breaking news tend to get picked up \
by multiple outlets, while niche pieces usually run on just one site).

Select the {desired_count} stories that represent genuinely MAJOR gaming news for today - \
prioritizing stories tied to games/platforms with a large player base (e.g. major franchises, \
top platforms, widely-played live-service games) or stories clearly getting significant \
attention right now. Use coverage_count as one input, but weigh it alongside your own \
knowledge of which franchises/platforms/publishers are actually significant - a high-coverage \
story about a minor patch note matters less than a single-source story about a major console \
or franchise announcement. Do not just default to the most recent items - prioritize genuine \
significance over recency.

For each selected story, write ONE 2-sentence voiceover line that:
- Explains what happened and why it matters, in plain conversational spoken language
- Is written entirely in your own words - never copy phrasing from the snippet
- Is short enough to read aloud in about 8-10 seconds (roughly 25-35 words)
- Has no "Story #1:" label, no restated headline - just the line itself

Return ONLY a JSON array of exactly {desired_count} objects, ordered most-to-least significant, \
each shaped like: {{"index": <candidate's 0-based position in the input list>, "script": "..."}}
No markdown formatting, no code fences, no extra commentary - just the raw JSON array."""


def select_and_summarize(candidates: list[dict], desired_count: int = 5, api_key: str | None = None) -> list[dict]:
    """
    Given a larger candidate pool (e.g. 20 stories from fetch_from_multiple_feeds,
    each with a coverage_count), asks Claude to pick the desired_count most
    significant ones and write a script line for each.

    Returns [{"story": <original candidate dict>, "script": "..."}], ordered
    most-to-least significant. Uses index-based selection (Claude returns the
    candidate's position number) rather than matching by title text, since
    that's more reliable than hoping the model echoes titles back verbatim.
    """
    client = Anthropic(api_key=api_key or os.environ["ANTHROPIC_API_KEY"])

    candidates_text = "\n\n".join(
        f"{i}. Headline: {c['title']}\n"
        f"   Outlets covering this: {', '.join(c['sources'])} (coverage_count={c['coverage_count']})\n"
        f"   Snippet: {c['summary']}"
        for i, c in enumerate(candidates)
    )

    response = client.messages.create(
        model=MODEL,
        max_tokens=2048,
        system=SELECT_SYSTEM_PROMPT.format(desired_count=desired_count),
        messages=[{"role": "user", "content": candidates_text}],
    )

    text = response.content[0].text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()

    selections = json.loads(text)

    results = []
    for sel in selections[:desired_count]:
        idx = sel["index"]
        if not (0 <= idx < len(candidates)):
            continue  # guard against an out-of-range index in the model's response
        results.append({"story": candidates[idx], "script": sel["script"]})

    return results


if __name__ == "__main__":
    from fetch_news import fetch_top_stories

    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    stories = fetch_top_stories(n)
    summaries = summarize_stories(stories)

    for s, line in zip(stories, summaries):
        print(f"- {s['title']} ({s['source']})\n  -> {line}\n")
