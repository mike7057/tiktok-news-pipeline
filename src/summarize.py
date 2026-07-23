"""
Turns raw headlines + RSS snippets into short, original voiceover scripts
using the Claude API. Uses Haiku since this is a simple, short task and
keeps per-run cost tiny (a handful of cents at most for 5 stories/day).

Each story becomes 3 script parts, matching the 3-tile video format:
  parts[0] = hook       - one short sentence, read on the title tile
  parts[1] = detail_1   - one to two sentences, read on tile 2
  parts[2] = detail_2   - one to two sentences, read on tile 3
"""

import json
import os
import sys

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

MODEL = "claude-haiku-4-5-20251001"

SYSTEM_PROMPT = """You write short voiceover scripts for a daily TikTok news recap video. \
Each story gets its own 3-screen mini-segment:
  - a title screen (just the headline, shown on screen - you don't write this part)
  - a "hook" line, read aloud right under the headline
  - two follow-up "detail" lines, each shown on its own screen after that

For EACH story you're given, write a "parts" array of exactly 3 strings:
  1. hook - ONE short sentence (roughly 10-15 words, ~4-5 seconds read aloud) that \
restates or introduces the headline conversationally - not a verbatim repeat of the headline
  2. detail_1 - one to two sentences (roughly 25-35 words, ~8-10 seconds read aloud) giving \
more specifics about what happened
  3. detail_2 - one to two sentences (roughly 25-35 words, ~8-10 seconds read aloud) adding \
context, implications, or what happens next

All 3 parts must together read naturally as one continuous mini-story, each written entirely \
in your own words - never copy phrasing from the source snippet. No "Story #1:" labels, no \
preamble - just the lines themselves.

IMPORTANT - if a source snippet/candidate is itself a review or opinion piece (not a \
factual news event), do not present that individual reviewer's personal verdict as an \
established fact or as this video's own conclusion - you are reporting on news/reception, \
not delivering your own review, and you never played or evaluated the thing yourself. \
Instead:
  - Lead with objective, checkable elements: scores/ratings (e.g. "sitting at 81 on \
Metacritic"), confirmed features, release details, or claims the piece states as fact \
rather than opinion.
  - If you convey sentiment, frame it as general reception, not one reviewer's individual \
take ("critics are split on whether it justifies a remake" rather than "the reviewer \
feels..." or a bare claim like "it's hard to recommend" stated as if it's just true).

Return ONLY a JSON array of objects, one per story, in the same order given, each shaped like:
  {{"parts": ["hook line", "detail line 1", "detail line 2"]}}
No markdown formatting, no code fences, no extra commentary - just the raw JSON array."""


def _parse_json_response(text: str):
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    return json.loads(text)


def summarize_stories(stories: list[dict], api_key: str | None = None) -> list[list[str]]:
    """Returns a list of 3-part script lists, one per story, same order as `stories`."""
    client = Anthropic(api_key=api_key or os.environ["ANTHROPIC_API_KEY"])

    stories_text = "\n\n".join(
        f"{i + 1}. Headline: {s['title']}\n"
        f"   Source: {s['source']}\n"
        f"   Snippet: {s['summary']}"
        for i, s in enumerate(stories)
    )

    response = client.messages.create(
        model=MODEL,
        max_tokens=2048,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": stories_text}],
    )

    parsed = _parse_json_response(response.content[0].text)

    if len(parsed) != len(stories):
        raise ValueError(f"Expected {len(stories)} entries back, got {len(parsed)}. Raw: {parsed}")

    parts_lists = []
    for entry in parsed:
        parts = entry["parts"]
        if len(parts) != 3:
            raise ValueError(f"Expected exactly 3 parts, got {len(parts)}: {parts}")
        parts_lists.append(parts)

    return parts_lists


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

Each selected story gets its own 3-screen mini-segment in the video:
  - a title screen (just the headline, shown on screen - you don't write this part)
  - a "hook" line, read aloud right under the headline
  - two follow-up "detail" lines, each shown on its own screen after that

For each selected story, write a "parts" array of exactly 3 strings:
  1. hook - ONE short sentence (roughly 10-15 words, ~4-5 seconds read aloud) that \
restates or introduces the headline conversationally - not a verbatim repeat of the headline
  2. detail_1 - one to two sentences (roughly 25-35 words, ~8-10 seconds read aloud) giving \
more specifics about what happened
  3. detail_2 - one to two sentences (roughly 25-35 words, ~8-10 seconds read aloud) adding \
context, implications, or what happens next

All 3 parts must together read naturally as one continuous mini-story, each written entirely \
in your own words - never copy phrasing from the snippet.

IMPORTANT - if a source snippet/candidate is itself a review or opinion piece (not a \
factual news event), do not present that individual reviewer's personal verdict as an \
established fact or as this video's own conclusion - you are reporting on news/reception, \
not delivering your own review, and you never played or evaluated the thing yourself. \
Instead:
  - Lead with objective, checkable elements: scores/ratings (e.g. "sitting at 81 on \
Metacritic"), confirmed features, release details, or claims the piece states as fact \
rather than opinion.
  - If you convey sentiment, frame it as general reception, not one reviewer's individual \
take ("critics are split on whether it justifies a remake" rather than "the reviewer \
feels..." or a bare claim like "it's hard to recommend" stated as if it's just true).

Return ONLY a JSON array of exactly {desired_count} objects, ordered most-to-least significant, \
each shaped like:
  {{"index": <candidate's 0-based position in the input list>, "parts": ["hook line", "detail line 1", "detail line 2"]}}
No markdown formatting, no code fences, no extra commentary - just the raw JSON array."""


def select_and_summarize(
    candidates: list[dict], desired_count: int = 5, api_key: str | None = None
) -> list[dict]:
    """
    Given a larger candidate pool (e.g. 20 stories from fetch_from_multiple_feeds,
    each with a coverage_count), asks Claude to pick the `desired_count` most
    significant ones and write a 3-part script for each.

    Returns a list of dicts: [{"story": <original candidate dict>, "parts": [p1, p2, p3]}],
    ordered most-to-least significant, length == desired_count (or fewer if the
    candidate pool itself was smaller). Uses index-based selection (Claude returns
    the candidate's position number) rather than matching by title text, since
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
        max_tokens=3072,
        system=SELECT_SYSTEM_PROMPT.format(desired_count=desired_count),
        messages=[{"role": "user", "content": candidates_text}],
    )

    selections = _parse_json_response(response.content[0].text)

    results = []
    for sel in selections[:desired_count]:
        idx = sel["index"]
        if not (0 <= idx < len(candidates)):
            continue  # guard against an out-of-range index in the model's response
        parts = sel["parts"]
        if len(parts) != 3:
            continue  # guard against a malformed parts array
        results.append({"story": candidates[idx], "parts": parts})

    return results


if __name__ == "__main__":
    from fetch_news import fetch_top_stories

    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    stories = fetch_top_stories(n)
    parts_lists = summarize_stories(stories)

    for s, parts in zip(stories, parts_lists):
        print(f"- {s['title']} ({s['source']})")
        for p in parts:
            print(f"    {p}")
        print()
