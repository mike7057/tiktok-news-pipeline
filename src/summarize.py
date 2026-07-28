"""
Turns raw headlines + RSS snippets into short, original voiceover scripts
using the Claude API. Uses Sonnet 5, not Haiku, specifically for hook-
writing quality: testing showed Haiku was inconsistently redundant with
the headline (roughly half of hooks across multiple test runs just
restated the headline in different words, despite explicit prompt
instructions and examples against it) - a genuine instruction-following
ceiling, not a prompt-wording problem, confirmed by testing progressively
more explicit prompt fixes with only partial improvement. Sonnet 5
followed the same instructions reliably. Cost impact is still small in
absolute terms (~2-3x Haiku's per-token rate on a short task), so the
quality gain was judged worth the small added cost - a deliberate choice,
not an oversight to "optimize" back to Haiku later.

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

from fetch_news import _tokenize

load_dotenv()

MODEL = "claude-sonnet-5"

SYSTEM_PROMPT = """You write short voiceover scripts for a daily TikTok news recap video. \
Each story gets its own 3-screen mini-segment:
  - a title screen (just the headline, shown on screen - you don't write this part)
  - a "hook" line, read aloud right under the headline
  - two follow-up "detail" lines, each shown on its own screen after that

For EACH story you're given, write a "parts" array of exactly 3 strings:
  1. hook - ONE short sentence (roughly 10-15 words, ~4-5 seconds read aloud). The viewer has \
ALREADY read the headline before this line plays - never restate what the headline already \
conveys (the subject, the name, the timing, what happened). Instead, continue the story: add \
a new fact, a specific detail, or an implication the headline didn't already say. If you can't \
think of anything genuinely new to add, it's better to move straight into detail_1's content \
than to pad the hook with a rephrased headline.
  2. detail_1 - one to two sentences (roughly 25-35 words, ~8-10 seconds read aloud) giving \
more specifics about what happened - assume the hook already moved past the headline's own \
facts, so keep building forward (specifics, mechanics, numbers) rather than circling back to \
re-confirm what/when/who
  3. detail_2 - one to two sentences (roughly 25-35 words, ~8-10 seconds read aloud) adding \
context, implications, or what happens next

  Too redundant (hook just repeats the headline's own facts):
  Headline: "Pokopia's First Major DLC, Bubbly Basin, Is Out Next Week"
  Hook: "Pokemon Pokopia's getting its first major expansion next week."

  Better (hook continues rather than restates):
  Hook: "It's an underwater-themed expansion, and it's just the first of a whole planned \
content pass for the game."

  A hook can look like it's following the rule while still mostly failing it - watch for a new \
clause bolted onto an otherwise-restated sentence, not just full-sentence restatement.

  Still too redundant (only the back half is new - the front half repeats "First Major DLC," \
"Bubbly Basin," and "Next Week" almost word-for-word):
  Headline: "Pokopia's First Major DLC, Bubbly Basin, Is Out Next Week"
  Hook: "The first big expansion pack, Bubbly Basin, launches next week with an underwater \
theme and more content on the way."

  Better (drops the restated setup entirely, leads with what's new):
  Hook: "It's an underwater-themed expansion, and it's just the first of a planned content \
pass for the game."

All 3 parts must together read naturally as one continuous mini-story, each written entirely \
in your own words - never copy phrasing from the source snippet. No "Story #1:" labels, no \
preamble - just the lines themselves.

IMPORTANT - stay factual; never adopt a source's opinions as your own:
  - If a source snippet is itself a review or opinion piece (not a factual news event), do not \
present that individual reviewer's personal verdict as an established fact or as this video's \
own conclusion - you are reporting on news/reception, not delivering your own review, and you \
never played, watched, or evaluated the thing yourself.
  - This also applies to hype or excitement baked into a source's own headline or writing. \
Outlets often editorialize in first person ("...and holy cow I think this one might actually \
be good") - report the underlying fact (a trailer dropped, a game launched, a feature was \
announced) and attribute any enthusiasm to its source, never voicing it yourself.
  - Never state quality judgments ("looks pretty solid", "feels fresh", "actually good") as \
your own assessment. Either attribute them ("early reactions are positive", "the outlet called \
it...") or leave them out.
  - Watch for conversational hedge phrases that smuggle in a personal reaction even with no \
direct quality-judgment word present - "let's say," "to put it mildly," "if we're honest," \
"surprise surprise," and similar all function as a wink to the audience, which is still you \
editorializing, just quieter about it. If a line reads like you're raising an eyebrow at the \
audience, cut the hedge and either state the underlying fact plainly or attribute the opinion.

  Too personal: "...though the track record for translating arcade classics to film remains, \
let's say, mixed."
  Factual: "...though video game movie adaptations have a historically spotty track record."

  - Lead with objective, checkable elements: scores/ratings (e.g. "sitting at 81 on Metacritic"), \
confirmed features, release details, or claims the piece states as fact rather than opinion.
  - If you convey sentiment, frame it as attributed reception, not one person's take voiced as \
truth ("critics are split on whether it justifies a remake" rather than "the reviewer feels..." \
or a bare claim like "it's hard to recommend" stated as if it's just true).

  Too personal: "The full trailer just dropped, and it actually looks pretty solid."
  Factual: "The full trailer just dropped, and early reactions are surprisingly positive."

STYLE - write the way someone would actually explain this to a friend out loud, not the way a \
news article or press release would state it:
  - Use contractions (it's, can't, they're, doesn't) - avoid phrasing you'd only ever see written down
  - Skip stiff openers like "This represents..." or "The approval from X clears the path forward..." \
- just say what happened plainly
  - Vary sentence rhythm - not every line needs the same "X did Y, which means Z" shape
  - It's fine to start a line with "So," "Basically," "Turns out," or similar - conversational \
connective language reads more natural spoken aloud than written-style transitions
  - Read each line back in your head as if you were saying it to a friend - if it sounds like \
something you'd only see in a press release, rewrite it
  - Conversational applies to the DELIVERY, not the substance - casual phrasing of facts is the \
goal, not casual insertion of your own opinions or reactions

  Too stiff: "This represents one of the gaming industry's largest takeover bids in recent years."
  More natural: "That makes it one of the biggest deals gaming has ever seen."

  Too stiff: "The approval from Europe clears the path forward, though other regulatory reviews \
in additional markets may still be required before the deal fully closes."
  More natural: "Europe's on board now, but the deal still needs the green light in a few other \
countries before it's actually final."

Return ONLY a JSON array of objects, one per story, in the same order given, each shaped like:
  {{"parts": ["hook line", "detail line 1", "detail line 2"]}}
No markdown formatting, no code fences, no extra commentary - just the raw JSON array."""


def _response_text(response) -> str:
    """Returns the first text block's content, skipping any thinking blocks."""
    for block in response.content:
        if block.type == "text":
            return block.text
    raise ValueError(f"No text block in response content: {response.content}")


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

    parsed = _parse_json_response(_response_text(response))

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

You'll get a pool of candidate gaming news stories. Each includes a headline, one or more \
outlet snippets (candidates already merged from multiple outlets covering the identical event \
show more than one snippet), the list of outlets covering it, and a coverage_count (how many \
separate publications are running essentially this same story - a higher count is one signal \
of real-world reach, since stories about major franchises/platforms or big breaking news tend \
to get picked up by multiple outlets, while niche pieces usually run on just one site).

Select the {desired_count} MOST SIGNIFICANT, MOST DISTINCT stories for today:
- Prioritize stories tied to games/platforms with a large player base (major franchises, top \
platforms, widely-played live-service games) or stories clearly getting real attention right \
now. Use coverage_count as one input, but weigh it alongside your own knowledge of which \
franchises/platforms/publishers are actually significant.
- CRITICAL - never spend more than one of your {desired_count} slots on the same underlying \
subject. If two or more candidates in the pool are about the same underlying subject even \
when they cover different angles - e.g. one candidate is a bug report about a game's launch \
and another is a review of that same game - MERGE them into a single selection instead of \
picking both separately. Combine the specifics from all of them into one richer story that \
covers the fuller picture (e.g. "it launched with some early bugs, but reviews have otherwise \
been positive") rather than giving the same subject two separate screens-worth of runtime.
- Do not just default to the most recent items - prioritize genuine significance and variety \
of subject matter over recency.

Each selected story gets its own 3-screen mini-segment in the video:
  - a title screen (just the headline, shown on screen - you don't write this part)
  - a "hook" line, read aloud right under the headline
  - two follow-up "detail" lines, each shown on its own screen after that

For each selection, write a "parts" array of exactly 3 strings:
  1. hook - ONE short sentence (roughly 10-15 words, ~4-5 seconds read aloud). The viewer has \
ALREADY read the headline before this line plays - never restate what the headline already \
conveys (the subject, the name, the timing, what happened). Instead, continue the story: add \
a new fact, a specific detail, or an implication the headline didn't already say. If you can't \
think of anything genuinely new to add, it's better to move straight into detail_1's content \
than to pad the hook with a rephrased headline.
  2. detail_1 - one to two sentences (roughly 25-35 words, ~8-10 seconds read aloud) giving \
more specifics about what happened, drawing on every snippet/candidate you merged into this \
selection - not just whichever one happened to be listed first. Assume the hook already moved \
past the headline's own facts, so keep building forward (specifics, mechanics, numbers) rather \
than circling back to re-confirm what/when/who
  3. detail_2 - one to two sentences (roughly 25-35 words, ~8-10 seconds read aloud) adding \
context, implications, or what happens next - if you merged multiple candidates, this is a \
good place to bring in the second angle (e.g. critical reception) alongside the main event

  Too redundant (hook just repeats the headline's own facts):
  Headline: "Pokopia's First Major DLC, Bubbly Basin, Is Out Next Week"
  Hook: "Pokemon Pokopia's getting its first major expansion next week."

  Better (hook continues rather than restates):
  Hook: "It's an underwater-themed expansion, and it's just the first of a whole planned \
content pass for the game."

  A hook can look like it's following the rule while still mostly failing it - watch for a new \
clause bolted onto an otherwise-restated sentence, not just full-sentence restatement.

  Still too redundant (only the back half is new - the front half repeats "First Major DLC," \
"Bubbly Basin," and "Next Week" almost word-for-word):
  Headline: "Pokopia's First Major DLC, Bubbly Basin, Is Out Next Week"
  Hook: "The first big expansion pack, Bubbly Basin, launches next week with an underwater \
theme and more content on the way."

  Better (drops the restated setup entirely, leads with what's new):
  Hook: "It's an underwater-themed expansion, and it's just the first of a planned content \
pass for the game."

All 3 parts must together read naturally as one continuous mini-story, each written entirely \
in your own words - never copy phrasing from any snippet.

IMPORTANT - stay factual; never adopt a source's opinions as your own:
  - If a candidate is itself a review or opinion piece (not a factual news event), do not \
present that individual reviewer's personal verdict as an established fact or as this video's \
own conclusion - you are reporting on news/reception, not delivering your own review, and you \
never played, watched, or evaluated the thing yourself.
  - This also applies to hype or excitement baked into a source's own headline or writing. \
Outlets often editorialize in first person ("...and holy cow I think this one might actually \
be good") - report the underlying fact (a trailer dropped, a game launched, a feature was \
announced) and attribute any enthusiasm to its source, never voicing it yourself.
  - Never state quality judgments ("looks pretty solid", "feels fresh", "actually good") as \
your own assessment. Either attribute them ("early reactions are positive", "the outlet called \
it...") or leave them out.
  - Watch for conversational hedge phrases that smuggle in a personal reaction even with no \
direct quality-judgment word present - "let's say," "to put it mildly," "if we're honest," \
"surprise surprise," and similar all function as a wink to the audience, which is still you \
editorializing, just quieter about it. If a line reads like you're raising an eyebrow at the \
audience, cut the hedge and either state the underlying fact plainly or attribute the opinion.

  Too personal: "...though the track record for translating arcade classics to film remains, \
let's say, mixed."
  Factual: "...though video game movie adaptations have a historically spotty track record."

  - Lead with objective, checkable elements: scores/ratings (e.g. "sitting at 81 on Metacritic"), \
confirmed features, release details, or claims the piece states as fact rather than opinion.
  - If you convey sentiment, frame it as attributed reception, not one person's take voiced as \
truth ("critics are split on whether it justifies a remake" rather than "the reviewer feels..." \
or a bare claim like "it's hard to recommend" stated as if it's just true).

  Too personal: "The full trailer just dropped, and it actually looks pretty solid."
  Factual: "The full trailer just dropped, and early reactions are surprisingly positive."

STYLE - write the way someone would actually explain this to a friend out loud, not the way a \
news article or press release would state it:
  - Use contractions (it's, can't, they're, doesn't) - avoid phrasing you'd only ever see written down
  - Skip stiff openers like "This represents..." or "The approval from X clears the path forward..." \
- just say what happened plainly
  - Vary sentence rhythm - not every line needs the same "X did Y, which means Z" shape
  - It's fine to start a line with "So," "Basically," "Turns out," or similar - conversational \
connective language reads more natural spoken aloud than written-style transitions
  - Read each line back in your head as if you were saying it to a friend - if it sounds like \
something you'd only see in a press release, rewrite it
  - Conversational applies to the DELIVERY, not the substance - casual phrasing of facts is the \
goal, not casual insertion of your own opinions or reactions

  Too stiff: "This represents one of the gaming industry's largest takeover bids in recent years."
  More natural: "That makes it one of the biggest deals gaming has ever seen."

  Too stiff: "The approval from Europe clears the path forward, though other regulatory reviews \
in additional markets may still be required before the deal fully closes."
  More natural: "Europe's on board now, but the deal still needs the green light in a few other \
countries before it's actually final."

Return ONLY a JSON array of exactly {desired_count} objects, ordered most-to-least significant, \
each shaped like:
  {{"indices": [<one or more 0-based candidate positions from the input list - use more than \
one only when merging candidates about the same underlying subject>], "parts": ["hook line", \
"detail line 1", "detail line 2"]}}
No markdown formatting, no code fences, no extra commentary - just the raw JSON array."""


def _format_candidate(index: int, candidate: dict) -> str:
    snippets = candidate.get("snippets") or [
        {"source": candidate.get("source", ""), "summary": candidate.get("summary", "")}
    ]
    lines = [
        f"{index}. Headline: {candidate['title']}",
        f"   Outlets covering this: {', '.join(candidate['sources'])} "
        f"(coverage_count={candidate['coverage_count']})",
    ]
    for snip in snippets:
        lines.append(f"   Snippet ({snip['source']}): {snip['summary']}")
    return "\n".join(lines)


def _parts_reference_candidate(parts_text: str, candidate: dict, min_shared_tokens: int = 1) -> bool:
    """
    Cheap coherence check: does the written script (parts) share at least
    some distinctive vocabulary with this specific candidate's own title?

    This exists to catch a confirmed real failure mode: the model can
    attach the same candidate index to two different selections in one
    response, writing genuine content for only one of them - the other
    selection ends up with a headline that doesn't match its own body text
    at all (observed directly: a "Xbox ad-supported streaming" headline
    paired with a script entirely about Sony/Ubisoft and discs). Not a
    rigorous check, just enough to catch a total mismatch.
    """
    candidate_tokens = set(_tokenize(candidate.get("title", "")))
    if not candidate_tokens:
        return True  # nothing to meaningfully check against, don't block on it
    parts_tokens = set(_tokenize(parts_text))
    return len(candidate_tokens & parts_tokens) >= min_shared_tokens


def _merge_candidates(candidate_list: list[dict]) -> dict:
    """Combine 2+ pool candidates that Claude flagged as the same underlying
    subject into one story dict - keeps the lead candidate's headline/link
    (by convention, whichever index came first in Claude's list), but rolls
    up every distinct source/link across all of them so the output script
    and the final script.txt reflect every outlet actually used."""
    primary = candidate_list[0]

    all_sources: list[str] = []
    all_links: list[str] = []
    for c in candidate_list:
        for s in c.get("sources", [c.get("source", "")]):
            if s and s not in all_sources:
                all_sources.append(s)
        link = c.get("link")
        if link and link not in all_links:
            all_links.append(link)

    return {
        "title": primary["title"],
        "source": ", ".join(all_sources) if all_sources else primary.get("source", ""),
        "sources": all_sources,
        "link": primary.get("link", ""),
        "links": all_links,
        "media_url": primary.get("media_url"),
    }


def select_and_summarize(
    candidates: list[dict], desired_count: int = 5, api_key: str | None = None
) -> list[dict]:
    """
    Given a larger candidate pool (e.g. 20 stories from fetch_from_multiple_feeds,
    each with a coverage_count and one or more outlet snippets), asks Claude to
    pick the `desired_count` most significant AND DISTINCT stories, merging pool
    entries that cover the same underlying subject from different angles into a
    single selection, and write a 3-part script for each.

    Returns a list of dicts: [{"story": <merged story dict>, "parts": [p1, p2, p3]}],
    ordered most-to-least significant, length == desired_count (or fewer if the
    candidate pool itself was smaller). Uses index-based selection (Claude returns
    candidate position numbers, one or more per selection) rather than matching by
    title text, since that's more reliable than hoping the model echoes titles
    back verbatim.
    """
    client = Anthropic(api_key=api_key or os.environ["ANTHROPIC_API_KEY"])

    candidates_text = "\n\n".join(_format_candidate(i, c) for i, c in enumerate(candidates))

    response = client.messages.create(
        model=MODEL,
        max_tokens=3072,
        system=SELECT_SYSTEM_PROMPT.format(desired_count=desired_count),
        messages=[{"role": "user", "content": candidates_text}],
    )

    selections = _parse_json_response(_response_text(response))

    used_indices: set[int] = set()
    results = []
    for sel in selections[:desired_count]:
        raw_indices = sel.get("indices")
        if raw_indices is None and "index" in sel:
            raw_indices = [sel["index"]]  # tolerate an older single-index response shape
        if not raw_indices:
            continue

        idxs = [i for i in raw_indices if isinstance(i, int) and 0 <= i < len(candidates)]
        if not idxs:
            continue  # guard against all-out-of-range indices in the model's response

        parts = sel.get("parts")
        if not parts or len(parts) != 3:
            continue  # guard against a malformed parts array

        # Guard 1: never let the same candidate index anchor two different
        # selections. Confirmed real failure mode: the model reused one
        # index across two separate merges in the same response, and the
        # written content only actually matched one of them.
        idxs = [i for i in idxs if i not in used_indices]
        if not idxs:
            continue

        # Guard 2: drop any index whose own candidate isn't actually
        # reflected in the written parts text at all - catches an index
        # that survives guard 1 (first use) but the model still wrote
        # content describing a different candidate entirely.
        combined_text = " ".join(parts)
        idxs = [i for i in idxs if _parts_reference_candidate(combined_text, candidates[i])]
        if not idxs:
            continue  # nothing in this selection actually matches any of its claimed sources

        used_indices.update(idxs)

        merged_story = _merge_candidates([candidates[i] for i in idxs])
        results.append({"story": merged_story, "parts": parts})

    return results


CAPTION_SYSTEM_PROMPT = """You write TikTok captions and hashtags for short news-recap videos, \
one caption/hashtag set per story, working from each story's final headline and script.

For each story, write:
  - caption: a short, scroll-stopping caption (1-2 sentences, an emoji or two is fine) that \
makes someone want to watch or comment - NOT a repeat of the video's own narration, since this \
is the text that appears under the post, not something read aloud.
  - hashtags: 4-6 relevant hashtags as plain words (no # symbol, no spaces within a tag) - mix \
broad discovery tags (e.g. gaming, news, tiktoknews) with specific ones tied to this story's \
actual subject (the game/company/franchise/platform named in the headline).

Return ONLY a JSON array, one object per story in the same order given, each shaped like:
  {"caption": "...", "hashtags": ["tag1", "tag2", "tag3", "tag4"]}
No markdown formatting, no code fences, no extra commentary - just the raw JSON array."""


def generate_captions(
    stories: list[dict], parts_lists: list[list[str]], api_key: str | None = None
) -> list[dict]:
    """
    Generates a TikTok caption + hashtag list for each story, in one batched
    API call (not one call per story, to keep cost low). Meant to run on the
    FINAL story/parts lists - i.e. after select_and_summarize's merge and
    coherence guards have already run - so captions always match what
    actually made it into the video, not a pre-validation draft.

    Returns a list of {"caption": str, "hashtags": list[str]}, one per story,
    same length and order as `stories`. Falls back to an empty caption/
    hashtags entry for any story the model's response doesn't cover, rather
    than raising, since a missing caption shouldn't block the whole run.
    """
    client = Anthropic(api_key=api_key or os.environ["ANTHROPIC_API_KEY"])

    stories_text = "\n\n".join(
        f"{i}. Headline: {s['title']}\n"
        f"   Hook: {parts[0]}\n"
        f"   Detail 1: {parts[1]}\n"
        f"   Detail 2: {parts[2]}"
        for i, (s, parts) in enumerate(zip(stories, parts_lists))
    )

    response = client.messages.create(
        model=MODEL,
        max_tokens=2048,
        system=CAPTION_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": stories_text}],
    )

    try:
        raw_results = _parse_json_response(_response_text(response))
    except (json.JSONDecodeError, ValueError, IndexError):
        raw_results = []

    normalized = []
    for i in range(len(stories)):
        entry = raw_results[i] if i < len(raw_results) and isinstance(raw_results[i], dict) else {}
        normalized.append(
            {
                "caption": entry.get("caption", ""),
                "hashtags": entry.get("hashtags", []) if isinstance(entry.get("hashtags"), list) else [],
            }
        )

    return normalized


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
