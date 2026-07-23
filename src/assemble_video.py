"""
Assembles narrated story segments into one vertical (1080x1920) TikTok-ready
video. Each story becomes 3 tiles:
  1. Title tile   - the headline (large) + a short one-sentence hook
  2-3. Detail tiles - one to two more descriptive sentences each

Text position rotates through a small set of vertical-start templates so
consecutive tiles don't all place the headline/body in the exact same spot.
Every text element is kept inside a safe margin sized for TikTok's own UI
overlay - engagement icons run down the right edge, and the caption/username/
sound-disc sit along the bottom - so nothing gets crowded out once this is
actually posted as a TikTok video rather than just previewed as a bare mp4.

Backgrounds are generated on the fly with Pillow (simple gradient + accent
bar) so the pipeline needs zero stock-footage/image API and stays free.
"""

import os
import platform

from moviepy import (
    AudioFileClip,
    CompositeVideoClip,
    ImageClip,
    TextClip,
    concatenate_videoclips,
)
from PIL import Image, ImageDraw

W, H = 1080, 1920

# Font paths are OS-specific: the GitHub Actions runner is Linux (DejaVu),
# but this also needs to run locally on macOS/Windows for testing.
_FONT_CANDIDATES = {
    "Linux": {
        True: "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        False: "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    },
    "Windows": {
        True: r"C:\Windows\Fonts\arialbd.ttf",
        False: r"C:\Windows\Fonts\arial.ttf",
    },
    "Darwin": {
        True: "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        False: "/System/Library/Fonts/Supplemental/Arial.ttf",
    },
}


def _resolve_font(bold: bool) -> str:
    path = _FONT_CANDIDATES.get(platform.system(), {}).get(bold)
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(
            f"No usable {'bold' if bold else 'regular'} font found for "
            f"platform {platform.system()!r} (looked for {path!r})."
        )
    return path


FONT_BOLD = _resolve_font(bold=True)
FONT_REGULAR = _resolve_font(bold=False)

# Keep all text inside this box. Right/bottom margins are extra generous to
# stay clear of TikTok's own UI overlay once posted as a real TikTok video.
SAFE_LEFT = 72
SAFE_RIGHT = 168
SAFE_TOP = 140
SAFE_BOTTOM = 300
CONTENT_W = W - SAFE_LEFT - SAFE_RIGHT  # 840

ACCENT_COLORS = [
    (255, 71, 87),    # red
    (255, 165, 2),    # orange
    (46, 213, 115),   # green
    (30, 144, 255),   # blue
    (162, 89, 255),   # purple
]
BG_TOP = (18, 18, 24)
BG_BOTTOM = (32, 32, 42)

# Dynamic content (the part that moves between tiles) is confined to this
# vertical band, computed from the actual rendered text height each time -
# so a position can never push content past the safe bottom margin, no
# matter how long the headline/hook/detail text turns out to be.
CONTENT_TOP = 160
CONTENT_BOTTOM = H - SAFE_BOTTOM  # 1620

# Rotating fractional anchors (0.0 = flush with CONTENT_TOP, 1.0 = flush with
# CONTENT_BOTTOM once the block's real height is subtracted) so the content
# block doesn't always land in the same spot. TITLE_Y_FRACTIONS has 4 entries;
# DETAIL_Y_FRACTIONS has 5 specifically because detail tiles advance by 2 per
# story (2 detail tiles/story) - an even-length list here would fall into a
# short repeating cycle (e.g. a 4-entry list repeats the same pair of
# positions every other story); 5 is coprime with 2, so all 5 positions get
# used before anything repeats.
TITLE_Y_FRACTIONS = [0.0, 0.45, 0.85, 0.2]
DETAIL_Y_FRACTIONS = [0.0, 0.25, 0.5, 0.75, 1.0]


def _anchored_y(fraction: float, block_height: float) -> int:
    """Convert a 0-1 fraction into an actual y position, guaranteeing the
    block (of the given height) never starts before CONTENT_TOP or ends
    after CONTENT_BOTTOM."""
    available = max(CONTENT_BOTTOM - CONTENT_TOP - block_height, 0)
    return int(round(CONTENT_TOP + fraction * available))


def _make_gradient_background(accent: tuple[int, int, int]) -> Image.Image:
    """Vertical dark gradient with a colored accent bar down the left edge."""
    img = Image.new("RGB", (W, H))
    draw = ImageDraw.Draw(img)

    for y in range(H):
        t = y / H
        r = int(BG_TOP[0] + (BG_BOTTOM[0] - BG_TOP[0]) * t)
        g = int(BG_TOP[1] + (BG_BOTTOM[1] - BG_TOP[1]) * t)
        b = int(BG_TOP[2] + (BG_BOTTOM[2] - BG_TOP[2]) * t)
        draw.line([(0, y), (W, y)], fill=(r, g, b))

    draw.rectangle([0, 0, 24, H], fill=accent)
    return img


def _story_badge(text: str, duration: float, accent: tuple[int, int, int]):
    """Small persistent nav marker (e.g. "2/5") anchored to the same corner
    on every tile - a stable orientation cue while the main content below
    it is free to move around. Right-aligned to its own rendered width so
    it never drifts past the safe boundary regardless of digit count."""
    clip = TextClip(font=FONT_BOLD, text=text, font_size=40, color=f"rgb{accent}", method="label")
    clip = clip.with_duration(duration)
    x = W - SAFE_RIGHT - clip.size[0]
    return clip.with_position((x, 56))


def _progress_dots(tile_index: int, total_tiles: int, duration: float, accent: tuple[int, int, int]):
    """Tiny filled/hollow dot row showing which of the 3 tiles (per story)
    this is - helps the viewer read 3 screens as one continuing story."""
    dots = "  ".join("\u25cf" if i == tile_index else "\u25cb" for i in range(total_tiles))
    clip = TextClip(font=FONT_REGULAR, text=dots, font_size=26, color=f"rgb{accent}", method="label")
    clip = clip.with_duration(duration)
    x = W - SAFE_RIGHT - clip.size[0]
    return clip.with_position((x, 108))


def _build_title_tile(
    story_number: int,
    total_stories: int,
    headline: str,
    hook: str,
    audio_path: str,
    accent: tuple[int, int, int],
    tmp_dir: str,
    template_index: int,
    pad_seconds: float = 0.6,
):
    """The first tile for a story: headline + a short hook line stacked
    beneath it. Hook position is computed from the headline's actual
    rendered height (not a hardcoded offset), so a headline that wraps to
    3 lines can never overlap the hook text below it."""
    audio = AudioFileClip(audio_path)
    duration = audio.duration + pad_seconds

    bg_path = os.path.join(tmp_dir, f"bg_{story_number}_title.png")
    _make_gradient_background(accent).save(bg_path)
    background = ImageClip(bg_path).with_duration(duration)

    y_fraction = TITLE_Y_FRACTIONS[template_index % len(TITLE_Y_FRACTIONS)]

    # Build both text clips first (unpositioned) so we know the real combined
    # height before choosing where the block starts - this is what makes the
    # positioning overflow-safe regardless of how long the text wraps.
    headline_clip = TextClip(
        font=FONT_BOLD,
        text=headline,
        font_size=68,
        color="white",
        size=(CONTENT_W, None),
        method="caption",
        text_align="left",
    ).with_duration(duration)
    headline_h = headline_clip.size[1]

    hook_clip = TextClip(
        font=FONT_REGULAR,
        text=hook,
        font_size=46,
        color=(210, 210, 210),
        size=(CONTENT_W, None),
        method="caption",
        text_align="left",
    ).with_duration(duration)
    hook_h = hook_clip.size[1]

    gap = 48
    block_height = headline_h + gap + hook_h
    y_start = _anchored_y(y_fraction, block_height)

    headline_clip = headline_clip.with_position((SAFE_LEFT, y_start))
    hook_clip = hook_clip.with_position((SAFE_LEFT, y_start + headline_h + gap))

    badge = _story_badge(f"{story_number}/{total_stories}", duration, accent)
    dots = _progress_dots(0, 3, duration, accent)

    segment = CompositeVideoClip(
        [background, headline_clip, hook_clip, badge, dots], size=(W, H)
    ).with_duration(duration)
    return segment.with_audio(audio)


def _build_detail_tile(
    story_number: int,
    total_stories: int,
    tile_index: int,
    text: str,
    audio_path: str,
    accent: tuple[int, int, int],
    tmp_dir: str,
    template_index: int,
    pad_seconds: float = 0.6,
):
    """A follow-up tile for a story: just the descriptive text, larger and
    given more of the screen than the title tile, at a rotating start
    position so tile 2 and tile 3 don't look identical."""
    audio = AudioFileClip(audio_path)
    duration = audio.duration + pad_seconds

    bg_path = os.path.join(tmp_dir, f"bg_{story_number}_detail{tile_index}.png")
    _make_gradient_background(accent).save(bg_path)
    background = ImageClip(bg_path).with_duration(duration)

    y_fraction = DETAIL_Y_FRACTIONS[template_index % len(DETAIL_Y_FRACTIONS)]

    body_clip = TextClip(
        font=FONT_REGULAR,
        text=text,
        font_size=54,
        color="white",
        size=(CONTENT_W, None),
        method="caption",
        text_align="left",
    ).with_duration(duration)
    body_h = body_clip.size[1]
    y_start = _anchored_y(y_fraction, body_h)
    body_clip = body_clip.with_position((SAFE_LEFT, y_start))

    badge = _story_badge(f"{story_number}/{total_stories}", duration, accent)
    dots = _progress_dots(tile_index, 3, duration, accent)

    segment = CompositeVideoClip(
        [background, body_clip, badge, dots], size=(W, H)
    ).with_duration(duration)
    return segment.with_audio(audio)


def _build_intro_card(title: str, subtitle: str, tmp_dir: str, duration: float = 2.5):
    bg_path = os.path.join(tmp_dir, "bg_intro.png")
    _make_gradient_background(ACCENT_COLORS[0]).save(bg_path)
    background = ImageClip(bg_path).with_duration(duration)

    title_clip = (
        TextClip(
            font=FONT_BOLD,
            text=title,
            font_size=100,
            color="white",
            size=(W - 2 * SAFE_LEFT, None),
            method="caption",
            text_align="center",
        )
        .with_duration(duration)
        .with_position(("center", 780))
    )

    subtitle_clip = (
        TextClip(
            font=FONT_REGULAR,
            text=subtitle,
            font_size=50,
            color=(200, 200, 200),
            size=(W - 2 * SAFE_LEFT, None),
            method="caption",
            text_align="center",
        )
        .with_duration(duration)
        .with_position(("center", 1000))
    )

    return CompositeVideoClip([background, title_clip, subtitle_clip], size=(W, H)).with_duration(
        duration
    )


def assemble_video(
    stories: list[dict],
    parts_lists: list[list[str]],
    audio_paths_lists: list[list[str]],
    output_path: str,
    tmp_dir: str,
    video_title: str = "Top News Today",
    video_subtitle: str = "",
) -> str:
    """
    stories: list of story dicts (needs at least a "title" key)
    parts_lists: one entry per story, each a list of exactly 3 strings:
                 [hook, detail_1, detail_2]
    audio_paths_lists: one entry per story, each a list of exactly 3 audio
                        file paths, matching parts_lists 1:1
    """
    os.makedirs(tmp_dir, exist_ok=True)
    total = len(stories)

    clips = [_build_intro_card(video_title, video_subtitle, tmp_dir)]

    # Separate counters for title vs. detail tiles. Sharing one counter
    # across both would be a trap here: every story has exactly 3 tiles, and
    # if that period ever evenly divides a template list's length, the
    # shared counter's modulo lands on the same index every time - e.g. a
    # 3-entry title template list would make every story's title tile use
    # index 0, defeating the rotation entirely. Independent counters avoid
    # any accidental syncing between tile cadence and template-list length.
    title_template_counter = 0
    detail_template_counter = 0

    for i, (story, parts, audio_paths) in enumerate(zip(stories, parts_lists, audio_paths_lists)):
        if len(parts) != 3 or len(audio_paths) != 3:
            raise ValueError(
                f"Story {i} needs exactly 3 parts and 3 audio paths, "
                f"got {len(parts)} parts / {len(audio_paths)} audio paths."
            )

        accent = ACCENT_COLORS[i % len(ACCENT_COLORS)]

        clips.append(
            _build_title_tile(
                i + 1, total, story["title"], parts[0], audio_paths[0], accent, tmp_dir, title_template_counter
            )
        )
        title_template_counter += 1

        for tile_idx in (1, 2):
            clips.append(
                _build_detail_tile(
                    i + 1,
                    total,
                    tile_idx,
                    parts[tile_idx],
                    audio_paths[tile_idx],
                    accent,
                    tmp_dir,
                    detail_template_counter,
                )
            )
            detail_template_counter += 1

    final = concatenate_videoclips(clips, method="compose")
    final.write_videofile(
        output_path,
        fps=30,
        codec="libx264",
        audio_codec="aac",
        threads=4,
        preset="medium",
        # Without this, moviepy drops its TEMP_MPY_wvf_snd audio file next to
        # the process's cwd instead of next to output_path - put it in
        # tmp_dir so it's swept up by the caller's tmp_dir cleanup regardless.
        temp_audiofile_path=tmp_dir,
    )

    return output_path
