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
import random
import re

import numpy as np
from moviepy import (
    AudioFileClip,
    CompositeVideoClip,
    ImageClip,
    TextClip,
    concatenate_videoclips,
)
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps

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

# Reserved band for the secondary visual (image), pinned at a fixed position
# so pacing stays consistent across a video even when a story's image
# generation failed (that band just shows background in that case, not a
# layout shift). Text's dynamic zone shrinks to make room above it.
IMAGE_BAND_TOP = 1080
IMAGE_BAND_BOTTOM = 1620  # was CONTENT_BOTTOM's old value
CONTENT_BOTTOM = 1040  # shrunk from 1620 to leave room for the image band above

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


def _wrap_text(text: str, font_path: str, font_size: int, max_width: int) -> str:
    """
    Word-wrap `text` to fit within max_width pixels, measuring actual
    rendered glyph width per candidate line and only ever breaking between
    words - never mid-word.

    This exists because moviepy's own TextClip(method="caption") wrapper has
    a confirmed bug (moviepy 2.1.2, VideoClip.py's private __break_text):
    it tracks a line-break position (`last_space`) and the loop's char
    index as positions in the ENTIRE original string, but resets the
    accumulated line buffer to a short remainder after each break without
    resetting those position counters to match. Every line after the first
    then gets sliced using an index meant for the full string, silently
    landing wherever the width threshold triggers rather than at an actual
    space - producing mid-word corruption like "More" -> "Mo" / "re". The
    first line is never affected since, up to that point, the buffer's own
    length coincidentally still matches its position in the full string.
    Confirmed by calling that method directly with real content from this
    project and seeing the exact corruption reproduced.

    Returns the text with explicit '\\n' between lines, meant to be passed
    to TextClip with method="label" (which renders text as-is and does not
    attempt to re-wrap it) rather than method="caption".
    """
    font = ImageFont.truetype(font_path, font_size)
    measure_img = Image.new("RGB", (1, 1))
    draw = ImageDraw.Draw(measure_img)

    lines = []
    current_line = ""

    for word in text.split():
        candidate = f"{current_line} {word}".strip()
        left, _, right, _ = draw.textbbox((0, 0), candidate, font=font)
        width = right - left
        if width <= max_width or not current_line:
            current_line = candidate
        else:
            lines.append(current_line)
            current_line = word

    if current_line:
        lines.append(current_line)

    return "\n".join(lines)


def _measure_wrapped_text_height(wrapped_text: str, font_path: str, font_size: int, spacing: int = 4) -> int:
    """
    Independently measure the true rendered height of already-wrapped
    (newline-separated) text by actually rendering it and finding the real
    ink bounding box - not by trusting any Pillow/moviepy bbox API.

    Why not just call draw.multiline_textbbox() (which is what moviepy's own
    __find_text_size does when its preferred _multiline_spacing method is
    missing, which it is as of Pillow 11.3.0)? Confirmed by direct testing:
    multiline_textbbox's reported height is *also* wrong for multi-line text
    with anchor="ls" (the anchor moviepy's TextClip actually draws with) -
    for a real 3-line example from this project, multiline_textbbox reported
    125px, while rendering the exact same draw call and measuring actual ink
    pixels showed content extending to 134px. The bbox API undercounts by
    roughly one line's worth once you're past the first line - a distinct
    bug from moviepy's own already-broken fallback, not something the bbox
    API happens to fix. Rendering and measuring real pixels sidesteps the
    question entirely, since there's no more-authoritative source than the
    actual rendered output.
    """
    font = ImageFont.truetype(font_path, font_size)
    ascent, descent = font.getmetrics()
    n_lines = wrapped_text.count("\n") + 1

    # Generous canvas: worst case is every line needing its full ascent+descent.
    pad = 200
    canvas_h = pad * 2 + n_lines * (ascent + descent + spacing) + 100
    canvas_w = 4000  # wide enough that no realistic line gets clipped horizontally

    img = Image.new("L", (canvas_w, canvas_h), 0)
    draw = ImageDraw.Draw(img)
    # Same anchor and y-reference TextClip itself draws with (see moviepy's
    # VideoClip.py: y = ascent, anchor="ls"), just offset by `pad` so content
    # above the nominal y=0 line (ascenders, spacing quirks) isn't clipped
    # by the top of our own measuring canvas.
    draw.multiline_text(
        (0, ascent + pad), wrapped_text, font=font, fill=255, spacing=spacing, align="left", anchor="ls"
    )

    ink_rows = np.any(np.array(img) > 0, axis=1)
    if not ink_rows.any():
        return 0
    last_ink_row = int(np.where(ink_rows)[0][-1])

    # Height needed from the canvas's y=0 reference (= `pad` in our
    # measuring canvas) down to the last row that actually has ink, plus a
    # small safety pad - a clipped descender is far worse than a few extra
    # pixels of breathing room.
    return (last_ink_row - pad) + 6


def _make_grid_texture() -> Image.Image:
    """A very faint dot-grid overlay for subtle depth/texture, so the
    background reads as designed rather than a flat gradient. Neutral
    white-on-transparent so it works under any accent color. Generated once
    at import time and reused for every tile - it's identical every time,
    so there's no reason to regenerate it per story."""
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    spacing = 64
    for gx in range(0, W, spacing):
        for gy in range(0, H, spacing):
            draw.ellipse([gx - 1, gy - 1, gx + 1, gy + 1], fill=(255, 255, 255, 14))
    return img


_GRID_TEXTURE = _make_grid_texture()


def _make_glow_blob(diameter: int, color: tuple[int, int, int], max_opacity: int = 70) -> Image.Image:
    """A large, heavily-blurred soft-edged circle in the given color - an
    ambient light shape, not a recognizable object, so it adds visual depth
    without depicting anything (no risk of resembling a copyrighted
    character/logo, since it's just a blurred gradient blob)."""
    img = Image.new("RGBA", (diameter, diameter), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    inset = diameter * 0.15
    draw.ellipse([inset, inset, diameter - inset, diameter - inset], fill=(*color, max_opacity))
    return img.filter(ImageFilter.GaussianBlur(diameter * 0.16))


def _glow_blob_clips(accent: tuple[int, int, int], duration: float, tmp_dir: str, seed: int):
    """2 large soft glow blobs that drift slowly across the frame for the
    tile's duration - real per-frame motion (not just a static card) using
    only originally-generated shapes. Positions/speeds are seeded per tile
    so every tile looks a little different but is still reproducible."""
    rng = random.Random(seed)
    clips = []
    for i in range(2):
        diameter = rng.randint(520, 780)
        blob_path = os.path.join(tmp_dir, f"blob_{seed}_{i}.png")
        _make_glow_blob(diameter, accent).save(blob_path)

        start_x = rng.randint(-diameter // 2, W - diameter // 2)
        start_y = rng.randint(-diameter // 2, H - diameter // 2)
        drift_x = rng.randint(-90, 90)
        drift_y = rng.randint(-90, 90)

        blob_clip = ImageClip(blob_path).with_duration(duration)
        blob_clip = blob_clip.with_position(
            lambda t, sx=start_x, sy=start_y, dx=drift_x, dy=drift_y, dur=duration: (
                sx + dx * (t / dur if dur else 0),
                sy + dy * (t / dur if dur else 0),
            )
        )
        clips.append(blob_clip)
    return clips


def _make_gradient_background(accent: tuple[int, int, int]) -> Image.Image:
    """Vertical dark gradient + colored accent bar + a faint dot-grid
    texture, all generated on the fly with Pillow - zero external assets."""
    img = Image.new("RGB", (W, H))
    draw = ImageDraw.Draw(img)

    for y in range(H):
        t = y / H
        r = int(BG_TOP[0] + (BG_BOTTOM[0] - BG_TOP[0]) * t)
        g = int(BG_TOP[1] + (BG_BOTTOM[1] - BG_TOP[1]) * t)
        b = int(BG_TOP[2] + (BG_BOTTOM[2] - BG_TOP[2]) * t)
        draw.line([(0, y), (W, y)], fill=(r, g, b))

    draw.rectangle([0, 0, 24, H], fill=accent)

    img = img.convert("RGBA")
    img.alpha_composite(_GRID_TEXTURE)
    return img.convert("RGB")


def _build_image_panel(image_path: str, accent: tuple[int, int, int]) -> Image.Image:
    """
    Loads the generated image, center-crops it to fill the reserved image
    band exactly (no letterboxing), and adds a thin accent-colored border so
    it feels like a designed element rather than a pasted rectangle.
    """
    panel_w = CONTENT_W
    panel_h = IMAGE_BAND_BOTTOM - IMAGE_BAND_TOP

    img = Image.open(image_path).convert("RGB")
    img = ImageOps.fit(img, (panel_w, panel_h), method=Image.LANCZOS)

    # Rounded-corner mask
    mask = Image.new("L", (panel_w, panel_h), 0)
    mask_draw = ImageDraw.Draw(mask)
    mask_draw.rounded_rectangle([0, 0, panel_w, panel_h], radius=16, fill=255)

    panel = Image.new("RGBA", (panel_w, panel_h), (0, 0, 0, 0))
    panel.paste(img, (0, 0), mask)

    border_draw = ImageDraw.Draw(panel)
    border_draw.rounded_rectangle(
        [1, 1, panel_w - 2, panel_h - 2], radius=16, outline=(*accent, 255), width=3
    )
    return panel


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
    image_path: str | None = None,
    total_tiles: int = 3,
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
    blobs = _glow_blob_clips(accent, duration, tmp_dir, seed=story_number * 10)

    y_fraction = TITLE_Y_FRACTIONS[template_index % len(TITLE_Y_FRACTIONS)]

    # Build both text clips first (unpositioned) so we know the real combined
    # height before choosing where the block starts - this is what makes the
    # positioning overflow-safe regardless of how long the text wraps. Height
    # is measured independently (_measure_wrapped_text_height) and fed back in
    # via size=(...) rather than left for TextClip to auto-compute, since its
    # own auto-sizing is what clips the last line in the first place (see
    # that function's docstring).
    wrapped_headline = _wrap_text(headline, FONT_BOLD, 68, CONTENT_W)
    headline_h = _measure_wrapped_text_height(wrapped_headline, FONT_BOLD, 68)
    headline_clip = TextClip(
        font=FONT_BOLD,
        text=wrapped_headline,
        font_size=68,
        color="white",
        size=(CONTENT_W, headline_h),
        method="label",
        text_align="left",
    ).with_duration(duration)

    wrapped_hook = _wrap_text(hook, FONT_REGULAR, 46, CONTENT_W)
    hook_h = _measure_wrapped_text_height(wrapped_hook, FONT_REGULAR, 46)
    hook_clip = TextClip(
        font=FONT_REGULAR,
        text=wrapped_hook,
        font_size=46,
        color=(210, 210, 210),
        size=(CONTENT_W, hook_h),
        method="label",
        text_align="left",
    ).with_duration(duration)

    gap = 48
    block_height = headline_h + gap + hook_h
    y_start = _anchored_y(y_fraction, block_height)

    headline_clip = headline_clip.with_position((SAFE_LEFT, y_start))
    hook_clip = hook_clip.with_position((SAFE_LEFT, y_start + headline_h + gap))

    badge = _story_badge(f"{story_number}/{total_stories}", duration, accent)

    layers = [background, *blobs]
    if image_path:
        panel_img = _build_image_panel(image_path, accent)
        panel_path = os.path.join(tmp_dir, f"panel_{story_number}_title.png")
        panel_img.save(panel_path)
        panel_clip = (
            ImageClip(panel_path)
            .with_duration(duration)
            .with_position((SAFE_LEFT, IMAGE_BAND_TOP))
        )
        layers.append(panel_clip)

    layers.extend([headline_clip, hook_clip, badge])
    if total_tiles > 1:
        layers.append(_progress_dots(0, total_tiles, duration, accent))

    segment = CompositeVideoClip(layers, size=(W, H)).with_duration(duration)
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
    image_path: str | None = None,
    total_tiles: int = 3,
):
    """A follow-up tile for a story: just the descriptive text, larger and
    given more of the screen than the title tile, at a rotating start
    position so tile 2 and tile 3 don't look identical."""
    audio = AudioFileClip(audio_path)
    duration = audio.duration + pad_seconds

    bg_path = os.path.join(tmp_dir, f"bg_{story_number}_detail{tile_index}.png")
    _make_gradient_background(accent).save(bg_path)
    background = ImageClip(bg_path).with_duration(duration)
    blobs = _glow_blob_clips(accent, duration, tmp_dir, seed=story_number * 10 + tile_index)

    y_fraction = DETAIL_Y_FRACTIONS[template_index % len(DETAIL_Y_FRACTIONS)]

    wrapped_body = _wrap_text(text, FONT_REGULAR, 54, CONTENT_W)
    body_h = _measure_wrapped_text_height(wrapped_body, FONT_REGULAR, 54)
    body_clip = TextClip(
        font=FONT_REGULAR,
        text=wrapped_body,
        font_size=54,
        color="white",
        size=(CONTENT_W, body_h),
        method="label",
        text_align="left",
    ).with_duration(duration)
    y_start = _anchored_y(y_fraction, body_h)
    body_clip = body_clip.with_position((SAFE_LEFT, y_start))

    badge = _story_badge(f"{story_number}/{total_stories}", duration, accent)

    layers = [background, *blobs]
    if image_path:
        panel_img = _build_image_panel(image_path, accent)
        panel_path = os.path.join(tmp_dir, f"panel_{story_number}_{tile_index}.png")
        panel_img.save(panel_path)
        panel_clip = (
            ImageClip(panel_path)
            .with_duration(duration)
            .with_position((SAFE_LEFT, IMAGE_BAND_TOP))
        )
        layers.append(panel_clip)

    layers.extend([body_clip, badge])
    if total_tiles > 1:
        layers.append(_progress_dots(tile_index, total_tiles, duration, accent))

    segment = CompositeVideoClip(layers, size=(W, H)).with_duration(duration)
    return segment.with_audio(audio)


def _build_intro_card(title: str, subtitle: str, tmp_dir: str, duration: float = 2.5):
    bg_path = os.path.join(tmp_dir, "bg_intro.png")
    _make_gradient_background(ACCENT_COLORS[0]).save(bg_path)
    background = ImageClip(bg_path).with_duration(duration)
    blobs = _glow_blob_clips(ACCENT_COLORS[0], duration, tmp_dir, seed=0)

    intro_w = W - 2 * SAFE_LEFT
    wrapped_title = _wrap_text(title, FONT_BOLD, 100, intro_w)
    title_h = _measure_wrapped_text_height(wrapped_title, FONT_BOLD, 100)
    title_clip = (
        TextClip(
            font=FONT_BOLD,
            text=wrapped_title,
            font_size=100,
            color="white",
            size=(intro_w, title_h),
            method="label",
            text_align="center",
        )
        .with_duration(duration)
        .with_position(("center", 780))
    )

    wrapped_subtitle = _wrap_text(subtitle, FONT_REGULAR, 50, intro_w)
    subtitle_h = _measure_wrapped_text_height(wrapped_subtitle, FONT_REGULAR, 50)
    subtitle_clip = (
        TextClip(
            font=FONT_REGULAR,
            text=wrapped_subtitle,
            font_size=50,
            color=(200, 200, 200),
            size=(intro_w, subtitle_h),
            method="label",
            text_align="center",
        )
        .with_duration(duration)
        .with_position(("center", 1000))
    )

    return CompositeVideoClip(
        [background, *blobs, title_clip, subtitle_clip], size=(W, H)
    ).with_duration(duration)


def group_parts_for_slides(parts: list[str], slides_per_story: int) -> list[list[int]]:
    """
    Groups a story's script parts (index 0 = hook, the rest = supporting
    detail sentences) into `slides_per_story` slide-groups, each a list of
    part-indices to combine onto one screen.

    Slide 1 always includes the hook (title-style slide, showing the
    headline); any remaining parts distribute evenly across additional
    slides as detail-style slides (body text only, no headline).

    slides_per_story=1 merges every part into a single slide - the hook
    still displays as on-screen text alongside the headline, but the
    supporting detail sentences become audio-only narration under that same
    still slide rather than getting their own screens.

    slides_per_story >= len(parts) gives one part per slide (maximum
    granularity - the original 3-tiles-per-story behavior when there are 3
    parts). This same function is called from both main.py (to know how to
    group text before generating narration audio) and the assemble_*
    functions (to know how to group text for display) - both sides derive
    the same grouping from the same (parts, slides_per_story) inputs, so
    there's nothing extra to pass between them.
    """
    n = len(parts)
    if slides_per_story <= 1:
        return [list(range(n))]
    if slides_per_story >= n:
        return [[i] for i in range(n)]

    groups = [[0]]
    remaining = list(range(1, n))
    chunk_size = -(-len(remaining) // (slides_per_story - 1))  # ceil division
    for i in range(0, len(remaining), chunk_size):
        groups.append(remaining[i : i + chunk_size])
    return groups


def assemble_video(
    stories: list[dict],
    parts_lists: list[list[str]],
    audio_paths_lists: list[list[str]],
    output_path: str,
    tmp_dir: str,
    video_title: str = "Top News Today",
    video_subtitle: str = "",
    image_paths: list[str | None] | None = None,
) -> str:
    """
    stories: list of story dicts (needs at least a "title" key)
    parts_lists: one entry per story, each a list of exactly 3 strings:
                 [hook, detail_1, detail_2]
    audio_paths_lists: one entry per story, each a list of exactly 3 audio
                        file paths, matching parts_lists 1:1
    image_paths: one entry per story (a path or None), or None to mean "no
                 images for any story"
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
                i + 1,
                total,
                story["title"],
                parts[0],
                audio_paths[0],
                accent,
                tmp_dir,
                title_template_counter,
                image_path=image_paths[i] if image_paths else None,
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
                    image_path=image_paths[i] if image_paths else None,
                )
            )
            detail_template_counter += 1

    final = concatenate_videoclips(clips, method="compose")
    final.write_videofile(
        output_path,
        # 15fps, not 30: this content is static text + slow-drifting glow
        # blobs, never fast motion, so the lower rate is visually
        # indistinguishable here but roughly halves render time - moviepy
        # re-composites every layer on every output frame with no caching,
        # so frame count is the dominant cost, not encode settings.
        fps=15,
        codec="libx264",
        audio_codec="aac",
        threads=4,
        preset="veryfast",
        # Without this, moviepy drops its TEMP_MPY_wvf_snd audio file next to
        # the process's cwd instead of next to output_path - put it in
        # tmp_dir so it's swept up by the caller's tmp_dir cleanup regardless.
        temp_audiofile_path=tmp_dir,
    )

    return output_path


def _slugify(text: str, max_len: int = 60) -> str:
    """Turn a headline into a safe, readable filename fragment."""
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:max_len].rstrip("-") or "story"


def assemble_individual_videos(
    stories: list[dict],
    parts_lists: list[list[str]],
    audio_paths_lists: list[list[str]],
    output_dir: str,
    tmp_dir: str,
    filename_prefix: str = "story",
    slides_per_story: int = 1,
    image_paths: list[str | None] | None = None,
) -> list[str]:
    """
    Builds one standalone vertical video per story. slides_per_story
    controls how many screens each story's video has - see
    group_parts_for_slides() for the exact grouping rule. audio_paths_lists
    must already be grouped to match: one audio path per SLIDE (i.e.
    len(audio_paths_lists[i]) == len(group_parts_for_slides(parts_lists[i],
    slides_per_story))), not one per original script part.

    image_paths: one entry per story (a path or None), or None to mean "no
                 images for any story"

    Returns the list of output file paths, one per story, in order.
    """
    os.makedirs(tmp_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    total = len(stories)

    title_template_counter = 0
    detail_template_counter = 0
    output_paths = []

    for i, (story, parts, audio_paths) in enumerate(zip(stories, parts_lists, audio_paths_lists)):
        groups = group_parts_for_slides(parts, slides_per_story)
        if len(audio_paths) != len(groups):
            raise ValueError(
                f"Story {i} has {len(groups)} slide-groups (slides_per_story="
                f"{slides_per_story}) but {len(audio_paths)} audio paths - these "
                f"must match 1:1. Audio must be generated over the same grouped "
                f"text, not one clip per original script part."
            )

        accent = ACCENT_COLORS[i % len(ACCENT_COLORS)]
        total_tiles = len(groups)
        image_path = image_paths[i] if image_paths else None

        clips = []
        for slide_idx, (group, audio_path) in enumerate(zip(groups, audio_paths)):
            if 0 in group:
                clips.append(
                    _build_title_tile(
                        i + 1,
                        total,
                        story["title"],
                        parts[0],
                        audio_path,
                        accent,
                        tmp_dir,
                        title_template_counter,
                        image_path=image_path,
                        total_tiles=total_tiles,
                    )
                )
                title_template_counter += 1
            else:
                combined_text = " ".join(parts[j] for j in group)
                clips.append(
                    _build_detail_tile(
                        i + 1,
                        total,
                        slide_idx,
                        combined_text,
                        audio_path,
                        accent,
                        tmp_dir,
                        detail_template_counter,
                        image_path=image_path,
                        total_tiles=total_tiles,
                    )
                )
                detail_template_counter += 1

        final = concatenate_videoclips(clips, method="compose")
        slug = _slugify(story["title"])
        output_path = os.path.join(output_dir, f"{filename_prefix}_{i + 1:02d}_{slug}.mp4")
        final.write_videofile(
            output_path,
            fps=15,
            codec="libx264",
            audio_codec="aac",
            threads=4,
            preset="veryfast",
            # Same fix as assemble_video(): keep moviepy's intermediate
            # audio-mux temp file inside tmp_dir instead of littering cwd.
            temp_audiofile_path=tmp_dir,
        )
        output_paths.append(output_path)

    return output_paths
