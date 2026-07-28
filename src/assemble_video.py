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
from PIL import Image, ImageDraw, ImageFilter, ImageFont

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

# SAFE_LEFT/SAFE_RIGHT still keep the story badge/dots and intro card text
# inset from the true left/right edges for readability. SAFE_TOP/SAFE_BOTTOM
# are used the same way for the badge/dots' own position, but - per explicit
# instruction - no longer carve out a dedicated top/bottom margin that
# images and the text zone stay out of: images now cover the ENTIRE frame
# top to bottom (CONTENT_TOP=0, CONTENT_BOTTOM=H), and the badge/dots simply
# render on top of whatever image or text-backdrop content is there,
# accepting that TikTok's own top/bottom UI chrome may overlap image content
# once actually posted - the same tradeoff already made for the right-edge
# icon column a few rounds back.
SAFE_LEFT = 72
SAFE_RIGHT = 168
SAFE_TOP = 140
SAFE_BOTTOM = 300

ACCENT_COLORS = [
    (255, 71, 87),    # red
    (255, 165, 2),    # orange
    (46, 213, 115),   # green
    (30, 144, 255),   # blue
    (162, 89, 255),   # purple
]
BG_TOP = (18, 18, 24)
BG_BOTTOM = (32, 32, 42)

BORDER_COLOR = (49, 168, 124)  # #31a87c - used ONLY as a thin divider line
# at the seam between two adjacent images (see _build_two_image_layers),
# not as a border around each panel's own outer edges.

# When a tile has no narration audio (generate_audio disabled - see
# main.py's --audio flag), there's nothing to time its duration against -
# it's a fixed 10 seconds flat (no reading-speed estimate), since these
# tiles are read-only with no audio track at all. Only applies when audio
# is actually disabled; an audio-driven tile's duration is unaffected.
NO_AUDIO_DURATION = 10.0

# Images/text now cover the full frame height - zero blank background
# margin at the top or bottom of the tile.
CONTENT_TOP = 0
CONTENT_BOTTOM = H

# Two images per tile, with the text block landing either above both (text_top)
# or sandwiched between them (text_middle). Zone sizes are NOT fixed - text
# gets exactly the height its actual content measures to, and the 2 images
# split whatever space remains. This is deliberate: an earlier version of
# this layout used a fixed text-zone height (360px) sized off one example
# headline, and testing across several realistic headline+hook combinations
# found real ones needing anywhere from ~450px up to 760+px - a fixed
# allocation would have silently overflowed into the image zone for any
# longer-than-average headline. Computing zones from the real measured
# text height removes that risk entirely: longer text simply leaves less
# (but never negative) room for images, rather than any zone ever running
# out of the space it was promised.
MIN_IMAGE_PANEL_H = 220  # floor so images never shrink to an awkward sliver

# Text glyphs (not the backdrop, which is full-width) sit inset from the
# frame edges by this much, so letters never touch the very edge of the
# frame even though the backdrop and images behind them go edge-to-edge.
TEXT_INSET = 32
TEXT_X = TEXT_INSET
TEXT_W = W - 2 * TEXT_INSET

# Rotating variant list, weighted 2:1 toward text_top so text_middle ("mix it
# up" per the sandwiched-between-images look) shows up occasionally rather
# than as often as text_top.
LAYOUT_VARIANTS = ["text_top", "text_top", "text_middle"]


def _zone_bounds(variant: str, text_height: float) -> dict:
    """
    Returns the {text: (top, bottom), image1: (top, bottom), image2: (top,
    bottom)} pixel bounds for a given layout variant and the CALLER'S ACTUAL
    measured text height - not a fixed guess. All 6 values are rounded to
    INTEGERS before returning, once, here - not left as floats for each
    downstream consumer (text position, backdrop position, image panel
    height) to round independently. Two boundaries that are meant to be
    flush (e.g. image1's bottom and text's top) are the exact same float
    value at this point, so rounding them together, the same way, is what
    actually guarantees they land on the same pixel row - rounding them
    separately in different callers (one via round(), another left for
    moviepy's own float-to-pixel handling) was confirmed via pixel sampling
    to reintroduce a ~1px seam even after the zones themselves were already
    flush in floating point.

    Both variants place text as a REAL reserved zone that images are
    computed AROUND, never into - images always end up as complete,
    uninterrupted rectangles; there's no carved hole or overlay involved.

    "text_middle": the text zone sits in the middle, with image1 filling
    everything above it and image2 filling everything below - image1's
    bottom and image2's top land EXACTLY on the text zone's own edges,
    flush, zero gap. "text_top": the text zone is pinned to CONTENT_TOP
    instead of a computed middle position - images fill ALL remaining
    space below it continuously, split evenly between image1/image2 if
    both exist, flush with the text zone's bottom edge, zero gap. No image
    content is ever positioned above the text block in this variant.
    """
    usable = CONTENT_BOTTOM - CONTENT_TOP

    if variant == "text_middle":
        each_image_h = max((usable - text_height) / 2, MIN_IMAGE_PANEL_H)
        img1_top = CONTENT_TOP
        text_top = img1_top + each_image_h
        img2_top = text_top + text_height
        zones = {
            "text": (text_top, text_top + text_height),
            "image1": (img1_top, img1_top + each_image_h),
            "image2": (img2_top, img2_top + each_image_h),
        }
    else:  # "text_top" - text pinned to CONTENT_TOP, images fill the rest
        text_top = CONTENT_TOP
        images_top = text_top + text_height
        each_image_h = max((CONTENT_BOTTOM - images_top) / 2, MIN_IMAGE_PANEL_H)
        img1_top = images_top
        img2_top = img1_top + each_image_h
        zones = {
            "text": (text_top, text_top + text_height),
            "image1": (img1_top, img1_top + each_image_h),
            "image2": (img2_top, img2_top + each_image_h),
        }

    return {key: (round(top), round(bottom)) for key, (top, bottom) in zones.items()}


def _truncate_to_fraction(text: str, fraction: float) -> str:
    """
    Cuts `text` down to roughly `fraction` of its original word count, then
    trims back to the last complete sentence within that cut so it never
    ends mid-thought. Falls back to a hard word-count cut (with a trailing
    "...") only if no sentence-ending punctuation exists within the target
    cut at all (e.g. one long run-on sentence).
    """
    words = text.split()
    target_word_count = max(1, int(len(words) * fraction))
    truncated = " ".join(words[:target_word_count])

    last_sentence_end = max(truncated.rfind(". "), truncated.rfind("! "), truncated.rfind("? "))
    if last_sentence_end > 0:
        return truncated[: last_sentence_end + 1]

    stripped = truncated.rstrip(".!? ")
    if stripped and stripped[-1] not in ".!?":
        return stripped + "..."
    return stripped


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
    # real safety margin - a clipped descender is far worse than a bit of
    # extra breathing room. Empirically, a small pad here (+6) was found to
    # be fully absorbed by a discrepancy between this measuring canvas and
    # the real TextClip's own render: across several real headline/hook
    # examples, the actual rendered ink consistently reached the *exact*
    # last pixel row of a box sized with only +6 - zero effective margin,
    # not the intended buffer. 24px reliably reproduces genuine breathing
    # room across the same test cases.
    return (last_ink_row - pad) + 36


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


BLOB_COLORS = [(255, 71, 87), (255, 165, 2)]  # red, orange - fixed pair used
# for every tile's glow blobs (cycled via i % len(BLOB_COLORS), so with
# NUM_BLOBS=4 that's 2 red + 2 orange), independent of that story's own
# accent color, so every tile consistently shows red/orange orbs rather
# than following the 5-color ACCENT_COLORS rotation.
NUM_BLOBS = 4  # doubled from the original 2, per explicit instruction.
# Blob motion is a bounded back-and-forth wobble (via _triangle_wave, same
# mechanism as the image pan) around each blob's start position, NOT a
# one-way slide - an earlier attempt at "10x faster" simply multiplied the
# old start-to-start+drift distance by 10x, which let blobs drift far
# enough off-frame to hit a real moviepy crash (a broadcast-shape
# ValueError from compositing a clip whose visible region had gone to
# zero) - confirmed via a real failed render, not a hypothetical concern.
# Bounding the motion within BLOB_MOTION_RANGE avoids that entirely while
# still reading as genuinely fast, continuous movement (several full
# wobble cycles across one tile's duration) - and, as a side benefit,
# keeps the anchor_zone-anchored blob lingering near the text zone instead
# of sliding away from it for most of the clip.
BLOB_MOTION_RANGE = 150
BLOB_SPEED_PX_PER_SEC = 90  # ~10x the original implied speed (90px drift
# over a ~10s clip works out to roughly 9px/sec)


def _make_glow_blob(diameter: int, color: tuple[int, int, int], max_opacity: int = 170) -> Image.Image:
    """A large, heavily-blurred soft-edged circle in the given color - an
    ambient light shape, not a recognizable object, so it adds visual depth
    without depicting anything (no risk of resembling a copyrighted
    character/logo, since it's just a blurred gradient blob). max_opacity
    bumped twice now (70 -> 110 -> 170) to make the orbs more prominent, per
    explicit instruction each time that they were still too hard to see."""
    img = Image.new("RGBA", (diameter, diameter), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    inset = diameter * 0.15
    draw.ellipse([inset, inset, diameter - inset, diameter - inset], fill=(*color, max_opacity))
    return img.filter(ImageFilter.GaussianBlur(diameter * 0.16))


def _glow_blob_clips(
    duration: float,
    tmp_dir: str,
    seed: int,
    anchor_zone: tuple[float, float] | None = None,
):
    """NUM_BLOBS large soft glow blobs (alternating red/orange - see
    BLOB_COLORS) that drift slowly across the frame for the tile's
    duration - real per-frame motion (not just a static card) using only
    originally-generated shapes. Positions/speeds are seeded per tile so
    every tile looks a little different but is still reproducible. Count
    doubled from the original 2 to NUM_BLOBS, per explicit instruction.

    anchor_zone: optional (top, bottom) y-range - typically the text zone -
    that must have real, guaranteed blob coverage, for tiles where the text
    backdrop is translucent specifically so this shows through it. Without
    this, every blob starts at a uniformly random position across the WHOLE
    frame height (0-H); confirmed in practice that for a seed whose random
    draw happens to land all blobs far from a given (often much smaller)
    text zone, none ever drifts into it for the tile's whole duration -
    since the per-story seed is fixed (story_number * 10), this isn't rare
    bad luck that varies run to run, it reproduces every single time for
    that story number. When given, blob 0 is deliberately centered on this
    zone instead of the full frame, guaranteeing real overlap; the rest
    stay randomly placed across the whole frame as before, for background
    variety outside the text zone.
    """
    rng = random.Random(seed)
    clips = []
    for i in range(NUM_BLOBS):
        # Diameter range halved from the original 520-780 (50% smaller),
        # per explicit instruction.
        diameter = rng.randint(260, 390)
        blob_path = os.path.join(tmp_dir, f"blob_{seed}_{i}.png")
        _make_glow_blob(diameter, BLOB_COLORS[i % len(BLOB_COLORS)]).save(blob_path)

        if i == 0 and anchor_zone is not None:
            zone_top, zone_bottom = anchor_zone
            zone_center_y = (zone_top + zone_bottom) / 2
            start_x = rng.randint(0, W) - diameter // 2
            start_y = int(zone_center_y - diameter / 2)
        else:
            start_x = rng.randint(-diameter // 2, W - diameter // 2)
            start_y = rng.randint(-diameter // 2, H - diameter // 2)
        phase_x = rng.uniform(0, 2 * BLOB_MOTION_RANGE)
        phase_y = rng.uniform(0, 2 * BLOB_MOTION_RANGE)

        blob_clip = ImageClip(blob_path).with_duration(duration)
        blob_clip = blob_clip.with_position(
            lambda t, sx=start_x, sy=start_y, px=phase_x, py=phase_y: (
                sx - _triangle_wave(t, BLOB_SPEED_PX_PER_SEC, BLOB_MOTION_RANGE, px),
                sy - _triangle_wave(t, BLOB_SPEED_PX_PER_SEC, BLOB_MOTION_RANGE, py),
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


def _find_focal_point(img: Image.Image, grid: int = 8) -> tuple[float, float]:
    """
    Returns (cx, cy) as 0-1 fractions of the image's own width/height,
    biased toward the most visually "busy" region (highest local pixel
    variance) - a lightweight, dependency-free proxy for "the interesting
    part of the image." Not true saliency/face detection (deliberately
    skipped - most source images here are stylized game renders/
    screenshots, not real photos, and face-detection models are unreliable
    on illustrated characters). Good enough to usually avoid centering a
    crop on a flat sky/background when there's a busier subject elsewhere.
    """
    gray = np.array(img.convert("L"))
    h, w = gray.shape
    bh, bw = max(h // grid, 1), max(w // grid, 1)

    best_score = -1.0
    best_cx, best_cy = 0.5, 0.5
    for gy in range(grid):
        for gx in range(grid):
            block = gray[gy * bh : (gy + 1) * bh, gx * bw : (gx + 1) * bw]
            if block.size == 0:
                continue
            score = float(block.std())
            if score > best_score:
                best_score = score
                best_cx = (gx + 0.5) / grid
                best_cy = (gy + 0.5) / grid
    return best_cx, best_cy


def _oversized_crop(image_path: str, panel_w: int, panel_h: int, headroom: float = 1.15) -> Image.Image:
    """
    Crops a region LARGER than the target panel size (by `headroom`),
    centered on the image's focal point (_find_focal_point), clamped so the
    crop never extends outside the source image's actual bounds. This
    oversized crop is what gets panned across during playback - the pan
    stays roughly centered on whatever was judged interesting rather than
    drifting toward a blank edge.

    Height-first: the crop region uses the source's FULL height, with a
    width chosen so that, once scaled to fill panel_h * headroom, the width
    also reaches panel_w * headroom - cropped off on whichever side(s)
    exceed it, not letterboxed. For an unusually portrait-shaped source
    where that would need more width than the source actually has, falls
    back to using the full WIDTH instead (guaranteed to overshoot the
    height target in that case) - the actual guarantee is "never smaller
    than the target in either dimension," not "always height-first"
    specifically.

    Crops FIRST, in the source image's own (small) coordinate space, and
    resizes only that already-small crop up to the final target size -
    deliberately NOT the other order (resize the whole source first, then
    crop), which would run LANCZOS resampling over the entire source image
    instead of just the crop - confirmed via direct render-time comparison
    to add real, avoidable per-image cost, especially for large source
    photos.
    """
    img = Image.open(image_path).convert("RGB")
    src_w, src_h = img.size

    target_w = panel_w * headroom
    target_h = panel_h * headroom

    cx_frac, cy_frac = _find_focal_point(img)

    crop_w = target_w * src_h / target_h
    if crop_w <= src_w:
        crop_h = src_h
    else:
        crop_w = src_w  # portrait source: width-first instead
        crop_h = target_h * src_w / target_w

    cx_px, cy_px = cx_frac * src_w, cy_frac * src_h
    left = max(0, min(cx_px - crop_w / 2, src_w - crop_w))
    top = max(0, min(cy_px - crop_h / 2, src_h - crop_h))

    box = (int(left), int(top), int(left + crop_w), int(top + crop_h))
    oversized_w = int(round(target_w))
    oversized_h = int(round(target_h))
    result = img.crop(box).resize((oversized_w, oversized_h), Image.LANCZOS)

    assert result.size[0] >= oversized_w and result.size[1] >= oversized_h, (
        f"_oversized_crop produced {result.size}, smaller than the required "
        f"({oversized_w},{oversized_h}) - this would leave a gap in the panel."
    )
    return result


PAN_SPEED_PX_PER_SEC = 28  # constant visual pan speed, independent of clip
# duration. The original version picked one random start/end point and
# linearly interpolated across the WHOLE clip duration - for a short tile
# that looked fine, but a merged multi-part narration (slides_per_story=1)
# can easily run 10-20+ seconds, and spreading the same fixed pixel range
# across that much more time made the pan nearly imperceptible (confirmed
# by user report: "panning too slowly"). A constant px/sec speed keeps the
# same visually-apparent pace no matter how long the tile plays.


def _triangle_wave(t: float, speed: float, max_range: float, phase: float) -> float:
    """
    Bounces a value back and forth within [0, max_range] at a constant
    `speed` (px/sec), offset by `phase` px along the same wave at t=0 - a
    continuous back-and-forth drift rather than a single start-to-end
    traverse, so the pan keeps moving for the tile's entire duration
    instead of reaching its endpoint early and going static.
    """
    if max_range <= 0:
        return 0.0
    period = 2 * max_range
    pos = (t * speed + phase) % period
    return pos if pos <= max_range else period - pos


def _build_panning_image_clip(
    image_path: str,
    panel_w: int,
    panel_h: int,
    accent: tuple[int, int, int],
    duration: float,
    tmp_dir: str,
    cache_key: str,
    headroom: float = 1.15,
    seed: int = 0,
):
    """
    A panning image clip fit exactly to (panel_w, panel_h): crops an
    oversized, focal-point-centered region (_oversized_crop), then animates
    its position within a fixed-size viewport at a constant PAN_SPEED_PX_PER_SEC
    (see _triangle_wave) so a different portion is visible throughout the
    clip's duration - Ken Burns motion, applied on top of the smart crop so
    the pan stays roughly centered on whatever was judged interesting
    rather than drifting toward a blank edge.

    No border is drawn around the panel itself - per explicit instruction,
    images no longer get a full outline; the only place the accent color
    (BORDER_COLOR) appears is a thin divider drawn separately at the seam
    between two adjacent images (see _build_two_image_layers), not around
    each panel's own outer edges.
    """
    oversized = _oversized_crop(image_path, panel_w, panel_h, headroom)
    oversized_path = os.path.join(tmp_dir, f"pan_src_{cache_key}.jpg")
    oversized.save(oversized_path, quality=90)

    ow, oh = oversized.size
    max_dx, max_dy = ow - panel_w, oh - panel_h

    rng = random.Random(seed)
    phase_x = rng.uniform(0, 2 * max_dx) if max_dx > 0 else 0
    phase_y = rng.uniform(0, 2 * max_dy) if max_dy > 0 else 0

    raw_clip = ImageClip(oversized_path).with_duration(duration)

    def pan_position(t):
        x = _triangle_wave(t, PAN_SPEED_PX_PER_SEC, max_dx, phase_x)
        y = _triangle_wave(t, PAN_SPEED_PX_PER_SEC, max_dy, phase_y)
        return (-x, -y)

    panning_clip = raw_clip.with_position(pan_position)
    return CompositeVideoClip([panning_clip], size=(panel_w, panel_h)).with_duration(duration)


BACKDROP_ALPHA = 120  # translucent (~47% opacity): the drifting glow blobs
# (see _glow_blob_clips) show through the text backdrop instead of being
# hidden behind a flat black box. A past round tried this and reverted it
# because real IMAGE content was bleeding through in odd slivers - but that
# was a symptom of the *old* zone design, where images were carved with a
# hole cut for the text and could end up adjacent to it in a way that read
# as visual noise. Under the CURRENT zone design (see _zone_bounds),
# image1/image2 are computed to fill the space AROUND the text zone and
# never extend into it at all - the only things ever drawn under the text
# zone are the plain gradient background and the soft blurred blobs
# (already earlier in every tile's layer stack), so there is no image
# content this can expose, by construction, not just in practice.


def _build_text_backdrop(height: int) -> Image.Image:
    """
    A translucent black bar spanning the FULL frame width (W), edge-to-edge
    like the images behind it - sized to EXACTLY the text zone's own
    height, no more. Sits on TOP of the background+blob layers in the
    z-order (images never extend into the text zone - see _zone_bounds),
    letting the drifting glow blobs show through at BACKDROP_ALPHA opacity
    rather than being hidden behind a flat, fully opaque box. An earlier
    version padded this taller than the text zone for "breathing room,"
    but that padding overhung past the zone's own bounds into the adjacent
    image panel, painting over part of that panel (and its border) with
    solid black - confirmed via pixel sampling to be a real bug ("photo
    doesn't reach the panel's own edges"), not a stylistic choice worth
    keeping; _measure_wrapped_text_height already bakes in its own safety
    margin, so no extra padding is needed here. Deliberately NOT inset
    horizontally either: only the text glyphs drawn on top of this (see
    TEXT_INSET/TEXT_X) get their own left/right margin, so the black box
    itself tiles flush against the images exactly like the images tile
    against each other and the frame edges - no rounded corners either,
    for the same edge-to-edge-coverage reason as
    _build_panning_image_clip's border.
    """
    return Image.new("RGBA", (W, height), (0, 0, 0, BACKDROP_ALPHA))


def _image_source_regions(
    image_path_1: str | None, image_path_2: str | None, zones: dict
) -> list[tuple[float, float, str]]:
    """
    Returns the (top, bottom, path) vertical spans image content should
    cover, straight from the zone math.

    Both paths present: each source photo covers its own zone1/zone2 span.

    Only one path present: rather than leaving a dead gap where the missing
    image would have been (or, worse, showing the same photo pasted into
    both slots side by side - confirmed to read as an obvious, lazy-looking
    repeat when a story only has one genuine source image), that single
    photo is given the FULL combined region (zone1's top through zone2's
    bottom) as ONE span, so it reads as one deliberately large image.

    Neither present: returns an empty list.
    """
    if image_path_1 and image_path_2:
        return [
            (zones["image1"][0], zones["image1"][1], image_path_1),
            (zones["image2"][0], zones["image2"][1], image_path_2),
        ]

    single_path = image_path_1 or image_path_2
    if not single_path:
        return []
    return [(zones["image1"][0], zones["image2"][1], single_path)]


def _build_two_image_layers(
    image_path_1: str | None,
    image_path_2: str | None,
    zones: dict,
    accent: tuple[int, int, int],
    duration: float,
    tmp_dir: str,
    cache_key: str,
    seed_base: int = 0,
):
    """
    Builds positioned, slowly-panning image clips covering image1/image2's
    zones, full frame width (W), edge-to-edge with each other and the frame
    boundary - each a single COMPLETE, uninterrupted rectangle (Ken
    Burns-style pan happens WITHIN that fixed rectangle, via
    _build_panning_image_clip - it never changes these zone bounds). The
    text zone is never inside an image's own zone (see _zone_bounds - both
    variants compute images AROUND the text zone, not into it), and the
    opaque text backdrop simply draws on top of whichever image ends up
    behind it in the z-order, the same way any ordinary caption sits on
    top of a photo - no hole is ever cut into an image.

    seed_base: the same per-tile seed already used for this tile's glow
    blobs (story/tile index) - offset per region_idx (image1 vs image2) so
    the two panels in one tile pan differently from each other, while
    staying reproducible.
    """
    regions = _image_source_regions(image_path_1, image_path_2, zones)
    layers = []
    for region_idx, (r_top, r_bottom, path) in enumerate(regions):
        panel_h = int(round(r_bottom - r_top))
        if panel_h <= 0:
            continue
        panel_clip = _build_panning_image_clip(
            path,
            W,
            panel_h,
            accent,
            duration,
            tmp_dir,
            f"{cache_key}_{region_idx}",
            seed=seed_base * 10 + region_idx,
        )
        layers.append(panel_clip.with_position((0, r_top)))

    # Thin divider line in BORDER_COLOR, ONLY at a seam where two images
    # directly touch each other with zero gap (e.g. text_top's continuous
    # 2-image stack) - NOT around each panel's own outer edges (that
    # per-panel border was removed per explicit instruction). text_middle's
    # two images are separated by the text zone, not adjacent to each
    # other, so no divider applies there - there's nothing to mark a seam
    # between.
    DIVIDER_THICKNESS = 3
    for (top_a, bottom_a, _), (top_b, _, _) in zip(regions, regions[1:]):
        if abs(top_b - bottom_a) > 0.5:
            continue  # not directly adjacent - no seam to mark
        seam_y = int(round(bottom_a))
        divider_img = Image.new("RGBA", (W, DIVIDER_THICKNESS), (*BORDER_COLOR, 255))
        divider_path = os.path.join(tmp_dir, f"divider_{cache_key}_{seam_y}.png")
        divider_img.save(divider_path)
        divider_clip = ImageClip(divider_path).with_duration(duration)
        layers.append(divider_clip.with_position((0, seam_y - DIVIDER_THICKNESS // 2)))

    return layers


def _story_badge(text: str, duration: float, accent: tuple[int, int, int]):
    """Small persistent nav marker (e.g. "2/5") anchored to the same corner
    on every tile - a stable orientation cue while the main content below
    it is free to move around. Right-aligned to its own rendered width so
    it never drifts past the safe boundary regardless of digit count.
    Vertically anchored to SAFE_TOP (not a hardcoded pixel value) so it
    never sits in the zone reserved for TikTok's own top UI chrome."""
    clip = TextClip(font=FONT_BOLD, text=text, font_size=40, color=f"rgb{accent}", method="label")
    clip = clip.with_duration(duration)
    x = W - SAFE_RIGHT - clip.size[0]
    return clip.with_position((x, SAFE_TOP))


def _progress_dots(tile_index: int, total_tiles: int, duration: float, accent: tuple[int, int, int]):
    """Tiny filled/hollow dot row showing which of the 3 tiles (per story)
    this is - helps the viewer read 3 screens as one continuing story.
    Sits just below the badge, both anchored relative to SAFE_TOP."""
    dots = "  ".join("\u25cf" if i == tile_index else "\u25cb" for i in range(total_tiles))
    clip = TextClip(font=FONT_REGULAR, text=dots, font_size=26, color=f"rgb{accent}", method="label")
    clip = clip.with_duration(duration)
    x = W - SAFE_RIGHT - clip.size[0]
    return clip.with_position((x, SAFE_TOP + 52))


def _build_title_tile(
    story_number: int,
    total_stories: int,
    headline: str,
    hook: str,
    audio_path: str | None,
    accent: tuple[int, int, int],
    tmp_dir: str,
    template_index: int,
    pad_seconds: float = 0.6,
    image_path: str | None = None,
    image_path_2: str | None = None,
    total_tiles: int = 3,
):
    """The first tile for a story: headline + a short hook line, plus up to
    2 real story images, laid out in one of LAYOUT_VARIANTS ("text_top" or
    "text_middle" - sandwiched between the 2 images). Hook position is
    computed from the headline's actual rendered height (not a hardcoded
    offset), so a headline that wraps to 3 lines can never overlap the hook
    text below it.

    audio_path=None means narration is disabled for this run (see main.py's
    --audio flag) - the tile is silent and its duration is a flat
    NO_AUDIO_DURATION instead of audio length."""
    if audio_path:
        audio = AudioFileClip(audio_path)
        duration = audio.duration + pad_seconds
    else:
        audio = None
        duration = NO_AUDIO_DURATION

    bg_path = os.path.join(tmp_dir, f"bg_{story_number}_title.png")
    _make_gradient_background(accent).save(bg_path)
    background = ImageClip(bg_path).with_duration(duration)

    # Single-page stories (total_tiles==1) always use "text_middle" - text
    # sandwiched between the 2 images - never "text_top", per explicit
    # instruction. Multi-page stories keep the existing rotation for
    # variety across tiles.
    variant = "text_middle" if total_tiles == 1 else LAYOUT_VARIANTS[template_index % len(LAYOUT_VARIANTS)]

    # Build both text clips first (unpositioned) so we know the real combined
    # height before asking for zone positions - this is what makes the
    # layout overflow-safe regardless of how long the text wraps. Height is
    # measured independently (_measure_wrapped_text_height) and fed back in
    # via size=(...) rather than left for TextClip to auto-compute, since its
    # own auto-sizing is what clips the last line in the first place (see
    # that function's docstring).
    wrapped_headline = _wrap_text(headline, FONT_BOLD, 68, TEXT_W)
    headline_h = _measure_wrapped_text_height(wrapped_headline, FONT_BOLD, 68)
    headline_clip = TextClip(
        font=FONT_BOLD,
        text=wrapped_headline,
        font_size=68,
        color="white",
        size=(TEXT_W, headline_h),
        method="label",
        text_align="left",
    ).with_duration(duration)

    wrapped_hook = _wrap_text(hook, FONT_REGULAR, 46, TEXT_W)
    hook_h = _measure_wrapped_text_height(wrapped_hook, FONT_REGULAR, 46)
    hook_clip = TextClip(
        font=FONT_REGULAR,
        text=wrapped_hook,
        font_size=46,
        color=(210, 210, 210),
        size=(TEXT_W, hook_h),
        method="label",
        text_align="left",
    ).with_duration(duration)

    gap = 32
    block_height = headline_h + gap + hook_h
    zones = _zone_bounds(variant, block_height)
    y_start = int(round(zones["text"][0]))

    # Blobs are built AFTER zones (not before) specifically so one can be
    # anchored to the text zone - see _glow_blob_clips's anchor_zone
    # docstring for why this matters.
    blobs = _glow_blob_clips(duration, tmp_dir, seed=story_number * 10, anchor_zone=zones["text"])

    headline_clip = headline_clip.with_position((TEXT_X, y_start))
    hook_clip = hook_clip.with_position((TEXT_X, y_start + headline_h + gap))

    # Full-width OPAQUE backdrop behind the text block, sized EXACTLY to
    # the text zone (flush against the adjacent image panel, same as
    # image1/image2 are flush against the text zone) - NOT padded past the
    # zone's own bounds, since that would overhang into the adjacent
    # panel's own bordered rectangle and paint over it with solid black
    # (confirmed via pixel sampling to be the actual cause of a prior
    # "photo doesn't reach the panel's edges" bug). Both the position AND
    # the height are derived from zones["text"] itself (already rounded to
    # integers by _zone_bounds), NOT from block_height directly - rounding
    # the zone's top and bottom independently can shave a pixel off the
    # true span (e.g. top rounds up, bottom rounds down), so a backdrop
    # built from the original unrounded block_height can end up 1px taller
    # than the zone it's supposed to exactly fill, reintroducing the same
    # 1px seam on whichever image panel is adjacent.
    zone_text_top, zone_text_bottom = zones["text"]
    backdrop_img = _build_text_backdrop(zone_text_bottom - zone_text_top)
    backdrop_path = os.path.join(tmp_dir, f"backdrop_{story_number}_title.png")
    backdrop_img.save(backdrop_path)
    backdrop_clip = (
        ImageClip(backdrop_path).with_duration(duration).with_position((0, zone_text_top))
    )

    layers = [background, *blobs]
    layers.extend(
        _build_two_image_layers(
            image_path, image_path_2, zones, accent, duration, tmp_dir, f"{story_number}_title",
            seed_base=story_number * 10,
        )
    )
    layers.extend([backdrop_clip, headline_clip, hook_clip])
    # Badge + dots are a progress indicator through THIS story's own pages -
    # meaningless (and confusing, implying more parts exist) on a single-page
    # post, so both only render when there's actually more than one page.
    if total_tiles > 1:
        layers.append(_story_badge(f"{story_number}/{total_stories}", duration, accent))
        layers.append(_progress_dots(0, total_tiles, duration, accent))

    segment = CompositeVideoClip(layers, size=(W, H)).with_duration(duration)
    return segment.with_audio(audio) if audio else segment


def _build_detail_tile(
    story_number: int,
    total_stories: int,
    tile_index: int,
    text: str,
    audio_path: str | None,
    accent: tuple[int, int, int],
    tmp_dir: str,
    template_index: int,
    pad_seconds: float = 0.6,
    image_path: str | None = None,
    image_path_2: str | None = None,
    total_tiles: int = 3,
):
    """A follow-up tile for a story: just the descriptive text, plus up to 2
    real story images, laid out in one of LAYOUT_VARIANTS the same way
    _build_title_tile is.

    audio_path=None means narration is disabled for this run (see
    _build_title_tile's docstring) - the tile is silent and its duration
    is a flat NO_AUDIO_DURATION instead of audio length."""
    if audio_path:
        audio = AudioFileClip(audio_path)
        duration = audio.duration + pad_seconds
    else:
        audio = None
        duration = NO_AUDIO_DURATION

    bg_path = os.path.join(tmp_dir, f"bg_{story_number}_detail{tile_index}.png")
    _make_gradient_background(accent).save(bg_path)
    background = ImageClip(bg_path).with_duration(duration)

    variant = LAYOUT_VARIANTS[template_index % len(LAYOUT_VARIANTS)]

    wrapped_body = _wrap_text(text, FONT_REGULAR, 54, TEXT_W)
    body_h = _measure_wrapped_text_height(wrapped_body, FONT_REGULAR, 54)
    body_clip = TextClip(
        font=FONT_REGULAR,
        text=wrapped_body,
        font_size=54,
        color="white",
        size=(TEXT_W, body_h),
        method="label",
        text_align="left",
    ).with_duration(duration)

    zones = _zone_bounds(variant, body_h)
    y_start = int(round(zones["text"][0]))
    # Blobs are built AFTER zones (not before) specifically so one can be
    # anchored to the text zone - see _glow_blob_clips's anchor_zone
    # docstring for why this matters.
    blobs = _glow_blob_clips(
        duration, tmp_dir, seed=story_number * 10 + tile_index, anchor_zone=zones["text"]
    )
    body_clip = body_clip.with_position((TEXT_X, y_start))

    # Full-width OPAQUE backdrop, position AND height both derived from
    # zones["text"] itself (not body_h directly) - see _build_title_tile
    # for why that 1px distinction matters.
    zone_text_top, zone_text_bottom = zones["text"]
    backdrop_img = _build_text_backdrop(zone_text_bottom - zone_text_top)
    backdrop_path = os.path.join(tmp_dir, f"backdrop_{story_number}_detail{tile_index}.png")
    backdrop_img.save(backdrop_path)
    backdrop_clip = (
        ImageClip(backdrop_path).with_duration(duration).with_position((0, zone_text_top))
    )

    layers = [background, *blobs]
    layers.extend(
        _build_two_image_layers(
            image_path, image_path_2, zones, accent, duration, tmp_dir, f"{story_number}_{tile_index}",
            seed_base=story_number * 10 + tile_index,
        )
    )
    layers.extend([backdrop_clip, body_clip])
    # Badge + dots are a progress indicator through THIS story's own pages -
    # meaningless (and confusing, implying more parts exist) on a single-page
    # post, so both only render when there's actually more than one page.
    if total_tiles > 1:
        layers.append(_story_badge(f"{story_number}/{total_stories}", duration, accent))
        layers.append(_progress_dots(tile_index, total_tiles, duration, accent))

    segment = CompositeVideoClip(layers, size=(W, H)).with_duration(duration)
    return segment.with_audio(audio) if audio else segment


def _build_intro_card(title: str, subtitle: str, tmp_dir: str, duration: float = 2.5):
    bg_path = os.path.join(tmp_dir, "bg_intro.png")
    _make_gradient_background(ACCENT_COLORS[0]).save(bg_path)
    background = ImageClip(bg_path).with_duration(duration)
    blobs = _glow_blob_clips(duration, tmp_dir, seed=0)

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
    image_paths: list[tuple[str | None, str | None]] | None = None,
) -> str:
    """
    stories: list of story dicts (needs at least a "title" key)
    parts_lists: one entry per story, each a list of exactly 3 strings:
                 [hook, detail_1, detail_2]
    audio_paths_lists: one entry per story, each a list of exactly 3 audio
                        file paths, matching parts_lists 1:1
    image_paths: one (image_path_1, image_path_2) tuple per story - either
                 element may be None - or None to mean "no images for any
                 story". Each tile shows up to 2 real story images.
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
        img1, img2 = image_paths[i] if image_paths else (None, None)

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
                image_path=img1,
                image_path_2=img2,
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
                    image_path=img1,
                    image_path_2=img2,
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
    audio_paths_lists: list[list[str | None]],
    output_dir: str,
    tmp_dir: str,
    filename_prefix: str = "story",
    slides_per_story: int = 1,
    image_paths: list[tuple[str | None, str | None]] | None = None,
) -> list[str]:
    """
    Builds one standalone vertical video per story. slides_per_story
    controls how many screens each story's video has - see
    group_parts_for_slides() for the exact grouping rule. audio_paths_lists
    must already be grouped to match: one audio path per SLIDE (i.e.
    len(audio_paths_lists[i]) == len(group_parts_for_slides(parts_lists[i],
    slides_per_story))), not one per original script part. Entries may be
    None (narration disabled - see main.py's --audio flag), in which case
    that tile is silent with a flat NO_AUDIO_DURATION instead of an
    audio-driven one; either every entry is None or none are, never mixed,
    since main.py generates audio for a whole run or not at all.

    image_paths: one (image_path_1, image_path_2) tuple per story - either
                 element may be None - or None to mean "no images for any
                 story". Each tile shows up to 2 real story images.

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
        img1, img2 = image_paths[i] if image_paths else (None, None)

        clips = []
        for slide_idx, (group, audio_path) in enumerate(zip(groups, audio_paths)):
            if 0 in group:
                # Normally only the hook (parts[0]) shows on screen for a
                # merged group - the rest of the group's parts are only
                # ever heard via narration (audio_path was generated from
                # ALL of them combined). With no audio at all, that
                # narration doesn't exist, so showing only the hook would
                # silently drop the rest of the story - but showing every
                # part in full ran too long for a read-only single-page
                # video, so it's cut to about half its combined length
                # (trimmed to a full sentence, never mid-thought) instead
                # of the complete, uncut text.
                if audio_path:
                    title_hook_text = parts[0]
                else:
                    full_text = " ".join(parts[j] for j in group)
                    title_hook_text = _truncate_to_fraction(full_text, 0.5)
                clips.append(
                    _build_title_tile(
                        i + 1,
                        total,
                        story["title"],
                        title_hook_text,
                        audio_path,
                        accent,
                        tmp_dir,
                        title_template_counter,
                        image_path=img1,
                        image_path_2=img2,
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
                        image_path=img1,
                        image_path_2=img2,
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
