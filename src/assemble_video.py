"""
Assembles narrated story segments into one vertical (1080x1920) TikTok-ready
video: a title card, one segment per story (numbered background + headline +
script text + narration audio), and an outro card.

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

# One accent color per story slot, cycled if there are more than 5 stories
ACCENT_COLORS = [
    (255, 71, 87),    # red
    (255, 165, 2),    # orange
    (46, 213, 115),   # green
    (30, 144, 255),   # blue
    (162, 89, 255),   # purple
]
BG_TOP = (18, 18, 24)
BG_BOTTOM = (32, 32, 42)


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


def _build_segment(
    index: int,
    number_label: str,
    headline: str,
    script_line: str,
    audio_path: str,
    accent: tuple[int, int, int],
    tmp_dir: str,
    pad_seconds: float = 0.6,
):
    """One story: background + number badge + headline + script text, timed to its audio."""
    audio = AudioFileClip(audio_path)
    duration = audio.duration + pad_seconds

    bg_path = os.path.join(tmp_dir, f"bg_{index}.png")
    _make_gradient_background(accent).save(bg_path)
    background = ImageClip(bg_path).with_duration(duration)

    number_clip = (
        TextClip(
            font=FONT_BOLD,
            text=number_label,
            font_size=140,
            color=f"rgb{accent}",
            size=(W - 160, None),
            method="label",
        )
        .with_duration(duration)
        .with_position((80, 220))
    )

    headline_clip = (
        TextClip(
            font=FONT_BOLD,
            text=headline,
            font_size=64,
            color="white",
            size=(W - 160, None),
            method="caption",
            text_align="left",
        )
        .with_duration(duration)
        .with_position((80, 470))
    )

    script_clip = (
        TextClip(
            font=FONT_REGULAR,
            text=script_line,
            font_size=48,
            color=(220, 220, 220),
            size=(W - 160, None),
            method="caption",
            text_align="left",
        )
        .with_duration(duration)
        .with_position((80, 850))
    )

    segment = CompositeVideoClip(
        [background, number_clip, headline_clip, script_clip], size=(W, H)
    ).with_duration(duration)
    segment = segment.with_audio(audio)

    return segment


def _build_title_card(title: str, subtitle: str, tmp_dir: str, duration: float = 2.5):
    bg_path = os.path.join(tmp_dir, "bg_title.png")
    _make_gradient_background(ACCENT_COLORS[0]).save(bg_path)
    background = ImageClip(bg_path).with_duration(duration)

    title_clip = (
        TextClip(
            font=FONT_BOLD,
            text=title,
            font_size=100,
            color="white",
            size=(W - 120, None),
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
            size=(W - 160, None),
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
    script_lines: list[str],
    audio_paths: list[str],
    output_path: str,
    tmp_dir: str,
    video_title: str = "Top 5 News Today",
    video_subtitle: str = "",
) -> str:
    """
    stories: list of dicts with at least a "title" key
    script_lines: the narration line for each story (same order as stories)
    audio_paths: path to the rendered mp3 for each story (same order)
    """
    os.makedirs(tmp_dir, exist_ok=True)

    clips = [_build_title_card(video_title, video_subtitle, tmp_dir)]

    for i, (story, line, audio_path) in enumerate(zip(stories, script_lines, audio_paths)):
        accent = ACCENT_COLORS[i % len(ACCENT_COLORS)]
        clips.append(
            _build_segment(
                index=i,
                number_label=f"{i + 1}/{len(stories)}",
                headline=story["title"],
                script_line=line,
                audio_path=audio_path,
                accent=accent,
                tmp_dir=tmp_dir,
            )
        )

    final = concatenate_videoclips(clips, method="compose")
    final.write_videofile(
        output_path,
        fps=30,
        codec="libx264",
        audio_codec="aac",
        threads=4,
        preset="medium",
    )

    return output_path
