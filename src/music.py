"""
Optional background music for assembled videos, sourced from a curated
local folder of instrumental tracks the user provides (see
assets/music/README.md) - this module ships with NO actual audio files and
has no way to fetch any automatically. It's a no-op (silently skipped, not
an error) until the user manually drops license-clear mp3/wav/m4a files
into assets/music/.

Track selection is a deterministic rotation keyed by a seed, not
random.choice - consistent with how blob/pan motion elsewhere in this
pipeline (see assemble_video.py's _triangle_wave usage) favors seeded
reproducibility over unseeded randomness, so it's easy to reason about
which track played for a given run/story rather than that being
unpredictable run to run.

v1 is intentionally simple: one constant volume for the whole track, no
dynamic ducking synced to narration segments - a reasonable later
enhancement, out of scope here.
"""

import os

from moviepy import AudioFileClip, CompositeAudioClip, afx

# Resolved relative to this file, not the caller's cwd - the project runs
# from both `src/` locally and CI, and this must work from either.
MUSIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets", "music")

_MUSIC_EXTENSIONS = (".mp3", ".wav", ".m4a")

# Never fade longer than this many seconds, regardless of MUSIC_FADE_SECONDS
# below - see add_background_music's fade-length clamp.
MUSIC_FADE_SECONDS = 1.5


def _list_music_tracks() -> list[str]:
    """Every .mp3/.wav/.m4a file directly inside MUSIC_DIR, sorted for a
    stable, predictable rotation order. Returns an empty list - never
    raises - if the folder doesn't exist or has no audio files yet; this
    feature must be a no-op until the user actually adds tracks."""
    if not os.path.isdir(MUSIC_DIR):
        return []
    return sorted(
        os.path.join(MUSIC_DIR, name)
        for name in os.listdir(MUSIC_DIR)
        if name.lower().endswith(_MUSIC_EXTENSIONS)
    )


def _pick_music_track(seed: int) -> str | None:
    """Deterministic rotation through the available tracks keyed by `seed`
    (e.g. a story index or run counter) - NOT random.choice, so which
    track plays for a given seed is reproducible and easy to reason about.
    Returns None if no tracks exist."""
    tracks = _list_music_tracks()
    if not tracks:
        return None
    return tracks[seed % len(tracks)]


def add_background_music(clip, tmp_dir: str, volume: float = 0.15, seed: int = 0):
    """
    Returns `clip` with a looping/trimmed background music track mixed
    into its audio, picked deterministically by `seed` (see
    _pick_music_track). If no tracks are available in MUSIC_DIR at all,
    returns `clip` completely UNCHANGED and skips silently - the same
    "optional media, not a fatal error" pattern used throughout this
    codebase (see generate_image.py, fetch_image.py) for anything that
    depends on content the user may not have provided yet.

    - The track is looped or trimmed (afx.AudioLoop(duration=clip.duration))
      to fit `clip`'s exact duration, whichever it actually needs - confirmed
      via AudioLoop's own implementation: it concatenates enough copies of
      the track to cover `duration` (so a short track loops), then always
      truncates to exactly `duration` (so a long track gets trimmed too,
      since a single untouched copy already exceeds the target).
    - Volume is scaled down (afx.MultiplyVolume) to sit well under narration
      by default (0.15 = 15% of the SOURCE track's own volume, not
      narration's - the right absolute level still depends on how the
      source file itself was mastered).
    - A short fade in/out (afx.AudioFadeIn/AudioFadeOut) avoids an abrupt
      start/stop at the clip's edges - clamped to a quarter of the clip's
      own duration so it can't exceed the clip on something unusually short.
    - If `clip.audio` is already set (narration was on), the two are mixed
      via CompositeAudioClip - narration is left untouched, music plays
      underneath it. If `clip.audio` is None (narration disabled - see
      main.py's --audio flag), the music becomes the entire audio track.

    `tmp_dir` is accepted for interface consistency with the rest of this
    codebase's media-fetching functions (tmp_dir, seed, ...) but isn't
    currently used - AudioFileClip reads directly from MUSIC_DIR and no
    intermediate file needs to be written.
    """
    track_path = _pick_music_track(seed)
    if not track_path:
        return clip

    duration = clip.duration
    fade = min(MUSIC_FADE_SECONDS, duration / 4)

    music = AudioFileClip(track_path).with_effects(
        [
            afx.AudioLoop(duration=duration),
            afx.MultiplyVolume(volume),
            afx.AudioFadeIn(fade),
            afx.AudioFadeOut(fade),
        ]
    )

    combined_audio = CompositeAudioClip([clip.audio, music]) if clip.audio is not None else music
    return clip.with_audio(combined_audio)
