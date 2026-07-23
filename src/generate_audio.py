"""
Free text-to-speech via edge-tts (uses Microsoft Edge's online TTS voices,
no API key or account needed). Swap VOICE for any voice from:
    edge-tts --list-voices
Good options for a news-recap tone: en-US-GuyNeural, en-US-AriaNeural,
en-US-ChristopherNeural, en-GB-RyanNeural
"""

import asyncio
import sys

import edge_tts

VOICE = "en-US-GuyNeural"


async def _generate(text: str, output_path: str, voice: str) -> None:
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_path)


def generate_audio(text: str, output_path: str, voice: str = VOICE) -> str:
    """Renders `text` to speech and saves it as an mp3 at `output_path`."""
    asyncio.run(_generate(text, output_path, voice))
    return output_path


if __name__ == "__main__":
    sample = sys.argv[1] if len(sys.argv) > 1 else "This is a test of the news narration voice."
    out = sys.argv[2] if len(sys.argv) > 2 else "test.mp3"
    generate_audio(sample, out)
    print(f"Saved: {out}")
