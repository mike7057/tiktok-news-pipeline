"""
Generates one secondary visual per story using Google's Gemini image
generation API ("Nano Banana" family). Requires a GEMINI_API_KEY - free to
create (no credit card needed just to get a key), but whether image
generation itself is covered by a free quota has changed multiple times in
2026 and may require billing enabled. Check https://ai.google.dev/gemini-api/docs/pricing
directly rather than assuming a specific free quota.

Failures here (quota, billing not enabled, safety filter, network error)
are treated as "skip the image for this story," not a fatal pipeline error -
the video still gets made, just without a secondary visual for that one
story.
"""
import base64
import os

import requests
from dotenv import load_dotenv

load_dotenv()

GEMINI_MODEL = "gemini-2.5-flash-image"
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"

# Appended to every prompt - keeps generated images clear of copyright/likeness
# risk regardless of what the per-story prompt itself says.
SAFETY_SUFFIX = (
    " Abstract, stylized illustration only - no readable text, no logos or "
    "brand marks, no depiction of real named public figures, no recognizable "
    "copyrighted characters or franchise-specific character designs."
)


def generate_image(prompt: str, output_path: str, api_key: str | None = None) -> str | None:
    """
    Calls Gemini's image generation API and saves the first returned image to
    output_path. Returns output_path on success, or None on any failure
    (logged to stderr) - callers should treat None as "no image for this
    story" and continue, not crash the run.
    """
    import sys

    key = api_key or os.environ.get("GEMINI_API_KEY")
    if not key:
        print("    (no GEMINI_API_KEY set - skipping image generation)", file=sys.stderr)
        return None

    payload = {
        "contents": [{"parts": [{"text": prompt.strip() + SAFETY_SUFFIX}]}],
        "generationConfig": {"responseModalities": ["TEXT", "IMAGE"]},
    }

    try:
        response = requests.post(GEMINI_URL, params={"key": key}, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()

        for part in data["candidates"][0]["content"]["parts"]:
            if "inlineData" in part:
                image_bytes = base64.b64decode(part["inlineData"]["data"])
                with open(output_path, "wb") as f:
                    f.write(image_bytes)
                return output_path

        print(f"    (Gemini returned no image for prompt: {prompt[:80]!r})", file=sys.stderr)
        return None

    except Exception as e:
        print(f"    (image generation failed: {e})", file=sys.stderr)
        return None


if __name__ == "__main__":
    import sys
    result = generate_image(
        sys.argv[1] if len(sys.argv) > 1 else "A dramatic abstract gaming-themed illustration",
        sys.argv[2] if len(sys.argv) > 2 else "test_image.png",
    )
    print(f"Result: {result}")
