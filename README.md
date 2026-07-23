# Daily Top-5 News Video Pipeline

Automatically generates a ~60-90 second vertical (1080x1920) video every day
covering the top 5 news stories, narrated with TTS — ready to upload to
TikTok. Runs for free on GitHub Actions; the only cost is a few cents/day
in Claude API usage.

## How it works

1. **Fetch** — pulls today's top headlines from Google News RSS (free, no
   API key, no rate limit).
2. **Summarize** — Claude (Haiku) rewrites each headline + snippet into a
   short, original 2-sentence voiceover line (never copies source text).
3. **Narrate** — `edge-tts` renders each line to speech (free, no API key).
4. **Assemble** — a Python script builds a numbered background card per
   story and stitches everything into one MP4 with `moviepy` + `ffmpeg`.
5. **Runs daily** via a GitHub Actions scheduled workflow, which uploads
   the finished video + a script/sources text file as a downloadable
   artifact each morning.

Posting to TikTok itself is a manual step (TikTok's public API for
creator self-posting is restricted) — you download the artifact and
upload it from your phone or TikTok Studio.

## One-time setup

### 1. Push this folder to a new GitHub repo
```bash
cd tiktok-news-pipeline
git init
git add .
git commit -m "Initial pipeline"
gh repo create your-repo-name --private --source=. --push
# (or create the repo on github.com and `git remote add origin ...` + push)
```

### 2. Add your Anthropic API key as a repo secret
GitHub repo → **Settings → Secrets and variables → Actions → New repository secret**
- Name: `ANTHROPIC_API_KEY`
- Value: your key from https://console.anthropic.com

### 3. Enable the workflow
Go to the **Actions** tab in your repo — GitHub Actions is enabled by
default for new repos, so `daily_news.yml` will already be picked up.
It runs automatically at 12:00 UTC daily, or you can click **Run workflow**
to trigger it manually any time (useful for testing).

### 4. Get your video
After a run finishes (Actions tab → the run → bottom of the page), download
the `daily-news-video-N` artifact — it contains the `.mp4` and a
`_script.txt` with headlines/sources/what was said, handy for writing your
caption before you post.

## Running it locally (optional, for testing)

```bash
pip install -r requirements.txt
# also needs ffmpeg installed locally: `sudo apt install ffmpeg` / `brew install ffmpeg`

cp .env.example .env   # then fill in your real key
export ANTHROPIC_API_KEY=sk-ant-...
cd src
python main.py --count 5
```

Output lands in `output/` at the repo root.

## Customizing

- **Topic filter**: `python main.py --topic technology` (also: business,
  sports, entertainment, science, health, world). Edit `TOPIC_FEEDS` in
  `src/main.py` to add more.
- **Story count**: `--count 3` for a shorter video.
- **Voice**: change `VOICE` in `src/generate_audio.py` — run
  `edge-tts --list-voices` to see all options.
- **Visual style**: colors, fonts, and layout are all in
  `src/assemble_video.py` (`ACCENT_COLORS`, `FONT_BOLD`, position tuples).
  Everything is generated on the fly with Pillow, so there's no stock
  footage/image API dependency to pay for.
- **Posting schedule**: edit the `cron:` line in
  `.github/workflows/daily_news.yml` (cron times are always UTC).

## Cost breakdown

| Piece                | Cost                                    |
|-----------------------|------------------------------------------|
| GitHub Actions compute | Free (well under the 2,000 free min/mo) |
| Google News RSS        | Free, no key                            |
| Claude API (Haiku)     | Roughly a few cents/day for 5 short summaries |
| edge-tts               | Free, no key                            |
| Storage (artifact)     | Free, auto-deleted after 14 days        |

Total: realistically **a few cents to a couple dollars a month**, driven
almost entirely by Claude API usage.
