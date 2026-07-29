# Background music

Drop license-clear **instrumental** mp3/wav/m4a files directly in this
folder to enable background music - one file per track, any length. The
pipeline automatically loops short tracks or trims long ones to fit each
video, so there's no need to pre-edit anything.

This folder ships empty. Background music is a no-op (silently skipped,
not an error) until you actually add at least one track here.

## Where to find license-clear tracks

Instrumental (no vocals - they'd compete with narration) tracks from
sources that offer clear, redistributable licensing, e.g.:

- [Pixabay Music](https://pixabay.com/music/) - free, no attribution
  required (but check the license on each individual track page)
- [Incompetech](https://incompetech.com/music/royalty-free/) (Kevin
  MacLeod) - free under Creative Commons, attribution required
- Any other library where you've personally confirmed the license permits
  this use (commercial video, redistribution via TikTok, etc.)

**Verify the license yourself before adding a track.** The pipeline has
no way to check licensing automatically - that's a human judgment call per
track, not something the code assumes or validates.

## Track record-keeping

Consider keeping a `CREDITS.md` alongside your tracks in this folder,
listing for each file: the track name, source URL, license, and whether
attribution is required (and where you've put it, e.g. in the video
description). This is just a suggestion for your own reference - nothing
in the pipeline reads or requires this file.
