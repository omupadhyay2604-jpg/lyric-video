---
name: glass-to-neon-lyrics
description: Make an emotional 16:9 lyric video (1920x1080, 30 fps) that changes mood with the song - red Apple-Music-style glass lyrics card over a wall of giant red words that cracks and shatters, full-screen clips with lyrics in the empty space (starry field, hill, clouds), a key line cut out as giant letters with fire inside, and a pink neon ending where words drop, splash and reflect in rain-wet water. Use this skill whenever the user runs /glass-to-neon-lyrics or asks for a lyric video, kinetic typography video, "Arz Kiya Hai" style video, glass card lyrics, or neon rain lyrics from an MP3 (or clip), lyrics, cover art and a few clips or photos - even if they do not name the skill.
argument-hint: [song.mp3] [lyrics.txt] [cover.jpg] [4-6 clips/photos, each with a mood note]
---

# Glass to Neon Lyrics

Turn one song into a lyric video that moves through four moods: **glass card -> shatter -> clips with lyrics in the empty space -> key line with fire inside giant letters -> pink neon rain ending**.

Everything visual is drawn in code: an HTML canvas rendered frame by frame in headless Chrome, encoded with ffmpeg. Only the user's own cover art, clips and photos are used as pictures. No stock footage, no image generators, no AI video, so the look is identical on every run.

## Inputs

Take them from `$ARGUMENTS`, attached files, or pasted text:

| Input | Notes |
|---|---|
| Song | MP3 (or a clip of it). The video length equals the audio length. |
| Lyrics | Text exactly as the user wrote it, line breaks included. This is what is shown on screen. |
| Cover art | Image for the card. |
| 4-6 clips / photos | Each has a mood note (in the message or filename), e.g. "starry red night, lonely", "sunny hill, hope", "man against clouds, longing", "fire, the peak". |
| Title and artists | Read ID3 tags with `ffprobe`; if missing, ask once. |

If something is missing or you cannot tell which file is which, ask one short question. Do not invent inputs.

## Rules for every run

1. **Draw everything in code.** No stock footage, no generated images. The only pictures are the user's cover, clips and photos.
2. **Never guess timings.** Every word is timed by forced alignment with torchaudio `MMS_FA` on the separated vocal track. The lyrics are shown exactly as written (spelling, case, punctuation, line breaks); only the alignment copy is normalised.
3. **Ask before overwriting** any existing file or folder (outputs, plan, alignment, frames, segments). List what exists and ask: overwrite, keep, or use a new name. Reusing a cached alignment is a choice to offer, not a default.
4. **Show a contact sheet of stills before the full render** and wait for the user's go-ahead.
5. **Deterministic frames.** `renderFrame(t)` must be a pure function of `t` and the timeline. No `Date.now`, no `Math.random` (use a seeded PRNG such as mulberry32), no particle state that accumulates frame to frame. Particles, rain, ripples, sparks and stars are closed-form functions of `(seed, t)` so parallel workers can render any frame in any order.
6. **Outputs:** a 1920x1080, 30 fps MP4 with the song's audio, plus a 720p copy under 20 MB for WhatsApp.
7. Tell the user before installing anything. Install Python packages in a venv inside the work folder.

## Work layout

Create `./glass-to-neon-work/` next to the inputs (ask first if it exists):

```
glass-to-neon-work/
  input/            copies/links of mp3, lyrics.txt, cover, clips
  audio/            vocals.wav, vocals16k.wav, analysis.json
  align/            words.json
  plan/             plan.json, timeline.json
  clips/<id>/       %05d.jpg frame sequences (video) or still.jpg (photo)
  engine/           scene.html, engine.js, fonts/, render.mjs
  segments/         seg_000.mp4 ...
  contact_sheet.png
out/                <slug>_glass-to-neon.mp4, <slug>_whatsapp.mp4
```

## Step 1 - Analyse the song

### 1a. Setup check
- `ffmpeg`, `ffprobe`, Chrome/Chromium (or `npx puppeteer browsers install chrome`), Node 18+, Python 3.9+.
- venv packages: `demucs torch torchaudio librosa soundfile numpy pillow uroman` (uroman only if the lyrics are not in Latin script).
- Fonts: **Inter** (800 and 900) and **Great Vibes**. Look with `fc-list`; otherwise `npm i @fontsource/inter @fontsource/great-vibes` and embed the woff2 files as base64 `@font-face` in `scene.html`, then `await document.fonts.load(...)` for each weight before rendering. If offline and fonts are missing, tell the user and ask. If the lyrics are not Latin script (Hindi, Urdu...), Inter and Great Vibes will not cover them: ask whether to display a Roman transliteration, or pick a matching Noto font and say the script word will use it too.

### 1b. Separate the vocals
```bash
python -m demucs --two-stems=vocals -o audio/demucs input/song.mp3
ffmpeg -i audio/demucs/htdemucs/song/vocals.wav -ac 1 -ar 16000 audio/vocals16k.wav
```
Keep the full-quality `vocals.wav` too.

### 1c. Force-align every word (MMS_FA)
Tokenise the lyrics into display words, keeping the line index and original text of each. Build an alignment copy of each word: lowercase, romanise with `uroman` if needed, keep only characters in the MMS dictionary (a-z and `'`). Strip section labels like `[Chorus]` from alignment (and tell the user). A display word with no alignable letters (a dash, a number) gets a time interpolated between its neighbours and is flagged `interpolated: true` in the report.

```python
import torch, torchaudio
bundle = torchaudio.pipelines.MMS_FA
dev = "cuda" if torch.cuda.is_available() else "cpu"
model = bundle.get_model().to(dev)
dictionary = bundle.get_dict(star=None)          # '-' is blank = 0
wav, sr = torchaudio.load("audio/vocals16k.wav")
wav = torchaudio.functional.resample(wav.mean(0, keepdim=True), sr, bundle.sample_rate)
with torch.inference_mode():
    emission, _ = model(wav.to(dev))
tokens = [dictionary[c] for w in align_words for c in w]   # one flat list, word order = lyric order
targets = torch.tensor([tokens], dtype=torch.int32, device=dev)
aligned, scores = torchaudio.functional.forced_align(emission, targets, blank=0)
spans = torchaudio.functional.merge_tokens(aligned[0], scores[0].exp())
sec_per_frame = wav.size(1) / emission.size(1) / bundle.sample_rate
# group spans back into words by letter count: start = first span.start, end = last span.end
```
If memory runs out on a long song, align in overlapping windows of about 60 s and stitch, never by guessing.

Write `align/words.json`: `[{i, line, text, start, end, score, interpolated}]`. Then check: starts are non-decreasing, `end > start`, nothing past the audio length. Report the mean score and list words with score below about 0.5. If many are low, say so (usually wrong lyrics or a language mismatch) and ask before continuing. Trust the aligner; never "fix" times by hand.

### 1d. Beats and energy (librosa)
On the full mix: `beat_track` for beat times and tempo; an RMS / onset-strength curve resampled to 30 fps (0-1 normalised); a rough downbeat guess (every 4th beat from the strongest of the first four). Save to `audio/analysis.json`. Beats are used to snap cuts, flashes, the shatter hit and bounces, and the energy curve drives subtle pulses (wall brightness, bloom, glow).

## Step 2 - Read the lyrics and plan the parts

Read the lyrics for meaning, not only length. Decide where the emotional shifts are, then split the song, using the aligned times, into:

| Part | Typical share | What goes there |
|---|---|---|
| 1 Glass card | 25-35% | the opening verse(s): intimate, setting the scene. Ends right after a line ends. |
| Crack + shatter | ~2.5 s | covers the end of part 1 and flows into clip 1. |
| 2 Clips | 30-40% | the build: longing, memory, hope, each clip owns a few lines (6-25 s each). |
| Key line | 6-15 s | the single line that carries the song's thesis or hook, the emotional peak. |
| 3 Pink neon | 15-25% (at least ~12 s) | the final stanza / outro; ends on the last sung word. |

For a short clip of a song, shrink proportionally (card at least ~8 s). Put every boundary in a gap between lines, then move it to the nearest beat within +/-0.4 s without cutting through a sung word.

Look at the user's clips and photos (view a 3x3 grid of stills for each video, the photo itself for stills) and match each clip to lines using the user's mood notes plus what you see. **Use every clip only once** in part 2 and the key line (the fire or brightest clip is the key-line clip; if no clip suits, use procedural fire and say so). Part 3 then re-uses all the clips, but as a pink duotone, so it still looks like a new scene.

For each clip decide and record:
- the **empty space** for text: `{x, y, w, h, align}` in 1920x1080 pixels: above the person, below the horizon, in open sky. **Never over a face or the main subject.** Verify by looking at stills.
- a crop **focus point** (important for portrait clips) and a start offset into the clip.
- per lyric line: the **small row** (lead-in words), the **big row** (main phrase) and the **script word** (the single most emotional word, usually a key noun or the last word). A one-word line becomes script only.

Save `plan/plan.json` and **tell the user the plan** as a short table: part, time range, lines, clip, mood. Then continue (the contact sheet is the real checkpoint).

Build `plan/timeline.json` from the plan, words, beats and clips: every part with start/end, every line with word timings, each transition with type (`whiteFlash` | `redBurn`), time and beat snap.

## Step 3 - Build the engine

Write `engine/scene.html` + `engine/engine.js`: one 1920x1080 canvas (layered canvases or DOM for the 3D card are fine as long as the screenshot captures everything), loading `timeline.json`, fonts, cover and clip frames, and exposing:

```js
window.renderFrame = async (n) => { const t = n / 30; /* draw everything for time t */ };
```

Clip frames: extract video clips to JPEG sequences at 30 fps, cover-cropped around the focus point at about 2112x1188 (headroom for the push-in): `ffmpeg -i clip.mp4 -vf "fps=30,scale=2112:1188:force_original_aspect_ratio=increase,crop=2112:1188:..." -q:v 3 clips/<id>/%05d.jpg`. Map local time to a frame index; if the clip is shorter than its slot, ping-pong (forward, then backward), never hard-loop. Photos are drawn from one image. Load frames on demand with a small cache and `await img.decode()` before drawing.

Final pass on **every** frame: film grain (a pre-made noise tile at a seeded random offset per frame, about 6-8% overlay) and a soft vignette. Colours: wall and card reds `#4a070c`-`#8f1219`, highlight `#ff4b57`, script red `#e0202e`, neon pink `#ff2e93` with white-pink core `#ffd6ec`.

### Part 1 - Glass card
- **Wall:** the lyric words (unique words, repeated to fill) set in Inter 900 caps at 220-300 px in deep red, low contrast, in rows that scroll slowly in alternating directions. Slight brightness breathing from the energy curve.
- **Card:** a rounded 16:9-friendly card (about 1100x700) of dark red frosted glass like the Apple Music lyrics screen: blur a copy of the wall behind it (`ctx.filter = 'blur(28px)'`, clipped to the rounded rect), tint it deep red, add a thin lighter edge, an inner top highlight, and fine noise. Tilt it slightly in 3D (about 6-10 deg on Y and 2-3 deg on X, with a very slow sway). Use a WebGL textured quad, CSS 3D on a canvas element, or strip-sliced perspective: your choice, but it must render the same in headless Chrome.
- **Card header:** cover art (rounded, small), song title, artists, top-left; lyrics below, left-aligned, bold, large.
- **Lyrics:** the current line is large. Each **word fills from dim to bright exactly from its `start` to its `end`** (a left-to-right wipe with a soft edge, minimum 0.12 s). Words already sung stay bright, words to come stay dim. The next line waits **blurred** (about 6 px blur, lower alpha) below. On a line change the lines glide up with an ease-out and **a soft diagonal shine sweeps across the card** (about 0.8 s).
- Before the first word, show the card with the title and a breathing glow.

### Crack and shatter (end of part 1)
- **Crack (about 1.4 s):** from an impact point, thin bright light lines grow across the card as seeded branching polylines (growth driven by `t`), with a white-red glow that gets stronger with the energy curve. Land the first crack on a beat.
- **Shatter (about 0.9 s):** on a beat, split the card (frozen as a texture) into shards using a jittered grid triangulated into triangles (no library needed). Each shard flies **toward the camera**: scale up, spin, drift away from the impact point, motion trail, fading late. A **red bloom** (radial `lighter` overlay and a red-to-white flash) peaks at the start of the shatter and washes into the first clip, which is already under it.

### Part 2 - Clips with lyrics in the empty space
- Each clip is full screen in its **natural colours**, with a slow push-in (scale 1.00 to about 1.08, tiny drift). A very soft local darkening behind text is allowed for legibility, but keep the colours natural.
- **Text block** inside that clip's empty space, all word by word on the voice (each word fades and rises over about 0.25 s at its `start`):
  1. small tracked caps row (Inter 600, wide letter-spacing, about 28-34 px, white at 85%);
  2. one big caps row (**Inter 800**, auto-fit to the box, white);
  3. one word in **red Great Vibes script** that **writes itself on**: a left-to-right reveal with a slightly slanted soft edge (and a tiny glow at the pen point) over the word's duration, at least 0.6 s, so it still reads as writing for short words.
  The block leaves about 0.3 s after the last word of its line and then fades.
- **Still photos (e.g. a starry field)** must be alive: twinkling stars (from seeded points plus the photo's brightest pixels, each with its own sine and phase), 3-5 shooting stars at seeded times with a bright head and fading tail (never during a key word), and slow drifting light motes or glints (`lighter` composite).
- **Transitions** between shots: a **soft white flash** (peak 50-65% white, about 0.3 s) or a **red burn** (red radial overlay, `screen`/`color-dodge`, about 0.4 s) placed on the beat; the new shot is revealed under the peak. Alternate them, choosing red burn for the darker moments.

### Key line
- On black, set the key line in **Inter 900 caps**, stacked in 1-3 rows to fit about 90% of the width, and **cut the words out**: draw the clip full screen, then cover it with black that has the text knocked out (`destination-out`), or draw black first and the clip with `source-atop`. The fire (or other bright clip) is visible **only inside the letters**, with a faint warm glow just outside the edges, and it is shifted by a slow parallax so it seems to burn.
- Letters appear word by word on the voice.
- After the last word, **grow the letters** (exponential scale around the centre of a letter with a big counter, so the black strips thin out) until the clip fills the screen. Finish with a short fade of any remaining black so it is certain the picture is full, then a white flash into part 3.
- With no fire clip, build procedural fire inside the letters: layered noise columns rising, drawn with `lighter` in black-red-orange-yellow-white.

### Part 3 - Pink neon ending
- **Backdrop:** all clips in rotation (crossfade about 0.6 s, 4-8 s each, in the order of the film), each in a **pink duotone**: grayscale, then map shadows `#1a0014` to mid `#ff2e93` to highlights `#ffc2e0` (for example `grayscale` filter, then `multiply`/`color` pink layers), then a vertical gradient that gets **darker toward the bottom**.
- **Words:** each lyric line (or half-line if long) is laid out centred in 1-2 rows above the water line (about 70% of the height). The words **drop in one by one** at their `start`, falling from above with gravity, with a small damped **bounce** (2 bounces) and **flicker on like a neon sign** (seeded on/off pattern over about 0.35 s, then steady with a faint hum). Draw them as neon: white-pink core, 2-3 `shadowBlur` layers of hot pink, plus a thin outer glow. Lines clear with a quick flicker-off before the next line.
- **Landing:** when a word touches down it makes a **splash ripple** (2-3 expanding, fading ellipses on the floor), a few **droplets** thrown up in parabolas, and a small light kick on the floor.
- **Wet floor:** the bottom 28% mirrors the scene above (backdrop + words), flipped, at about 35% alpha with a slight blur, a fading gradient, and a horizontal-strip sine **distortion that is stronger near ripples**. Subtle specular streaks.
- **Rain:** light diagonal streaks (seeded, thin, low alpha), small splashes on the floor, and **a few drops on a glass plane** in front (small beads with a bright rim, now and then one slides down leaving a thin trail).
- **Last word:** it lands with a big splash, then **hold the frame** (only the rain, ripples and neon hum keep moving, very softly) and **fade to black** together with the audio fade.

## Step 4 - Contact sheet (before the full render)

Render stills with the final engine at about 18 key times and tile them into `contact_sheet.png` (4 columns, each labelled with the time and part, via ffmpeg `drawtext` or Pillow):

card with first words filling, line-change shine, a blurred next line, first crack, shatter mid-way, bloom, each clip's text block (at least one with script writing on), a star-field frame with a shooting star, key line letters with fire, the letters growing, a neon word mid-drop, a splash with reflection, rain on glass, the last word hold, the fade.

**View `contact_sheet.png` yourself first** and fix obvious problems: text over a face, clipped text, illegible contrast, a wrong fill timing. Then show it to the user with the plan, and **stop and ask for a go-ahead** (or changes: swap clip order, move a text block, change the key line). Re-render the sheet after changes. Do not start the full render until they agree.

## Step 5 - Full render

`render.mjs` (Puppeteer, headless Chrome, `deviceScaleFactor: 1`, viewport 1920x1080, args `--hide-scrollbars --enable-unsafe-swiftshader --use-gl=angle`):
- total frames = `round(duration * 30)`; split into contiguous chunks across `min(8, cores/2)` workers, each with its own page;
- per frame: `await page.evaluate(n => window.renderFrame(n), n)`, then capture the canvas (`canvas.toDataURL('image/jpeg', 0.95)` or a PNG screenshot) and pipe it to that worker's ffmpeg: `ffmpeg -f image2pipe -framerate 30 -i - -c:v libx264 -crf 14 -preset fast -pix_fmt yuv420p -g 30 segments/seg_XXX.mp4`;
- print progress (frames done, fps, ETA); skip segments that already exist and are complete only if the user chose to keep them (rule 3);
- concat the segments with the concat demuxer (`-c copy`).

Check that frame counts match, and take a few frames from the segments to confirm they are not black or duplicated.

## Step 6 - Encode

Main file, with the song's audio fading in at the start and out at the end (short in, longer out):
```bash
ffmpeg -i video_all.mp4 -i input/song.mp3 -map 0:v -map 1:a -t <duration> \
  -af "afade=t=in:st=0:d=0.6,afade=t=out:st=<duration-2.0>:d=2.0" \
  -c:v libx264 -crf 15 -preset slow -pix_fmt yuv420p -r 30 \
  -c:a aac -b:a 320k -movflags +faststart out/<slug>_glass-to-neon.mp4
```
If the song is a clip, use its full length. Keep the audio otherwise unprocessed (no loudness change, no EQ).

WhatsApp copy, 1280x720, under 20 MB (aim for about 17.5 MB): compute the video bitrate as `17.5e6*8/duration/1000 - 96` kbit/s, then two-pass x264:
```bash
ffmpeg -y -i main.mp4 -vf "scale=1280:720:flags=lanczos,hqdn3d=1.5:1.5:6:6" -c:v libx264 -preset slow -b:v <vkbps>k -maxrate <1.3*vkbps>k -bufsize <2*vkbps>k -pass 1 -an -f null /dev/null
ffmpeg -i main.mp4 -vf "scale=1280:720:flags=lanczos,hqdn3d=1.5:1.5:6:6" -c:v libx264 -preset slow -b:v <vkbps>k -maxrate <1.3*vkbps>k -bufsize <2*vkbps>k -pass 2 -c:a aac -b:a 96k -pix_fmt yuv420p -movflags +faststart out/<slug>_whatsapp.mp4
```
(`hqdn3d` softens the grain so it compresses well.) If the file is 20 MB or more, lower the bitrate by 10% and redo, until it fits.

Verify both files with `ffprobe`: size 1920x1080 and 1280x720, 30 fps, duration within 0.05 s of the audio, and the sizes. Ask before overwriting existing outputs (rule 3).

## Final message to the user

Keep it short:
- the two output paths, the durations and file sizes;
- the part timeline (card, shatter, each clip with its lines, key line, neon ending);
- alignment quality (mean score, any low-score or interpolated words);
- anything you changed or could not do (fonts, a missing fire clip, a long-lyric split) and an offer to adjust clips, key line or colours and re-render only what changed.
