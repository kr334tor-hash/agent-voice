# acquire-voice

Fully guided pipeline to clone any voice and wire it into the TARS voice assistant.
Mirrors the exact process used to build the original TARS voice: download → extract → Demucs separation → A/B test → configure → test generation.

## Pipeline script
All audio processing is handled by:
`voice_pipeline.py` (in the repo root)

Run it via the venv Python:
- Windows: `venv\Scripts\python.exe voice_pipeline.py <command>`
- Mac/Linux: `venv/bin/python voice_pipeline.py <command>`

Commands: `search`, `download`, `extract`, `separate`, `play`, `abtest`, `update-config`

---

## When invoked — ask these questions in order

### Step 1 — What voice?
Ask: "What voice do you want to clone? Give me a character name, a person, or describe it."

### Step 2 — Source
Ask: "Do you want me to search for the voice, or will you provide the file or URL yourself?"

- **"Search for it"** → ask: "What should I search for? (e.g. 'Morgan Freeman monologue', 'TARS Interstellar voice')"
  Then run: `voice_pipeline.py search "<query>" "samples/downloads"`
  The script lists up to 5 results with titles — ask the user to pick one, or it defaults to the first.

- **"I have a URL"** → run: `voice_pipeline.py download <url> "samples/downloads"`

- **"I have a local file"** → go straight to Step 3 with the provided path

### Step 3 — Extract audio
Run: `voice_pipeline.py extract <source_file> "samples/raw_voice.wav"`

Tell the user: "Audio extracted. Running background separation now — this takes a few minutes."

### Step 3b — Character line extraction (optional but recommended)
Ask: "Do you want me to isolate only the lines spoken by this specific character? I will play each detected speech segment and you tell me which ones sound like them."

If yes → run: `voice_pipeline.py clip-character "samples/raw_voice.wav" "<character name>" "samples/clips"`

This plays each detected speech segment one by one. User presses Y to keep or N to skip. Approved clips are concatenated into a single clean reference file with auto-generated reference text. Skip to Step 6 (reference text already generated).

If no → continue to Step 4.

### Step 4 — Demucs separation
Run: `voice_pipeline.py separate "samples/raw_voice.wav" "samples/demucs_out"`

This produces `vocals_clean.wav` with background music/noise removed.

### Step 5 — A/B test (exactly as we did it)
Say: "I will play the original first, then the cleaned version. Tell me which sounds clearer."

Run: `voice_pipeline.py abtest "samples/raw_voice.wav" "samples/demucs_out/vocals_clean.wav"`

Wait for user choice. The script prints `CHOSEN: <path>` — use that path as `REF_CLIP`.

### Step 6 — Reference text
Ask: "Now I need the exact words being spoken in that clip — type them out verbatim. This is what the model uses to align the voice."

Store the answer as `REF_TEXT`.

### Step 7 — Update config
Run: `voice_pipeline.py update-config "<REF_CLIP>" "<REF_TEXT>" "tars_control.py"`

### Step 8 — Test generation
Say: "Config updated. Restart the TARS Control panel, wait for all systems go, toggle Voice ON, and say something. Tell me if it sounds like the voice you wanted."

If user says it sounds off: go back to Step 5 and try different clip length or source.
If user says it sounds good: done.

---

## Notes
- Demucs model: `htdemucs_ft` (4-stem, best vocal isolation)
- F5-TTS works best with 6–30 seconds of clean reference audio
- Reference text must be verbatim — not a description, the actual words spoken
- If background noise is mild, the original (pre-Demucs) may actually sound better — always A/B test
- Soundboard.com audio is compressed and works poorly — use YouTube instead
- ffmpeg must be installed and on your PATH (see README setup)

## Critical patch (must reapply if f5-tts is reinstalled)
`venv/Lib/site-packages/f5_tts/model/__init__.py` — remove the Trainer import:
```python
# Remove this line:
from f5_tts.model.trainer import Trainer
```
Reason: Trainer → datasets → pandas causes an ACCESS_VIOLATION crash on Windows at import.
