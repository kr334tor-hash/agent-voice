# Agent Voice

Give any AI agent a cloned voice — fully local, no API keys, no cloud.
Built on F5-TTS, faster-whisper, and Silero VAD.

Originally built to give [Claude Code](https://claude.ai/code) the voice of TARS from *Interstellar*, but works with any voice and any agent that can write to a file.

## What it does

- **Listens** via microphone using Silero VAD (voice activity detection)
- **Transcribes** your speech with faster-whisper
- **Auto-sends** your words to Claude Code (or any target app) via clipboard + pyautogui
- **Speaks responses** in a cloned voice using F5-TTS
- **Clones any voice** via the `/acquire-voice` Claude Code skill — YouTube video → Demucs separation → F5-TTS reference

## Demo

Speak → VAD detects end of sentence → Whisper transcribes → Claude Code receives the text → Claude responds → response auto-plays in the cloned voice.

## Requirements

- Python 3.10–3.12
- CUDA GPU recommended (CPU works but is slow for TTS)
- ffmpeg on your PATH
- [Claude Code](https://claude.ai/code) (for the voice-to-agent loop; the TTS parts work standalone)

## Setup

### 1. Install ffmpeg

**Windows:** `winget install Gyan.FFmpeg`  
**Mac:** `brew install ffmpeg`  
**Linux:** `sudo apt install ffmpeg`

### 2. Create a virtual environment and install dependencies

```bash
python -m venv venv

# Windows
venv\Scripts\pip install -r requirements.txt

# Mac/Linux
venv/bin/pip install -r requirements.txt
```

### 3. Linux dependencies (Linux only)

```bash
sudo apt install python3-tk xclip xdotool
```

- `python3-tk` — required for the tkinter GUI
- `xclip` — required for clipboard (pyperclip)
- `xdotool` — required for auto-sending text to Claude Code on X11

**Wayland note:** Auto-send to Claude Code uses `xdotool` which only works on X11. On Wayland (Ubuntu 22+, modern GNOME), auto-send is not supported — transcribed text is copied to clipboard instead, and you paste it manually. Everything else (TTS, STT, VAD, voice cloning) works fine on Wayland.

### 3b. Fix F5-TTS Windows crash (Windows only)

F5-TTS has a known Windows crash caused by its `Trainer` class importing pandas. Remove the import:

```bash
# Open this file:
venv/Lib/site-packages/f5_tts/model/__init__.py

# Remove or comment out this line:
from f5_tts.model.trainer import Trainer
```

### 4. Clone a voice with /acquire-voice

Open your project in [Claude Code](https://claude.ai/code) and run:

```
/acquire-voice
```

This launches a guided pipeline:
1. You tell it what voice you want (TARS, Morgan Freeman, your own voice, anything)
2. It searches YouTube or takes a URL/local file
3. Downloads audio, runs Demucs vocal separation, lets you A/B test original vs. cleaned
4. You type out the exact words spoken in the reference clip
5. It updates `tars_control.py` with your `REF_CLIP` and `REF_TEXT` automatically

**Best reference audio:** 6–30 seconds of clean speech, minimal background noise.

### 5. Launch the control panel

```bash
# Windows
venv\Scripts\pythonw.exe tars_control.py

# Mac/Linux
venv/bin/python tars_control.py
```

Toggle **MIC ON** to start listening, **VOICE ON** to speak responses.

## Optional: Claude Code Stop Hook (auto-speak responses)

The `tars_stop_hook.py` script fires after every Claude Code response and auto-writes the response text to `tars_input.txt`, which the agent then speaks.

To wire it up, add this to `.claude/settings.local.json` in your project:

```json
{
  "hooks": {
    "Stop": [
      {
        "matcher": "",
        "hooks": [
          {
            "type": "command",
            "command": "python /path/to/agent-voice/tars_stop_hook.py"
          }
        ]
      }
    ]
  }
}
```

Note: this adds latency on every response (F5-TTS synthesis time). Useful for voice-first workflows; skip it for normal coding sessions.

## Project structure

```
tars_control.py        — Tkinter control panel (mic toggle, voice toggle, mic selector)
voice_pipeline.py      — Acquire-voice pipeline (search, download, Demucs, A/B test)
voice_enroll.py        — Speaker enrollment (SpeechBrain ECAPA-TDNN)
speak.py               — Standalone TTS test script
tars_stop_hook.py      — Claude Code Stop hook (auto-speak responses)
samples/               — Put your reference .wav here (gitignored)
.claude/commands/
  acquire-voice.md     — /acquire-voice skill for Claude Code
```

## Stack

| Component | Library |
|-----------|---------|
| TTS | F5-TTS (local voice clone) |
| STT | faster-whisper (`small.en`, beam_size=1) |
| VAD | Silero VAD |
| Voice separation | Demucs `htdemucs_ft` |
| YouTube download | yt-dlp |
| Audio I/O | sounddevice + soundfile |
| GUI | tkinter |
| Speaker ID | SpeechBrain ECAPA-TDNN (optional) |

---

## Support

If this saved you time, feel free to send some sats:

**Bitcoin (on-chain)**
```
bc1q6xd2vh8qsempgtafp8wc7het7h4sxuz4atd5py
```

**Lightning**
```
lno1pgqppmsrse80qf0aara4slvcjxrvu6j2rp5ftmjy4yntlsmsutpkvkt6878s8gyys87lvc5ht84g6splqtktvur0k7mgr2g2xsz9fxdclr6adkdlqgp7yfdkgq0unlpdrlu28z9v4aukaeurxhjh9umsw9kj7674qxdlzdcqx0jztcrn9xc9qeeexx9nzenkg36p5vc75ew9dzn95x0h6050pqgtess6hkupcvuf6vyu86l0ghnack4tg6uqxs4r4zcrg05mg68t9cvu8f0p7zj2jjf88t3c6tlv7d8fdup7rzsrqqezzsk3ckkkz7qlj4w53r75hpjz6s7dsv0jg8h0v33tk3egpe2lz8fz4hhq4tpgs0fu3q8dyqz0p25q3gzs
```

## License

MIT
