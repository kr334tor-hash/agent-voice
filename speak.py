"""
speak.py
Called by TARS after each response when voice mode is active.
Usage: python speak.py  (reads text from tars_speak_input.txt)
"""

import os, sys, tempfile, sounddevice as sd, soundfile as sf

_HERE = os.path.dirname(os.path.abspath(__file__))

import shutil as _shutil
_ffmpeg = _shutil.which("ffmpeg")
if _ffmpeg:
    os.environ["PATH"] = os.path.dirname(_ffmpeg) + os.pathsep + os.environ.get("PATH", "")

REF_CLIP = os.path.join(_HERE, "samples", "reference.wav")
REF_TEXT = "Replace this with the exact words spoken in your reference clip."
INPUT_FILE = os.path.join(_HERE, "tars_input.txt")
OUT_WAV    = os.path.join(_HERE, "output", "tars_auto.wav")

if not os.path.exists(INPUT_FILE):
    print("No input file found.")
    sys.exit(0)

with open(INPUT_FILE, "r", encoding="utf-8-sig") as f:
    text = f.read().strip()

if not text:
    sys.exit(0)

# Trim very long responses — F5-TTS works best under ~200 words
words = text.split()
if len(words) > 180:
    text = " ".join(words[:180]) + "..."

print(f"[TARS] Speaking {len(words)} words...")

from f5_tts.api import F5TTS
tts = F5TTS()
wav, sr, _ = tts.infer(
    ref_file=REF_CLIP,
    ref_text=REF_TEXT,
    gen_text=text,
    remove_silence=True,
)
os.makedirs(os.path.dirname(OUT_WAV), exist_ok=True)
sf.write(OUT_WAV, wav, sr)
data, samplerate = sf.read(OUT_WAV)
sd.play(data, samplerate)
sd.wait()
print("[TARS] Done.")
