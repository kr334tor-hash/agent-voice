"""
voice_pipeline.py — Acquire Voice Pipeline
Called by the /acquire-voice skill to handle audio extraction and Demucs separation.

Usage:
  python voice_pipeline.py download <url> <out_dir>
  python voice_pipeline.py extract <video_file> <out_wav>
  python voice_pipeline.py separate <wav_file> <out_dir>
  python voice_pipeline.py play <wav_file>
  python voice_pipeline.py abtest <file_a> <file_b>
"""

import sys
import os
import subprocess
import shutil
import sounddevice as sd
import soundfile as sf
import numpy as np

_ffmpeg_which = shutil.which("ffmpeg")
if not _ffmpeg_which:
    raise RuntimeError(
        "ffmpeg not found. Install it and ensure it's on your PATH.\n"
        "  Windows: winget install Gyan.FFmpeg\n"
        "  Mac:     brew install ffmpeg\n"
        "  Linux:   sudo apt install ffmpeg"
    )
FFMPEG = _ffmpeg_which

VENV_PY = os.path.join(os.path.dirname(__file__), "venv", "Scripts", "python.exe")


def search(query, out_dir, max_results=5):
    """Search YouTube for query, list results, let user pick one to download."""
    os.makedirs(out_dir, exist_ok=True)
    search_query = f"ytsearch{max_results}:{query}"

    # Get titles and URLs without downloading
    result = subprocess.run(
        ["yt-dlp", "--get-title", "--get-url", "--no-playlist",
         "-f", "bestaudio", search_query],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"ERROR: {result.stderr}")
        sys.exit(1)

    lines = [l.strip() for l in result.stdout.strip().splitlines() if l.strip()]
    # yt-dlp alternates: title, url, title, url ...
    results = []
    for i in range(0, len(lines) - 1, 2):
        results.append((lines[i], lines[i + 1]))

    if not results:
        print("ERROR: No results found.")
        sys.exit(1)

    print(f"\nFound {len(results)} results for '{query}':")
    for i, (title, url) in enumerate(results, 1):
        print(f"  [{i}] {title}")
        print(f"      {url}")

    choice = input(f"\nPick a number (1-{len(results)}), or press Enter to use [1]: ").strip()
    idx = (int(choice) - 1) if choice.isdigit() else 0
    chosen_url = results[idx][1]
    print(f"\nDownloading: {results[idx][0]}")
    return download(chosen_url, out_dir)


def download(url, out_dir):
    """Download audio from URL using yt-dlp."""
    os.makedirs(out_dir, exist_ok=True)
    out_template = os.path.join(out_dir, "%(title)s.%(ext)s")
    result = subprocess.run(
        ["yt-dlp", "-f", "bestaudio", "--extract-audio",
         "--audio-format", "wav", "-o", out_template, url],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"ERROR: {result.stderr}")
        sys.exit(1)
    for f in os.listdir(out_dir):
        if f.endswith(".wav"):
            path = os.path.join(out_dir, f)
            print(f"DOWNLOADED: {path}")
            return path
    print("ERROR: No wav found after download.")
    sys.exit(1)


def extract(video_file, out_wav):
    """Extract audio from any video/audio file to 22050Hz mono wav."""
    result = subprocess.run([
        FFMPEG, "-y", "-i", video_file,
        "-ac", "1", "-ar", "22050", out_wav
    ], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ERROR: {result.stderr}")
        sys.exit(1)
    print(f"EXTRACTED: {out_wav}")


def separate(wav_file, out_dir):
    """Run Demucs htdemucs_ft to isolate vocals from the wav file."""
    os.makedirs(out_dir, exist_ok=True)
    result = subprocess.run([
        VENV_PY, "-m", "demucs",
        "-n", "htdemucs_ft",
        "--two-stems", "vocals",
        "-o", out_dir,
        wav_file
    ], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ERROR: {result.stderr}")
        sys.exit(1)

    # Find vocals.wav in Demucs output
    base = os.path.splitext(os.path.basename(wav_file))[0]
    vocals_path = None
    for root, dirs, files in os.walk(out_dir):
        for f in files:
            if f == "vocals.wav" and base in root:
                vocals_path = os.path.join(root, f)
                break

    if not vocals_path:
        print("ERROR: vocals.wav not found after separation.")
        sys.exit(1)

    # Convert to 22050Hz mono
    final_path = os.path.join(out_dir, "vocals_clean.wav")
    subprocess.run([
        FFMPEG, "-y", "-i", vocals_path,
        "-ac", "1", "-ar", "22050", final_path
    ], capture_output=True)

    print(f"VOCALS: {final_path}")
    return final_path


def play(wav_file, label=""):
    """Play a wav file and wait for it to finish."""
    data, sr = sf.read(wav_file)
    if label:
        print(f"\nPlaying: {label}")
    sd.play(data, sr)
    sd.wait()


def abtest(file_a, file_b):
    """Play A then B, ask user which is clearer."""
    print("\n--- A/B Test ---")
    print("Playing option A...")
    play(file_a, "A")
    input("Press Enter to play B...")
    play(file_b, "B")
    choice = input("\nWhich sounded clearer? (A/B): ").strip().upper()
    if choice == "A":
        print(f"CHOSEN: {file_a}")
        return file_a
    else:
        print(f"CHOSEN: {file_b}")
        return file_b


def clip_character(audio_file, character_name, out_dir, min_duration=2.0, max_duration=15.0):
    """
    Use Whisper to detect speech segments, then let user pick which ones
    belong to the target character. Saves approved clips to out_dir.
    Best clips: 6-30s, clean, no overlapping voices.
    """
    import soundfile as sf
    from faster_whisper import WhisperModel

    os.makedirs(out_dir, exist_ok=True)
    print(f"\nLoading Whisper to detect speech in: {os.path.basename(audio_file)}")
    model = WhisperModel("small.en", device="cuda", compute_type="float16")
    segments, _ = model.transcribe(audio_file, beam_size=1, word_timestamps=False)

    data, sr = sf.read(audio_file)
    approved = []

    print(f"\nReview each segment — keep lines that sound like {character_name}.")
    print("Press Y to keep, N to skip, Q to stop reviewing.\n")

    for i, seg in enumerate(segments):
        duration = seg.end - seg.start
        if duration < min_duration or duration > max_duration:
            continue

        print(f"[{i:03d}] {seg.start:.1f}s → {seg.end:.1f}s ({duration:.1f}s)")
        print(f"      \"{seg.text.strip()}\"")

        # Play the segment
        start_sample = int(seg.start * sr)
        end_sample   = int(seg.end * sr)
        chunk = data[start_sample:end_sample]
        sd.play(chunk, sr)
        sd.wait()

        choice = input("      Keep? (Y/N/Q): ").strip().upper()
        if choice == "Q":
            break
        if choice == "Y":
            out_path = os.path.join(out_dir, f"clip_{i:03d}.wav")
            sf.write(out_path, chunk, sr)
            approved.append((out_path, seg.text.strip()))
            print(f"      Saved: {out_path}")

    if not approved:
        print("No clips approved.")
        return None

    # Concatenate approved clips into one reference file
    all_audio = np.concatenate([sf.read(p)[0] for p, _ in approved])
    ref_text  = " ".join(t for _, t in approved)
    ref_path  = os.path.join(out_dir, f"{character_name.lower().replace(' ', '_')}_reference.wav")
    sf.write(ref_path, all_audio, sr)

    print(f"\nReference saved: {ref_path}")
    print(f"Reference text:  {ref_text}")
    print(f"REFERENCE: {ref_path}")
    print(f"REF_TEXT: {ref_text}")
    return ref_path, ref_text


def update_tars_config(ref_clip, ref_text, target_file):
    """Patch REF_CLIP and REF_TEXT in a tars_control.py file."""
    with open(target_file, "r", encoding="utf-8") as f:
        content = f.read()

    import re
    content = re.sub(
        r'REF_CLIP\s*=\s*r?"[^"]*"',
        f'REF_CLIP   = r"{ref_clip}"',
        content
    )
    content = re.sub(
        r'REF_TEXT\s*=\s*"[^"]*"',
        f'REF_TEXT   = "{ref_text}"',
        content
    )

    with open(target_file, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"UPDATED: {target_file}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""

    if cmd == "search":
        search(sys.argv[2], sys.argv[3])

    elif cmd == "download":
        download(sys.argv[2], sys.argv[3])

    elif cmd == "extract":
        extract(sys.argv[2], sys.argv[3])

    elif cmd == "separate":
        separate(sys.argv[2], sys.argv[3])

    elif cmd == "play":
        play(sys.argv[2])

    elif cmd == "abtest":
        abtest(sys.argv[2], sys.argv[3])

    elif cmd == "clip-character":
        clip_character(sys.argv[2], sys.argv[3], sys.argv[4])

    elif cmd == "update-config":
        update_tars_config(sys.argv[2], sys.argv[3], sys.argv[4])

    else:
        print(__doc__)
