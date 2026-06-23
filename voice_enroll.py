"""
voice_enroll.py — Voice Registration & Identification
Enroll a speaker by name, then identify who is speaking in real-time.

Usage:
  python voice_enroll.py enroll <name>        # record and save a voice profile
  python voice_enroll.py list                 # show enrolled voices
  python voice_enroll.py delete <name>        # remove a voice profile
  python voice_enroll.py test                 # identify who is speaking (live test)
"""

import os
import sys
import json
import time
import numpy as np
import sounddevice as sd
import soundfile as sf

VOICES_DIR   = os.path.join(os.path.dirname(__file__), "voices")
SAMPLE_RATE  = 16000
ENROLL_SECS  = 25    # seconds of speech to record for enrollment

ENROLL_PHRASES = [
    "Tell me the odds — and be completely honest.",
    "What's our current position and estimated arrival time?",
    "Engaging flight plan. Stand by for the launch sequence.",
    "The quick brown fox jumps over the lazy dog.",
    "Houston, do you read me? We are go for departure.",
    "I can operate on sixty percent honesty, if that helps.",
    "Preparing interstellar coordinates. All systems are nominal.",
]


def get_encoder():
    import os, torch
    os.environ['SPEECHBRAIN_LOCAL_FETCH_STRATEGY'] = 'copy'
    from speechbrain.inference.speaker import EncoderClassifier
    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=os.path.join(os.path.dirname(__file__), "models", "spkrec")
    )
    return classifier


def embed(encoder, audio, sr=16000):
    """Generate speaker embedding from audio array."""
    import torch
    tensor = torch.tensor(audio).unsqueeze(0).float()
    with torch.no_grad():
        embedding = encoder.encode_batch(tensor)
    return embedding.squeeze().numpy()


def enroll(name):
    os.makedirs(VOICES_DIR, exist_ok=True)
    print(f"\nEnrolling: {name}")
    print(f"\nRead these phrases aloud during the {ENROLL_SECS}s recording:")
    for i, phrase in enumerate(ENROLL_PHRASES, 1):
        print(f"  {i}. {phrase}")
    print("\nRecording in 3...")
    time.sleep(1)
    print("2...")
    time.sleep(1)
    print("1...")
    time.sleep(1)
    print("GO — speak now!\n")

    audio = sd.rec(int(ENROLL_SECS * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                   channels=1, dtype="float32")
    sd.wait()
    print("Done recording.")

    audio = audio[:, 0]
    encoder = get_encoder()
    embedding = embed(encoder, audio)

    profile_path = os.path.join(VOICES_DIR, f"{name.lower().replace(' ', '_')}.npy")
    np.save(profile_path, embedding)
    print(f"Voice profile saved: {profile_path}")


def list_voices():
    if not os.path.exists(VOICES_DIR):
        print("No voices enrolled yet.")
        return []
    profiles = [f for f in os.listdir(VOICES_DIR) if f.endswith(".npy")]
    if not profiles:
        print("No voices enrolled yet.")
        return []
    print("\nEnrolled voices:")
    names = []
    for p in profiles:
        name = p.replace(".npy", "").replace("_", " ").title()
        print(f"  - {name}")
        names.append(name)
    return names


def delete(name):
    path = os.path.join(VOICES_DIR, f"{name.lower().replace(' ', '_')}.npy")
    if os.path.exists(path):
        os.remove(path)
        print(f"Deleted: {name}")
    else:
        print(f"Not found: {name}")


def load_profiles():
    """Load all enrolled voice profiles. Returns {name: embedding}."""
    profiles = {}
    if not os.path.exists(VOICES_DIR):
        return profiles
    for f in os.listdir(VOICES_DIR):
        if f.endswith(".npy"):
            name = f.replace(".npy", "").replace("_", " ").title()
            profiles[name] = np.load(os.path.join(VOICES_DIR, f))
    return profiles


def identify(audio, profiles, encoder, threshold=0.75):
    """
    Identify speaker from audio. Returns (name, confidence) or ("Unknown", score).
    Uses cosine similarity between audio embedding and enrolled profiles.
    Only re-checks when called — caller decides when to check.
    """
    if not profiles:
        return "Unknown", 0.0

    embedding = embed(encoder, audio)
    best_name  = "Unknown"
    best_score = 0.0

    for name, profile in profiles.items():
        # Cosine similarity
        score = float(np.dot(embedding, profile) /
                      (np.linalg.norm(embedding) * np.linalg.norm(profile)))
        if score > best_score:
            best_score = score
            best_name  = name

    if best_score < threshold:
        return "Unknown", best_score
    return best_name, best_score


def test():
    """Live identification test — speak and see who it thinks you are."""
    import torch
    from faster_whisper import WhisperModel

    profiles = load_profiles()
    if not profiles:
        print("No voices enrolled. Run: python voice_enroll.py enroll <name>")
        return

    encoder = get_encoder()
    print(f"\nLoaded {len(profiles)} voice(s): {', '.join(profiles.keys())}")
    print("Loading Whisper...")
    whisper = WhisperModel("small.en", device="cuda", compute_type="float16")
    print("Speak — Ctrl+C to stop.\n")

    CHUNK = 512
    buffer = []
    silence_count = 0
    recording = False

    vad_model, _ = torch.hub.load("snakers4/silero-vad", "silero_vad",
                                   force_reload=False, onnx=False)

    def callback(indata, frames, time_info, status):
        nonlocal silence_count, recording
        chunk = indata[:, 0].astype(np.float32).copy()
        tensor = torch.from_numpy(chunk)
        prob = vad_model(tensor, SAMPLE_RATE).item()

        if prob > 0.45:
            if not recording:
                recording = True
            silence_count = 0
            buffer.append(chunk)
        elif recording:
            silence_count += 1
            buffer.append(chunk)
            if silence_count >= 60:
                recording = False
                audio = np.concatenate(buffer[:])
                buffer.clear()
                silence_count = 0

                speaker, conf = identify(audio, profiles, encoder)
                segs, _ = whisper.transcribe(audio, beam_size=1, language="en")
                text = " ".join(s.text.strip() for s in segs).strip()
                if text:
                    print(f"{speaker} ({conf:.0%}): {text}")

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, blocksize=CHUNK,
                        dtype="float32", callback=callback):
        try:
            while True:
                sd.sleep(100)
        except KeyboardInterrupt:
            print("\nStopped.")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "list"

    if cmd == "enroll":
        enroll(sys.argv[2])
    elif cmd == "list":
        list_voices()
    elif cmd == "delete":
        delete(sys.argv[2])
    elif cmd == "test":
        test()
    else:
        print(__doc__)
