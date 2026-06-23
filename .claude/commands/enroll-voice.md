# enroll-voice

Register a speaker's voice so TARS can identify who is talking in real-time.
Uses SpeechBrain ECAPA-TDNN voice embeddings — same concept as Shazam's audio fingerprinting but for speaker identity.

## How it works
- **Enroll phase:** record 25 seconds of someone speaking → creates a voice embedding (a compact mathematical fingerprint of their voice) → saved to `voices/<name>.npy`
- **Recognition phase:** smart matching — identifies the speaker on first utterance, then only re-checks if the voice signature drifts (speaker changed). No redundant matching mid-conversation.
- **Labeling:** transcriptions in the TARS Control log show `Sati: [text]` instead of `You: [text]`

## Script
`voice_enroll.py` (in the repo root)

Run via venv:
- Windows: `venv\Scripts\python.exe voice_enroll.py <command>`
- Mac/Linux: `venv/bin/python voice_enroll.py <command>`

## IMPORTANT — SpeechBrain conflict
SpeechBrain conflicts with F5-TTS at the Python process level via the `k2_fsa` lazy module.
**Before running voice_enroll.py:** ensure `tars_control.py` is closed, then install SpeechBrain:
```
pip install speechbrain
```
**After enrollment:** uninstall SpeechBrain before starting tars_control.py again:
```
pip uninstall speechbrain -y
```

## When invoked — steps to follow

### Step 1 — Check existing profiles
Run: `voice_enroll.py list`
Show the user who is already enrolled.

### Step 2 — Ask what to do
"Do you want to enroll a new voice, delete one, or run a live identification test?"

- **Enroll** → ask: "What name should I use for this voice?"
  Then run: `voice_enroll.py enroll "<name>"`
  The script counts down and records 25 seconds. Tell the user to read the phrases shown on screen aloud during the countdown.
  After saving, ask: "Do you want to enroll another voice?"

- **Delete** → ask which name, then run: `voice_enroll.py delete "<name>"`

- **Test** → run: `voice_enroll.py test`
  This opens a live mic session. Each detected utterance is transcribed and labeled with the identified speaker name and confidence. Press Ctrl+C to stop.

### Step 3 — Restart TARS Control
After enrolling, tell the user: "Restart the TARS Control panel. It will automatically load your voice profiles on startup and label each speaker in the log."

## Notes
- 25 seconds gives a solid embedding — read all 7 phrases during enrollment
- Speak in your natural voice during enrollment, not slowly or artificially clearly
- Confidence threshold is 75% — below that it shows "Unknown"
- Re-check threshold is 82% — if current speaker confidence stays above this, it skips re-identification (efficient for long conversations)
- Voice profiles survive restarts — enrolled once, recognized forever until deleted
