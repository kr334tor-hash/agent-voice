"""
tars_control.py  —  TARS Voice Control Panel (v2)
Toggles: MIC (listen + transcribe) | VOICE (speak responses)
Mic device selector — switch anytime without restart.

v2 improvements over v1:
  - SILENCE_CHUNKS = 60 (~1.9s — more breathing room)
  - beam_size=1 (faster transcription)
  - remove_silence=False + trailing period + 0.5s silence padding (no cutoff)
  - TTS pre-warm at startup
  - Watchdog-based file watcher (instant detection)
"""

import os, sys, threading, time, queue, traceback
import tkinter as tk
from tkinter import ttk
import numpy as np
import sounddevice as sd
import soundfile as sf
import pyperclip
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))

# Set ffmpeg — install globally or add your ffmpeg/bin to PATH.
# Windows: winget install Gyan.FFmpeg   Mac: brew install ffmpeg   Linux: apt install ffmpeg
import shutil as _shutil
_ffmpeg = _shutil.which("ffmpeg")
if _ffmpeg:
    os.environ["PATH"] = os.path.dirname(_ffmpeg) + os.pathsep + os.environ.get("PATH", "")

# ── Voice reference ─────────────────────────────────────────────────────────
# Run /acquire-voice in Claude Code to clone your chosen voice.
# That skill updates REF_CLIP and REF_TEXT automatically.
REF_CLIP   = os.path.join(_HERE, "samples", "reference.wav")
REF_TEXT   = "Replace this with the exact words spoken in your reference clip."
# ────────────────────────────────────────────────────────────────────────────

INPUT_FILE = os.path.join(_HERE, "tars_input.txt")
OUT_WAV    = os.path.join(_HERE, "output", "tars_auto.wav")
CRASH_LOG  = os.path.join(_HERE, "tars_crash.log")

SAMPLE_RATE    = 16000
CHUNK_SAMPLES  = 512    # Silero VAD minimum for 16kHz
SILENCE_CHUNKS = 60     # ~1.9s silence = end of utterance
MIN_SPEECH     = 6

mic_on        = False
voice_on      = False
speaking      = False
tts           = None
tts_ready     = False
whisper       = None
whisper_ready = False
vad_model     = None
vad_ready     = False
stream        = None
recording_now = False

speak_queue = queue.Queue()
audio_queue = queue.Queue(maxsize=200)
log_queue   = queue.Queue()


def log_crash(context, exc):
    msg = f"[CRASH in {context}] {type(exc).__name__}: {exc}\n{traceback.format_exc()}"
    log_queue.put(f"ERROR: {type(exc).__name__}: {exc}")
    try:
        with open(CRASH_LOG, "a") as f:
            f.write(msg + "\n")
    except Exception:
        pass


def load_models():
    global tts, tts_ready, whisper, whisper_ready, vad_model, vad_ready
    try:
        log_queue.put("Loading Whisper...")
        from faster_whisper import WhisperModel
        whisper = WhisperModel("small.en", device="cuda", compute_type="float16")
        whisper_ready = True
        log_queue.put("Whisper ready.")
    except Exception as e:
        log_crash("load_whisper", e)

    try:
        log_queue.put("Loading Silero VAD...")
        vad_model, _ = torch.hub.load("snakers4/silero-vad", "silero_vad",
                                       force_reload=False, onnx=False)
        vad_ready = True
        log_queue.put("VAD ready.")
    except Exception as e:
        log_crash("load_vad", e)

    try:
        log_queue.put("Loading TARS voice...")
        from f5_tts.api import F5TTS
        tts = F5TTS()
        tts.infer(ref_file=REF_CLIP, ref_text=REF_TEXT,
                  gen_text="Warming up.", remove_silence=False)
        tts_ready = True
        log_queue.put("TARS voice ready. All systems go.")
    except Exception as e:
        log_crash("load_tts", e)


def audio_callback(indata, frames, time_info, status):
    if not mic_on or speaking:
        return
    try:
        chunk = indata[:, 0].astype(np.float32).copy()
        audio_queue.put_nowait(chunk)
    except queue.Full:
        pass
    except Exception:
        pass


def vad_worker():
    global recording_now
    audio_buffer  = []
    silence_count = 0
    speech_chunks = 0

    while True:
        try:
            chunk = audio_queue.get(timeout=1.0)
        except queue.Empty:
            continue

        if not vad_ready or not mic_on:
            continue

        try:
            tensor    = torch.from_numpy(chunk)
            prob      = vad_model(tensor, SAMPLE_RATE).item()
            is_speech = prob > 0.45
        except Exception as e:
            log_crash("vad", e)
            continue

        if is_speech:
            if not recording_now:
                recording_now = True
                speech_chunks = 0
            silence_count = 0
            speech_chunks += 1
            audio_buffer.append(chunk)
        elif recording_now:
            silence_count += 1
            audio_buffer.append(chunk)
            if silence_count >= SILENCE_CHUNKS:
                recording_now = False
                if speech_chunks >= MIN_SPEECH and whisper_ready:
                    audio = np.concatenate(audio_buffer)
                    threading.Thread(target=transcribe, args=(audio,), daemon=True).start()
                audio_buffer.clear()
                silence_count = 0
                speech_chunks = 0


def transcribe(audio):
    try:
        segs, _ = whisper.transcribe(audio, beam_size=1, language="en")
        text = " ".join(s.text.strip() for s in segs).strip()
        if text:
            log_queue.put(f"You: {text}")
            send_to_claude(text)
    except Exception as e:
        log_crash("transcribe", e)


def send_to_claude(text):
    try:
        import platform
        pyperclip.copy(text)

        if platform.system() == "Windows":
            import pygetwindow as gw
            import pyautogui
            wins = gw.getWindowsWithTitle("Claude")
            if not wins:
                log_queue.put("Claude window not found.")
                return
            win = wins[0]
            if win.isMinimized:
                win.restore()
            win.activate()
            time.sleep(0.4)
            x = win.left + win.width // 2
            y = win.top + win.height - 70
            pyautogui.click(x, y)
            time.sleep(0.2)
            pyautogui.hotkey("ctrl", "v")
            time.sleep(0.15)
            pyautogui.press("enter")
        else:
            # Linux — use xdotool (X11). On Wayland, falls back to clipboard only.
            import subprocess
            result = subprocess.run(
                ["xdotool", "search", "--name", "Claude"],
                capture_output=True, text=True
            )
            if result.returncode != 0 or not result.stdout.strip():
                log_queue.put("Copied to clipboard (auto-send requires xdotool on X11 / not supported on Wayland)")
                return
            win_id = result.stdout.strip().split("\n")[0]
            subprocess.run(["xdotool", "windowfocus", "--sync", win_id])
            time.sleep(0.3)
            subprocess.run(["xdotool", "key", "--clearmodifiers", "ctrl+v"])
            time.sleep(0.15)
            subprocess.run(["xdotool", "key", "Return"])

        log_queue.put(f"Sent: {text[:60]}...")
    except Exception as e:
        log_crash("send_to_claude", e)


def start_stream(device_idx):
    global stream
    stop_stream()
    try:
        stream = sd.InputStream(
            device=device_idx,
            samplerate=SAMPLE_RATE,
            channels=1,
            blocksize=CHUNK_SAMPLES,
            dtype="float32",
            callback=audio_callback,
        )
        stream.start()
        log_queue.put(f"Mic open [{device_idx}].")
    except Exception as e:
        log_crash("start_stream", e)


def stop_stream():
    global stream
    if stream:
        try:
            stream.stop()
            stream.close()
        except Exception:
            pass
        stream = None


def speaker_loop():
    global speaking
    while True:
        try:
            text = speak_queue.get()
            if not tts_ready or not voice_on:
                continue
            speaking = True
            words = text.split()
            if len(words) > 180:
                text = " ".join(words[:180]) + "..."
            if text and text[-1] not in ".!?,":
                text = text + "."
            wav, sr, _ = tts.infer(ref_file=REF_CLIP, ref_text=REF_TEXT,
                                    gen_text=text, remove_silence=False)
            silence = np.zeros(int(sr * 0.5), dtype=wav.dtype)
            wav = np.concatenate([wav, silence])
            os.makedirs(os.path.dirname(OUT_WAV), exist_ok=True)
            sf.write(OUT_WAV, wav, sr)
            data, samplerate = sf.read(OUT_WAV)
            sd.play(data, samplerate)
            sd.wait()
            time.sleep(0.6)
        except Exception as e:
            log_crash("speaker", e)
        finally:
            speaking = False


def inputfile_watcher():
    try:
        last_mtime = os.path.getmtime(INPUT_FILE)
    except Exception:
        last_mtime = 0
    while True:
        time.sleep(0.4)
        if not voice_on or not tts_ready:
            continue
        try:
            mt = os.path.getmtime(INPUT_FILE)
            if mt != last_mtime:
                last_mtime = mt
                with open(INPUT_FILE, "r", encoding="utf-8-sig") as f:
                    text = f.read().strip()
                if text:
                    while not speak_queue.empty():
                        try:
                            speak_queue.get_nowait()
                        except Exception:
                            break
                    speak_queue.put(text)
        except Exception:
            pass


def clipboard_watcher():
    pass  # disabled


# ── UI ─────────────────────────────────────────────────────────────────────
BG      = "#0d0d0d"
BTN_OFF = ("#1c1c1c", "#666666")
MIC_ON  = ("#001a0d", "#00ff88")
VOC_ON  = ("#00001a", "#4488ff")

class TARSPanel:
    def __init__(self, root):
        self.root = root
        root.title("TARS Control")
        root.configure(bg=BG)
        root.resizable(False, False)
        root.attributes("-topmost", True)

        tk.Label(root, text="T A R S", bg=BG, fg="#333333",
                 font=("Consolas", 11, "bold"), pady=8).pack()

        frame_dev = tk.Frame(root, bg=BG)
        frame_dev.pack(fill="x", padx=14, pady=(0, 6))
        tk.Label(frame_dev, text="MIC:", bg=BG, fg="#444444",
                 font=("Consolas", 8)).pack(side="left")
        self.device_var = tk.StringVar()
        self.device_map = {}
        choices = []
        default_idx = 0
        for i, d in enumerate(sd.query_devices()):
            if d["max_input_channels"] > 0:
                label = f"[{i}] {d['name'][:35]}"
                choices.append(label)
                self.device_map[label] = i
                if "G435" in d["name"] and default_idx == 0:
                    default_idx = len(choices) - 1
        self.device_cb = ttk.Combobox(frame_dev, textvariable=self.device_var,
                                       values=choices, state="readonly", width=34,
                                       font=("Consolas", 8))
        self.device_cb.pack(side="left", padx=(6, 0))
        if choices:
            self.device_cb.current(default_idx)
        self.device_cb.bind("<<ComboboxSelected>>", self.on_device_change)

        frame_btns = tk.Frame(root, bg=BG)
        frame_btns.pack(fill="x", padx=14, pady=4)
        self.mic_btn = tk.Button(
            frame_btns, text="MIC: OFF", command=self.toggle_mic,
            bg=BTN_OFF[0], fg=BTN_OFF[1],
            font=("Consolas", 11, "bold"), relief="flat", bd=0,
            padx=10, pady=12, cursor="hand2", width=14,
        )
        self.mic_btn.pack(side="left", padx=(0, 6))
        self.voice_btn = tk.Button(
            frame_btns, text="VOICE: OFF", command=self.toggle_voice,
            bg=BTN_OFF[0], fg=BTN_OFF[1],
            font=("Consolas", 11, "bold"), relief="flat", bd=0,
            padx=10, pady=12, cursor="hand2", width=14,
        )
        self.voice_btn.pack(side="left")

        self.log = tk.Text(root, bg="#1a1a1a", fg="#888888",
                           font=("Consolas", 8), height=10, width=56,
                           relief="flat", bd=0, state="disabled", wrap="word")
        self.log.pack(padx=14, pady=(4, 2))

        self.status_var = tk.StringVar(value="Loading models...")
        tk.Label(root, textvariable=self.status_var, bg=BG, fg="#444444",
                 font=("Consolas", 8), pady=4).pack()

        threading.Thread(target=load_models,       daemon=True).start()
        threading.Thread(target=speaker_loop,      daemon=True).start()
        threading.Thread(target=vad_worker,        daemon=True).start()
        threading.Thread(target=clipboard_watcher, daemon=True).start()
        threading.Thread(target=inputfile_watcher, daemon=True).start()

        root.after(300, self.poll)

    def toggle_mic(self):
        global mic_on
        if not (vad_ready and whisper_ready):
            self.append_log("Still loading — please wait...")
            return
        mic_on = not mic_on
        if mic_on:
            dev = self.device_map.get(self.device_var.get(), 1)
            start_stream(dev)
            self.mic_btn.config(text="MIC:  ON", bg=MIC_ON[0], fg=MIC_ON[1])
        else:
            stop_stream()
            self.mic_btn.config(text="MIC: OFF", bg=BTN_OFF[0], fg=BTN_OFF[1])

    def toggle_voice(self):
        global voice_on
        if not tts_ready:
            self.append_log("TARS voice still loading — please wait...")
            return
        voice_on = not voice_on
        if voice_on:
            self.voice_btn.config(text="VOICE:  ON", bg=VOC_ON[0], fg=VOC_ON[1])
        else:
            sd.stop()
            self.voice_btn.config(text="VOICE: OFF", bg=BTN_OFF[0], fg=BTN_OFF[1])

    def on_device_change(self, _=None):
        if mic_on:
            dev = self.device_map.get(self.device_var.get(), 1)
            start_stream(dev)

    def poll(self):
        while not log_queue.empty():
            self.append_log(log_queue.get_nowait())
        parts = []
        if not (tts_ready and vad_ready and whisper_ready):
            parts.append("loading...")
        else:
            if mic_on:
                parts.append("recording" if recording_now else "listening")
            if voice_on:
                parts.append("speaking" if speaking else "voice on")
        self.status_var.set("  |  ".join(parts) if parts else "ready")
        self.root.after(350, self.poll)

    def append_log(self, msg):
        self.log.config(state="normal")
        self.log.insert("end", msg + "\n")
        self.log.see("end")
        self.log.config(state="disabled")


if __name__ == "__main__":
    root = tk.Tk()
    style = ttk.Style(root)
    style.theme_use("default")
    style.configure("TCombobox", fieldbackground="#1c1c1c", background="#1c1c1c",
                    foreground="#666666", selectbackground="#222222")
    TARSPanel(root)
    root.mainloop()
