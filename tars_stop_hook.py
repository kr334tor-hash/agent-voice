"""
TARS Stop Hook
==============
Runs after every Claude Code response. Extracts the assistant's text,
strips markdown/code blocks, and writes the conversational portion to
tars_input.txt so TARS speaks it automatically.

Wired via .claude/settings.local.json → hooks → Stop.
"""

import json
import os
import re
import sys

TARS_INPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tars_input.txt")
MAX_CHARS  = 400   # max chars to speak — longer responses get last paragraph only


def strip_markdown(text: str) -> str:
    text = re.sub(r"```[\s\S]*?```", "", text)          # fenced code blocks
    text = re.sub(r"`[^`\n]+`", "", text)               # inline code
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)  # headers
    text = re.sub(r"\*{1,3}([^*\n]+)\*{1,3}", r"\1", text)     # bold/italic
    text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.MULTILINE)  # bullets
    text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.MULTILINE)  # numbered lists
    text = re.sub(r"^[-*_]{3,}$", "", text, flags=re.MULTILINE)   # hr rules
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)          # links
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Normalise common Unicode punctuation to ASCII for clean TTS output
    text = text.replace("—", " - ").replace("–", " - ")  # em/en dash
    text = text.replace("‘", "'").replace("’", "'")       # curly single quotes
    text = text.replace("“", '"').replace("”", '"')       # curly double quotes
    text = text.replace("…", "...")                             # ellipsis
    return text.strip()


def extract_last_assistant_text(data: dict) -> str:
    transcript = data.get("transcript", [])
    for msg in reversed(transcript):
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text", ""))
            return "\n".join(parts)
    return ""


def smart_truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    # Try to use the last substantial paragraph
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    for para in reversed(paragraphs):
        if 20 < len(para) <= max_chars:
            return para
    # Fall back to hard truncate at word boundary
    truncated = text[:max_chars]
    last_space = truncated.rfind(" ")
    return (truncated[:last_space] if last_space > 0 else truncated).rstrip(".,;:") + "."


try:
    raw = sys.stdin.read().strip()
    data = json.loads(raw) if raw else {}

    # Payload has last_assistant_message directly — no transcript traversal needed
    text = data.get("last_assistant_message", "")
    text = strip_markdown(text)
    text = smart_truncate(text, MAX_CHARS)

    if text and len(text) > 15:
        with open(TARS_INPUT, "w", encoding="utf-8") as f:
            f.write(text)
except Exception:
    pass   # never crash the session
