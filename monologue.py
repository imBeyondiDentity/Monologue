#!/usr/bin/env python3
"""
Monologue
=========
Clone a voice from a solo singing vocal (a cappella, no music, no
harmonies) and read any text in that voice: the song's own lyrics
(transcribed automatically) or your own text.

Engines:
  --engine elevenlabs   cloud, needs ELEVENLABS_API_KEY and a paid
                        plan (Starter+), best likeness
  --engine xtts         local Coqui XTTS-v2, free, no key, slow on
                        CPU (a GPU helps a lot)

Examples
--------
    python monologue.py take.wav --engine elevenlabs -o out.wav
    python monologue.py take.wav --engine xtts -o out.wav --language ru
    python monologue.py take.wav --engine elevenlabs -o out.wav \
        --text-file words.txt

Install
-------
    pip install -r requirements.txt
    pip install TTS torch      # only for --engine xtts
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path



# --------------------------------------------------------------------------
# Transcribe (optional), then speak in a clone of the same voice
# --------------------------------------------------------------------------

def transcribe(path: str, language: str | None = None) -> tuple[str, str]:
    """Returns (text, language_code). Tries faster-whisper, then whisper."""
    try:
        from faster_whisper import WhisperModel
        model = WhisperModel("small", compute_type="int8")
        segments, info = model.transcribe(path, language=language)
        text = " ".join(seg.text.strip() for seg in segments)
        return text.strip(), (language or info.language)
    except ImportError:
        pass
    import whisper
    model = whisper.load_model("small")
    result = model.transcribe(path, language=language)
    return result["text"].strip(), result.get("language", language or "en")


def synth_elevenlabs(sample_path: str, text: str, out_path: str,
                      voice_name: str = "monologue-temp",
                      keep_voice: bool = False,
                      model_id: str = "eleven_multilingual_v2"):
    import requests
    api_key = os.environ.get("ELEVENLABS_API_KEY")
    if not api_key:
        sys.exit("Set the ELEVENLABS_API_KEY environment variable to use "
                  "--engine elevenlabs.")
    headers = {"xi-api-key": api_key}

    with open(sample_path, "rb") as f:
        r = requests.post(
            "https://api.elevenlabs.io/v1/voices/add",
            headers=headers,
            data={"name": voice_name},
            files={"files": (Path(sample_path).name, f, "audio/wav")},
        )
    r.raise_for_status()
    voice_id = r.json()["voice_id"]

    try:
        r = requests.post(
            f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
            headers={**headers, "Content-Type": "application/json"},
            json={"text": text, "model_id": model_id},
        )
        r.raise_for_status()
        with open(out_path, "wb") as f:
            f.write(r.content)
    finally:
        if not keep_voice:
            requests.delete(f"https://api.elevenlabs.io/v1/voices/{voice_id}",
                             headers=headers)


def synth_xtts(sample_path: str, text: str, out_path: str, language: str = "en"):
    from TTS.api import TTS
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    tts.tts_to_file(text=text, speaker_wav=sample_path,
                     language=language, file_path=out_path)


def ai_convert(input_path: str, out_path: str, engine: str,
               language: str | None, keep_voice: bool,
               text: str | None = None):
    """
    The vocal is only the voice sample. The words come from `text` if
    given, otherwise they're transcribed from the vocal itself.
    """
    detected_lang = None
    if text:
        text = text.strip()
    else:
        print("Transcribing...", file=sys.stderr)
        text, detected_lang = transcribe(input_path, language)
        if not text:
            sys.exit("Got no words back from transcription — pass your "
                      "own text with --text or --text-file instead.")
        print(f"Transcript ({detected_lang}): {text}", file=sys.stderr)

    if engine == "xtts" and not (language or detected_lang):
        sys.exit("With --text/--text-file and --engine xtts, pass "
                 "--language too (e.g. --language ru).")
    lang = language or detected_lang or "en"
    if engine == "elevenlabs":
        synth_elevenlabs(input_path, text, out_path, keep_voice=keep_voice)
    elif engine == "xtts":
        synth_xtts(input_path, text, out_path, language=lang)
    else:
        raise ValueError(f"unknown engine: {engine}")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", help="solo vocal file used as the voice sample")
    p.add_argument("-o", "--output", required=True, help="output audio path")
    p.add_argument("--engine", choices=["elevenlabs", "xtts"], required=True)
    p.add_argument("--language", default=None,
                   help="language code; default: auto-detect")
    src = p.add_mutually_exclusive_group()
    src.add_argument("--text", default=None,
                     help="text to read (default: transcribe the vocal)")
    src.add_argument("--text-file", default=None,
                     help="read the text from a UTF-8 file")
    p.add_argument("--keep-voice", action="store_true",
                   help="keep the temporary ElevenLabs voice clone")
    args = p.parse_args()

    text = args.text
    if args.text_file:
        text = Path(args.text_file).read_text(encoding="utf-8")
    ai_convert(args.input, args.output, args.engine,
               args.language, args.keep_voice, text=text)
    print(f"Wrote {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
