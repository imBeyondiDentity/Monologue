#!/usr/bin/env python3
"""
Monologue
=========
Turn a solo singing vocal (a cappella, no music, no harmonies) into the
same voice just speaking the words.

Two modes:

  dsp   Pure signal processing on the recording itself. Flattens the
        pitch contour, smooths out vibrato, and shortens held notes
        down to spoken-length syllables. No transcription, no cloning,
        no internet, no API key. Uses the WORLD vocoder.

  ai    Transcribes the vocal, then speaks the transcript back in a
        clone of the same voice. Two engines:
          --engine elevenlabs   cloud, needs ELEVENLABS_API_KEY, best
                                 likeness, costs per character
          --engine xtts         local Coqui XTTS-v2, free, needs a
                                 decent CPU (or a GPU, much faster)

Examples
--------
    python monologue.py take_04.wav dsp -o take_04_spoken.wav
    python monologue.py take_04.wav dsp -o out.wav --flatten 0.5
    python monologue.py take_04.wav ai --engine elevenlabs -o out.wav
    python monologue.py take_04.wav ai --engine xtts -o out.wav --language lv

Install
-------
    pip install -r requirements.txt
    # then, only for the ai engine you actually want:
    pip install faster-whisper          # or: pip install openai-whisper
    pip install TTS torch               # only needed for --engine xtts
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np


# --------------------------------------------------------------------------
# Shared I/O
# --------------------------------------------------------------------------

def load_mono(path: str):
    """Load an audio file as float64 mono. Returns (signal, sample_rate)."""
    import soundfile as sf
    x, sr = sf.read(path, always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x.astype(np.float64), sr


def save_audio(path: str, x: np.ndarray, sr: int):
    import soundfile as sf
    peak = np.max(np.abs(x)) or 1.0
    if peak > 0.999:
        x = x / peak * 0.98
    sf.write(path, x.astype(np.float32), sr)


# --------------------------------------------------------------------------
# DSP mode — WORLD vocoder pitch flattening
# --------------------------------------------------------------------------

def _voiced_runs(voiced: np.ndarray):
    """Contiguous [start, end) index ranges where voiced is True."""
    runs = []
    i, n = 0, len(voiced)
    while i < n:
        if voiced[i]:
            j = i
            while j < n and voiced[j]:
                j += 1
            runs.append((i, j))
            i = j
        else:
            i += 1
    return runs


def _sustain_keep_indices(f0: np.ndarray, voiced: np.ndarray, frame_rate: float,
                           sustain_ms: float, trigger_ms: float) -> np.ndarray:
    """
    Which frame indices to keep. A voiced run that's long AND steady in
    pitch (a held note, not a fast melodic run) gets frames dropped from
    its middle until it's down to roughly `sustain_ms` long — like a
    spoken syllable instead of a sung one. Everything else is kept as-is.
    """
    n = len(f0)
    trigger_frames = int(trigger_ms / 1000 * frame_rate)
    target_frames = max(1, int(sustain_ms / 1000 * frame_rate))

    keep = np.ones(n, dtype=bool)
    for start, end in _voiced_runs(voiced):
        length = end - start
        if length <= trigger_frames:
            continue
        seg = f0[start:end]
        seg = seg[seg > 0]
        if seg.size == 0:
            continue
        cents = 1200 * np.log2(seg / np.median(seg))
        if np.std(cents) > 80:  # more than ~a semitone of drift: a real
            continue           # melodic move, not a held note — leave it
        n_drop = length - target_frames
        if n_drop <= 0:
            continue
        drop_idx = np.linspace(start, end - 1, n_drop).round().astype(int)
        keep[drop_idx] = False

    idx = np.nonzero(keep)[0]
    return idx if len(idx) else np.arange(n)


def dsp_convert(x: np.ndarray, fs: int, flatten: float = 0.75,
                vibrato_cutoff_hz: float = 4.5,
                sustain_ms: float = 180.0,
                sustain_trigger_ms: float = 320.0) -> np.ndarray:
    """
    flatten             0 = keep the natural pitch shape (just de-vibrato'd)
                         1 = compress fully onto each phrase's median pitch
    vibrato_cutoff_hz    wobble faster than this is treated as vibrato and
                         smoothed away
    sustain_ms           target length a held note gets compressed to
    sustain_trigger_ms   a steady-pitch run longer than this counts as
                         "held" and is a candidate for compression
    """
    import pyworld as pw
    from scipy.signal import butter, filtfilt

    x = np.ascontiguousarray(x)
    f0, t = pw.dio(x, fs)
    f0 = pw.stonemask(x, f0, t, fs)
    sp = pw.cheaptrick(x, f0, t, fs)
    ap = pw.d4c(x, f0, t, fs)

    hop = t[1] - t[0] if len(t) > 1 else 0.005
    frame_rate = 1.0 / hop
    voiced = f0 > 0
    f0_smooth = f0.copy()

    if voiced.any():
        # 1. de-vibrato: low-pass the pitch contour (in log-freq / cents
        #    space, so the filter behaves the same across registers)
        log_f0 = np.full_like(f0, np.nan)
        log_f0[voiced] = np.log2(f0[voiced])
        idx = np.arange(len(f0))
        filled = np.interp(idx, idx[voiced], log_f0[voiced])

        nyq = frame_rate / 2
        cutoff = min(vibrato_cutoff_hz / nyq, 0.99)
        b, a = butter(2, cutoff, btype="low")
        smooth = filtfilt(b, a, filled)

        # 2. flatten: pull each sung phrase toward its own median pitch
        for start, end in _voiced_runs(voiced):
            seg = smooth[start:end]
            med = np.median(seg)
            smooth[start:end] = seg + (med - seg) * flatten

        f0_smooth = np.where(voiced, 2 ** smooth, 0.0)

    # 3. compress held notes into spoken-length syllables by dropping
    #    WORLD frames from the middle of long steady-pitch runs. This must
    #    look at the RAW pitch, not f0_smooth — flattening already squashes
    #    cents-variance toward 0, so checking the flattened contour here
    #    made almost every phrase look "held" and wrongly got compressed.
    keep_idx = _sustain_keep_indices(f0, voiced, frame_rate,
                                      sustain_ms, sustain_trigger_ms)
    f0_w = f0_smooth[keep_idx]
    sp_w = sp[keep_idx]
    ap_w = ap[keep_idx]

    return pw.synthesize(f0_w, sp_w, ap_w, fs, frame_period=hop * 1000)


# --------------------------------------------------------------------------
# AI mode — transcribe, then re-speak in a clone of the same voice
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
               language: str | None, keep_voice: bool):
    print("Transcribing...", file=sys.stderr)
    text, detected_lang = transcribe(input_path, language)
    if not text:
        sys.exit("Got no words back from transcription — is this really "
                  "an a cappella vocal, and is there enough of it?")
    print(f"Transcript ({detected_lang}): {text}", file=sys.stderr)

    lang = language or detected_lang
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
    p.add_argument("input", help="solo vocal file (wav/mp3/flac/...)")
    p.add_argument("-o", "--output", required=True, help="output audio path")
    sub = p.add_subparsers(dest="mode", required=True)

    dsp_p = sub.add_parser("dsp", help="offline signal-processing mode")
    dsp_p.add_argument("--flatten", type=float, default=0.75,
                        help="0=keep melody shape, 1=fully flat per phrase (default 0.75)")
    dsp_p.add_argument("--vibrato-cutoff", type=float, default=4.5,
                        help="Hz above which pitch wobble is smoothed out (default 4.5)")
    dsp_p.add_argument("--sustain-ms", type=float, default=180,
                        help="target length for compressed held notes, ms (default 180)")
    dsp_p.add_argument("--sustain-trigger-ms", type=float, default=320,
                        help="notes held longer than this get compressed (default 320)")

    ai_p = sub.add_parser("ai", help="transcribe + voice-clone mode")
    ai_p.add_argument("--engine", choices=["elevenlabs", "xtts"], required=True)
    ai_p.add_argument("--language", default=None,
                       help="force a language code instead of auto-detecting")
    ai_p.add_argument("--keep-voice", action="store_true",
                       help="don't delete the temporary ElevenLabs voice clone afterwards")

    args = p.parse_args()

    if args.mode == "dsp":
        x, fs = load_mono(args.input)
        y = dsp_convert(x, fs, flatten=args.flatten,
                         vibrato_cutoff_hz=args.vibrato_cutoff,
                         sustain_ms=args.sustain_ms,
                         sustain_trigger_ms=args.sustain_trigger_ms)
        save_audio(args.output, y, fs)
    else:
        ai_convert(args.input, args.output, args.engine,
                   args.language, args.keep_voice)

    print(f"Wrote {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
