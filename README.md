# Monologue

Upload a solo vocal — a cappella, no music underneath. Monologue clones
the voice from it and reads any text in that same voice: the song's own
lyrics, or anything else you paste in.

## Browser — `monologue.html`

A single self-contained page, same design as Peel, with an RU/EN
toggle. No server or build step: open it, or serve it from GitHub Pages.

1. Drop in the vocal. It's used only as the voice sample.
2. Paste the text to read — or press **Fill in from the vocal** to pull
   the lyrics out of the recording with ElevenLabs Scribe.
3. Press **Read it in this voice**. Listen, compare with the original,
   download as WAV.

It runs on the ElevenLabs API with your own key (paste it into the page;
tick "remember" to keep it in this browser only). The vocal and text go
straight from your browser to ElevenLabs — nothing passes through any
other server. The temporary voice clone is deleted from your account
afterwards unless you tick "keep the cloned voice".

**Getting a key:** elevenlabs.io → Developers → API Keys → Create API
Key, with Text to Speech, Speech to Text and Voices (write) enabled.
Instant voice cloning needs a paid plan (Starter or higher).

**Getting a closer voice:** a clone made from singing can carry a bit of
the sung colour. Longer, cleaner takes (1–2 minutes, no effects or
harmonies) give the closest match.

**iOS:** if the file picker greys out your audio files, drag the file
onto the dropzone instead.

## CLI — `monologue.py`

Same idea from the command line, with a second engine that's free and
fully local.

```bash
pip install -r requirements.txt

# ElevenLabs — best likeness
export ELEVENLABS_API_KEY=sk_...
python monologue.py take.wav --engine elevenlabs -o out.wav

# your own text instead of the lyrics
python monologue.py take.wav --engine elevenlabs -o out.wav \
    --text-file words.txt

# local Coqui XTTS-v2 — free, no key
pip install TTS torch
python monologue.py take.wav --engine xtts -o out.wav --language ru
```

Without `--text` / `--text-file` the lyrics are transcribed from the
vocal with `faster-whisper` (or `openai-whisper`). With `--engine xtts`
and your own text, pass `--language` as well. XTTS is slow on CPU; a
GPU helps a lot.

Run `python monologue.py --help` for all flags.
