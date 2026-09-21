# Monologue

Upload a solo vocal — a cappella, no music underneath — and get the same
voice back, reading the words instead of singing them.

Two ways to do it, and two tools:

|              | **Signal** (`dsp`)                              | **Clone** (`ai`)                                     |
|--------------|--------------------------------------------------|-------------------------------------------------------|
| How          | Reshapes the take itself: flattens the melody, smooths vibrato, shrinks held notes to spoken length | Transcribes the vocal, then speaks the text back in a clone of the same voice |
| Needs        | Nothing — fully offline                          | An engine: ElevenLabs (cloud, paid) or Coqui XTTS (local, free) |
| Sounds like  | The original take, "de-sung"                     | A fresh, natural reading in the same voice             |
| Where        | Browser app *or* CLI                             | Browser app (ElevenLabs only) *or* CLI (either engine) |

## Browser app — `index.html`

A single self-contained page. Open it directly, or publish it as a static
site (GitHub Pages works fine — no server or build step needed).

- **Signal** mode runs entirely in the browser: nothing is uploaded
  anywhere. Four sliders (flatten amount, vibrato cutoff, held-note
  target length, and the threshold for what counts as "held") let you
  tune it by ear.
- **Clone** mode calls the ElevenLabs API directly from the page with an
  API key you paste in (kept in memory, or in this browser's local
  storage only if you tick "remember"). It transcribes with Scribe, lets
  you fix any misheard words, clones the voice, generates the reading,
  and — unless you tick "keep the cloned voice" — deletes the temporary
  clone from your account afterwards.

The in-browser signal path is a simplified pitch-shift (autocorrelation
pitch tracking + a windowed resample-and-overlap-add). It's solid for
typical singing ranges, but wide melodic leaps or belted high notes can
come out a little "chipmunked," since a simple resample shifts the vocal
formants along with the pitch. For those, use the CLI's `dsp` mode
instead — it separates pitch from timbre properly (see below).

## Python CLI — `cli/monologue.py`

```bash
pip install -r cli/requirements.txt

# signal mode — offline, no API key
python cli/monologue.py take_04.wav dsp -o take_04_spoken.wav

# tune it
python cli/monologue.py take_04.wav dsp -o out.wav --flatten 0.5 --sustain-ms 150

# clone mode — pick an engine
export ELEVENLABS_API_KEY=sk_...
python cli/monologue.py take_04.wav ai --engine elevenlabs -o out.wav

pip install TTS torch   # only needed for the local engine
python cli/monologue.py take_04.wav ai --engine xtts -o out.wav --language lv
```

`dsp` mode uses the [WORLD vocoder](https://github.com/mmorise/World) to
split the recording into pitch, spectral envelope, and aperiodicity, so
it can flatten the melody and shrink held notes without touching the
timbre or formants — cleaner results than the browser version, especially
on wide vocal ranges.

`ai` mode transcribes with `faster-whisper` (or `openai-whisper` as a
fallback), then either:
- **`--engine elevenlabs`** — instant-clones the voice from the sample
  and generates the reading via the ElevenLabs API (needs
  `ELEVENLABS_API_KEY`, costs a small amount per character, best
  likeness), or
- **`--engine xtts`** — clones and generates locally with Coqui
  XTTS-v2 (free, no internet needed once the model's downloaded, but
  slow on CPU — a GPU helps a lot).

Run `python cli/monologue.py --help` for the full flag list, including
the four signal-mode knobs also exposed as sliders in the browser app.

## Repo layout

```
monologue/
├── index.html            browser app
├── README.md
└── cli/
    ├── monologue.py
    └── requirements.txt
```

Suggested repo name, matching the others: `imBeyondiDentity/monologue`.
