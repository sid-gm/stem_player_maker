# stem_player_maker

A local test harness for turning a song into stems, analyzing its structure, and
**replacing parts of it with your own raw sound bites** — one part at a time or as a
messy multi-part medley.

The pipeline:

1. **Separate** a song into 4 stems (`vocals / drums / bass / other`) with `htdemucs`.
2. **Analyze** tempo, key, beats, bars, sections, per-stem loudness, and **repeated
   patterns** per layer (so each recurring "part" gets a label like `B3`, `O1`, `V12`).
3. **Replace** a part's sound with an uploaded sample — placed **raw** (no time-stretch,
   no pitch-shift, no grid-quantize, so the sound bite is preserved exactly) at every
   occurrence, then bounced back to a full mix.
4. **Medley** — stack many parts → many sounds (across any stems) into one messy bounce.

A browser **visualizer** drives the whole thing: section ribbon, per-stem waveforms,
clickable pattern bands, current/new-sound waveforms with a **trim** selector, and the
render/medley controls.

Everything lives in [`test-harness/`](test-harness/) — see its
[README](test-harness/README.md) for setup (Python 3.11 + ffmpeg), the module map, and
how to run the server and CLI.

```bash
cd test-harness
./.venv/bin/python server.py    # http://localhost:8753/visualizer.html
```

> Sample songs and generated stems/output are intentionally **not** committed (large and
> often copyrighted). Bring your own audio — drag it into the visualizer to process it.
