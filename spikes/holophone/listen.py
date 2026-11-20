"""Writes listen.html: the kill-test audio, embedded, for listening by ear."""
import base64, io, json, wave
import numpy as np
import killtest as K

def b64(x):
    x = x / (np.max(np.abs(x)) + 1e-9) * 0.9
    buf = io.BytesIO(); w = wave.open(buf, "wb"); w.setnchannels(1); w.setsampwidth(2); w.setframerate(K.SR)
    w.writeframes((x * 32767).astype(np.int16).tobytes()); w.close()
    return "data:audio/wav;base64," + base64.b64encode(buf.getvalue()).decode()

sents = K.sentences(K.load("osr.wav"))
pick = sents[:3]
n = max(len(s) for s in pick)
h = K.Holophone(max(n, len(sents[9])))
for s in pick:
    h.store(s)
L = h.L
rng = np.random.default_rng(1950)
keep50 = (rng.random(L // 2 + 1) >= 0.5).astype(float)
freqs = np.fft.rfftfreq(L, 1 / K.SR)
keepband = ((freqs < 500) | (freqs >= 1500)).astype(float)
rows = []
for i, s in enumerate(pick):
    c = int(0.25 * len(s))
    full = h.recall(s[:c])[: len(s)]
    half = h.recall(s[:c], keep50)[: len(s)]
    band = h.recall(s[:c], keepband)[: len(s)]
    rows.append((i, [("1. The whole sentence that was stored", s), ("2. The cue: only its first quarter", s[:c]),
                     ("3. What the holophone plays back after the cue", full[c:]), ("4. The true rest of the sentence, for comparison", s[c:]),
                     ("5. Playback with half of the resonators cut out at random", half[c:]), ("6. Playback with the 500-1500 Hz resonators cut out", band[c:])]))
s9 = sents[9]; c9 = int(0.25 * len(s9))
ctrl = h.recall(s9[:c9])[c9 : c9 + 16000]
html = ["<!doctype html><meta charset=utf-8><title>Holophone kill test: listen</title>",
        "<style>body{font:15px/1.5 system-ui;max-width:760px;margin:2rem auto;padding:0 1rem;background:#111;color:#ddd}audio{width:100%}h2{font-size:1.05rem;margin-top:2rem}small{color:#999}</style>",
        "<h1>Longuet-Higgins's 1968 holophone: listening test</h1>",
        "<p>Three spoken sentences were stored in one holophone: a bank of 32,769 resonators. Storing changes only each resonator's gain, by the energy that rang in its band. No waveform and no timing are kept. Each sentence was then cued with its first quarter. The holophone's answer after the cue is clip 3. Clips 5 and 6 are Lashley-style lesions.</p>",
        "<p><small>Speech: Open Speech Repository (Harvard sentences, OSR_us_000_0010). Model: the ideal holophone as formalised by Willshaw (1971, Ch. 3). Measured, 3 sentences: the recall matches its own sentence best for 3 of 3.</small></p>"]
for i, clips in rows:
    html.append(f"<h2>Sentence {i + 1}</h2>")
    for label, x in clips:
        html.append(f"<p>{label}<br><audio controls src='{b64(x)}'></audio></p>")
html.append("<h2>Control: cue from a sentence that was never stored</h2><p>What the holophone plays when it has no memory of the cue<br><audio controls src='" + b64(ctrl) + "'></audio></p>")
open("listen.html", "w").write("\n".join(html))
print("listen.html", round(len("\n".join(html)) / 1e6, 2), "MB")
