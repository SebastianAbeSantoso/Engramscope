"""Follow-up to killtest.py (same functions): lesions on voiced speech, and a control cue from an unstored sentence."""
import json
import numpy as np
import killtest as K

sents = K.sentences(K.load("osr.wav"))
pick = sents[:3]
n = max(len(s) for s in pick)
out = {"rows_speech_3": None, "control": None, "lesions_speech": []}
res, outs = K.run(pick, "speech")
out["rows_speech_3"] = res["rows"]
for r in res["rows"]:
    print(f"sentence {r['sentence']}: right {r['right']:.3f}  best wrong {r['best_wrong']:.3f}")
# waveform SNR of the recall after the cue, after least-squares scaling
for i, (s, c, o) in enumerate(outs):
    a, b = o[c:], s[c:]
    g = (a @ b) / (a @ a)
    snr = 10 * np.log10((b @ b) / np.sum((b - g * a) ** 2))
    print(f"sentence {i}: waveform SNR of recall {snr:.2f} dB, waveform corr {np.corrcoef(a, b)[0,1]:.3f}")
# control: cue taken from a sentence that was never stored (sentence 9), compared with the stored ones
h = K.Holophone(max(n, len(sents[9])))
for s in pick:
    h.store(s)
s9 = sents[9]
c = int(0.25 * len(s9))
o = h.recall(s9[:c])
rec = K.zspec(o[c : c + min(len(t) for t in pick) - c])
ctrl = [K.corr(rec, K.zspec(t[c:])) for t in pick]
out["control"] = ctrl
print("control (unstored cue) vs stored continuations:", [round(x, 3) for x in ctrl])
# lesions on voiced speech
L = K.Holophone(n).L
freqs = np.fft.rfftfreq(L, 1 / K.SR)
rng = np.random.default_rng(1950)
for frac in (0.25, 0.5, 0.75, 0.9, 0.97):
    sc = []
    for rep in range(5):
        keep = (rng.random(L // 2 + 1) >= frac).astype(float)
        r, _ = K.run(pick, "x", keep=keep)
        sc.append((sum(x["identified"] for x in r["rows"]), r["mean_margin"]))
    line = {"lesion": f"{int(frac*100)}% of resonators removed at random (5 draws)", "identified": [a for a, _ in sc], "mean_margin": round(float(np.mean([m for _, m in sc])), 3)}
    out["lesions_speech"].append(line); print(json.dumps(line))
for lo, hi in ((0, 500), (500, 1500), (1500, 4000)):
    keep = ((freqs < lo) | (freqs >= hi)).astype(float)
    r, outs = K.run(pick, "x", keep=keep)
    line = {"lesion": f"band {lo}-{hi} Hz removed ({100*((freqs>=lo)&(freqs<hi)).mean():.0f}% of resonators)", "identified": sum(x["identified"] for x in r["rows"]), "mean_margin": round(r["mean_margin"], 3)}
    out["lesions_speech"].append(line); print(json.dumps(line))
    if lo == 500:
        for i, (s, c, o) in enumerate(outs):
            K.wav(f"speech_{i}_recall_band500-1500removed.wav", o)
json.dump(out, open("followup.json", "w"), indent=1)
