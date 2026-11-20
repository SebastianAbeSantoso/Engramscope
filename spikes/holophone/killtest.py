"""Kill test: does Longuet-Higgins's 1968 holophone recall real sound from a fragment?

The ideal holophone, as formalised by Willshaw (1971, PhD thesis, Edinburgh, Ch. 3, eqs. 3.10-3.14):
  - a dense bank of resonators covering the frequency range; each feeds an amplifier;
  - storing signals F_r raises each amplifier's gain by the energy in its pass-band: D(w) = sum_r |Phi_r(w)|^2;
  - recall: a cue F' passes through the bank; the extra output is dG = IFFT(D * FFT(F')).
    Equivalently dG = F (*) (cue cross-correlated with F): a noisy copy of the stored signal,
    aligned with the cue, which continues past the cue's end.

Criteria, written before the first run:
  Identification: 3 sentences stored; each is cued with its first 25%. The recall after the cue
    must correlate best with the right sentence's continuation (log-spectrogram, z-scored per
    frequency) for 3 of 3 sentences, with a mean margin over the best wrong sentence of at least 0.05.
  Reported for three kinds of sound: white noise (the case the 1968 theory assumes), the original
    speech, and the same speech made noise-excited (a whisper: same spectral envelope, noise source).
  Listening: WAV files of cue, recall and truth for the builder to judge by ear.
  Lesion (descriptive): identification and correlation after removing a random share of the
    resonators (frequency bins of D), and after removing one band.
"""

import json
import sys
import wave

import numpy as np

SR = 8000
RNG = np.random.default_rng(1968)


def load(path):
    w = wave.open(path)
    x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float64) / 32768
    return x


def sentences(x, min_len=1.2):
    """Split on pauses: frames of 20 ms with energy below 2% of the 95th percentile."""
    f = 160
    e = np.array([np.mean(x[i : i + f] ** 2) for i in range(0, len(x) - f, f)])
    thr = 0.02 * np.percentile(e, 95)
    voiced = e > thr
    segs, start, gap = [], None, 0
    for i, v in enumerate(voiced):
        if v:
            if start is None:
                start = i
            gap = 0
        elif start is not None:
            gap += 1
            if gap > 15:  # 300 ms of silence ends a sentence
                end = i - gap + 1
                if (end - start) * f / SR >= min_len:
                    segs.append(x[start * f : end * f])
                start, gap = None, 0
    return [s / np.sqrt(np.mean(s**2)) for s in segs]


def stft_mag(x, n=256, hop=64):
    win = np.hanning(n)
    frames = [x[i : i + n] * win for i in range(0, len(x) - n, hop)]
    return np.abs(np.fft.rfft(np.array(frames), axis=1)).T  # [freq, time]


def whisper(x, n=256, hop=64):
    """Same smoothed spectral envelope, noise excitation (removes pitch harmonics)."""
    win = np.hanning(n)
    mag = stft_mag(x, n, hop)
    k = 9  # smooth across frequency to remove harmonic ripple
    env = np.apply_along_axis(lambda c: np.convolve(c, np.ones(k) / k, mode="same"), 0, mag)
    out = np.zeros(len(x) + n)
    norm = np.zeros(len(x) + n)
    for t in range(env.shape[1]):
        noise = np.fft.rfft(RNG.standard_normal(n) * win)
        frame = np.fft.irfft(env[:, t] * noise / (np.abs(noise) + 1e-9), n) * win
        out[t * hop : t * hop + n] += frame
        norm[t * hop : t * hop + n] += win**2
    y = (out / np.maximum(norm, 1e-6))[: len(x)]
    return y / np.sqrt(np.mean(y**2))


def zspec(x):
    s = np.log(stft_mag(x) + 1e-6)
    return (s - s.mean(axis=1, keepdims=True)) / (s.std(axis=1, keepdims=True) + 1e-9)


def corr(a, b):
    m = min(a.shape[1], b.shape[1])
    a, b = a[:, :m].ravel(), b[:, :m].ravel()
    return float(np.corrcoef(a, b)[0, 1])


class Holophone:
    def __init__(self, length):
        self.L = 1 << int(np.ceil(np.log2(2 * length + 1)))
        self.D = np.zeros(self.L // 2 + 1)

    def store(self, f):
        self.D += np.abs(np.fft.rfft(f, self.L)) ** 2

    def recall(self, cue, keep=None):
        d = self.D if keep is None else self.D * keep
        return np.fft.irfft(d * np.fft.rfft(cue, self.L), self.L)


def run(signals, label, cue_frac=0.25, keep=None):
    n = max(len(s) for s in signals)
    h = Holophone(n)
    for s in signals:
        h.store(s)
    rows, outs = [], []
    for i, s in enumerate(signals):
        c = int(cue_frac * len(s))
        out = h.recall(s[:c], keep)[: len(s)]
        rec = zspec(out[c:])
        scores = [corr(rec, zspec(t[c:])) if len(t) > c + 512 else -1.0 for t in signals]
        wrong = max(sc for j, sc in enumerate(scores) if j != i)
        rows.append({"sentence": i, "right": scores[i], "best_wrong": wrong, "identified": scores[i] > wrong})
        outs.append((s, c, out))
    ident = sum(r["identified"] for r in rows)
    margin = float(np.mean([r["right"] - r["best_wrong"] for r in rows]))
    return {"label": label, "identified": f"{ident}/{len(rows)}", "mean_margin": margin, "rows": rows}, outs


def wav(path, x):
    x = x / (np.max(np.abs(x)) + 1e-9) * 0.9
    w = wave.open(path, "wb")
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(SR)
    w.writeframes((x * 32767).astype(np.int16).tobytes())
    w.close()


def main():
    sents = sentences(load("osr.wav"))
    print(f"{len(sents)} sentences, lengths (s):", [round(len(s) / SR, 2) for s in sents])
    pick = sents[:3]
    n = max(len(s) for s in pick)
    noise = [RNG.standard_normal(len(s)) for s in pick]
    whispers = [whisper(s) for s in pick]
    results = {"criteria": "3/3 identified and mean margin >= 0.05 (see docstring)", "runs": []}
    for kind, sig in [("white noise", noise), ("speech", pick), ("whispered speech", whispers)]:
        res, outs = run(sig, kind)
        res["pass"] = res["identified"] == "3/3" and res["mean_margin"] >= 0.05
        results["runs"].append(res)
        print(json.dumps({k: v for k, v in res.items() if k != "rows"}))
        for i, (s, c, out) in enumerate(outs):
            tag = kind.replace(" ", "_")
            wav(f"{tag}_{i}_truth.wav", s)
            wav(f"{tag}_{i}_cue.wav", s[:c])
            wav(f"{tag}_{i}_recall.wav", out)
    # more sentences stored: capacity
    for r in (5, 8):
        if len(sents) >= r:
            for kind, sig in [("speech", sents[:r]), ("whispered speech", [whisper(s) for s in sents[:r]])]:
                res, _ = run(sig, f"{kind}, {r} stored")
                results["runs"].append(res)
                print(json.dumps({k: v for k, v in res.items() if k != "rows"}))
    # lesions on the whispered set (descriptive)
    L = Holophone(n).L
    lesions = []
    for frac in (0.25, 0.5, 0.75, 0.9):
        keep = (RNG.random(L // 2 + 1) >= frac).astype(float)
        res, _ = run(whispers, f"whisper, {int(frac * 100)}% of resonators removed", keep=keep)
        lesions.append(res)
        print(json.dumps({k: v for k, v in res.items() if k != "rows"}))
    freqs = np.fft.rfftfreq(L, 1 / SR)
    for lo, hi in ((0, 1000), (1000, 2000), (2000, 4000)):
        keep = ((freqs < lo) | (freqs >= hi)).astype(float)
        res, outs = run(whispers, f"whisper, band {lo}-{hi} Hz removed", keep=keep)
        lesions.append(res)
        print(json.dumps({k: v for k, v in res.items() if k != "rows"}))
    results["lesions"] = lesions
    json.dump(results, open("results.json", "w"), indent=1)


if __name__ == "__main__":
    sys.exit(main())
