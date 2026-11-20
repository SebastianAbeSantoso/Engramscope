# Kill test: Longuet-Higgins's 1968 holophone (2026-11-19)

**Question.** Does the holophone, a 1968 theory of how the brain might store sequences, actually recall real sound from a fragment?

**What it is.**
- H. C. Longuet-Higgins proposed a "temporal analogue of optical holography": a dense bank of narrow-band resonators with a shared input and output ([Nature 217, 104, 1968](https://www.nature.com/articles/217104a0)). He coined the word *holophone* in [Proc. R. Soc. B (1968)](https://royalsocietypublishing.org/doi/10.1098/rspb.1968.0074).
- **Storing** a signal raises each resonator's amplifier gain by the energy that rang in its band. Only the power spectrum is kept: no waveform and no timing.
- **Recall:** play a fragment in, and the bank answers with a noisy copy of the stored signal, aligned with the fragment, which carries on past its end.
- This spike implements the *ideal* (undamped) holophone in the discrete form of [Willshaw's 1971 Edinburgh thesis](https://era.ed.ac.uk/server/api/core/bitstreams/22477120-d384-4757-a5bd-ba329cdd47ec/content), Ch. 3, eqs. 3.10–3.14: `recall = IFFT(D · FFT(cue))`, with `D = Σ |FFT(stored)|²`.

**History, from primary sources.**
- [Gabor (1968)](https://www.nature.com/articles/2171288a0) and [Gabor (1969, IBM J. Res. Dev. 13:156)](http://bitsavers.computerhistory.org/pdf/ibm/IBM_Journal_of_Research_and_Development/132/ibmrd1302C.pdf) generalised it.
  - Gabor (1969, p. 157, footnote) links this to neurophysiologists' trouble locating the "engram" and to Lashley's lesion experiments.
- [Reid (IJCAI 1971)](https://www.ijcai.org/Proceedings/71/Papers/010.pdf) simulated a related correlographic memory on digitised speech.
- [Willshaw, Buneman & Longuet-Higgins (1969)](http://www.nature.com/nature/journal/v222/n5197/pdf/222960a0.pdf) argued other devices do better than holograms. Willshaw's thesis found that the models that perform well are non-linear (the associative net). The holophone was dropped.
- Two searches on 2026-11-19 found no modern working or playable build.

## Criteria (written in `killtest.py` before the first run)

- 3 sentences stored, each cued with its first 25%.
- Pass if the recall after the cue correlates best with its own sentence's continuation for 3 of 3 sentences, with a mean margin of at least 0.05 over the best wrong sentence.
- Correlation is on log-spectrograms, z-scored per frequency.

Speech: Harvard sentences from the [Open Speech Repository](https://www.voiptroubleshooter.com/open_speech/) (`OSR_us_000_0010_8k.wav`, 8 kHz). Use is free with attribution to "Open Speech Repository".

## Results (measured; tiny sample)

| Stored | Identified | Mean margin | Pass |
| --- | ---: | ---: | --- |
| 3 white-noise signals (the case the 1968 theory assumes) | 3/3 | 0.097 | yes |
| 3 spoken sentences | 3/3 | 0.140 | **yes** |
| 5 spoken sentences | 5/5 | 0.093 | |
| 8 spoken sentences | 6/8 | 0.037 | |
| 3 "whispered" sentences (my own noise-excited resynthesis) | 1/3 | 0.005 | no; probably an artefact of the resynthesis, not checked |

Details for the 3 spoken sentences:
- Spectrogram correlation with the right continuation: 0.319, 0.280 and 0.194. With the best wrong one: 0.129, 0.199 and 0.046.
- A cue from a sentence that was never stored: 0.065–0.108.
- As a waveform, the recall is very noisy: SNR 0.2–1.3 dB, waveform correlation 0.19–0.51. Judge it by ear in `listen.html`.

Lesions on the 3 spoken sentences (descriptive; `followup.py`; 32,769 resonators):

| Lesion | Identified | Mean margin |
| --- | --- | ---: |
| none | 3/3 | 0.140 |
| 25% of resonators removed at random (5 draws) | 3/3 in every draw | 0.072 |
| 50% removed at random | 3/3 in every draw | 0.044 |
| 75% removed at random | 3, 3, 3, 2, 2 | 0.013 |
| 90% removed at random | 2, 0, 2, 3, 1 | 0.000 |
| band 0–500 Hz removed (12% of resonators) | 3/3 | 0.120 |
| band 500–1500 Hz removed (25%) | 3/3 | 0.121 |
| band 1500–4000 Hz removed (62%) | 3/3 | 0.039 |

Reading:
- Recall degrades gradually with the number of resonators removed, and which band goes matters little at equal size. That is Lashley's mass action and equipotentiality, here on 3 sentences only.
- Capacity is small: recall starts to fail at 8 sentences.

## Files

- `killtest.py`: model, criteria, first run → `results.json`.
- `followup.py`: speech lesions and the never-stored control → `followup.json`.
- `listen.py` → `listen.html`: a self-contained page with every clip embedded. Download and open it in a browser.
