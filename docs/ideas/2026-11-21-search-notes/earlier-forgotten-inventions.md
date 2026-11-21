# Forgotten practical inventions: idea search (2026-11-21)

Angle: old inventions that were "ahead of their hardware" and are trivial in a browser now. One candidate survived all seven filters. "Not found" means not found in the searches listed, not proof of absence.

## 1. "Hum": the SiVo voice-pitch lipreading aid (UCL, 1980s)

**Belief.** "A sound with no words in it can't help anyone understand speech." People assume understanding comes from hearing the words, and the hum by itself is, in the authors' words, "completely unintelligible".

**Breaker.** With only a hum that follows the talker's voice pitch, normal listeners lipread continuous text up to 2.5 times faster than with lipreading alone ([Rosen, Fourcin & Moore, Nature 1981](https://pubmed.ncbi.nlm.nih.gov/7231534/)). Could fail: pitch-tracking errors on noisy real calls; "up to" means the typical gain is smaller.

**Old source.** Walliker, Rosen & Fourcin, [US 4,809,329](https://patents.google.com/patent/US4809329A/en) (priority 1986), "Apparatus for use in conjunction with lipreading by the profoundly deaf". A pitch extractor drives a sine wave at the talker's larynx frequency, shifted down ("mapitch") to fit the user's residual hearing, with optional noise for voiceless sounds. The patent says the straightforward design needed about 100 kHz sampling, too power-hungry for a wearable, so it under-sampled a 128-entry sine table instead. The SiVo aid was announced for sale in 1990 at about £1,000 ([New Scientist](https://www.newscientist.com/article/1820011-technology-sound-comes-to-the-aid-of-lip-readers/)) and field-tested with 22 patients in four countries ([ICPhS 1995](https://www.coli.uni-saarland.de/groups/FK/speech_science/icphs/ICPhS1995/13_ICPhS_1995_Vol_3/p13.3_206.pdf)). Why it stopped is not documented in what I found. Google Patents lists the patent as expired for unpaid fees. Cochlear implants taking over is my guess (unverified).

**Users and job.**
- Profoundly deaf adults who lipread, keep some hearing below about 500 Hz ([SIVO candidate tests](https://pubmed.ncbi.nlm.nih.gov/2470311/)) and have no implant. Job: follow a December family video call or an online lecture with their eyes on the face.
- Cochlear-implant users with some low-frequency hearing. A tone carrying pitch and loudness gave them significant benefit ([Brown & Bacon 2009](https://pubmed.ncbi.nlm.nih.gov/19546806)); the authors expected new devices, and I found none.
- Deaf speakers of tone languages, where pitch carries word meaning that lips can't show ([ERIC ED335901](https://eric.ed.gov/?id=ED335901)).
- Caveat: in [Faulkner et al. 1992](https://discovery.ucl.ac.uk/id/eprint/19347/2/Faulkner%20et%20al%201992.pdf), 4 of 11 users did better with the pitch tone, 5 did better with plain amplification, and 5 of 11 kept using SiVo afterwards. In noise, the later SiVo-3 "analytic" mode degraded less than amplification ([TIDE OSCAR report](https://discovery.ucl.ac.uk/id/eprint/19341/1/19341.pdf)).

**What we build.** A browser SiVo:
- an AudioWorklet pitch and voicing detector that outputs one sine cycle per larynx period, shifted down and set to a level from a 2-minute low-frequency hearing check;
- the patent's voiceless-noise channel and the 1992 compound aid's loudness channel, as options;
- input from any tab (video calls, YouTube) or a built-in WebRTC call that delays the video so lips and tone stay in sync;
- a stereo mode for implant users: the tone to the hearing-aid ear, the speech to the implant;
- a lipreading test that repeats the 1981 experiment on the user's own videos.

No deep learning.

**5-second silent clip.**
1. 0–1.5 s: muted close-up of a woman talking. Under her, a lipreader's transcript full of gaps.
2. 1.5–3 s: her dense speech waveform collapses into one smooth sine line labelled "only her pitch".
3. 3–5 s: the gaps fill in, with a word counter taken from your own kill test (not invented).

Caption: "This hum has no words. It still makes lipreading up to 2.5× faster."

**Existing-software check.** Queries:
- web: `app converts speech to voice pitch tone for lipreading profound hearing loss F0 sine wave software`
- web/GitHub: `github F0 modulated tone hearing aid lipreading SiVo implementation open source WebAudio`
- app stores (via web search): `"lipreading" app "pitch" tone hum aid hearing loss Google Play OR App Store speech to tone`
- `Laryngograph Ltd SiVo aid product sine voice hearing aid for profoundly deaf available`

Found: voice-pitch meters for voice training ([VoiceScope](https://www.github.com/Lemelson/VoiceScope)), open hearing-aid platforms that amplify ([openMHA prototype](https://github.com/hendrikkayser/hearingaid-prototype), [Open Speech Platform](https://github.com/tzubatiy/OpenSpeechPlatform-UCSD)), and AI lip-reading transcribers. **No working software found that turns speech into a pitch tone to aid lipreading.** Not checked: openMHA's full plugin list, and a direct app-store search.

**Feasibility, risk, kill test.** Ten days is enough: pitch tracking and oscillators are standard WebAudio. Main risk: in 10 days you can test only hearing volunteers, and their gain doesn't prove a deaf user's gain. Other risks:
- Bluetooth delay can put the tone out of sync with the lips in face-to-face use.
- Loud low tones need a hard output limiter.
- You may count this as "sonification". I'd call it a hearing-aid speech code, not data sonification.

One-day kill test: build pitch-to-sine, record 20 short sentences, and have 4 hearing volunteers lipread 10 muted and 10 with the tone, in counterbalanced order. Kill it if the mean keyword gain is under 15 points or the added delay is over 40 ms (my thresholds).

**Verdict.** The only survivor. A patented, briefly sold 1980s aid that its hardware held back, with a surprising Nature result anyone can feel in 30 seconds. The users are a niche and the clinical evidence is mixed.

## Died in checks

- Model-based face video (CANDIDE 1987, MPEG-4 FBA 1999): [open GFVC encoder/decoder code](https://github.com/Berlin0610/Awesome-Generative-Face-Video-Coding) and a [JVET standard](https://arxiv.org/abs/2410.15105) already do this job.
- Autocuer (Cornett, EP0050443) and MIT automatic cueing (66% vs 35% keywords): CNRS's open-source [Auto-CS](https://autocs.sourceforge.io/) generates cued-speech video offline, and [interact.ai](https://devpost.com/software/interact-ai) did live speech-to-avatar cueing (2021). This is an active research field, not a dead one.
- Upton's 1967 eyeglass lights and Massaro's iGlasses: [Massaro 2013](http://psyentificmind.com/wp-content/uploads/2013/10/iGlassesR2R.pdf) found the cues usable for single words but not continuous speech, and switched to captions. That reason still applies.
- SYNFACE talking-face phone (KTH/UCL, 2001–04): about +20% with ideal control ([ICPhS 2003](https://www.internationalphoneticassociation.org/icphs-proceedings/ICPhS2003/p15_0131.html)); captioned calls now do this job.
- BUGGY (Brown & Burton 1978): [MalruleLib](https://arxiv.org/abs/2601.03217) and [Eedi](https://www.eedi.com/diagnostic-engine) already infer the wrong rule and predict the next answer.
- Randomized response (Warner 1965): for a class of 30 the 95% interval is about ±36 points, so the demo fails.
- ClearBoard (Ishii 1992): the gaze cue needs the half-mirror; a webcam above the screen breaks it.
- Ogilby strip maps (1675): [QGIS Strip Map Maker](https://plugins.qgis.org/plugins/strip_map_maker/) and [mapline](https://github.com/krmr/mapline) exist.
- Words in Color (Gattegno 1962): still taught, with an app ([iReadBetter](https://www.bronxbetterlearning.org/learning-apps/ireadbetter)).
