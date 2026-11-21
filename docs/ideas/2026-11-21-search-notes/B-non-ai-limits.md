# Working notes: non-AI old-theory candidates (2026-11-21)

## Checked so far

### Sign-language "cartoon" codec (Sperling et al. 1985; Pearson & Robinson 1985) -> DIES
- Source read: Sperling, Landy, Cohen, Pavel, "Intelligible encoding of ASL image sequences at extremely low information rates", CVGIP 31:335-391 (1985), https://www.cns.nyu.edu/~msl/papers/sperlingetal85.pdf
  - 38 deaf subjects, 96x64 base images.
  - Polygonal cartoon + vectorgraph code: normalized intelligibility I = 0.56 at 3,900 bps, 0.70 at 6,000 bps; binquad binary I = 0.68 at 7,500 bps.
  - BUT their own data: 24x16 gray at 15 fps = I 0.744 raw, 0.856 normalized (p.363, p.384); they call it "the premier candidate". 48x32 gray ~0.95 normalized.
  - Gray digital codes were only *estimated* at 3 bits/pixel (Sect. 7.3), i.e. 1985 coding. A modern codec (AV1) at 24x16-48x32 gray, 15 fps, would very likely fit in ~3-6 kbps -> modern baseline probably beats cartoons. What is left is a codec setting, not an engine.
- Prior art for the purpose: VC4ASL (arXiv 2510.23056, 2025): pose keypoints over the audio channel of Zoom/Meet at 1.5-3 kbps, user study with ASL users. MobileASL (UW) H.264 ROI work; Tran et al. ASSETS'14 tested 5 fps/25 kbps ... 30 fps/150 kbps.

- MEASURED (av1_tiny.py, CC0 clip "Alphabets - Ghanaian Sign Language.webm", portrait crop, gray, 15 fps, 51 s, libaom-av1 usage=realtime cpu-used=8 lag-in-frames=0, IVF container incl. 12 B/frame):
  - 64x96: 5.54 kbps -> 26.2 dB / SSIM 0.80; 9.70 kbps -> 30.3 dB / 0.89
  - 32x48: 7.25 kbps -> 30.9 dB / 0.944; 9.68 kbps -> 33.4 dB / 0.966
  - 16x24: 7.26 kbps -> 36.6 dB / 0.996
  - libaom realtime has a ~5.5-7 kbps floor at 15 fps (per-frame overhead). Over IP, IPv4/UDP alone at 15 packets/s is 28 B x 15 x 8 = 3.4 kbps before RTP/SRTP.
  - Reading: at ~7 kbps AV1 delivers near-transparent 16x24 and good 32x48 gray, the conditions Sperling found 86% / ~95% as intelligible as 96x64 gray; the best 1985 cartoon code reached 70% at 6 kbps. Modern baseline wins or ties. PSNR is a proxy, not intelligibility.

### Possible rescue: event-boundary frame selection (Parish, Sperling & Landy 1990)
- JEP:HPP 16(2):282-294, https://sites.socsci.uci.edu/~whipl/staff/sperling/PDFs/Parish+Sperling_ASL_JEP_1990.pdf
- 32 experienced ASL signers. Activity index = pick frames where successive-frame difference is at a local minimum.
- Full gray-scale dynamic: 80% correct at slightly <11 new fps vs ~20-25 fps for every-nth-frame -> 1.8-2.25x fewer frames (p.291-292). ~8% better on average at matched fps, ~15% if rest positions excluded. Binary images: smaller advantage.
- Why it might matter now: at tiny sizes AV1's cost is dominated by per-frame overhead, so fewer frames ~ proportionally fewer bits.
- Caveats: isolated signs, not conversation; needs 1-2 frames lookahead; wire overhead (IPv4/UDP + DTLS + SCTP ~85 B per data-channel packet; SRTP media ~60-70 B) eats much of the saving unless frames are bundled.

- Rescue fails on filter 3: pose keypoints are the obvious modern alternative. VC4ASL (2025) sends quantized pose (56 B/frame) inside Zoom/Meet audio at 1.5-3 kbps; classifier accuracy 96.2%/95.75% ASL alphabet, 88.64% signs over Zoom. Tiny AV1 video needs ~5.5-7 kbps payload at 15 fps plus wire overhead. MobileASL (UW, 2009) already did activity-based variable frame rate (lower fps when not signing) and intelligible ASL at 30 kbps.

### Other ideas killed in my head (need one-line reasons in final list)
- Antiphasic speech (N0Spi BMLD, Licklider 1948 / Levitt & Rabiner 1967) for earbuds in noise: engine is a polarity flip; ANC already removes the coherent low-frequency noise where BMLD is largest; audio fails the silent clip.
- Lattice-gas fluids (FHP 1986) for WebGPU bandwidth: abandoned for noise; LBM wins at equal resolution.
- Remote-RAM paging over Wi-Fi (GMS 1995): Wi-Fi latency (ms) loses to SSD swap (~0.1 ms).
- BitWeaving / bit-transposed files (Wong et al. 1985): code exists (Quickstep, research repos).
- Interpolation search over HTTP range: rejected category, and B-tree fanout already gives ~4-5 fetches.
- Table Lens (Rao & Card 1994) on phones: LineUp/Taggle exist; also a visualisation.
- Touch prediction with minimum-jerk (Flash & Hogan 1985): Android/Chrome/iOS predictors already ship.
- Talisman / priority rendering (1994-96): impostors standard in engines.
- Night-quota hoarding (Coda 1991): YouTube smart downloads etc.; pricing, not hardware.

- Color cycling auto-encoder: "Automated Palette Cycling Animations" (SciTePress 2024, https://www.scitepress.org/Papers/2024/124325/124325.pdf) and Emergent Mind palette-cycling demo exist -> taken.
- WebGPU mixed-precision IR: well-known (Wilkinson/Moler; LAPACK dsgesv), fea_app does WebGPU iterative solvers; users thin -> weak.

- App-store check (via web search): "deaf sign language video calling app for slow internet low data mode" -> VRS apps (Convo, SignVideo, TTRS), generic deaf video chat; none advertises sub-10 kbps signing video.
- Palette cycling confirmed taken: Sattari Javid & Mould 2024 automatically convert input video into index map + per-frame palettes.

## Conclusion (2026-11-21)
No idea passed all seven filters. Closest miss: event-boundary tiny-video sign calls; dropped on filter 3 (pose keypoints likely win at matched bytes).

## Reproduce the AV1 measurement
python3 -m venv venv && ./venv/bin/pip install imageio-ffmpeg
Download the CC0 clip "File:Alphabets - Ghanaian Sign Language.webm" from Wikimedia Commons as gsl_alphabet.webm, then:
./venv/bin/python av1_tiny.py   (results saved in av1_tiny_results.txt)

## To check next (one at a time, with searches)
1. WebGPU f64: mixed-precision iterative refinement / defect correction. Existing WebGPU code? Users?
2. Color-cycling (palette rotation, 1985-1990s) auto-encoder for looping animation: existing tools?
3. Fresh angles still open: Bluetooth latency, background-tab throttling, storage eviction, flash wear.
