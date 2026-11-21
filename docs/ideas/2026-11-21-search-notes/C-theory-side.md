# Theory-side idea search: working notes (in progress)

Scratch file for the theory-first search. Final report goes in the chat answer.

## Screened in my head, dropped before searching (reason)

- Stochastic computing, LNS/Gaussian logs, Mitchell 1962: GPUs have single-cycle FMA and int8 tensor cores; no win.
- Winograd 1968 inner product: revived for FPGA/systolic hardware (FFIP, Pogue & Nicolici); needs 9-bit sums, breaks int8 dot-product units.
- RNS / CRT: Ozaki scheme II (INT8 tensor cores emulate FP64, GEMMul8 code) already exists.
- Distributed arithmetic (1973-74): this is the LUT matmul in T-MAC / bitnet.cpp (taken).
- Belady MIN: per-token weight access is cyclic, so MIN = static partial offload (llama.cpp -ngl); training-data case done by NoPFS/CoorDL.
- Overlays: layer-wise loading (AirLLM, taken).
- Dataflow, systolic arrays: alive.
- Sigma-delta / error diffusion for weights: GPTQ-style error feedback (taken).
- Progressive transmission: bit-plane any-precision (taken); JPEG XL progressive is strong.
- SPSA (Spall 1992): MeZO (taken). Bennett reversible computing: RevNets (taken).
- Iterative refinement, Dekker double-float, Kulisch accumulator: alive in cuSOLVER IRS, QD libs, ExBLAS.
- Richardson extrapolation on diffusion steps: high-order solvers and distillation already win; check if needed.

## Shortlist to check with searches

1. Liu & Layland 1973 + slack stealing (Lehoczky & Ramos-Thuel 1992) / sporadic server (Sprunt, Sha, Lehoczky 1989):
   local AI (voice changer, LLM, captions) sharing one consumer GPU with a game. Metric: game p99 frame time and AI deadline misses.
   Baseline: Windows WDDM time slicing; NVIDIA NVIGI "CUDA in Graphics" for in-game AI. Prior art to check: Heimdall (MobiCom 2020), GCAPS, NVIGI.
2. Online arithmetic (Ercegovac & Trivedi 1977) MSB-first for exact top-k of the LM head with interval pruning. Risk: close to bit-plane any-precision (taken).
3. Open: look for one more strong candidate outside AI.

## Search log

- C1 GPU co-tenancy (Liu & Layland / slack stealing): **DEAD, taken.** NVIDIA NVIGI "GPU Scheduling for AI" + CiG, with a llama.cpp CiG patch
  (https://docs.nvidia.com/nvigi-sdk/1.7.0/docs/nvigi_core/docs/GpuSchedulingForAI.html, https://docs.nvidia.com/nvigi-sdk/1.6.0/docs/nvigi_core/docs/llamacpp_CIG_patch.html),
  and Atelico's Compute Graphics Scheduler product (https://atelico.studio/ai-engine/gpu-sharing/).
  Queries: "schedule DNN inference in gaps between game rendering frames GPU frame time stutter co-running";
  "NVIDIA In-Game Inferencing SDK "CUDA in Graphics" scheduling AI alongside rendering frame rate";
  "llama.cpp while gaming stutter frame drops GPU sharing local LLM game same GPU".
- C2 Online arithmetic for LM-head top-k: MSDF early termination for DNNs is an active hardware research line
  (ECHO https://www.mdpi.com/2079-9292/13/10/1893, MINT https://arxiv.org/abs/2606.31514, DEX https://arxiv.org/abs/2610.11748, FGIE FPT 2019);
  FlashSampling fuses exact sampling into the LM-head matmul (https://arxiv.org/abs/2603.15854). Software gain limited to LM-head bytes; wow weak. **Weak, likely drop.**

## More ideas screened (no search yet)

- ABFT (Huang & Abraham 1984) + undervolting past the stable point: GreenMM (ICS 2019) did ABFT+undervolt GEMM on GPUs; laptop voltage control locked; crashes not silent errors. Drop.
- Lattice gas automata (FHP 1986) vs LBM for VRAM: noise needs coarse-graining; LBM (FluidX3D FP16) wins per byte. Drop.
- Slotted ALOHA / Capetanakis tree splitting (1979) for Meshtastic LoRa congestion: needs radios + collision feedback; not software-only. Drop.
- Deliberation scheduling (Boddy & Dean 1994) for reasoning-token budgets: early-stop-CoT papers with code exist. Drop.
- Frame-budgeted WebGPU inference in browsers (Liu & Layland in the browser): possible gap, weak wow. Maybe.
- Interpolation search (Peterson 1957) over HTTP range requests on sorted static files: 2-3 requests vs PMTiles' 2; small win. Drop.
- Ski-rental keep-alive (Karlin et al. 1988/1994) for Ollama-style model unloading on laptops: trivial policy, low wow. Drop.
- Richardson over quantization bit-width: quantization error is not a smooth expansion in b; no clean extrapolation. Drop.
- Strategy note: papers without code are allowed, so a revival paper with no public code is an opening.

## Results of D1-D3
- D3 3DGS offload: **DEAD.** CLM (https://arxiv.org/abs/2511.04951), GS-Scale (https://arxiv.org/abs/2509.15645), A LoD of Gaussians (https://arxiv.org/abs/2507.01110), and a 2026 paper treating GPU memory as a working-set cache for 3DGS training (https://arxiv.org/abs/2605.20150).
- D1 local LLM QoS scheduling: **DEAD.** "Local AI Server for Distributed LLM Inference" moves interactive requests ahead of agent work (https://arxiv.org/abs/2611.17753); clairvoyant-scheduler SJF sidecar (https://github.com/Aravind0403/clairvoyant-scheduler).
- D2 WebGPU jank: no direct prior art in one search; the literature says dispatch overhead is the batch-1 bottleneck (https://arxiv.org/abs/2608.08730). Same purpose as NVIGI/Atelico, ported to browsers; weak.
- Out-of-process variant of C1 (external AI app + unmodified game): need to check whether NVIGI's CUDA-vs-D3D priority interface works across processes.

- NVIGI scheduling needs the game's own D3D12 queue (in-process), so out-of-process co-tenancy is not covered by NVIGI.
  But the obvious baseline (Windows GPU priority / HAGS) may already give slack to a low-priority process; slack stealing = idle priority for 2 tasks. Weak.
- Swapping-based load control (Denning et al. 1976 "Optimal multiprogramming") across GPU apps via process suspension, exploiting WDDM evicting idle allocations (builder's measurement, feasibility-report 8.2): ~1.8x total throughput for two heavy jobs (estimate), niche users. Weak.
- Program slicing (Weiser 1981; dynamic slicing Korel & Laski 1988) as context reduction for local coding agents: hardware link is prefill time/context on laptops. Check prior art.

- Program slicing for LLM context: **DEAD.** Katana (slicing-based context for repair, code: https://github.com/saltlab/Katana), CausalRepair dual slicing (https://arxiv.org/abs/2608.10613), llm-tldr dependency tracing (https://github.com/parcadei/llm-tldr).

- Load control across GPU apps: partly done crudely by ComfyFlush (recycles ComfyUI to free VRAM for LLMs, https://github.com/hugoremington/ComfyFlush) and gpu-suspend-resume-runtime (https://github.com/manishklach/gpu-suspend-resume-runtime); Salus/Gandiva in datacenters. **Weak/dead.**
- Agents-as-processes (tool calls = I/O, KV = working set, Denning load control): KV is on the taken list; InferCept/Continuum/KVFlow/Autellix cover it in research. Dead.
- "Mult is expensive again" (FHE, MPC, ZK) for Winograd 1968 / Strassen: SIMD packing, matrix triples and sum-check already beat it. Dead.

- D2 WebGPU jank: common advice is "put the model in a Web Worker so the page never freezes"; no complaints of GPU-level jank found. Problem may not exist. **Drop.**

## Current best (medium): VRAM load control across AI apps (Denning 1968 thrashing; Denning et al. 1976 optimal multiprogramming; PFF, Chu & Opderbeck 1972)
- Builder's own data (feasibility-report.md 8.2): with an active co-tenant, llama-bench decode fell to 0.36x and 0.20x; Windows pages the less active app silently.
- Mechanism: detect thrashing per process (ETW VidMm paging events or PCIe throughput), suspend/resume whole processes in quanta so only working sets that fit run (swap-based load control), interactive first (CTSS MLFQ, Corbato 1962).
- Baseline: run both (Windows default); smarter users unload models manually (Ollama keep_alive=0, ComfyUI unload nodes).
- Overlap: same theory family as MaltOE (Denning). Prior art partial: ComfyFlush, gpu-suspend-resume-runtime.

- Novelty queries for load control: "tool suspend background GPU process when VRAM full avoid thrashing Windows ComfyUI LM Studio time slicing";
  "GPU memory oversubscription suspend resume time-slicing deep learning jobs Gandiva Salus consumer desktop";
  "automatically pause or suspend GPU apps when VRAM oversubscribed Windows thrashing detector ETW VidMm paging per process".
  Found: ComfyFlush (periodic kill/restart), gpu-suspend-resume-runtime (Linux research scaffold with simulation, cluster/K8s), Salus (TF-integrated, datacenter), manual unload guides. No load controller for consumer Windows found.

- FP64 GEMM in the browser via RNS/Ozaki-II on WGSL dot4I8Packed: Ozaki is a very active 2024-2026 line (Ozaki 2.5 https://arxiv.org/abs/2609.09095, FP8 Ozaki-II https://arxiv.org/abs/2603.10634, GEMMul8); a browser port is a port of a modern method and browser FP64 users are few. Drop.

- Verified sources: Denning 1968 AFIPS FJCC "Thrashing: its causes and prevention" (PDF https://cs.uwaterloo.ca/~brecht/courses/702/Possible-Readings/vm-and-gc/thrashing-denning-afips-1968.pdf);
  Denning, Kahn, Leroudier, Potier, Suri 1976 "Optimal multiprogramming", Acta Informatica 7:197-216 (https://link.springer.com/article/10.1007/BF00265771): knee criterion and L=S criterion (lifetime between faults L at least swap time S).
- Users: SillyTavern-style stacks (LLM + image gen + TTS on one GPU), ComfyUI + LM Studio guides (https://localai-kobo.com/comfyui-lmstudio/, https://markaicode.com/integrate/lm-studio-with-ollama/).

- Chu & Opderbeck 1972 PFF: AFIPS FJCC 1972 (https://dl.acm.org/doi/10.1145/1479992.1480077). Verified as a citation; content not read.
- "Why forgotten" for load control: no source found yet saying modern OSes dropped it; state as my inference.

- **Load control candidate now DEAD on filter 4 (same purpose, working code):** maxron84/Ollama-OpenWebUi-ComfyUi ("intelligent VRAM management" between LLM and image gen, https://github.com/maxron84/Ollama-OpenWebUi-ComfyUi),
  ComfyUI_OllamaUnload (https://github.com/bradsec/ComfyUI_OllamaUnload), cuda-vmm-swap (RAM<->VRAM model swap in ~2.3 s, https://github.com/bjivanovich/cuda-vmm-swap),
  vram-gpu-oom skill (OOM retry + auto-unload across Ollama/Whisper/ComfyUI), ComfyUI-VRAM-Manager, NVIDIA cuda-checkpoint (Linux). Mechanism differs (swap + thrash detection), purpose is the same.
- Atelico publishes frame-time benchmarks for AI next to games (https://atelico.studio/ai-engine/benchmarks/): co-tenancy is crowded.

- Energy pacing (Weiser 1994, YDS 1995) for local LLM decode: SM-clock locking for memory-bound decode is already measured (https://arxiv.org/abs/2605.11999, up to 32% decode energy) and in hobby guides. Drop.

- cuda-vmm-swap confirmed: Windows 11/WDDM, VMM sleep/wake of llama.cpp models between pinned RAM and VRAM in 1.7-2.3 s with pre-staging and a router, for multi-model agent pipelines on one GPU. maxron84 VRAM Manager: arbiter that calls ComfyUI /free when Ollama needs VRAM. => load control is taken in substance.
- Energy pacing re-check (my estimate): pacing to reading speed loses to race-to-idle when idle power is low (e.g. 70 W x 10 s + 10 W x 50 s = 1.2 kJ vs 30 W x 60 s = 1.8 kJ). Clock-locked race-to-idle is the known winner. DEAD.

- Out-of-process co-tenancy evidence: Atelico's own "Other baseline" (engine in a separate process contending with the game) shows OS context switching costs 45-55 FPS instead of 60 (https://atelico.studio/ai-engine/gpu-sharing/, https://atelico.studio/ai-engine/benchmarks/). CGS and NVIGI are in-process; SkyrimNet runs as an in-game DLL. npcvram: on 8 GB, modded Skyrim leaves no room for a local LLM (https://npcvram.com/).
  No out-of-process, frame-aligned scheduler for unmodified games found. Obvious baselines to beat: (a) default WDDM, (b) AI process at idle GPU priority (D3DKMTSetProcessSchedulingPriorityClass).
  => **Conditional survivor**: external slack-stealing scheduler (Liu & Layland 1973 periodic model; Lehoczky & Ramos-Thuel 1992 slack stealing). Risk: baseline (b) may already work.

- Verified: Lehoczky & Ramos-Thuel, RTSS 1992, slack stealing, proved optimal for soft-aperiodic response under fixed-priority periodic tasks (PDF https://igm.univ-mlv.fr/~masson/pdfANDps/lehoramos92.pdf).
  Heimdall (MobiCom 2020, https://dl.acm.org/doi/10.1145/3372224.3419192): in-app pseudo-preemption of DNNs on mobile AR, ~12 to ~30 fps. Paper, in-app; not a tool for external processes.

## Final outcome (2026-11-21)
- One conditional survivor: external slack-stealing scheduler for an AI process next to an unmodified, frame-capped game. Dies if Windows idle GPU priority already protects frames.
- Everything else died (see sections above).
- D1 Time-utility-function scheduling (Jensen, Locke & Tokuda 1985) for a local LLM server shared by autocomplete + chat + agents.
- D2 Frame-budgeted WebGPU inference in the browser.
- D3 3DGS training on 8 GB with known view order (Belady / Cao et al. 1995 integrated prefetching): check CLM.
