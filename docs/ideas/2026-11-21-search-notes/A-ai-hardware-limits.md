# Kiroween 2026 candidate search: working notes

## Plan
1. Read Molt/docs/ideas + handoff docs to avoid re-proposing ideas already checked.
2. Shortlist candidates where quantization / offload / speculation are NOT the obvious winner
   (lesson from MaltOE: "quantize harder until it fits" beats any VRAM-capacity trick).
3. Novelty-check each shortlisted candidate with web/GitHub/paper searches.
4. Write final report (<1,000 words), 1-3 ranked candidates + died list.

## Key structural insight so far
- Batch-1 decode bandwidth levers (fewer bytes, reuse bytes, skip bytes) are all on the taken list.
- Any VRAM-capacity scheme loses to "lower-bit quant that fits" unless the model still
  doesn't fit at ~3 bits. So look at limits quantization does not fix:
  compute-bound work (prefill, diffusion, encoders), deadlines (speech/video), energy, idle time.

## Died in head (before search)
- CPU+GPU concurrent row split (divisible load theory, Cheng & Robertazzi 1988): loses to
  "drop to Q3 so it fits"; PowerInfer/HeteGen also do concurrent CPU/GPU split.
- ABFT (Huang & Abraham 1984) + undervolting: decode is memory-bound, extra core undervolt
  margin saves ~5% board power; GreenMM (2019) paper already did ABFT+undervolt.
- GPU texture compression (BTC 1979 / BCn / ASTC) as weight format: same bytes as Q3/Q4,
  no 2D correlation in weights, no speed gain at batch 1 -> loses to K-quants.
- LM-head branch-and-bound top-k (Fukunaga & Narendra 1975): <=6.5% bytes on 8B models;
  FGD/L2S/FR-Spec exist.
- Progressive/embedded download for browser (Sloan & Tanimoto 1979): mechanism = bit-planes (taken).
- Zeroth-order fine-tuning (Kiefer-Wolfowitz 1952/SPSA): MeZO exists.
- Strassen/Winograd 1968 inner product: no gain on tensor cores / FMA.
- DVFS (Weiser 1994): static power cap baseline already captures most savings.
- Conditional replenishment (Mounts 1969) for streaming diffusion: Eventful Transformers, SIGE exist.

## Candidates still to check
A. Slack stealing (Lehoczky & Ramos-Thuel 1992) / real-time GPU reservation: LLM (or TTS/ASR)
   runs only in a game's per-frame GPU slack, read from PresentMon GPUBusy. Metric: game p99
   frame time at equal tok/s. Users: Skyrim Mantella / CHIM, streamers with local TTS.
   Baseline: WDDM time-slicing (+ process GPU priority). Prior art to check: NVIDIA In-Game
   Inferencing SDK, TimeGraph (2011), GPU priority tools.
   CHECKED: NVIGI SDK (NVIDIA) already schedules ASR/GPT/TTS alongside game graphics in-process
   (CUDA-in-Graphics + driver priority hints kPrioritizeGraphics/kBalance/kPrioritizeCompute),
   and ships a llama.cpp CIG patch. Doc says default CUDA = compute prioritized.
   https://docs.nvidia.com/nvigi-sdk/1.7.0/docs/nvigi_core/docs/GpuSchedulingForAI.html
   Remaining gap: unmodified games + separate LLM process. Theory: slack stealing gives the same
   throughput as background priority; only wins where no cross-process priority knob exists
   (CUDA on Windows). Vulkan has VK_EXT_global_priority -> modern baseline covers AMD.
   Searches (no cross-process frame-slack LLM scheduler found): "run local LLM in background
   without hurting game FPS GPU idle time between frames scheduler PresentMon";
   "github llama.cpp frame time game aware throttle inference yield GPU between frames".
   Found only how-to articles (itechguides/techyorker: "unload the model while you play").
   Risks: laptop power limit (filling slack raises power -> game clocks drop); PresentMon
   latency; CPU-inference baseline at matched tok/s.
   MORE: npcvram.com says on 8 GB modded Skyrim leaves no room for a local LLM (VRAM binds first,
   not GPU time). NVIDIA CIG llama.cpp patch tags llama.cpp launches "background" (in-process,
   needs the game's D3D device). Windows has D3DKMTSetProcessSchedulingPriorityClass (IDLE class)
   -> a preemptive priority scheduler gives the same throughput as slack stealing. A only wins if
   the driver ignores the priority class for CUDA. Verdict so far: conditional / weak.
   KILLED: NVIDIA App "GPU App Prioritization" (foreground/background GPU resource control,
   RTX Spark, HAGS required) per https://hdopti.com/articles/nvidia-gpu-app-prioritization-stutter-test ;
   Windows per-process GPU priority classes (Bitsum https://bitsum.com/gpu-priorities/ );
   NVIGI in-process. -> A = taken / ties a priority scheduler.
B. Belady MIN (1966) for model swapping in ComfyUI graphs -> DIED: gains ~1-5 s per item vs
   minutes of compute; community nodes (ComfyUI-UnloadModels) already free dead models.
   (original note:) Belady MIN (1966) for model swapping in ComfyUI graphs (future is known from the DAG).
   Cyclic multi-model workflows make LRU thrash. Check ComfyUI model_management policy,
   SwapAdvisor-style prior art.

## Verified links for the died list
- GreenMM (ICS 2019): ABFT + undervolting past Vmin for cuBLAS GEMM, energy down up to 19.8%:
  https://cs.ucr.edu/~dtrip003/publication/Website_GreenMM_ICS2019.pdf
- SIGE (code): computes only edited regions for GANs/diffusion: https://github.com/lmxyy/sige
- Power-throttle chunking, gains up to 2% on DGX Spark: https://arxiv.org/abs/2609.11936
- ComfyUI-UnloadModels: https://github.com/neezoy/ComfyUI-UnloadModels
- npcvram.com: 8 GB modded Skyrim leaves no room for a local LLM.
- Slack stealer primary-ish source: Davis, Tindell & Burns 1993 PDF
  http://igm.univ-mlv.fr/~masson/pdfANDps/davtinbur93.pdf (original: Lehoczky & Ramos-Thuel, RTSS 1992)

## Conclusion
No candidate passed all seven filters. Structural reason:
- batch-1 decode levers (fewer / reused / skipped bytes) are all on the taken list;
- VRAM capacity: "quantize until it fits" wins down to ~2.5 bits;
- where quantization doesn't apply (GPU time-sharing, idle time, power) a modern baseline exists
  (priority schedulers, prefix caching) or the measured gain is <= 2%.
