"""Exploratory diagnostic (not pre-registered): do the trained table rows carry n-gram-specific content?

Loads a trained memory checkpoint, then scores validation perplexity three ways:
  trained   the rows as trained
  shuffled  the same rows, randomly permuted (each n-gram gets another n-gram's trained row)
  mean      every row replaced by the mean trained row
If 'shuffled' and 'mean' score about the same as 'trained', the gain does not come from what
individual rows store but from the dense part (W_k, W_v, gate, norms) acting on generic rows.

  python diag_shuffle.py RUN_DIR MILESTONE
"""

import json
import sys
from pathlib import Path

import torch

import ownmem as om

run, milestone = Path(sys.argv[1]), int(sys.argv[2])
model, mem = om.load_model("mem")
sparse, dense = om.trainable(model, mem)
saved = torch.load(run / f"milestone-{milestone}.pt", weights_only=False)
for p, v in zip(sparse + dense, saved["params"], strict=True):
    p.data.copy_(v)
val = om.data()["val"]
trained = mem.table.weight.data.clone()
out = {"run": str(run), "milestone": milestone, "rows": int(trained.shape[0])}
out["trained"] = om.evaluate(model, mem, val)
g = torch.Generator().manual_seed(om.SEED)
mem.table.weight.data.copy_(trained[torch.randperm(trained.shape[0], generator=g)])
out["shuffled"] = om.evaluate(model, mem, val)
mem.table.weight.data.copy_(trained.mean(0, keepdim=True).expand_as(trained))
out["mean"] = om.evaluate(model, mem, val)
(run / f"diag-shuffle-{milestone}.json").write_text(json.dumps(out, indent=1))
print(json.dumps({k: (v["ppl"] if isinstance(v, dict) else v) for k, v in out.items()}))
