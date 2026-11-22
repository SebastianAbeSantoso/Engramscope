"""Kill test: an Engram-style memory bolted onto a frozen SmolLM2-135M, against LoRA (rules: PREREG.md).

Subcommands (run from this folder; ENGRAMSCOPE_OWNMEM = scratch dir for data, checkpoints, results):
  prep                       tokenize WikiText-103, build the streams and slices
  hashtest                   vectorised hash == engramscope.hashing.Hasher on real sequences
  smoke                      2 steps per method on random token ids (code check only)
  train METHOD LR TOKENS     train (resumable; stops after --budget seconds); METHOD = mem | lora
  pilot [METHOD]             the LR pilot (PREREG step 3), for one method or both
  eval RUN [MILESTONE]       evaluate a checkpoint on validation and the memorisation slice
  base                       evaluate the untrained model
"""

import json
import math
import os
import random
import re
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
from engramscope.hashing import EngramConfig, Hasher, bucket_primes, compressed_token_map  # noqa: E402

MODEL = "HuggingFaceTB/SmolLM2-135M"
SEED = 20261121
SEQ, BATCH = 512, 8
WARMUP = 20
INSERT_AFTER = 1  # 0-based decoder layer index (the module sits after layer 2)
N_HEADS, MAX_N, ROW_DIM = 4, 3, 64
PRIME_START = int(os.environ.get("OWNMEM_PRIME_START", "65536"))  # rows per column start here (bigger-table test: 1048576)
CKPT_EVERY = int(os.environ.get("OWNMEM_CKPT_EVERY", "300"))  # seconds between resumable checkpoints
SEL_TOKENS, MEMO_TOKENS, STREAM_TOKENS = 100_000, 200_000, 6_500_000
PILOT_TOKENS = 300_000
PILOT_LRS = {"mem": [3e-4, 1e-3, 3e-3], "lora": [1e-4, 3e-4, 1e-3]}
MILESTONES = [500_000, 1_000_000, 2_000_000, 6_000_000]
WORK = Path(os.environ.get("ENGRAMSCOPE_OWNMEM", "/projects/sandbox/_scratch/ownmem"))
torch.set_num_threads(8)


# ---------- data ----------
def tokenizer():
    from tokenizers import Tokenizer

    return Tokenizer.from_pretrained(MODEL)


def articles(parquet):
    import pyarrow.parquet as pq

    lines = pq.read_table(parquet).column("text").to_pylist()
    arts, cur = [], []
    for line in lines:
        if re.match(r"^ = [^=].* = \n?$", line):  # a top-level heading starts a new article
            if cur:
                arts.append("".join(cur))
            cur = [line]
        elif cur or line.strip():
            cur.append(line)
    if cur:
        arts.append("".join(cur))
    return [a for a in arts if a.strip()]


def pack(token_lists, eos, limit=None):
    flat = []
    for ids in token_lists:
        flat.extend(ids + [eos])
        if limit and len(flat) >= limit:
            break
    n = len(flat) // SEQ
    return np.array(flat[: n * SEQ], dtype=np.int64).reshape(n, SEQ)


def cmd_prep():
    tok = tokenizer()
    eos = tok.token_to_id("<|endoftext|>")
    arts = articles(WORK / "data/train-00000-of-00002.parquet")
    random.Random(SEED).shuffle(arts)
    enc = (tok.encode(a).ids for a in arts)
    sel, memo_and_stream, used = [], [], 0
    for ids in enc:  # selection slice: its own articles, never trained on
        sel.append(ids)
        used += len(ids) + 1
        if used >= SEL_TOKENS:
            break
    for ids in enc:  # the training stream (its first MEMO_TOKENS are the memorisation slice)
        memo_and_stream.append(ids)
        used += len(ids) + 1
        if used >= SEL_TOKENS + STREAM_TOKENS:
            break
    val = [tok.encode(a).ids for a in articles(WORK / "data/validation-00000-of-00001.parquet")]
    out = {"sel": pack(sel, eos), "stream": pack(memo_and_stream, eos), "val": pack(val, eos)}
    out["memo"] = out["stream"][: MEMO_TOKENS // SEQ]
    np.savez(WORK / "data/packed.npz", **out)
    print({k: f"{v.shape[0]} seq = {v.size:,} tokens" for k, v in out.items()}, f"articles: {len(arts):,}")


def data():
    return dict(np.load(WORK / "data/packed.npz"))


# ---------- memory ----------
class Hash:
    """Vectorised DeepSeek-style addresses for SmolLM2 (checked against engramscope.hashing.Hasher)."""

    def __init__(self):
        tok = tokenizer()
        cache = WORK / "data/token_map.npy"
        if not cache.exists():
            lookup, size = compressed_token_map(tok)
            np.save(cache, np.append(lookup, size))
        stored = np.load(cache)
        self.map, size = stored[:-1], int(stored[-1])
        pad = tok.token_to_id("<|endoftext|>")
        cfg0 = EngramConfig((INSERT_AFTER,), (0,), MAX_N, N_HEADS, ROW_DIM, PRIME_START, size, pad)
        self.cfg = replace(cfg0, num_embeddings=(int(bucket_primes(cfg0).sum()),))
        self.ref = Hasher(self.cfg, self.map, size)
        self.mult, self.primes, self.offs = self.ref.multipliers[0], self.ref.primes[0], self.ref.offsets[0]
        self.pad = self.ref.pad

    @property
    def rows(self):
        return self.cfg.num_embeddings[0]

    def __call__(self, ids):  # ids: int64 [B, T] -> rows [B, T, n_cols]
        c = self.map[ids]
        b, t = c.shape
        shifted = [c] + [np.concatenate([np.full((b, s), self.pad, dtype=np.int64), c[:, :-s]], axis=1) for s in range(1, MAX_N)]
        rolling = shifted[0] * self.mult[0]
        cols = []
        for i in range(1, MAX_N):
            rolling = np.bitwise_xor(rolling, shifted[i] * self.mult[i])
            for h in range(N_HEADS):
                col = (i - 1) * N_HEADS + h
                cols.append(rolling % self.primes[col] + self.offs[col])
        return np.stack(cols, axis=-1)


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.weight, self.eps = nn.Parameter(torch.ones(d)), eps

    def forward(self, x):
        x = x.float()
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class Memory(nn.Module):
    def __init__(self, rows, n_cols, hidden):
        super().__init__()
        self.table = nn.Embedding(rows, ROW_DIM, sparse=True)
        nn.init.normal_(self.table.weight, std=0.02)
        d = n_cols * ROW_DIM
        self.wk, self.wv = nn.Linear(d, hidden, bias=False), nn.Linear(d, hidden, bias=False)
        nn.init.zeros_(self.wv.weight)  # starts as an exact no-op
        self.qn, self.kn = RMSNorm(hidden), RMSNorm(hidden)
        self.hidden, self.rows_now, self.gate_sum, self.gate_n = hidden, None, 0.0, 0

    def forward(self, h):
        b, t, _ = h.shape
        e = self.table(self.rows_now).reshape(b, t, -1).float()
        k, v = self.wk(e), self.wv(e)
        gate = torch.sigmoid((self.qn(h) * self.kn(k)).sum(-1, keepdim=True) / math.sqrt(self.hidden))
        self.gate_sum += float(gate.detach().sum())
        self.gate_n += gate.numel()
        return h + (gate * v).to(h.dtype)


def load_model(method):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16)
    for p in model.parameters():
        p.requires_grad_(False)
    mem = None
    if method == "mem":
        hasher = Hash()
        mem = Memory(hasher.rows, (MAX_N - 1) * N_HEADS, model.config.hidden_size)

        def hook(_module, _inputs, output):
            if isinstance(output, tuple):
                return (mem(output[0]),) + tuple(output[1:])
            return mem(output)

        model.model.layers[INSERT_AFTER].register_forward_hook(hook)
        model.mem, model.hasher = mem, hasher
    elif method == "lora":
        from peft import LoraConfig, get_peft_model

        cfg = LoraConfig(r=64, lora_alpha=64, lora_dropout=0.0, bias="none",
                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
        model = get_peft_model(model, cfg)
    return model, mem


def trainable(model, mem):
    if mem is not None:
        return [mem.table.weight], [p for n, p in mem.named_parameters() if not n.startswith("table")]
    return [], [p for p in model.parameters() if p.requires_grad]


def step_loss(model, mem, batch):
    ids = torch.from_numpy(batch)
    if mem is not None:
        mem.rows_now = torch.from_numpy(model.hasher(batch))
    with torch.autocast("cpu", dtype=torch.bfloat16):
        return model(input_ids=ids, labels=ids).loss


@torch.no_grad()
def evaluate(model, mem, seqs):
    model.eval()
    if mem is not None:
        mem.gate_sum, mem.gate_n = 0.0, 0
    tot, n = 0.0, 0
    for i in range(0, len(seqs), BATCH):
        b = seqs[i : i + BATCH]
        loss = step_loss(model, mem, b)
        k = b.shape[0] * (SEQ - 1)
        tot, n = tot + float(loss) * k, n + k
    model.train()
    out = {"loss": tot / n, "ppl": math.exp(tot / n), "tokens": int(n)}
    if mem is not None:
        out["mean_gate"] = mem.gate_sum / max(1, mem.gate_n)
    return out


# ---------- training ----------
def run_dir(method, lr, tokens, tag=""):
    return WORK / "runs" / f"{tag}{method}-lr{lr:g}-{tokens}"


def train(method, lr, tokens, budget=1500, tag="", evals=None):
    d = run_dir(method, lr, tokens, tag)
    d.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    model, mem = load_model(method)
    sparse, dense = trainable(model, mem)
    w_lr = lr / 5 if method == "mem" else lr
    opts = [torch.optim.Adam(dense, lr=w_lr)]
    if sparse:
        opts.append(torch.optim.SparseAdam(sparse, lr=lr))
    base_lrs = [lr if isinstance(o, torch.optim.SparseAdam) else w_lr for o in opts]
    stream = data()["stream"]
    steps_total = -(-tokens // (SEQ * BATCH))  # round up so every milestone checkpoint is reached
    state = {"step": 0, "seconds": 0.0, "log": []}
    ck = d / "ckpt.pt"
    if ck.exists():
        saved = torch.load(ck, weights_only=False)
        params = sparse + dense
        for p, v in zip(params, saved["params"], strict=True):
            p.data.copy_(v)
        for o, s in zip(opts, saved["opts"], strict=True):
            o.load_state_dict(s)
        state = saved["state"]
    t_start, last_save = time.time(), time.time()

    def save(path, with_opt=True):
        blob = {"params": [p.detach() for p in sparse + dense], "state": state}
        if with_opt:
            blob["opts"] = [o.state_dict() for o in opts]
        torch.save(blob, path)

    while state["step"] < steps_total:
        s = state["step"]
        batch = stream[(s * BATCH) % len(stream) : (s * BATCH) % len(stream) + BATCH]
        scale = min(1.0, (s + 1) / WARMUP)
        for o, blr in zip(opts, base_lrs, strict=True):
            for g in o.param_groups:
                g["lr"] = blr * scale
        t0 = time.time()
        loss = step_loss(model, mem, batch)
        loss.backward()
        for o in opts:
            o.step()
            o.zero_grad(set_to_none=True)
        state["seconds"] += time.time() - t0
        state["step"] = s + 1
        if state["step"] % 10 == 0:
            state["log"].append([state["step"], float(loss)])
        done_tokens = state["step"] * SEQ * BATCH
        hit = [m for m in MILESTONES if m <= tokens and (s * SEQ * BATCH) < m <= done_tokens]
        for m in hit:
            save(d / f"milestone-{m}.pt", with_opt=False)
        if time.time() - last_save > CKPT_EVERY:
            save(ck)
            last_save = time.time()
        if time.time() - t_start > budget and state["step"] < steps_total:
            save(ck)
            print(f"paused at step {state['step']}/{steps_total} ({done_tokens:,} tokens, {state['seconds']:.0f}s)")
            return None
    save(ck)
    result = {"method": method, "lr": lr, "tokens": tokens, "train_seconds": state["seconds"], "log": state["log"]}
    if evals:
        for name, seqs in evals.items():
            result[name] = evaluate(model, mem, seqs)
    (d / "result.json").write_text(json.dumps(result, indent=1))
    print(json.dumps({k: v for k, v in result.items() if k != "log"}))
    return result


def eval_milestone(run, milestone):
    method = "mem" if "mem" in Path(run).name.split("-") else "lora"
    model, mem = load_model(method)
    sparse, dense = trainable(model, mem)
    saved = torch.load(Path(run) / f"milestone-{milestone}.pt", weights_only=False)
    for p, v in zip(sparse + dense, saved["params"], strict=True):
        p.data.copy_(v)
    dd = data()
    out = {"run": str(run), "milestone": milestone, "train_seconds": saved["state"]["seconds"],
           "val": evaluate(model, mem, dd["val"]), "memo": evaluate(model, mem, dd["memo"])}
    (Path(run) / f"eval-{milestone}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out))


def cmd_hashtest():
    h = Hash()
    seqs = data()["stream"][:2]
    fast = h(seqs)
    for i, s in enumerate(seqs):
        ref = h.ref.addresses(s.tolist())[:, 0, :]
        assert np.array_equal(fast[i], ref), f"mismatch in sequence {i}"
    print(f"hash ok: {seqs.size} positions x {fast.shape[-1]} columns match engramscope.hashing.Hasher;",
          f"compressed vocab {h.cfg.compressed_vocab_size:,}; table rows {h.rows:,}")


def cmd_smoke():
    rng = np.random.default_rng(0)
    fake = rng.integers(1, 40000, size=(BATCH, SEQ))
    for method in ("mem", "lora"):
        model, mem = load_model(method)
        sparse, dense = trainable(model, mem)
        n = sum(p.numel() for p in sparse + dense)
        opts = [torch.optim.Adam(dense, lr=1e-3)] + ([torch.optim.SparseAdam(sparse, lr=1e-3)] if sparse else [])
        for _ in range(2):
            t0 = time.time()
            loss = step_loss(model, mem, fake)
            loss.backward()
            for o in opts:
                o.step()
                o.zero_grad(set_to_none=True)
        print(f"{method}: {n:,} trainable params; step {time.time() - t0:.2f}s ({SEQ * BATCH / (time.time() - t0):.0f} tok/s)")


def cmd_pilot(only=None):
    sel = data()["sel"]
    choice_file = WORK / "runs/pilot-choice.json"
    best = json.loads(choice_file.read_text()) if choice_file.exists() else {}
    for method, lrs in PILOT_LRS.items():
        if only and method != only:
            continue
        scores = {}
        for lr in lrs:
            r = train(method, lr, PILOT_TOKENS, budget=10**9, tag="pilot-", evals={"sel": sel})
            scores[lr] = r["sel"]["ppl"]
        best[method] = min(scores, key=scores.get)
        print(method, "selection-slice PPL by LR:", scores, "-> chosen", best[method])
    choice_file.write_text(json.dumps(best, indent=1))


def cmd_base():
    model, mem = load_model("base")
    dd = data()
    out = {"val": evaluate(model, mem, dd["val"]), "memo": evaluate(model, mem, dd["memo"]),
           "sel": evaluate(model, mem, dd["sel"])}
    (WORK / "runs").mkdir(exist_ok=True)
    (WORK / "runs/base.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out))


if __name__ == "__main__":
    cmd, args = sys.argv[1], sys.argv[2:]
    budget, tag = 1500, ""
    if "--budget" in args:
        i = args.index("--budget")
        budget, args = int(args[i + 1]), args[:i] + args[i + 2 :]
    if "--tag" in args:
        i = args.index("--tag")
        tag, args = args[i + 1], args[:i] + args[i + 2 :]
    if cmd == "prep":
        cmd_prep()
    elif cmd == "hashtest":
        cmd_hashtest()
    elif cmd == "smoke":
        cmd_smoke()
    elif cmd == "pilot":
        cmd_pilot(args[0] if args else None)
    elif cmd == "base":
        cmd_base()
    elif cmd == "train":
        train(args[0], float(args[1]), int(args[2]), budget=budget, tag=tag)
    elif cmd == "eval":
        eval_milestone(args[0], int(args[1]))
    else:
        raise SystemExit(__doc__)
