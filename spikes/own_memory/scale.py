"""1.7B test: the bolted-on Engram-style memory against LoRA on a frozen SmolLM2-1.7B (rules: PREREG-1.7B.md).

Run on an NVIDIA GPU through laptop.ps1 (Windows) or rent.sh (Linux), or directly:
  python scale.py all     prep, check, base, both pilots, both main runs, report
Every step skips when its result exists, so rerunning the same command resumes after a stop.
Single steps: prep | check | base | pilot METHOD | train METHOD | report | gradcheck | dataequal
ENGRAMSCOPE_OWNMEM sets the working directory. SCALE_SMOKE=1 shrinks everything for a CPU code test.
"""

import gc
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

import ownmem as om

SMOKE = os.environ.get("SCALE_SMOKE") == "1"
MODEL = os.environ.get("SCALE_MODEL", "HuggingFaceTB/SmolLM2-1.7B")
if not MODEL.startswith("HuggingFaceTB/SmolLM2-"):
    raise SystemExit("only SmolLM2 models: the cached token map belongs to their shared tokenizer")
om.MODEL = MODEL
om.PRIME_START = int(os.environ.get("OWNMEM_PRIME_START", "242944"))  # 1,944,242 rows: 1.75x LoRA's parameters at 1.7B
WORK = om.WORK
SCALE = WORK / ("scale-smoke" if SMOKE else "scale")
REPO_RESULTS = Path(__file__).resolve().parent / "results" / "scale"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEQ, BATCH, SEED, WARMUP = om.SEQ, om.BATCH, om.SEED, om.WARMUP
STREAM_TOKENS = 20_600_000
BUDGETS = [20_000_000, 10_000_000, 5_000_000]
MAX_HOURS = 7.0
MILESTONES = [2_000_000, 5_000_000, 10_000_000, 20_000_000]
PILOT_TOKENS = 300_000
PILOT_LRS = {"mem": [1e-3, 3e-3, 1e-2], "lora": [3e-4, 1e-3, 3e-3]}
FLOOR = 0.1  # the cosine schedule ends at 10% of the peak learning rate
EVAL_BS = 4
CKPT_EVERY = 900
URL = "https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-103-raw-v1/"
FILES = {  # the files the 135M test used (sha256 from the sandbox)
    "train-00000-of-00002.parquet": "74da360f23826045b3e6ac6375411fdb15f003030aa74f2596ed08b857cb9212",
    "validation-00000-of-00001.parquet": "204929b7ff9d6184953f867dedb860e40aa69c078fc1e54b3baaa8fb28511c4c",
}
if SMOKE:
    BUDGETS, MILESTONES, PILOT_TOKENS = [4 * SEQ * BATCH], [2 * SEQ * BATCH, 4 * SEQ * BATCH], 2 * SEQ * BATCH
    PILOT_LRS, EVAL_BS = {"mem": [3e-3], "lora": [1e-3]}, 8


def say(*a):
    print(*a, flush=True)
    try:
        SCALE.mkdir(parents=True, exist_ok=True)
        with open(SCALE / "log.txt", "a", encoding="utf-8") as f:
            f.write(time.strftime("%Y-%m-%d %H:%M:%S ") + " ".join(str(x) for x in a) + "\n")
    except OSError:
        pass


def jwrite(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    tmp.replace(path)


def jread(path):
    return json.loads(Path(path).read_text())


def sync():
    if DEVICE == "cuda":
        torch.cuda.synchronize()


# ---------- data ----------
def download():
    import hashlib

    import requests

    for name, sha in FILES.items():
        path = WORK / "data" / name
        if not path.exists():
            say(f"downloading {name}")
            r = requests.get(URL + name, timeout=900)
            r.raise_for_status()
            path.write_bytes(r.content)
        got = hashlib.sha256(path.read_bytes()).hexdigest()
        if got != sha:
            raise SystemExit(f"{name}: sha256 {got} differs from the file the 135M test used ({sha})")


def prep():
    out = WORK / "data" / "packed-scale.npz"
    if out.exists():
        return
    (WORK / "data").mkdir(parents=True, exist_ok=True)
    download()
    tok = om.tokenizer()
    eos = tok.token_to_id("<|endoftext|>")
    arts = om.articles(WORK / "data/train-00000-of-00002.parquet")
    random.Random(SEED).shuffle(arts)  # the same order as the 135M test
    enc = (tok.encode(a).ids for a in arts)
    sel, stream, used = [], [], 0
    for ids in enc:
        sel.append(ids)
        used += len(ids) + 1
        if used >= om.SEL_TOKENS:
            break
    for ids in enc:
        stream.append(ids)
        used += len(ids) + 1
        if used >= om.SEL_TOKENS + STREAM_TOKENS:
            break
    val = [tok.encode(a).ids for a in om.articles(WORK / "data/validation-00000-of-00001.parquet")]
    packed = {"sel": om.pack(sel, eos), "stream": om.pack(stream, eos), "val": om.pack(val, eos)}
    packed["memo"] = packed["stream"][: om.MEMO_TOKENS // SEQ]
    np.savez(out, **packed)
    say("data:", {k: f"{v.shape[0]} sequences" for k, v in packed.items()})


_DATA = None


def data():
    global _DATA
    if _DATA is None:
        _DATA = dict(np.load(WORK / "data" / "packed-scale.npz"))
        if SMOKE:
            for k in ("sel", "val", "memo"):
                _DATA[k] = _DATA[k][:8]
    return _DATA


# ---------- model ----------
class CPUTableMemory(om.Memory):
    """om.Memory with the table kept in CPU RAM (Engram's design); only looked-up rows move to the GPU."""

    def forward(self, h):
        b, t, _ = h.shape
        e = self.table(self.rows_now).reshape(b, t, -1).to(h.device).float()
        k, v = self.wk(e), self.wv(e)
        gate = torch.sigmoid((self.qn(h) * self.kn(k)).sum(-1, keepdim=True) / math.sqrt(self.hidden))
        self.gate_sum += float(gate.detach().sum())
        self.gate_n += gate.numel()
        return h + (gate * v).to(h.dtype)


def load(method):
    from transformers import AutoModelForCausalLM

    free()  # release the previous model (also after an out-of-memory retry)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEVICE)
    model.config.use_cache = False
    for p in model.parameters():
        p.requires_grad_(False)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    mem = None
    if method == "mem":
        hasher = om.Hash()
        mem = CPUTableMemory(hasher.rows, (om.MAX_N - 1) * om.N_HEADS, model.config.hidden_size)
        for name in ("wk", "wv", "qn", "kn"):
            getattr(mem, name).to(DEVICE)

        def hook(_module, _inputs, output):
            if isinstance(output, tuple):
                return (mem(output[0]),) + tuple(output[1:])
            return mem(output)

        model.model.layers[om.INSERT_AFTER].register_forward_hook(hook)
        model.hasher = hasher
    elif method == "lora":
        from peft import LoraConfig, get_peft_model

        cfg = LoraConfig(r=64, lora_alpha=64, lora_dropout=0.0, bias="none",
                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
        model = get_peft_model(model, cfg)
    model.train()
    return model, mem


def trainable(model, mem):
    if mem is not None:
        return [mem.table.weight], [p for n, p in mem.named_parameters() if not n.startswith("table")]
    return [], [p for p in model.parameters() if p.requires_grad]


def step_loss(model, mem, batch):
    ids = torch.from_numpy(batch).to(DEVICE)
    if mem is not None:
        mem.rows_now = torch.from_numpy(model.hasher(batch))
    with torch.autocast(DEVICE, dtype=torch.bfloat16):
        return model(input_ids=ids, labels=ids).loss


@torch.no_grad()
def evaluate(model, mem, seqs):
    model.eval()
    if mem is not None:
        mem.gate_sum, mem.gate_n = 0.0, 0
    tot, n = 0.0, 0
    for i in range(0, len(seqs), EVAL_BS):
        b = seqs[i : i + EVAL_BS]
        k = b.shape[0] * (SEQ - 1)
        tot, n = tot + float(step_loss(model, mem, b)) * k, n + k
    model.train()
    out = {"loss": tot / n, "ppl": math.exp(tot / n), "tokens": int(n)}
    if mem is not None:
        out["mean_gate"] = mem.gate_sum / max(1, mem.gate_n)
    return out


def lr_scale(step, total):
    if step < WARMUP:
        return (step + 1) / WARMUP
    prog = min(1.0, (step - WARMUP) / max(1, total - WARMUP - 1))
    return FLOOR + (1 - FLOOR) * 0.5 * (1 + math.cos(math.pi * prog))


def free(*objs):
    del objs
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()


# ---------- training ----------
def train_run(method, lr, tokens, d, micro, milestones=(), evals=None, final_evals=None):
    d.mkdir(parents=True, exist_ok=True)
    if (d / "result.json").exists():
        return jread(d / "result.json")
    torch.manual_seed(SEED)
    model, mem = load(method)
    sparse, dense = trainable(model, mem)
    w_lr = lr / 5 if method == "mem" else lr
    opts = [(torch.optim.Adam(dense, lr=w_lr), w_lr)]
    if sparse:
        opts.append((torch.optim.SparseAdam(sparse, lr=lr), lr))
    stream = data()["stream"]
    steps_total = -(-tokens // (SEQ * BATCH))
    if steps_total * BATCH > len(stream):
        raise SystemExit("the training stream is too short for this budget")
    state = {"step": 0, "seconds": 0.0, "log": [], "evals": {}}
    ck = d / "ckpt.pt"
    if ck.exists():
        saved = torch.load(ck, weights_only=False)
        for p, v in zip(sparse + dense, saved["params"], strict=True):
            p.data.copy_(v.to(p.device))
        for (o, _), s in zip(opts, saved["opts"], strict=True):
            o.load_state_dict(s)
        state = saved["state"]
        say(f"{d.name}: resuming at step {state['step']}/{steps_total}")

    def save(path, with_opt=True):
        blob = {"params": [p.detach().cpu() for p in sparse + dense], "state": state}
        if with_opt:
            blob["opts"] = [o.state_dict() for o, _ in opts]
        torch.save(blob, str(path) + ".tmp")
        os.replace(str(path) + ".tmp", path)

    last_save = last_print = time.time()
    while state["step"] < steps_total:
        s = state["step"]
        batch = stream[s * BATCH : s * BATCH + BATCH]
        scale = lr_scale(s, steps_total)
        for o, base in opts:
            for g in o.param_groups:
                g["lr"] = base * scale
        t0 = time.time()
        total = 0.0
        for j in range(0, BATCH, micro):
            loss = step_loss(model, mem, batch[j : j + micro])
            (loss * (micro / BATCH)).backward()
            total += loss.item() * micro / BATCH
        for o, _ in opts:
            o.step()
            o.zero_grad(set_to_none=True)
        sync()
        state["seconds"] += time.time() - t0
        state["step"] = s + 1
        if state["step"] % 10 == 0:
            state["log"].append([state["step"], total])
        done = state["step"] * SEQ * BATCH
        for m in milestones:
            if s * SEQ * BATCH < m <= done:
                save(d / f"milestone-{m}.pt", with_opt=False)
                ev = {name: evaluate(model, mem, seqs) for name, seqs in (evals or {}).items()}
                ev["train_seconds"] = state["seconds"]
                state["evals"][str(m)] = ev
                jwrite(d / f"eval-{m}.json", ev)
                say(f"{d.name} at {m / 1e6:g}M tokens: " + ", ".join(f"{k} PPL {v['ppl']:.3f}" for k, v in ev.items() if isinstance(v, dict)))
        if time.time() - last_print > 60:
            tps = done / max(1e-9, state["seconds"])
            say(f"{d.name}: step {state['step']}/{steps_total}, {done / 1e6:.2f}M tokens, {tps:.0f} tok/s, about {(steps_total - state['step']) * SEQ * BATCH / tps / 3600:.1f} h left")
            last_print = time.time()
        if time.time() - last_save > CKPT_EVERY and state["step"] < steps_total:
            save(ck)
            last_save = time.time()
    save(ck)
    result = {"method": method, "lr": lr, "tokens": tokens, "steps": steps_total, "micro": micro,
              "train_seconds": state["seconds"], "evals": state["evals"], "log": state["log"]}
    for name, seqs in (final_evals or {}).items():
        result[name] = evaluate(model, mem, seqs)
    jwrite(d / "result.json", result)
    ck.unlink(missing_ok=True)  # finished: the milestones keep the trained parameters
    free(model, mem, opts)
    return result


# ---------- steps ----------
def env_info():
    import peft
    import transformers

    info = {"model": MODEL, "device": DEVICE, "torch": torch.__version__, "transformers": transformers.__version__,
            "peft": peft.__version__, "python": sys.version.split()[0], "prime_start": om.PRIME_START}
    if DEVICE == "cuda":
        info["gpu"] = torch.cuda.get_device_name(0)
        info["vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2)
        try:
            info["driver"] = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                                            capture_output=True, text=True).stdout.strip()
        except OSError:
            pass
    try:
        import psutil

        info["ram_gb"] = round(psutil.virtual_memory().total / 2**30, 1)
    except ImportError:
        pass
    return info


def speed(method, micro):
    steps, warm = (3, 1) if SMOKE else (12, 3)
    torch.manual_seed(SEED)
    model, mem = load(method)
    sparse, dense = trainable(model, mem)
    opts = [torch.optim.Adam(dense, lr=1e-4)] + ([torch.optim.SparseAdam(sparse, lr=1e-3)] if sparse else [])
    stream = data()["stream"]
    if DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats()
    times = []
    for s in range(steps):
        batch = stream[s * BATCH : (s + 1) * BATCH]
        t0 = time.time()
        for j in range(0, BATCH, micro):
            (step_loss(model, mem, batch[j : j + micro]) * (micro / BATCH)).backward()
        for o in opts:
            o.step()
            o.zero_grad(set_to_none=True)
        sync()
        times.append(time.time() - t0)
    out = {"micro": micro, "tok_per_s": SEQ * BATCH / float(np.mean(times[warm:]))}
    n = 8 if SMOKE else 32
    t0 = time.time()
    evaluate(model, mem, data()["val"][:n])
    sync()
    out["eval_tok_per_s"] = n * SEQ / (time.time() - t0)
    if DEVICE == "cuda":
        out["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    try:
        import psutil

        out["process_ram_gb"] = round(psutil.Process().memory_info().rss / 2**30, 2)
    except ImportError:
        pass
    out["trainable_params"] = sum(p.numel() for p in sparse + dense)
    free(model, mem, opts)
    return out


def gradcheck():
    """Gradient checkpointing must not change the memory's gradients."""
    prep()
    torch.manual_seed(SEED)
    model, mem = load("mem")
    with torch.no_grad():
        mem.wv.weight.normal_(0, 0.02)  # zero W_v would make most gradients zero
    batch = data()["stream"][:1]
    grads = []
    for ckpt in (True, False):
        if ckpt:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        else:
            model.gradient_checkpointing_disable()
        model.train()
        step_loss(model, mem, batch).backward()
        g = {n: p.grad.detach().float().clone() for n, p in mem.named_parameters() if not n.startswith("table")}
        g["table"] = mem.table.weight.grad.coalesce().values().float().clone()
        grads.append(g)
        for p in mem.parameters():
            p.grad = None
    rel = {k: float((grads[0][k] - grads[1][k]).norm() / (grads[1][k].norm() + 1e-30)) for k in grads[0]}
    norms = {k: float(grads[1][k].norm()) for k in grads[1]}
    free(model, mem)
    out = {"relative_difference": rel, "grad_norms": norms, "ok": max(rel.values()) < 2e-2 and min(norms.values()) > 0}
    say("gradcheck:", json.dumps(out))
    return out


def check():
    path = SCALE / "check.json"
    if path.exists():
        return jread(path)
    prep()
    say("checking speed and memory (about 10 minutes)")
    info = env_info()
    info["gradcheck"] = gradcheck()
    if not info["gradcheck"]["ok"]:
        raise SystemExit("gradient checkpointing changes the memory's gradients; stopping (see PREREG-1.7B.md)")
    for method in ("mem", "lora"):
        for micro in (4, 2, 1):
            try:
                info[method] = speed(method, micro)
                say(f"{method}: micro-batch {micro}, {info[method]['tok_per_s']:.0f} tok/s",
                    f"peak VRAM {info[method].get('peak_vram_gb')} GB" if DEVICE == "cuda" else "")
                break
            except torch.cuda.OutOfMemoryError:
                say(f"{method}: micro-batch {micro} ran out of GPU memory, trying a smaller one")
                free()
        else:
            raise SystemExit(f"{method} does not fit in GPU memory even at micro-batch 1")
    d = data()
    eval_tokens = (len(d["val"]) + len(d["memo"])) * SEQ

    def hours(method, budget):
        evals = sum(1 for m in MILESTONES if m <= budget) * eval_tokens / info[method]["eval_tok_per_s"]
        return (budget / info[method]["tok_per_s"] + evals) / 3600

    info["projected_hours"] = {f"{b // 1_000_000}M": {m: round(hours(m, b), 2) for m in ("mem", "lora")} for b in BUDGETS}
    info["budget"] = next((b for b in BUDGETS if max(hours(m, b) for m in ("mem", "lora")) <= MAX_HOURS), None)
    sel_tokens = len(d["sel"]) * SEQ
    info["projected_pilot_hours"] = round(sum(len(PILOT_LRS[m]) * (PILOT_TOKENS / info[m]["tok_per_s"] + sel_tokens / info[m]["eval_tok_per_s"])
                                              for m in ("mem", "lora")) / 3600, 2)
    jwrite(path, info)
    if info["budget"] is None:
        raise SystemExit("too slow: even 5M tokens would take over 7 hours per run; use a rented GPU (see LAPTOP.md)")
    b = info["budget"]
    say(f"budget: {b // 1_000_000}M tokens per method; pilot about {info['projected_pilot_hours']} h,",
        f"memory run about {hours('mem', b):.1f} h, LoRA run about {hours('lora', b):.1f} h")
    return info


def base():
    path = SCALE / "base.json"
    if path.exists():
        return jread(path)
    torch.manual_seed(SEED)
    model, _ = load("base")
    out = {name: evaluate(model, None, data()[name]) for name in ("val", "memo", "sel")}
    jwrite(path, out)
    say("base model:", ", ".join(f"{k} PPL {v['ppl']:.3f}" for k, v in out.items()))
    free(model)
    return out


def pilot(method):
    path = SCALE / "pilot-choice.json"
    choice = jread(path) if path.exists() else {}
    if method in choice:
        return choice[method]["chosen"]
    micro = check()[method]["micro"]
    scores = {}
    for lr in PILOT_LRS[method]:
        r = train_run(method, lr, PILOT_TOKENS, SCALE / "runs" / f"pilot-{method}-lr{lr:g}", micro,
                      final_evals={"sel": data()["sel"]})
        scores[f"{lr:g}"] = r["sel"]["ppl"]
        say(f"pilot {method} lr {lr:g}: selection PPL {r['sel']['ppl']:.3f}")
    chosen = float(min(scores, key=scores.get))
    choice[method] = {"chosen": chosen, "selection_ppl": scores}
    jwrite(path, choice)
    say(f"pilot {method}: chosen lr {chosen:g}")
    return chosen


def train(method):
    info = check()
    lr = pilot(method)
    budget = info["budget"]
    d = data()
    return train_run(method, lr, budget, SCALE / "runs" / f"main-{method}", info[method]["micro"],
                     milestones=[m for m in MILESTONES if m <= budget], evals={"val": d["val"], "memo": d["memo"]})


def report():
    info, b = check(), base()
    budget = info["budget"]
    res = {m: jread(SCALE / "runs" / f"main-{m}" / "result.json") for m in ("mem", "lora")}
    key = str(budget)
    pm, pl, pb = res["mem"]["evals"][key]["val"]["ppl"], res["lora"]["evals"][key]["val"]["ppl"], b["val"]["ppl"]
    r_mem, r_lora = 1 - pm / pb, 1 - pl / pb
    outcome = "PASS" if (r_mem >= 0.03 and pm <= pl) else "KILL"
    lines = [f"1.7B test ({info.get('gpu', DEVICE)}, torch {info['torch']}): budget {budget / 1e6:g}M tokens per method",
             f"base val PPL {pb:.3f}"]
    for m in ("mem", "lora"):
        ev = res[m]["evals"]
        lines.append(f"{m}: " + ", ".join(f"{int(k) / 1e6:g}M val {v['val']['ppl']:.3f} memo {v['memo']['ppl']:.3f}" for k, v in sorted(ev.items(), key=lambda kv: int(kv[0])))
                     + f"; trained {res[m]['train_seconds'] / 3600:.2f} h at lr {res[m]['lr']:g}")
    lines.append(f"r_mem {100 * r_mem:.2f}%, r_lora {100 * r_lora:.2f}%; PASS needs r_mem >= 3% and mem PPL <= LoRA PPL")
    lines.append(f"OUTCOME: {outcome}")
    summary = {"budget": budget, "base_val_ppl": pb, "mem_val_ppl": pm, "lora_val_ppl": pl,
               "r_mem": r_mem, "r_lora": r_lora, "outcome": outcome, "lines": lines}
    jwrite(SCALE / "summary.json", summary)
    if not SMOKE:
        REPO_RESULTS.mkdir(parents=True, exist_ok=True)
        for name in ("check.json", "base.json", "pilot-choice.json", "summary.json"):
            shutil.copy(SCALE / name, REPO_RESULTS / name)
        for m in ("mem", "lora"):
            src = SCALE / "runs" / f"main-{m}"
            r = jread(src / "result.json")
            jwrite(REPO_RESULTS / f"main-{m}.json", r)
    say("\n".join(["", "===== paste this into the chat =====", *lines, "===================================="]))
    return summary


EXTRA_LORA_LR = 3e-4  # exploratory follow-up, decided after the PASS (PREREG-1.7B.md, "Exploratory follow-up")


def extra():
    """Exploratory: LoRA at a lower learning rate, same budget, data, schedule and evaluations as main-lora."""
    path = SCALE / "extra.json"
    if path.exists():
        return jread(path)
    info, b, summary = check(), base(), jread(SCALE / "summary.json")
    budget, d = info["budget"], data()
    r = train_run("lora", EXTRA_LORA_LR, budget, SCALE / "runs" / f"extra-lora-lr{EXTRA_LORA_LR:g}", info["lora"]["micro"],
                  milestones=[m for m in MILESTONES if m <= budget], evals={"val": d["val"], "memo": d["memo"]})
    pe, pm, pb = r["evals"][str(budget)]["val"]["ppl"], summary["mem_val_ppl"], b["val"]["ppl"]
    verdict = "the PASS survives this check" if pm <= pe else "the PASS does not survive this check"
    lines = [f"Exploratory follow-up (not part of the pre-registered outcome, which stays {summary['outcome']})",
             f"lora at lr {EXTRA_LORA_LR:g}: " + ", ".join(f"{int(k) / 1e6:g}M val {v['val']['ppl']:.3f} memo {v['memo']['ppl']:.3f}"
                                                         for k, v in sorted(r["evals"].items(), key=lambda kv: int(kv[0])))
             + f"; trained {r['train_seconds'] / 3600:.2f} h",
             f"at {budget / 1e6:g}M: memory {pm:.3f}, lora lr {jread(SCALE / 'pilot-choice.json')['lora']['chosen']:g} {summary['lora_val_ppl']:.3f}, lora lr {EXTRA_LORA_LR:g} {pe:.3f}, base {pb:.3f}",
             f"RESULT: {verdict}"]
    out = {"lora_lr": EXTRA_LORA_LR, "budget": budget, "extra_lora_val_ppl": pe, "mem_val_ppl": pm,
           "main_lora_val_ppl": summary["lora_val_ppl"], "base_val_ppl": pb, "verdict": verdict, "lines": lines}
    jwrite(path, out)
    say("\n".join(["", "===== paste this into the chat =====", *lines, "===================================="]))
    return out


def dataequal():
    """Sandbox check: the 1.7B data starts exactly like the 135M test's data."""
    a, b = data(), dict(np.load(WORK / "data" / "packed.npz"))
    same = {k: bool(np.array_equal(a[k], b[k])) for k in ("sel", "val", "memo")}
    same["stream_prefix"] = bool(np.array_equal(a["stream"][: len(b["stream"])], b["stream"]))
    say("data equal to the 135M test:", same, f"stream {len(a['stream'])} vs {len(b['stream'])} sequences")
    return all(same.values())


def run_all():
    prep()
    check()
    base()
    pilot("mem")
    pilot("lora")
    train("mem")
    train("lora")
    report()


if __name__ == "__main__":
    SCALE.mkdir(parents=True, exist_ok=True)
    cmd, args = (sys.argv[1] if len(sys.argv) > 1 else "all"), sys.argv[2:]
    steps = {"all": run_all, "prep": prep, "check": check, "base": base, "report": report,
             "gradcheck": gradcheck, "dataequal": dataequal, "extra": extra}
    if cmd in ("pilot", "train"):
        {"pilot": pilot, "train": train}[cmd](args[0])
    elif cmd in steps:
        steps[cmd]()
    else:
        raise SystemExit(__doc__)
