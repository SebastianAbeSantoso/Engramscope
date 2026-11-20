"""Kill test for language-model SMS compression. Criteria in PREREG.md (written first)."""

import base64
import csv
import json
import math
import random
import statistics as st
import subprocess
import sys
import time
import zlib

import brotli

SEED = 20261120
SMS_BITS = 1120
GSM7 = set(
    "@£$¥èéùìòÇ\nØø\rÅåΔ_ΦΓΛΩΠΨΣΘΞÆæßÉ !\"#¤%&'()*+,-./0123456789:;<=>?¡ABCDEFGHIJKLMNOPQRSTUVWXYZÄÖÑÜ§¿abcdefghijklmnopqrstuvwxyzäöñüà"
)
GSM7_EXT = set("^{}\\[~]|€")  # cost 2 septets each
TS = "ts_sms-2024-12-26"


def plain_bits(m):
    if all(ch in GSM7 or ch in GSM7_EXT for ch in m):
        return 7 * sum(2 if ch in GSM7_EXT else 1 for ch in m)
    return 16 * len(m.encode("utf-16-le")) // 2


def load():
    rng = random.Random(SEED)
    en = [line.split("\t", 1)[1].rstrip("\n") for line in open("uci/SMSSpamCollection", encoding="utf-8") if line.startswith("ham\t")]
    en = [m for m in en if m.strip()]
    rows = list(csv.DictReader(open("emot_train.csv", encoding="utf-8")))
    idn = [r["tweet"].strip() for r in rows if r["tweet"].strip()]
    return {"en": rng.sample(en, 200), "id": rng.sample(idn, 200)}


def ts_sms(m):
    t0 = time.time()
    c = subprocess.run([f"./{TS}/ts_sms", "c", m, "-F", "base64"], capture_output=True, text=True, check=True).stdout.strip()
    d = subprocess.run([f"./{TS}/ts_sms", "d", c, "-F", "base64"], capture_output=True, text=True, check=True).stdout
    d = d[:-1] if d.endswith("\n") else d
    return 8 * len(base64.b64decode(c)), d == m, time.time() - t0


def lm_bits(name, msgs):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(8)
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float32)
    model.eval()
    ctx = tok("Message: ", add_special_tokens=False).input_ids
    bos = [tok.bos_token_id] if tok.bos_token_id is not None else []
    eos = tok.eos_token_id
    out, t0 = [], time.time()
    for m in msgs:
        ids = tok(m, add_special_tokens=False).input_ids + [eos]  # end marker so the decoder knows where to stop
        full = torch.tensor([bos + ctx + ids])
        with torch.no_grad():
            logits = model(full).logits[0, :-1].float()
        logp = torch.log_softmax(logits, -1)
        start = len(bos + ctx) - 1
        tgt = full[0, len(bos + ctx) :]
        nll = -logp[start : start + len(ids)].gather(1, tgt[:, None]).sum().item() / math.log(2)
        out.append(nll + 2)
    return out, (time.time() - t0) / len(msgs)


def main():
    data = load()
    res = {"prereg": "PREREG.md", "seed": SEED, "langs": {}}
    for lang, msgs in data.items():
        chars = [len(m) for m in msgs]
        r = {"n": len(msgs), "median_chars": st.median(chars), "methods": {}}
        def summ(bits, label, extra=None):
            bpc = [b / c for b, c in zip(bits, chars)]
            fit = sum(b <= SMS_BITS for b in bits) / len(bits)
            r["methods"][label] = {"median_bpc": round(st.median(bpc), 3), "mean_bpc": round(sum(bits) / sum(chars), 3), "fit_one_sms": round(fit, 3), **(extra or {})}
            print(lang, label, r["methods"][label], flush=True)
        summ([plain_bits(m) for m in msgs], "plain SMS")
        summ([8 * len(zlib.compress(m.encode(), 9)[2:-4]) for m in msgs], "zlib")
        summ([8 * len(brotli.compress(m.encode(), quality=11)) for m in msgs], "brotli")
        ts = [ts_sms(m) for m in msgs]
        summ([b for b, _, _ in ts], "ts_sms (RWKV 169M)", {"roundtrip_failures": sum(not ok for _, ok, _ in ts), "sec_per_msg": round(sum(t for _, _, t in ts) / len(ts), 3)})
        for name in sys.argv[1:]:
            bits, spm = lm_bits(name, msgs)
            summ(bits, f"{name} (ideal AC)", {"sec_per_msg": round(spm, 3)})
        res["langs"][lang] = r
    json.dump(res, open("results.json", "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
