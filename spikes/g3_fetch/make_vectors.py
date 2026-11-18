"""Gate G3 helper: write vectors.json with the byte ranges the browser prototype fetches.

For nine G3-only phrases (never benchmark or dev phrases), take the last token position and both Engram
layers: 24 rows per layer, each a 256-byte FP8 row plus its 8-byte E8M0 scale, so 96 ranges per
phrase. Also stores the SHA-256 of all fetched bytes in order, so the page can check it got the
right bytes.

Run where config.json and tokenizer.json are (see spikes/engram_readout):
  python make_vectors.py /path/to/Engramscope/spikes/g3_fetch/vectors.json
"""
import sys, json, hashlib, concurrent.futures as cf
import numpy as np
from tokenizers import Tokenizer
import killtest as K

REV = "314113f8a7dadb388215396ba5fbee2c610a04d6"
# Phrases used only by G3, so their rows were never fetched before (no warm CDN cache).
# Not in bench/phrases.jsonl or bench/dev_phrases.txt. The page runs each once (cold) and repeats the first (warm).
PHRASES = ["a lighthouse keeper", "the Andromeda galaxy", "Komodo National Park", "a bowl of ramen", "the Trojan horse",
           "Ki Hajar Dewantara", "a thunderstorm", "the Rosetta Stone", "Danau Kelimutu"]


def main(out):
    tok = Tokenizer.from_file("tokenizer.json")
    tmap, vocab = K.token_map(tok)
    pad = int(tmap[K.CFG["engram_pad_token_id"]])
    primes, offs = K.layout()
    mult = K.multipliers(vocab)
    heads = {lid: K.header(K.SHARD[lid]) for lid in K.LAYERS}
    res = {"revision": REV, "repo": "deepseek-ai/DeepSeek-V4.1-Flash", "phrases": []}
    for ph in PHRASES:
        ids = np.array(tok.encode(" " + ph).ids, dtype=np.int64)
        ranges = []
        for li, lid in enumerate(K.LAYERS):
            base, h = heads[lid]
            eo = base + h[f"layers.{lid}.engram.embed.weight"]["data_offsets"][0]
            so = base + h[f"layers.{lid}.engram.embed.scale"]["data_offsets"][0]
            rows = K.hash_ids(tmap[ids], pad, mult[li], primes[li], offs[li])[-1]
            for r in rows:
                ranges.append({"shard": K.SHARD[lid], "start": int(eo + int(r) * K.HD), "length": K.HD, "layer": lid, "row": int(r), "kind": "row"})
                ranges.append({"shard": K.SHARD[lid], "start": int(so + int(r) * (K.HD // 32)), "length": K.HD // 32, "layer": lid, "row": int(r), "kind": "scale"})
        with cf.ThreadPoolExecutor(16) as ex:
            blobs = list(ex.map(lambda g: K.rng(g["shard"], g["start"], g["length"]), ranges))
        res["phrases"].append({"phrase": ph, "token_ids": ids.tolist(), "ranges": ranges,
                               "sha256": hashlib.sha256(b"".join(blobs)).hexdigest(), "bytes": sum(map(len, blobs))})
        print(ph, len(ranges), "ranges", res["phrases"][-1]["bytes"], "bytes")
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
