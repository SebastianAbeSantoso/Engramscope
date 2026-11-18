# Gate G3: browser fetch prototype

`index.html` is a single self-contained page; no server is needed. Open it in a browser (download it and double-click, or open it from GitHub Pages later), press **Run**, then **Copy results**.

It resolves the CDN URL of the two Engram shards once, then fetches the 96 byte ranges (48 rows of 256 bytes plus their 8-byte scales, both layers) for three dev phrases. It does this at concurrency 8, 16 and 32 with the browser cache off, and checks the bytes against SHA-256 hashes computed in Python.

- `make_vectors.py` writes `vectors.json` (byte ranges and hashes) from the Python reference.
- `template.html` is the page; `index.html` is the template with `vectors.json` inlined:
  `python3 -c "import json;v=json.load(open('vectors.json'));t=open('template.html').read();open('index.html','w').write(t.replace('/*VECTORS*/null',json.dumps(v,separators=(',',':'))))"`
