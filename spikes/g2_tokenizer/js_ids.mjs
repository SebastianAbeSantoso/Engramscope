// Gate G2: compare browser-side tokenizers against the Python reference ids.
// Usage: node js_ids.mjs   (needs strings.json, py_ids.json, tokenizer.json, tokenizer_config.json here)
import fs from "node:fs";
import { Tokenizer } from "@huggingface/tokenizers";
import { AutoTokenizer, env } from "@huggingface/transformers";

const strings = JSON.parse(fs.readFileSync("strings.json", "utf8"));
const py = JSON.parse(fs.readFileSync("py_ids.json", "utf8"));
const tj = JSON.parse(fs.readFileSync("tokenizer.json", "utf8"));
const tc = JSON.parse(fs.readFileSync("tokenizer_config.json", "utf8"));

env.allowRemoteModels = false;
env.localModelPath = "./models/";
fs.mkdirSync("models/dsv41", { recursive: true });
for (const f of ["tokenizer.json", "tokenizer_config.json"]) fs.copyFileSync(f, `models/dsv41/${f}`);

const impls = {
  "@huggingface/tokenizers": new Tokenizer(tj, tc),
  "transformers.js AutoTokenizer": await AutoTokenizer.from_pretrained("dsv41"),
};

const same = (a, b) => a.length === b.length && a.every((x, i) => x === b[i]);
const report = {};
for (const [name, t] of Object.entries(impls)) {
  for (const mode of ["with_special", "no_special"]) {
    const opt = { add_special_tokens: mode === "with_special" };
    const t0 = performance.now();
    const got = strings.map((s) => Array.from(name.startsWith("transformers") ? t.encode(s, opt) : t.encode(s, opt).ids));
    const ms = performance.now() - t0;
    const bad = [];
    got.forEach((g, i) => { if (!same(g, py[mode][i])) bad.push({ i, s: strings[i], js: g, py: py[mode][i] }); });
    report[`${name} / ${mode}`] = { mismatches: bad.length, ms_total: Math.round(ms), examples: bad.slice(0, 8) };
    console.log(`${name} / ${mode}: ${bad.length} mismatches of ${strings.length}, ${ms.toFixed(0)} ms`);
    for (const b of bad.slice(0, 5)) console.log("  ", JSON.stringify(b.s), "js", JSON.stringify(b.js), "py", JSON.stringify(b.py));
  }
}
fs.writeFileSync("g2-report.json", JSON.stringify(report, null, 1));
