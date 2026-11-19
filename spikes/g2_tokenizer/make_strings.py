"""Gate G2: build the 1,000 test strings and the Python reference token ids.

Usage (from this directory, with tokenizer.json at the pinned revision next to it):
  python make_strings.py ../../bench
Writes strings.json and py_ids.json.
"""
import json, sys, random
from tokenizers import Tokenizer

BENCH = sys.argv[1] if len(sys.argv) > 1 else "../../bench"
rnd = random.Random(20261111)
tok = Tokenizer.from_file("tokenizer.json")

phrases = [json.loads(l)["phrase"] for l in open(f"{BENCH}/phrases.jsonl", encoding="utf-8")]
phrases += [l.strip() for l in open(f"{BENCH}/dev_phrases.txt", encoding="utf-8") if l.strip() and not l.startswith("#")]

hand = [
    "", " ", "  ", "\t", "\n", " \n\n ", "a", " a", "A", "  leading two spaces", "trailing space ", "tab\tinside", "line\nbreak", "crlf\r\nline",
    "北京是中国的首都。", " 北京", "我爱机器学习", "日本語のテキスト、カタカナとひらがな", "東京タワー", "한국어 문장입니다", "서울",
    "Saya suka makan nasi goreng di Yogyakarta.", "Selamat pagi, apa kabar?", "Bhinneka Tunggal Ika", "rumah-rumah", "anak2", "Gus Dur",
    "😀", " 😀", "👍🏽", "👨‍👩‍👧‍👦", "🇮🇩🇯🇵", "❤️", "🧠 memory 🧠", "🎃👻", "emoji😀inside",
    "café", "cafe\u0301", "naïve façade", "Søren Kierkegaard", "Zürich", "Ångström", "ﬁnance", "①②③", "ＡＢＣ　全角", "½ ¼", "™ ©",
    "straße", "İstanbul", "ΑΘΗΝΑ", "Москва", "القاهرة", "हिन्दी", "ภาษาไทย", "עברית", "Tiếng Việt",
    "1234567890", " 3.14159", "2026-11-11", "$1,000,000", "100%", "1e-20", "0x7fffffff",
    "def f(x):\n    return x**2", "if (a && b) { c(); }", "https://huggingface.co/deepseek-ai", "user@example.com", "#KiroweenHackathon26",
    "<｜begin▁of▁sentence｜>", "<｜end▁of▁sentence｜>hello", "<|endoftext|>", "Ġ", "▁", "\u200b", "\u00a0nbsp", "\ufeffbom",
    "DON'T can't I'm we'll", "\"quoted\" 'single'", "...", "!!!???", "—em dash–en dash", "a  b   c    d",
]

vocab = tok.get_vocab_size()
def vocab_mix():
    ids = [rnd.randrange(vocab) for _ in range(rnd.randint(1, 6))]
    return tok.decode(ids)

RANGES = [(0x20, 0x7E), (0xA0, 0x24F), (0x370, 0x3FF), (0x400, 0x4FF), (0x600, 0x6FF), (0x900, 0x97F), (0xE00, 0xE7F),
          (0x3040, 0x30FF), (0x4E00, 0x9FFF), (0xAC00, 0xD7A3), (0x1F300, 0x1FAFF), (0x2000, 0x206F), (0xFF00, 0xFFEF)]
def unicode_mix():
    out = []
    for _ in range(rnd.randint(1, 12)):
        a, b = rnd.choice(RANGES)
        out.append(chr(rnd.randint(a, b)))
        if rnd.random() < 0.2:
            out.append(" ")
    return "".join(out)

strings = []
for p in phrases:
    strings += [p, " " + p]
strings += hand
while len(strings) < 1000:
    strings.append(vocab_mix() if len(strings) % 2 else unicode_mix())
strings = strings[:1000]
assert len(strings) == 1000

ids = {"with_special": [tok.encode(s).ids for s in strings], "no_special": [tok.encode(s, add_special_tokens=False).ids for s in strings]}
json.dump(strings, open("strings.json", "w", encoding="utf-8"), ensure_ascii=False)
json.dump(ids, open("py_ids.json", "w"))
print(len(strings), "strings;", sum(map(len, ids["no_special"])), "tokens; first phrase ids:", ids["with_special"][1])
