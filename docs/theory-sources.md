# History sources: what we can say about the engram

The reference for every historical statement in the README, app, write-up and video (rule in `.kiro/steering/research-integrity.md`). If a claim isn't here, check the source first and add it.

Status 2026-11-11: **primary sources read** (Task 1.5, gate G5). Page numbers are the printed page numbers of the editions named. Quotes are short and exact; use them only as written here.

## Richard Semon (1859–1918), *Die Mneme* (1904)

Sources read:
- First edition: Semon, R. (1904). *Die Mneme als erhaltendes Prinzip im Wechsel des organischen Geschehens.* Leipzig: Wilhelm Engelmann. Scan of the University of Wisconsin–Madison copy: [archive.org/details/diemnemealserha01semogoog](https://archive.org/details/diemnemealserha01semogoog). Public domain.
- English translation: Semon, R. (1921). *The Mneme*, translated by Louis Simon. London: Allen & Unwin. Translated 1912–14 "with the benefit of the constant advice" of Semon (translator's preface). Scan of the Cornell copy: [archive.org/details/cu31924100387210](https://archive.org/details/cu31924100387210).

What he says:
- **Engram, defined** (1904, Ch. 2, p. 20; 1921, Ch. II, p. 24). Semon calls the lasting effect of a stimulus "engraphic", because it is, so to speak, dug or written into the organic substance ("sich in die organische Substanz sozusagen eingräbt oder einschreibt"). The change itself is the engram: "Die so bewirkte Veränderung der organischen Substanz bezeichne ich als das Engramm des betreffenden Reizes." The 1921 English: "I use the word engram to denote this permanent change wrought by a stimulus." The sum of an organism's engrams is its *Mneme*.
- **Why a new word** (same pages). He avoided "Gedächtnis" (memory) and "Erinnerungsbild" (memory image) because he needed a much wider sense than usual, and "memory image" is almost always tied to consciousness.
- **Association** (1904, Ch. 2, around p. 22; 1921, p. 27). A dog that has been beaten flinches at the mere sight of the whip. Engrams created at the same time are associated, so the return of one stimulus can retrieve the other's engram. Semon knew no exception to the rule that all simultaneously created engrams are associated.
- **Ecphory, defined** (1921, Ch. VI, p. 138): "the passage of an engram from a latent to a manifest state". The corresponding mnemic excitation "may be released by the partial recurrence of this stimulus-complex".
- **Localization** gets its own chapter (1921, Ch. V, "The Localisation of Engrams", from p. 113). Semon left open how engrams are physically stored, and called the question "hardly ripe yet for discussion" (p. 116).

## Karl Lashley (1890–1958), "In Search of the Engram" (1950)

Source read: Lashley, K. S. (1950). In search of the engram. *Symposia of the Society for Experimental Biology* 4 (*Physiological Mechanisms in Animal Behaviour*), 454–482. Cambridge University Press. A scan of the printed article is hosted at [github.com/dr-darryl-wright/Lashley_1950](https://github.com/dr-darryl-wright/Lashley_1950). It is still in copyright, so it is not copied into this repository. The scan is missing printed pages 476–477.

What he says:
- **Opening** (p. 454). A quote from Descartes about memory "traces" in the brain, and the remark that we seem little nearer to understanding the memory trace than Descartes was.
- **Equipotentiality** (p. 469). The evidence points to the memory trace being "located in all parts of the functional area", with the parts "equipotential for its maintenance and activation".
- **Mass action** (Section VII, from p. 469). Rats trained in a maze lost more of the habit the more cortex was removed. With 5–10% destroyed the loss "may be scarcely detectable"; with 50% or more the habit is completely lost.
- **The famous remark** (end on p. 478; it starts on p. 477, which is missing from the scan). The visible part reads: "that learning just is not possible. It is difficult to conceive of a mechanism which can satisfy the conditions set for it. Nevertheless, in spite of such evidence against it, learning does sometimes occur." **Quote only this part** until p. 477 is checked. The often-quoted opening of the sentence is not verified here.
- **Conclusions** (p. 478, numbered):
  - (2) "It is not possible to demonstrate the isolated localization of a memory trace anywhere within the nervous system." Also: "The engram is represented throughout the region."
  - (3) The associative areas "are not storehouses for specific memories."
- **Multiple representation** (conclusion 5, p. 479). The equivalence of cortical regions "points to multiple representation"; "equivalent traces are established throughout the functional area."

Criticism to read before the write-up (not read yet): a 1970 reanalysis argued that Lashley's retention data gave questionable support for mass action and equipotentiality, partly because of small samples ([PDF hosted by the University of Georgia](https://psyc.franklin.uga.edu/sites/default/files/inline-files/ReanalysisLashley1970.pdf)). **Unverified** until read; it matters because Engramscope can test these principles with bootstrap intervals over 123 phrases.

Secondary sources, kept for context:
- [Josselyn, Köhler & Frankland, "Heroes of the Engram", J. Neurosci. 37(18), 2017](https://pmc.ncbi.nlm.nih.gov/articles/PMC6596490/).
- [Bruce 2001, "Fifty years since Lashley's In search of the Engram", PubMed 11770197](https://pubmed.ncbi.nlm.nih.gov/11770197).

## Mapping to Engramscope

| Old idea | Engramscope | Label |
| --- | --- | --- |
| Engram: a lasting change written into a substance by a stimulus (Semon 1904, p. 20) | A row of DeepSeek-V4.1-Flash's Engram table, written by training | the model's designers chose the name; whether they meant Semon is not checked |
| Ecphory: an engram becomes active again when part of the original stimulus recurs (Semon 1921, p. 138) | The last 2–4 tokens of the text (part of the context) compute an address and retrieve a stored row | **ours**; an analogy that fits Semon's definition |
| Association of simultaneous engrams (Semon 1921, p. 27) | Rows retrieved together are summed into one value vector | **ours**; analogy only |
| Lashley could not find the isolated location of a memory (1950, p. 478) | Every row has an address computable from the text | **ours**; a contrast, not a claim about brains |
| Multiple representation: equivalent traces throughout an area (Lashley 1950, p. 479) | Each n-gram is stored under 8 independent hashes (heads), spread across the table | **ours**; a structural parallel only |
| Mass action: loss grows with the amount of cortex removed (Lashley 1950, Section VII) | Trace lesions of size 1–23 out of 24; retention curve (Req 9, protocol Q1) | **rerun of the method** on a machine memory |
| Equipotentiality: parts of an area are equivalent for a trace (Lashley 1950, p. 469; conclusion 2, p. 478) | Structured lesions (by n-gram size, by head) against random lesions of the same size; single-trace lesions (protocol Q2) | **rerun of the method** |
| Ecphory from a partial recurrence of the stimulus (Semon 1921, p. 138) | Retrieval from one n-gram size alone, and from the phrase after different preceding words (protocol Q3) | **rerun of the test** |
| Lesion experiments on the living animal (Lashley) | Zeroing traces in the running model (behavioural lesion, stretch, Req 10) | **adaptation**, if done |

Never claim that this says anything about biological memory. The experiments borrow the questions and methods; their results describe this model's memory only.
