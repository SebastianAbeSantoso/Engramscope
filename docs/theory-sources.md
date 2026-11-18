# History sources: what we can say about the engram

The reference for every historical statement in the README, app, write-up and video (rule in `.kiro/steering/research-integrity.md`). If a claim isn't here, check the source first and add it.

Status 2026-11-19: **checked against secondary sources only.** The primary sources (Semon 1904, Lashley 1950) have not been read yet; Task 1.5 does that.

## Richard Semon (1859–1918)

- Introduced the term *engram* for the substrate of a memory trace in *Die Mneme* (1904). Also coined *ecphory* for its retrieval. ([Josselyn, Köhler & Frankland, "Heroes of the Engram", J. Neurosci. 37(18), 2017](https://pmc.ncbi.nlm.nih.gov/articles/PMC6596490/); [Wikipedia: Richard Semon](https://en.wikipedia.org/wiki/Richard_Semon))
- **To read:** *Die Mneme* (1904), or its English translation *The Mneme* (1921), for his own definition of engram and ecphory.

## Karl Lashley (1890–1958)

- "In Search of the Engram", 1950, summarised about 30 years of experiments: rats learned mazes, then parts of their cortex were removed to find where the memory was stored. ([Bruce 2001, "Fifty years since Lashley's In search of the Engram", PubMed](https://pubmed.ncbi.nlm.nih.gov/11770197); [OpenStax Psychology 2e, 8.2](https://openstax.org/books/psychology-2e/pages/8-2-parts-of-the-brain-involved-with-memory))
- Conclusion as summarised by Bruce (2001): memories are not localized but distributed within functional areas of the cortex.
- Associated principles: *mass action* (the amount of cortex removed mattered, not its location) and *equipotentiality*.
- His closing remark that learning "just is not possible" is quoted in later reviews ([Springer chapter](https://link.springer.com/chapter/10.1007/978-1-4615-9843-5_1)).
- **To read:** Lashley, K. S. (1950). In search of the engram. *Symposia of the Society for Experimental Biology* 4, 454–482. Check the exact wording of the "learning just is not possible" remark before quoting it.

## Mapping to Engramscope

| Old idea | Engramscope | Label |
| --- | --- | --- |
| Engram: a physical memory trace (Semon) | A row of DeepSeek-V4.1-Flash's Engram table | the model's designers chose the name; the analogy is theirs |
| Ecphory: retrieving an engram (Semon) | Computing the address and fetching the row | **ours** (an analogy; to be checked against Semon's definition) |
| Lashley could not find where a memory lives | Here every row has an address computable from the text | **ours** (a contrast, not a claim about brains) |
| Lesion experiments (Lashley) | Zeroing a phrase's rows and testing the model (stretch, Req 9) | **adaptation** |

Never claim that this says anything about biological memory.
