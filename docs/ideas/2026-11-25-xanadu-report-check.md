# Idea check: the "Xanadu-Compile" report (2026-11-25)

The builder shared an outside report, "Xanadu-Compile Uses 1960 Transclusion Constraints to Make LLM Hallucination Structurally Impossible". It lists 30 ideas, each putting a hosted LLM inside an old deterministic system. It gives five finalists its own scores out of 50 and recommends Xanadu-Compile (46/50).

Checked against the bar in `2026-11-20-criteria-and-state.md`, starting with filter 3 (no existing working software).

## The five finalists: all taken

| Report pick | Mechanism | Already exists | Verdict |
| --- | --- | --- | --- |
| 1. Xanadu-Compile (46) | The LLM may output only source spans (offsets) plus a few connective words. A linker checks every span against the source bytes. | [Deterministic Quoting](https://simonwillison.net/2024/may/7/deterministic-quoting/) (Matt Yeung, Invetech, May 2024). [Anthropic Citations API](https://docs.claude.com/en/docs/build-with-claude/citations) (Jan 2025): citations "are guaranteed to contain valid pointers to the provided documents". [Google LangExtract](https://developers.googleblog.com/en/introducing-langextract-a-gemini-powered-information-extraction-library) (Jul 2025): every extraction is mapped to exact character offsets, with a highlighting viewer. [KR Labs VerbatimRAG](https://github.com/KRLabsOrg/verbatim-rag) (2025): the LLM only extracts exact spans, which are composed with no rewriting ([blog](https://huggingface.co/blog/adaamko/verbatimrag)). | out |
| 2. CapGate (44) | Capability tokens and taint labels block tool calls driven by injected text. | Google DeepMind CaMeL, ["Defeating Prompt Injections by Design"](https://arxiv.org/pdf/2503.18813) (Mar 2025), with code at [google-research/camel-prompt-injection](https://github.com/google-research/camel-prompt-injection). A [Codex CLI fork](https://github.com/nativ3ai/codex-cli-camel) already integrates it. | out |
| 3. MYCIN-Proof (42) | The LLM extracts facts. A MYCIN certainty-factor engine decides and shows the proof tree. | [Mopheshi/ExpertSystemforMedDiag](https://github.com/Mopheshi/ExpertSystemforMedDiag): Gemini plus a "MYCIN-inspired symbolic inference engine", with a GUI. | out |
| 4. Petri-Guard (41) | The LLM writes a workflow as a Petri net. A checker finds deadlocks and the LLM repairs them. | [ProMoAI](https://www.ijcai.org/Proceedings/2024/1014.pdf) (IJCAI 2024), a working tool that turns text into a Petri net or BPMN model, with error handling. [CIR+CVN](https://arxiv.org/html/2604.09318) (Apr 2026): LLM to Petri net, exhaustive search, counterexample-guided repair. | out |
| 5. Hole-Lock (40) | The LLM fills typed holes. The type checker keeps the rest of the program intact. | [Typed holes for LLMs](https://arxiv.org/abs/2409.00921) (OOPSLA 2024), built into the Hazel editor. | out |

## Problems with the report

- **Its prior-art check searched for the label, not the mechanism.** It found no app that enforces "1960 byte-range pointers as an anti-hallucination compiler pass". The mechanism itself ships in the four tools above.
- **Its own "Xanadu + agents" source shows the framing is taken.** The cited Hacker News item is Zed's ["Xanadu Was Waiting for Agents"](https://zed.dev/blog/agentic-xanadu) (Sep 2026), which applies "never copy, always reference" to agent work.
- **"Hallucination structurally impossible" is not an honest claim.** Span-only output prevents misquotes, not false conclusions. The model can still pick the wrong passage (the Deterministic Quoting write-up says so), leave out a qualifying sentence, or join two true quotes with "however" to suggest something neither says.
- **Every pick is a wrapper.** A hosted LLM sits inside an old checker, so the old idea guards someone else's model (filter 1). Nothing touches learning (filter 2), and none targets a hardware limit (the MaltOE-shaped filter). Guards like these are also the standard way LLM products are built, which is why the mechanisms already exist.
- **The scores are the report's own opinion**, not measurements.
- **Rules claims:** the five equally weighted criteria match Official Rules §6 (`Molt/docs/kiroween-rules.md`). The Dec 1 credit-request deadline is correct, and that step is already done. The 4,000-credit and $1,000 Kiro Crew amounts were not checked.

## The other 25

Not searched one by one. The report's own "skeptical judge" cut already drops them. Some repeat ideas that are already out:

- Baud-Telegraph and Morse-Brief are message compression, rejected earlier because `ts_sms` exists.
- TokenRing-85 is round-robin turn-taking between agents. AutoGen ships it as [RoundRobinGroupChat](https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/tutorial/teams.html).
- Cron-Batch-75 is a model router. [RouteLLM](https://github.com/lm-sys/RouteLLM) does this with trained routers.

## Result

Nothing in the report passes filter 3. The open decisions are unchanged: the 1.7B own-memory run (rented GPU or laptop), and when to start the Engramscope web app.
