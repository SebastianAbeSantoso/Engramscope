# Demo video script (under 3 minutes, public on YouTube)

Target length: about 2:50. Narration is about 300 words, read at a calm pace. Times are approximate.

## Before recording (checklist)

- **Add `?generic` to the end of every app address you record**, for example `https://sebastianabesantoso.github.io/Engramscope/?generic#/read?p=the%20Eiffel%20Tower`. The app then shows "an open-weight frontier model" in place of the model's name. Without it, the name appears on the pages. Before recording, open the Experiments and About pages and check the name doesn't appear.
- **No copyrighted music and no third-party logos.** Hide browser bookmarks and tabs. Don't show the Kiro logo or your GitHub avatar.
- **Device:** record on the phone, since the app is a web app. If you get the laptop back, add one desktop shot.
- **Kiro segment:** show files, not the GitHub repository page. The repository's description and README name the model, so don't show them.
- Upload to YouTube as **Public**, then paste the link into the Devpost form.

## Full script

**0:00 · Title card, then the landing page**
> In 1950, Karl Lashley spent about thirty years cutting parts of rats' brains to find where a learned habit was stored. He never found it.

**0:15 · Home page findings**
> An open-weight frontier model has a memory where every trace has an address. For each phrase, it keeps 24 traces. So we can do what Lashley couldn't: find every trace, read it, and remove it.

**0:35 · Read a memory: type "the Kremlin"**
> Type a phrase. Your browser works out the model's own addresses, and fetches about 25 kilobytes from a 189-gigabyte table. The words it decodes are on the left. The grey words on the right come from random rows. They mean nothing.

*Show the token list, the "Fetched … of 189 GiB" line, and both word panels.*

**1:05 · Live lesion: "the Eiffel Tower"**
> Now the experiment. I'll remove the two-word traces. Paris goes. Remove 8 random traces instead, and the concept mostly survives.

*Tap "Remove the 2-word traces", then read the recorded-result line aloud.*

**1:35 · Experiments page**
> Lashley's three questions, asked of this memory. Half the concepts survive until 19 of the 24 traces are removed. Any head can stand in for another. The two-word traces can't be replaced. Both predictions, written before the run, held.

*Show the mass-action curve, the head bars, then the n-gram bars. Scroll slowly.*

**2:00 · Side test and what this does not show**
> Not everything worked. We tried adding the same kind of memory to small frozen models, against LoRA training. LoRA won at the smaller size, and a better-tuned LoRA beat the memory at the larger one. This is a measurement, not a new method, and these are trace lesions on a readout, not a claim about brains.

*Show the About page, side-test section.*

**2:20 · Kiro: the spec**
> The whole project was built in Kiro, spec first. Here is the spec: the requirements, the design and the task list. The feasibility gates are ticked, and they ran before any app code.

*Open `.kiro/specs/engramscope/tasks.md` on GitHub. Scroll to Task 1, showing the gates, and pause on the G1 line that says the planned index failed.*

**2:32 · Kiro: steering, hook and tests**
> Steering files hold the research rules. A post-task hook runs the linters and tests. Property-based tests check the hash against the model's reference code.

*Open `.kiro/steering/research-integrity.md`, then `.kiro/hooks/checks-after-task.json`, then `tests/test_hashing.py`, scrolling to a Hypothesis test.*

**2:45 · Close**
> Everything is open source, and the full account of how Kiro was used is in the write-up. Try the app at the link below.

*Show the app URL on screen with the end card.*

## End card (last 5 seconds)

App: sebastianabesantoso.github.io/Engramscope · Source: github.com/SebastianAbeSantoso/Engramscope · MIT license
