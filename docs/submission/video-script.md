# Demo video script (under 3 minutes, public on YouTube)

Target length: 2:40. Narration is about 330 words at a calm pace.

## Before recording (checklist)

- **Add `?generic` to the end of every app address you record**, for example `https://sebastianabesantoso.github.io/Engramscope/?generic#/read?p=the%20Eiffel%20Tower`. The app then shows "an open-weight frontier model" in place of the model's name. Without it, the name appears on the pages. Before recording, open the Experiments and About pages and check the name doesn't appear.
- **No copyrighted music and no third-party logos.** Hide browser bookmarks and tabs, and don't show the Kiro logo or the GitHub avatar.
- **Device:** the rules want the device the project was built for. Record the phone (the app is a web app) and, if you get the laptop back, one desktop shot too.
- **Length:** keep it under 3:00 with a margin. The narration below is about 2:40 with pauses.
- Upload to YouTube as **Public**, then paste the link into the Devpost form.

## Shot list and narration

**0:00–0:15 · Title card, then the landing page**
> In 1950, Karl Lashley spent about thirty years cutting parts of rats' brains to find where a learned habit was stored. He never found it.

**0:15–0:35 · Home page, the findings paragraph**
> An open-weight frontier model has a memory where every trace has an address. For each phrase, it keeps 24 traces. So we can do what Lashley couldn't: find every trace, read it, and remove it.

**0:35–1:05 · Read a memory: type "the Kremlin"**
> Type a phrase. Your browser works out the model's own addresses, and fetches about 25 kilobytes from a 189-gigabyte table. The words it decodes are on the left. The grey words on the right come from random rows. They mean nothing.

Show: the token list, the bytes line ("Fetched … of 189 GiB"), the two word panels.

**1:05–1:35 · Live lesion: "the Eiffel Tower"**
> Now the experiment. I'll remove the two-word traces. Paris goes. Remove 8 random traces instead, and the concept mostly survives.

Show: tap **Remove the 2-word traces**, then read the recorded-result line aloud.

**1:35–2:00 · Experiments page**
> Lashley's three questions, asked of this memory. Half the concepts survive until 19 of the 24 traces are removed. Any head can stand in for another. The two-word traces can't be replaced. Both predictions, written before the run, held.

Show: the mass-action curve, then the head bars, then the n-gram bars. Scroll slowly.

**2:00–2:25 · The side test and what this does not show**
> Not everything worked. We tried adding the same kind of memory to small frozen models, against LoRA training. LoRA won at the smaller size, and a better-tuned LoRA beat the memory at the larger one. This is a measurement, not a new method, and these are trace lesions on a readout, not a claim about brains.

Show: the About page, the side-test section.

**2:25–2:40 · Built with Kiro, close**
> The whole project was built in Kiro, spec first. Feasibility gates ran before any app code. Steering held the research rules. Hooks are set up to run the tests. Everything is open source. Try it at the link below.

Show: only the `.kiro` folder's file list on GitHub, then the app URL on screen with the end card. The repository page's description and README name the model, so don't show them.

## End card (last 5 seconds)

App: sebastianabesantoso.github.io/Engramscope · Source: github.com/SebastianAbeSantoso/Engramscope · MIT license
