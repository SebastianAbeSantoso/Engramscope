# Kiroween 2026: rules check (pre-flight P3)

First checked 2026-11-16 against the handoff (`docs/elastic-kiroween-handoff.md`, "Kiroween rules and deadlines") from search excerpts, because Devpost returns HTTP 403 to automated fetches. Updated 2026-11-17 from the full text of the Devpost overview page and the **Official Rules** (version "Updated November 16, 2026, to include residents of Australia"), both pasted by the builder from a browser. Where the overview page and the Official Rules differ, the Official Rules prevail (Rules §11).

Entrant: **solo, a student** (builder, 2026-11-17).

Copied from the MaltOE repo on 2026-11-19, where it was written; the rules are the same for Engramscope. Project-specific notes below that mention MaltOE refer to the earlier project.

## Dates (Rules §1)

| Date (Pacific Time) | WITA (UTC+8) | Event |
| --- | --- | --- |
| Oct 13, 2026, 8:30 AM | Oct 13, 23:30 | Registration opens |
| Oct 30, 2026, 9:00 AM | Oct 31, 00:00 | Submission Period opens |
| Dec 1, 2026, 12:00 PM | Dec 2, 04:00 | Credit requests close (P1 done) |
| Dec 4, 2026, 2:00 PM | Dec 5, 06:00 | **Submission and registration close.** Unredeemed credit codes expire. No extensions, no late entries. |
| Dec 16, 2026, 1:00 PM | Dec 17, 05:00 | Judging Period opens |
| Dec 31, 2026, 11:59 PM | Jan 1, 15:59 | Redeemed credits expire |
| Jan 29, 2027, 1:00 PM | Jan 30, 05:00 | Judging Period ends. **The project must stay available to judges, free and unrestricted, until then.** |
| ~Feb 3, 2027, 9:00 AM | ~Feb 4, 01:00 | Winners announced |

October 13 and 30 are in PDT (UTC-7). The December and later dates are in PST (UTC-8).

**No difference from the handoff dates.** The handoff did not give the judging and announcement dates.

## Eligibility (Rules §3)

- Individuals who are at least the **age of majority where they reside**, at the time of entry. Teams of up to 3. A solo entrant is fine.
- **Excluded residents:** where US or local law prohibits taking part or receiving a prize, including Quebec, Argentina, Brazil, Italy, the Philippines, Russia, Cuba, Iran, North Korea, Syria, Belarus, Crimea, the DNR and LNR regions, the UAE, and other countries under comprehensive OFAC sanctions.
- Also excluded: employees, agents and household or immediate family of AWS, Devpost and other Promotional Entities; judges and their employers; and affiliates of any of these.
- Projects built with financial or preferential support from AWS or Devpost are excluded (§4).
- **The builder confirms their own eligibility.** This file does not infer it from the timezone (handoff rule). The Rules have a Bahasa Indonesia translation, if that helps.
- The rules also say Kiro features and models vary by region, and the entrant is responsible for regional restrictions (§4).

## Project requirements for Engramscope (Rules §4)

| Rule | What it means for Engramscope |
| --- | --- |
| A working app built with Kiro (IDE, CLI, Web, Crew); insufficient Kiro use can disqualify | Built in Kiro Web: specs, steering, hooks, sub-agents |
| One category | Resurrection (Semon 1904, Lashley 1950); see `docs/theory-sources.md` |
| Functionality on the intended platform | A static web app; intended platform: current desktop Chrome, Edge, Firefox, plus phone width (Req 7.2) |
| New project only; disclose pre-existing work | Started 2026-11-19. Disclose the model and reference code (MIT), tokenizer, libraries, and the MaltOE kill test (also built during the period) |
| Third-party integrations authorised | Model weights are MIT; fetched at runtime from Hugging Face, not redistributed |
| URL to the working app; test access until Jan 29, 2027 | GitHub Pages; model URLs pinned to a revision |

## Judging (Rules §6)

- **Stage One, pass/fail:** the project "reasonably fits the theme and reasonably applies the required tools". The README and video must make the Resurrection link and the Kiro use obvious early.
- **Stage Two:** five **equally weighted** criteria, 20% (10 points) each: Potential Impact, Technological Implementation (use of Kiro), Design ("not just a technical proof of concept"), Innovation/Idea, Presentation.
- Judging may use expert panels, peer review, **automated AI-driven analysis**, or a mix. The README, write-up and repo should therefore be easy to read cold, without the video.
- Ties are re-scored from scratch at 20 points per criterion.

## Prizes Engramscope can enter (Rules §8)

A project can win **one Overall Prize or one Category Prize, plus any number of Bonus Prizes.**

| Prize | Amount | How it is awarded | Engramscope |
| --- | --- | --- | --- |
| 1st / 2nd / 3rd | $30,000 / $20,000 / $10,000 | highest overall scores | yes |
| Best Resurrection | $5,000 | highest score in the category | yes |
| **Best Student Project: Resurrection** | $1,500 + 1-year Kiro Pro Max | highest-scoring eligible student entry in Resurrection | **yes (solo student).** The form needs LinkedIn, student email, year and degree, and school name. Proof of enrollment may be requested; reply in time or lose eligibility. |
| Best Demo | $1,000 | highest Presentation score among the top 10% overall | automatic |
| Social Blitz | 3-month Kiro Pro+ (5 winners) | post must **start with** `#KiroweenHackathon26`, posted during the period, linked in the form. Judged on "original thoughts vs AI writing" and post assets. The profile must match the submission details. | optional. **Write it yourself.** |
| Bonus Blog | 2,000 Kiro credits (top 100) | public post on a third-party platform during the period, saying it is written for Kiroween 2026 | optional |
| Startup, Ghost of Kiroween Past, Kiro Crew App | | | no |

Prizes need identity verification, a winner affidavit and tax forms (a W-8BEN for non-US residents) returned within 10 business days. Fees and taxes are the winner's.

## How the rules map to the spec

All submission rules are covered by Requirement 12 and tracked in `docs/submission/deliverables.md`:
- platform statement: a static web app on current desktop browsers and phone width (Req 7.2);
- testing access until Jan 29, 2027: GitHub Pages plus a pinned model revision (Req 11.5);
- disclosure of pre-existing work (Req 12.4);
- video rules (Req 12.3);
- the Kiro write-up (Req 12.6);
- the student prize details (D13).

## Open question: trademarks in the video

The rule says the video "must not include third party trademarks ... unless the Entrant has permission". Engramscope's demo would name DeepSeek-V4.1-Flash on screen. "DeepSeek" is used as a trademark; its registrations are contested (US filing raced by another applicant in January 2025, [TechCrunch](https://techcrunch.com/2025/01/29/deepseek-might-have-a-trademark-problem-in-the-u-s); EU refusal for software classes, [Knijff](https://iguana-parsnip-23eg.squarespace.com/en/blogs/deepseek-widespread-recognition-is-no-free-pass-to-trademark-registration)). Unregistered names can still be trademarks.

The Rules (§11) say an entrant who finds a term ambiguous must ask in writing. **Plan:** email `support@devpost.com` (pre-flight P3). Until answered: no logos; the model's name only in plain text where needed. This is not legal advice.
