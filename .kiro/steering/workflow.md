---
inclusion: always
---

# Workflow Rules

## Push at every checkpoint

- After every checkpoint (a finished task, a gate result, a script ready for the builder to run, or any point where work stops for review), commit and push before reporting back.
- Work on a branch and open or update a pull request. Do not commit directly to `main`.
- State where the work can be reviewed (branch or PR link) in the report.

## Attribution

- Commits are authored and committed as the builder: `Sebastian Abe Santoso <185665220+SebastianAbeSantoso@users.noreply.github.com>`, set through `GIT_AUTHOR_*` and `GIT_COMMITTER_*` environment variables for the command. Do not change git config.
- Pull requests and merges go through the GitHub API, which acts as the builder's account.

## Merging

- Merge a pull request only when the builder asks.

## Submission tracking

- At every checkpoint, update `docs/submission/deliverables.md` and add what happened to `docs/submission/kiro-writeup.md` (how Kiro was used, with evidence). Write only what actually happened; mark plans as planned and unverified items as "check".

## Builder steps

- When a step needs the builder's machine or browser, give exact commands or a link, say what to paste back, and say how long it takes.
