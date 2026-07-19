# Agent guidance (torchfits)

Pixi-first: use `pixi run …`, never bare `python` for project work.

## Verify tiers

| When | Command |
|---|---|
| During edits | `pixi run preflight-push` |
| Before push / PR | `pixi run ci-local` |
| Before tag | `pixi run release-gate` |

Durable notes live under `.cursor/harness/` (playbook, trajectories) — not long
chat scrollback. Deferred product work: [`.cursor/post-1.0-backlog.md`](.cursor/post-1.0-backlog.md).

## Humans / coding agents

- Docs must match the public façade (`docs/api*.md`); env vars must exist in `src/`.
- Prefer smallest correct diffs; no new dependencies without a clear need.
- Pre-tag public-API audit: [`.cursor/skills/release-api-freeze-review/SKILL.md`](.cursor/skills/release-api-freeze-review/SKILL.md).

## Jules / autonomous PR agents

Prefer **correctness** and **measurable performance** over style, a11y drive-bys,
renames, comment churn, or docs-only nits.

Before opening a PR:

1. Read [`.cursor/jules-ledger.md`](.cursor/jules-ledger.md) and search recently
   merged Jules PRs — **do not repeat a theme already landed**.
2. Cite evidence: a failing test / assert that fails before and passes after, a
   CFITSIO or public API contract, or before/after timing from an existing
   `pixi run bench-*` case (same host, same `case_id`).
3. One logical change per PR; title names the bug or the bench case.
4. No new dependencies; no SemVer bumps; no force-push.
5. Run `pixi run preflight-push` (or the smallest relevant pytest) before opening
   the PR.

Out of scope unless a human explicitly labels the issue: HTML / `repr_html`
cosmetics, markdown wording, “clean up” without a repro.

If research finds nothing serious: **open no PR**.

Weekly Jules prompt: [`.cursor/jules.md`](.cursor/jules.md).
