# Agent guidance (torchfits)

Pixi-first: use `pixi run …`, never bare `python` for project work.

## Jules / autonomous PR agents

Prefer **correctness** and **measurable performance** over style, a11y drive-bys, renames, comment churn, “hoist class”, or docs-only nits.

Before opening a PR:

1. Read [`docs/jules-ledger.md`](docs/jules-ledger.md) and search recently merged Jules PRs — **do not repeat a theme already landed**.
2. Cite evidence: a failing test / assert that fails before and passes after, a CFITSIO or public API contract, or before/after timing from an existing `pixi run bench-*` case (same host, same `case_id`).
3. One logical change per PR; title names the bug or the bench case.
4. No new dependencies; no SemVer bumps; no force-push.
5. Run `pixi run preflight-push` (or the smallest relevant pytest) before opening the PR.

Out of scope unless a human explicitly labels the issue: HTML / `repr_html` cosmetics, markdown wording, “clean up” without a repro.

If research finds nothing serious: **open no PR**.

Weekly Jules prompt source of truth: [`JULES.md`](JULES.md).

## Humans / coding agents shipping features

- Verify tiers: `pixi run preflight-push` during edits; `pixi run ci-local` / `release-gate` before push/tag.
- Durable notes: `.cursor/harness/` (playbook, trajectories), not long chat scrollback.
- Docs must match the public façade (`docs/api*.md`); env vars must exist in `src/`.
