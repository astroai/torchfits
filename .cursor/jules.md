# Jules — torchfits scheduled task

Source of truth for [Jules](https://jules.google) scheduled tasks on `astroai/torchfits`.

## Human setup (once per schedule change)

Jules cannot edit schedules in place: delete the old task, then create a new one.

1. Open https://jules.google → Scheduled tab for this repo.
2. **Delete** daily / Design / a11y / Suggested-Task spam schedules.
3. Create **one Weekly** scheduled task on branch `main`.
4. Paste the prompt in [Weekly prompt](#weekly-prompt) below.
5. Optional: disable Suggested Tasks on this repo if they keep proposing cosmetic TODOs.

## Weekly prompt

```text
Weekly torchfits deep pass (bug + performance only). Read AGENTS.md and .cursor/jules-ledger.md first.

Research (≥30 min of reading before coding):
- Recent commits / CHANGELOG / roadmap deferred items
- Failures or flaky tests; open issues labeled bug/perf
- Hot paths: src/torchfits/_io_engine/, _table/, data/, transforms/, CLI probe/remote
- Existing benches under benchmarks/ — pick ONE concrete case if optimizing

Then do EXACTLY ONE of:
A) Correctness: one real bug with a minimal failing test (or assert) that fails before and passes after
B) Performance: one optimization with before/after numbers from an existing pixi bench task (same host, same case_id)

Rules:
- No a11y, HTML polish, renames, comment-only, docstring-only, or "refactor for clarity"
- No repeating themes listed in .cursor/jules-ledger.md or already merged Jules PRs
- One PR; evidence in the PR body; leave unrelated files alone
- If nothing serious is found: open NO PR — comment on the scheduled-task run that research found nothing
```

## Out of scope for Jules weekly

Examples gallery expansion, docs persona polish, SemVer releases, dependency bumps.
