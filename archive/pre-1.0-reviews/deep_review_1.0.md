# Deep review 1.0 (executed)

Working-tree triage from the 1.0.0rc3 deep reviews. Findings were executed;
full Round 6–7 dump: [`deep_review_1.0-r2.md`](deep_review_1.0-r2.md)
(not published on the docs site).

## What shipped (first triage + round 2)

| Area | Action |
|------|--------|
| Root aliases / `core.py` / spectral+continuum | Hard-removed |
| Lazy import race / logging | `_ATTR_CACHE`+`RLock`; `NullHandler` |
| R7-MUT1 / R7-CLI1 / R7-MUT2 / R7-MUT3 / R7-PIPE1 | Error UX; CLI kwargs; dtype maps; cache barrier; dead args |
| Audit follow-ups | `insert_column` numpy import; NAXIS>9; HTTP Range; remote-cache lock; zero batch |

## Still deferred

See [`.cursor/post-1.0-backlog.md`](../../.cursor/post-1.0-backlog.md).

## Bench report (round 2)

`.cursor/harness/trajectories/2026-07-19-deep-review-r2-bench-report.md`
