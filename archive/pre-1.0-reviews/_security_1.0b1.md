# Release 1.0b1 — Security review (FITS I/O / CLI)

**Status:** complete (merged from security-review subagent notes)  
**Scope:** Branch diff for 1.0b1 (API aliases + docs + CLI transform fix)

## Verdict

No **Blocking** or **Important** security issues in the changed surface.

## Blocking

None.

## Important

None.

## Nits

- Thin aliases (`read_torch`, `read_arrow`) delegate to existing validated paths; no new trust boundaries.
- CLI still accepts local filesystem paths as before; no new remote URL handling in this diff.

## Evidence

Security Review Task on `release/1.0b1` vs main; focus FITS I/O and CLI.
