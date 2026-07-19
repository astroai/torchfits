# Thermo-nuclear review: CLI + docs polish (uncommitted, vs `HEAD`/`origin/main`)

Scope: ~401/300 across 24 files. Product code reviewed: `_io_engine/paths.py` (new),
`_io_engine/hdu_api.py`, `_hdu/hdu_list.py`, `cli/common.py`, `cli/cmds_convert.py`,
`cli/cmds_cutout.py`, and the inventory commands (`info`, `header`, `stats`, `table`,
`verify`, `probe`), plus `cli/cmds_transform.py`. Docs/README/zensical are cosmetic and
out of scope except where they lock in an API surface.

## Verdict: **Approve with required cleanup**

Behavior is fine and the tests cover the new paths. But this diff commits two structural
smells the rubric treats as presumptive blockers: **dead-on-arrival compatibility code**
and a **hand-rolled parser living in a shared helper module**. Both are cheap to remove
now and expensive to remove later. Clean up items 1 and 2 before committing; 3–5 are
strongly recommended.

---

## 1. Structural regression — the `emit_records` dual API is dead on arrival (required)

`emit_records` grew a `format=` parameter *plus* `json_mode=` / `jsonl=` "compatibility
aliases for callers not yet migrated" (`common.py:191-224`). But the **same diff migrated
every caller and every test** to `format=`:

```19:224:src/torchfits/cli/common.py
    if jsonl:
        format = "jsonl"
    elif json_mode:
        format = "json"
```

Grep confirms zero remaining users of the aliases anywhere in `src/` or `tests/`:

- `cmds_info`, `cmds_stats`, `cmds_probe` → `emit_records(records, format=resolve_emit_format(args))`
- `cmds_header`, `cmds_table`, `cmds_verify` → `emit_records(records, format=fmt)`

This is not a migration shim — there is nothing to migrate. `emit_records` is a **private
CLI helper** (`cli/common.py`), not a public API, so there is no external-compat argument
either. The `json_mode`/`jsonl` params and their normalization block are pure dead weight
that a future reader has to understand and a future refactor has to carry.

**Remedy (delete, don't polish):**

```python
def emit_records(
    records: Iterable[dict[str, Any]],
    *,
    format: str = "text",
    stream: TextIO | None = None,
) -> None:
    out = stream or sys.stdout
    items = list(records)
    if format == "jsonl":
        ...
```

Drop the two alias params, the docstring paragraph about them, and the `if jsonl / elif
json_mode` block. `resolve_emit_format` already collapses the flag surface into one
canonical string; `emit_records` should only ever speak that vocabulary.

Note: the *user-facing* `--json` / `--jsonl` flags in `add_emit_format_args` are a
different thing and are defensible (qfits/fitsverify muscle-memory), so keep those and keep
`resolve_emit_format`'s conflict checks — that is legitimate input validation at a trust
boundary.

## 2. Code-judo / boundary — hand-rolled YAML does not belong in `common.py` (required)

`_yaml_scalar` + `_yaml_dump` (~40 lines, `common.py:148-188`) reimplement a YAML emitter
inside a shared helper module, with a JSON round-trip for normalization. Two problems:

**a. Undocumented correctness ceiling.** `_yaml_scalar` only quotes on structural
special-chars. Plain strings that YAML 1.1 parsers coerce are emitted bare and read back
as the wrong type:

- `"yes"`, `"no"`, `"on"`, `"off"`, `"true"`, `"null"`, `"~"` → booleans/null
- numeric-looking strings like `"0123"`, `"1.0"`, `"1e3"` → numbers
- FITS logicals arrive as `"T"`/`"F"` strings and stay bare

The `ponytail:` comment claims the ceiling is "nested quirks," but the real ceiling is
**scalar type ambiguity**, which is exactly the class of bug that makes hand-rolled YAML a
trap. This is the rubric's "generic magic handling that hides simple data-shape
assumptions."

**b. It is bespoke where the layer should not own a serializer.** A structured-output
helper should dispatch to formatters, not *be* a YAML implementation.

**Remedy — pick one, in preference order:**

1. **YAGNI (preferred).** JSON already gives machine-readable output. Does the CLI actually
   need YAML? If nobody asked for it, drop `--format yaml`, delete `_yaml_dump`/`_yaml_scalar`,
   drop the doc rows, and the whole class of edge cases disappears. This is the "delete
   complexity rather than rearrange it" move.
2. If YAML output is genuinely wanted, take the already-installed-dependency rung: emit via
   a real library rather than hand-rolling a spec. Do not ship a partial YAML 1.1 emitter
   as a maintained surface.

At minimum, if it stays as-is, the `ponytail:` comment must name the *real* ceiling
(unquoted `yes`/`no`/`null`/numeric-looking strings deserialize to the wrong type), not
"nested quirks."

## 3. Spaghetti / branching — collapse the `cutout` branches (recommended)

`cmds_cutout.run` reads the header identically in both arms of the branch
(`cmds_cutout.py:54-62`):

```54:62:src/torchfits/cli/cmds_cutout.py
    if sectioned:
        tensor = torchfits.read_tensor(args.input, hdu=args.hdu)
        header = torchfits.get_header(args.input, args.hdu)
    else:
        x1, y1, x2, y2 = _parse_box(args.box)
        tensor = torchfits.read_subset(args.input, args.hdu, x1, y1, x2, y2)
        header = torchfits.get_header(args.input, args.hdu)
```

The only per-branch difference is how `tensor` is produced; `header` and `write_tensor`
are common. Hoist them so the branch carries only its real distinction:

```python
if sectioned:
    tensor = torchfits.read_tensor(args.input, hdu=args.hdu)
else:
    x1, y1, x2, y2 = _parse_box(args.box)
    tensor = torchfits.read_subset(args.input, args.hdu, x1, y1, x2, y2)
header = torchfits.get_header(args.input, args.hdu)
torchfits.write_tensor(args.output, tensor, header=header, overwrite=True)
```

The two guard clauses above (section-xor-box) are fine as-is — that is real input
validation, not spaghetti.

## 4. Boundary — `hdu_list.py` inline-imports a zero-dependency helper (recommended)

`hdu_list.py:31` imports the new helper *inside* `fromfile`:

```29:33:src/torchfits/_hdu/hdu_list.py
        import os

        from torchfits._io_engine.paths import cfitsio_base_path

        if mode == "r" and not os.path.exists(cfitsio_base_path(path)):
```

`hdu_api.py` imports the same helper at module top (`hdu_api.py:18`), which is the correct
form and violates no layering. `paths.py` has **zero imports** and
`_io_engine/__init__.py` is empty, so there is no circular-dependency justification for the
inline form — it just contradicts the no-inline-imports rule and the sibling module. Move
it to the top of `hdu_list.py` (the pre-existing inline `import os` is unrelated legacy;
promote it too while you are there). Keeping the same helper imported two different ways in
two files is exactly the "architectural drift" the rubric says not to normalize.

## 5. `paths.py` shape — fine, but keep the two helpers honest (minor)

`cfitsio_base_path` / `has_cfitsio_filter` are two tiny pure functions for two distinct
call sites (existence check vs CLI branching); keeping them separate is reasonable and not
over-abstraction. No change required. One caveat surfaced by the author's own honesty note:
`has_cfitsio_filter` is a pure `"[" in path` test, so it also returns True for pure-HDU
selectors like `file.fits[1]` / `[EVENTS]` that the docs admit do **not** work via
`open()`. `cmds_cutout` then routes those into the section branch where they will fail at
read time rather than at the guard. That is a latent sharp edge, not a blocker for this
diff (cutout is image-section only), but if HDU-selector paths ever reach these commands
the naming (`has_cfitsio_filter`) will over-promise. Leave a one-line note that this
detects *any* extended-filename bracket, not specifically an image section.

## File-size: N/A

No file is near 1k lines (`common.py` 242, `cmds_convert.py` 127, `cmds_cutout.py` 63,
`hdu_list.py` 305). The 1k-line rule is not triggered. `common.py` is the one to watch: it
is the catch-all CLI helper, and item 2 (delete the YAML emitter) is the right way to keep
it from accreting.

## `cmds_convert` / `cmds_transform` (no action)

- `cmds_convert` optional `--to` with `_EXT_TO_FORMAT` inference and the
  table-vs-png dispatch is clean, table-driven, and legible. Good.
- `cmds_transform` moving `import torchfits.transforms` into `run()` is a lazy-import for
  startup cost; that is a defensible *intentional* inline import (heavy optional submodule),
  unlike item 4. Fine.

---

### Top required cleanups before commit
1. Delete the dead `json_mode`/`jsonl` params from `emit_records` (item 1).
2. Remove or properly-source the hand-rolled YAML emitter; at minimum re-document its real
   ceiling (item 2).
3. Hoist the duplicated `header`/`write_tensor` out of the `cutout` branch and top-level the
   `hdu_list` import (items 3–4).
