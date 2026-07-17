# Release 1.0b1 — Rendered docs review

**Status:** complete
**Reviewer:** R1 rendered docs review
**Model:** claude-opus-4-8-thinking-high
**Branch:** release/1.0b1
**Built site:** `site/` (zensical-0.0.50, rebuilt 2026-07-17 14:29, newer than all reviewed `docs/*.md`)

## Verdict

**Ship with notes.**

The rendered site is coherent, on-message, and truthful. The tensors + dataframes
story is consistent from the hero through every reference page; return-type claims
match the code; KaTeX renders; the gallery images are present and embedded. One
user-visible RST role leak remains on the flagship API page — cosmetic, non-blocking,
one-line fix — so it ships with notes rather than clean.

- **Blocking:** 0
- **Should-fix:** 1
- **Defer:** 2

## Blocking

None.

## Should-fix

### S1 — RST `:mod:` role leaks literally on the API reference page
`docs/api.md` lines 131–132 use Sphinx/RST roles that mkdocs does not process, so
they render verbatim in the built HTML:

`site/api/index.html` (lines 2140–2142):
```html
<p>Transforms are not re-exported at the package root. Import them from
:mod:<code>torchfits.transforms</code>. HDU helpers are available as root names and via
:mod:<code>torchfits.hdu</code>.</p>
```
Readers see the raw `:mod:` prefix before each code span on the primary API landing
page. Fix: drop the `:mod:` roles and use plain inline code, e.g. “Import them from
`torchfits.transforms`. … also available via `torchfits.hdu`.” This is the only RST
leak in the docs tree (grep for `:class:` / `:mod:` / `:func:` / etc. across
`docs/*.md` returns only these two lines).

## Defer

### D1 — Benchmark provenance differs between the two migration pages
`docs/migration_astropy.md` cites `exhaustive_cuda_0.9.0_20260714_065950` while
`docs/migration_fitsio.md` cites the newer `exhaustive_cuda_20260717_042840`, and the
home page “Why torchfits?” table reuses the astropy (0.9.0-era) numbers. All are
internally labelled, but for a 1.0b1 release it would be cleaner to regenerate both
migration tables and the hero comparison from a single run. Non-blocking: numbers are
plausible, sourced, and clearly attributed.

### D2 — `torch.compile` support stated as uncertified
`docs/api-transforms.md` line 36 (“There is no certified compile matrix yet.”) is an
honest caveat, fine to ship. Track adding a compile-support table post-1.0.

## Evidence

### Site build / freshness
- `site/` already built; `git log` shows `docs/api.md` last touched 14:00, site
  rebuilt 14:29 — the inspected HTML reflects current sources. Did not need to re-run
  `pixi run docs-build`.
- All reviewed pages present under `site/`: `index.html`, `api/`, `api-core-io/`,
  `api-tables/`, `api-transforms/`, `api-data/`, `cli/`, `cli-recipes/`, `examples/`,
  `examples-transforms/`, `migration_astropy/`, `migration_fitsio/`, etc.

### Hero / home (`overrides/home.html`, `docs/index.md` → `site/index.html`)
- Hero lede (`site/index.html:399-400`): “to PyTorch tensors **and dataframes** —
  FITS tables as columnar catalogs.” Matches `site_description` in `zensical.toml:3`.
- Hero code sample (`site/index.html:408-409`) renders correctly, including the
  HTML-escaped predicate `torchfits.table.read("catalog.fits", where="MAG &lt; 20")`
  (displays as `MAG < 20` inside the `<code>` block — correct escaping, not a leak).
- Home body reinforces tensors↔images and dataframes↔tables (`docs/index.md:44-52,
  73-85`).

### Return-type truthfulness (docs vs. source)
- `table.read` documented as **`pyarrow.Table`** (“portable dataframe”), never as a
  `pandas.DataFrame`: `docs/api-tables.md:7-11,20-22,47-49`; `docs/api.md:19`.
  Source confirms: `src/torchfits/_table/read.py:1040` `read(...)` returns
  `pa.Table` (`pa.Table.from_arrays/from_batches`, lines 222-223, 643, 761, 1472).
- `read_arrow` is the same object as `read`: `src/torchfits/table.py:48`
  (`read_arrow = read`); doc claim `docs/api.md:20`, `docs/api-tables.md:65` verified.
- Root `read()` documented to return `dict[str, torch.Tensor]` for tables and to
  **not** return Arrow: `docs/api-core-io.md:26-27,33,39-40`. Source confirms:
  `src/torchfits/io.py:131-149` docstring returns `torch.Tensor` (images) or
  `dict[str, torch.Tensor]` (tables).
- `read_torch` → `dict[str, torch.Tensor]` (`docs/api-tables.md:83`, source
  `_table/read.py:1114`); `read_table` documented as root alias
  (`docs/api-core-io.md:164-166`, `docs/api.md:24`).
- `scale_on_device` kwarg referenced in `docs/migration_astropy.md:77` exists in code
  (`src/torchfits/_io_engine/options.py:16`, `_read_pipeline.py`), and the doc
  correctly notes `read_tensor` does not accept it.

### Table ≈ dataframe story coherence
- Consistent framing across `docs/api.md:7-10`, `docs/api-tables.md:1-14`,
  `docs/quickstart.md:72-94`, both migration pages, and `docs/examples.md:54-61`:
  Arrow default, Polars/Pandas one call away, tensor columns for training, namespace
  `torchfits.table` because it is the FITS name. No page claims the Arrow table *is* a
  pandas/Polars frame.

### KaTeX / arithmatex
- Config correct: `zensical.toml:95-96` enables `pymdownx.arithmatex` with
  `generic = true`; CSS/JS wired (`zensical.toml:10-19`) to katex + auto-render.
- `docs/javascripts/katex.js` delimiters (`$$`, `$`, `\(`, `\[`) match the generic
  arithmatex output. Built page emits matching markup, e.g.
  `site/api-transforms/index.html:2585` `<div class="arithmatex">\[…\]</div>` and
  `:2586` `<span class="arithmatex">\(…\)</span>` (70 arithmatex spans on that page).
- `katex.js` copied to `site/javascripts/katex.js`.

### Gallery images
- Source PNGs present: `docs/assets/gallery/` has all 7
  (`image_arcsinh`, `image_zscale`, `image_compose_pipeline`,
  `spectrum_continuum_normalize`, `spectrum_continuum_removal`,
  `spectrum_doppler_shift`, `lightcurve_phase_fold`).
- Copied into the build: `site/assets/gallery/` contains the same 7 files.
- Embedded and referenced in `docs/examples-transforms.md:25-62` with matching paths.

### RST leaks / backtick hygiene
- Only leak is S1 (`:mod:` in `docs/api.md:131-132`). No `:class:`/`:func:`/`:ref:`
  elsewhere.
- Apparent double-backticks in `docs/cli.md:89` (```` ``.arrow`` ````) render correctly
  as `<code>.arrow</code>` (`site/cli/index.html:1904`) — valid, not abuse.

### CLI docs (`docs/cli.md`, `docs/cli-recipes.md`)
- Subcommand table, exit codes, MEF defaults, and familiar-tool mapping are internally
  consistent; recipes reference real sample fixtures (`examples/_sample_data.py`,
  `examples/cli/imstat_imarith.sh`).

### Product code
- No product code modified. Only this review file was written.
