# Release 1.0b1 — API usefulness / adoption review

**Status:** complete  
**Model:** gpt-5.6-sol-high

## Verdict

**Block.** The API and documentation have a good adoption story, but the
release currently identifies itself as `0.9.3` in every version declaration.
A wheel presented as 1.0b1 while `torchfits.__version__` reports `0.9.3` is
misleading and must not ship.

After the version metadata is changed and checked from a built artifact, the
API verdict becomes **Ship with notes**. The front-door model is strong:
**images → tensors; tables → dataframes**, with explicit tensor-column paths
for training. No namespace rename is recommended: `torchfits.table` is the
locked and adequately explained FITS-domain name.

## Blocking

### 1. Runtime and package versions are still `0.9.3`

- `src/torchfits/__init__.py:17` sets `__version__ = "0.9.3"`.
- `pyproject.toml:14` sets the Python package version to `0.9.3`.
- `pixi.toml:11` sets the Pixi package version to `0.9.3`.
- `tests/test_release_smoke.py:28-29` only verifies that the runtime and
  declared versions agree. Because they all agree on the old value, that test
  passes without proving the artifact is 1.0b1.

Before tagging or building, set all release declarations to the intended beta
version and verify both installed package metadata and
`torchfits.__version__` from the built wheel.

There is no blocking API-design problem once release identity is corrected.

## Should-fix

### 1. Demote aliases in the 1.0b1 adoption path

Keep compatibility names callable in b1, but stop presenting each as a
separate choice. The primary decision tree should be:

1. Image, cube, or spectrum → `torchfits.read_tensor`
2. Table as a dataframe → `torchfits.table.read`
3. Table columns for training → `torchfits.table.read_torch`
4. Large table → `torchfits.table.scan` or `.scan_torch`, according to output

Concretely:

- Move `table.read_arrow` out of the `docs/api.md` “Which reader?” table. It is
  an exact synonym of `table.read`, useful for destination-qualified symmetry
  but not a distinct user decision.
- Move root `read_table`, `read_table_rows`, and `stream_table` out of the
  “Quick Paths” front door into a compatibility/root-helper note.
- Describe `stream_table` as the lower-level tensor streaming path, not simply
  a root alias of `scan_torch`: the signatures and capabilities differ
  (`chunk_rows`/`start_row` versus `batch_size`/`row_slice`, and only
  `scan_torch` exposes `device`, `non_blocking`, and `pin_memory`).
- Keep `read_polars` visible as a genuinely different destination, but after
  the Arrow default rather than as another competing default.

This is documentation demotion, not an API removal, and is suitable for b1.

### 2. Make explicit readers the examples everywhere

The quick start correctly begins with `read_tensor`, but the image-with-header
example and both migration guides still use generic `torchfits.read` for
ordinary image reads. Prefer `read_tensor(..., return_header=True)` in
adoption examples and reserve `read` for the explicitly labelled
auto-detect/convenience case. This avoids teaching a function whose return
type changes from `torch.Tensor` to `dict[str, torch.Tensor]` based on the HDU.

### 3. Improve interactive-help discoverability

The root `read` and `read_table` docstrings explain their role well, but
`read_tensor` has only a one-line docstring. The public `table.read`,
`table.scan`, and `table.scan_torch` implementations have no user-facing
docstrings despite being the recommended entry points. Before rc1, add short
docstrings covering return type, row-selection convention, device behavior,
and the adjacent alternative. Users adopting from an IDE or `help()` should
receive the same mental model as the rendered docs.

### 4. Show one table-to-training handoff

`docs/quickstart.md` reads tensor columns but stops before showing how they
enter training, while its DataLoader example covers images only. Add one small
table example using `FitsTableDataset` or explicitly stacking selected
homogeneous columns into a feature tensor. This removes the main PyTorch-user
question: why `read_torch` returns a column dictionary rather than one tensor.

## Defer

### Before rc1: settle compatibility names

- Retain `table.read_arrow` as an exact, low-prominence synonym if
  destination symmetry is desired; do not list it as a separate reader.
- Recommended: deprecate root `read_table`, `read_table_rows`, and
  `stream_table` in favor of `table.read_torch` and `table.scan_torch` before
  the 1.0 API freezes. `read_table` is especially ambiguous under the stated
  tables-as-dataframes model because it returns tensor columns, not the
  default Arrow dataframe.
- If backward compatibility requires keeping those root names for 1.0, label
  them compatibility helpers and keep them out of introductory material.
  Do not add more aliases.

### Before rc1: normalize or document row selection

The Arrow path uses Python-style `row_slice`; the tensor compatibility path
exposes `start_row=1` and `num_rows`; the streaming APIs also split between
`row_slice` and `start_row`. Prefer the Python-style vocabulary on the
canonical `table.*` API. If changing it is too disruptive, document indexing
bases next to every affected signature.

### Keep advanced Arrow surfaces reference-only

`reader`, `scanner`, and `dataset` are useful interop surfaces, but they should
remain on the table reference page rather than enter the initial reader
decision. Their presence is not an adoption blocker when the primary four
paths stay dominant.

Renaming `torchfits.table` to `torchfits.dataframe` is explicitly out of scope
and is not needed: all reviewed front-door docs explain the FITS-name /
dataframe-object distinction in one sentence.

## Evidence

- **10-second mental model: pass.** `docs/api.md:7-10`,
  `docs/quickstart.md:72-76`, `docs/api-tables.md:3-5`, and the module
  docstring in `src/torchfits/table.py:1-5` consistently say images become
  tensors and FITS tables are dataframes on disk.
- **Reader selection: pass with alias clutter.** The `docs/api.md` “Which
  reader?” box gives correct return types, but spends two of five table rows
  on `read`/`read_arrow` synonyms and immediately adds the root `read_table`
  alias. The later Quick Paths table repeats all three naming families.
- **Return-type honesty: pass.** `src/torchfits/io.py:141-148` states that
  generic `read` returns a tensor for images or a tensor-column dictionary for
  tables and points dataframe users to `table.read`. The `read_table`
  docstring at lines 216-220 likewise directs new code to `table.read_torch`
  and Arrow users to `table.read`.
- **PyTorch friction: low, with one missing bridge.** `read_tensor` directly
  supports `device`, mmap, reduced precision flags, and optional headers.
  `read_torch` returns named tensor columns and `scan_torch` adds device and
  pinned-memory controls. The quick start demonstrates these reads but not the
  final table-to-model shape conversion.
- **Migration consistency: pass.** Both Astropy and fitsio migration guides
  choose `table.read` for dataframe work, `table.read_torch` for tensor
  columns, and `table.scan_torch` for tensor streaming. The dataset migration
  also replaces auto mode with explicit image/table dataset classes.
- **Example consistency: pass.** `examples/example_table.py` leads with
  `table.read`, uses `table.read_torch` and `table.scan_torch`, and asserts
  that `read_arrow` is only a synonym. It does not steer users toward a new
  namespace.
- **Root breadth is manageable if curated.** The inventory reports 33 root
  exports and 16 names weak or absent in `docs/api.md`. That is not a reason
  to promote every export: adoption improves when the docs deliberately
  emphasize the few canonical entry points and leave operational helpers to
  reference pages.
- **Release identity: fail.** Runtime, Python packaging, and Pixi packaging
  all still declare `0.9.3`; this is the sole blocking finding.
