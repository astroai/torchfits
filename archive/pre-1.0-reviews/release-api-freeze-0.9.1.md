# Torchfits 0.9.1 release/API freeze review

## Verdict

Ship. This patch constrains the native ABI in dependency metadata, CI install
paths, CMake configuration, and extension import initialization. The
50-symbol root public API inventory is unchanged.

## Blocking

None.

The 0.9.0 wheel was built against PyTorch/libtorch 2.10 but declared
`torch>=2.0.0`. A Torchsky clean environment consequently selected PyTorch
2.12 and reproducibly exited 139 in native image and table conversion, both
under xdist and in a single serial test. Version 0.9.1 constrains build and
runtime metadata to `torch>=2.10,<2.11`. Its first remote matrix exposed a
second path: CI installed PyTorch 2.13, built without isolation, and then
dependency resolution downgraded the runtime to 2.10. All build workflows now
install 2.10 first; CMake rejects other build minors and the extension rejects
other runtime minors before reaching native tensor conversion.

## Should-fix

None for this patch release.

## Defer to next minor

- Remove or isolate the direct `libtorch_python` ABI dependency, or publish an
  explicit wheel strategy for each supported PyTorch minor. Do not widen the
  runtime range until a clean cross-minor native test proves it safe.

## Evidence gaps

None for the metadata correction. This release makes no new API, correctness,
or performance claim and therefore does not require a new exhaustive benchmark
campaign.

## Release evidence

- Public inventory: 50 root exports; all remain documented.
- Targeted contracts pass for manifest/workflow consistency, version sync,
  docs, native image/table smoke, and mismatched-runtime rejection.
- `pixi run preflight-push`: Ruff, formatting, mypy, and compileall passed.
- `pixi run release-gate`: 538 passed, 3 skipped.
- `pixi run ci-local`: docs build and release gate passed.
- Built CPython 3.13 macOS ARM wheel SHA-256:
  `999235f295a81be623424099e15dd942458c33def6f3e48dd34b45b169eab508`.
- Wheel METADATA: `Requires-Dist: torch<2.11,>=2.10`.
- Fresh dependency-resolved environment selected PyTorch 2.10.0 and passed
  native image and table write/read round-trips with Torchfits 0.9.1.

## Freeze criteria

| Criterion | Result |
|---|---|
| No undocumented public exports | Pass |
| Parity claims unchanged | Pass |
| Release gate | Pass |
| Examples and docs contracts | Pass |
| Version/changelog aligned | Pass |
| No scope or performance-claim expansion | Pass |
