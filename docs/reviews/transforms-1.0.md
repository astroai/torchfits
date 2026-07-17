# Transforms catalog review — 1.0 prep

**Date:** 2026-07-16  
**Scope:** `torchfits.transforms` public surface (33 `__all__` symbols: 29 classes + 4 helpers)  
**Code:** `src/torchfits/transforms.py` (3184 lines, 29 `FITSTransform` subclasses)  
**Verdict:** **Keep the catalog; settle namespace, docs, and file split before 1.0 freeze**

---

## Executive summary

The transforms module is feature-rich, well-tested, and aligned with torchfits’ FITS-native ML story. No class is a strong **remove** candidate — every public transform has unit tests and a documented use case. The 1.0 work is **structural and contractual**: confirm namespace-only imports (done in 0.9.2), fix doc lies (`nn.Module`, root re-exports), document mask/threading semantics, split the 3k-line monolith, and tier the spectral/continuum catalog so users are not overwhelmed.

**Keep:** 27 transform classes + `Compose` + `FITSTransform` + 4 public helpers.  
**Merge (defer API change):** `SigmaClip` / `AsymmetricSigmaClip` — same family, different defaults; consider unified class post-1.0.  
**Demote (docs/priority, not removal):** `AlphaShapeContinuum`, `AsymmetricLeastSquares`, `PhaseFold`, `BandMath` — niche or non-FITS-specific; move to “Advanced” in docs.  
**Remove:** none recommended for 1.0.

---

## 1. Verdict by family

### Infrastructure

| Symbol | Lines (approx) | Tests | Recommendation | Notes |
|---|---:|---|---|---|
| `FITSTransform` | 284–306 | `TestFITSTransform`, typing | **Keep** | Callable protocol (`forward` / `inverse` / `__call__`); not `nn.Module`. |
| `Compose` | 313–341 | `TestCompose`, e2e pipelines | **Keep** | Reverse-order `inverse()` is the core UX. |

### Stretches (stateless, invertible)

| Symbol | Tests | Recommendation |
|---|---|---|
| `ArcsinhStretch` | `TestArcsinhStretch` | **Keep** — flagship display stretch |
| `LogStretch` | `TestLogStretch` | **Keep** |
| `SqrtStretch` | `TestSqrtStretch` | **Keep** |

No merges warranted; three distinct compression curves.

### Normalizers (data-dependent, invertible)

| Symbol | Tests | Recommendation |
|---|---|---|
| `ZScaleNormalize` | `TestZScaleNormalize` | **Keep** — IRAF display default |
| `RobustNormalize` | `TestRobustNormalize` | **Keep** — ML default |
| `BackgroundSubtract` | `TestBackgroundSubtract` | **Keep** |
| `PercentileClipNormalize` | `TestPercentileClipNormalize` | **Keep** |
| `MinMaxNormalize` | `TestMinMaxNormalize` | **Keep** — document outlier sensitivity |
| `GlobalScalarNorm` | `TestGlobalScalarNorm` | **Keep** — minimal linear prep |

### FITS metadata (header-aware)

| Symbol | Tests | Recommendation |
|---|---|---|
| `FITSHeaderScale` | `TestFITSHeaderScale`, e2e | **Keep** — BSCALE/BZERO |
| `FITSScaleColumns` | `TestFITSScaleColumns`, e2e | **Keep** — TSCAL/TZERO per column |
| `TNullToNan` | `TestTNullToNan`, e2e | **Keep** — lossy by design |
| `FITSHeaderNormalize` | `TestFITSHeaderNormalize`, e2e | **Keep** — BITPIX-aware [0,1] for integers |

`FITSHeaderScale` and `FITSHeaderNormalize` overlap conceptually but serve different pipelines (explicit scale vs auto BITPIX policy). **Do not merge.**

### Spectral / time-domain

| Symbol | Tests | Recommendation |
|---|---|---|
| `ContinuumNormalize` | `TestContinuumNormalize` + vectorized + reference | **Keep** — divide-by-continuum |
| `ContinuumRemoval` | `TestContinuumRemoval` + vectorized + reference | **Keep** — subtract baseline (distinct semantics) |
| `DopplerShift` | `TestDopplerShift`, `TestResample1d` | **Keep** — augmentation / correction |
| `SpectralBinning` | `TestSpectralBinning` | **Keep** |
| `BandMath` | `TestBandMath` | **Keep** — **Demote** in docs (generic NDVI-style; not FITS-specific) |
| `PhaseFold` | `TestPhaseFold` + vectorized + reference | **Keep** — **Demote** (time-series; lossy) |

### Continuum / baseline estimators

| Symbol | Tests | Recommendation |
|---|---|---|
| `SavitzkyGolayFilter` | unit + vectorized + reference | **Keep** |
| `RunningPercentile` | unit + vectorized + reference | **Keep** |
| `UpperEnvelopeContinuum` | unit + vectorized + reference | **Keep** |
| `WaveletDecompose` | unit + vectorized + reference | **Keep** — fully invertible |
| `AsymmetricLeastSquares` | unit + vectorized + reference | **Keep** — **Demote** (Raman/NIR specialty) |
| `AlphaShapeContinuum` | unit + vectorized + reference | **Keep** — **Demote** (overlaps upper-envelope story; morphological variant) |

`UpperEnvelopeContinuum` vs `AlphaShapeContinuum`: similar “upper envelope” intent; keep both but document when to pick which. **Merge deferred** — different algorithms, both tested.

### Outlier rejection (lossy)

| Symbol | Tests | Recommendation |
|---|---|---|
| `SigmaClip` | `TestSigmaClip` + vectorized + reference | **Keep** |
| `AsymmetricSigmaClip` | `TestAsymmetricSigmaClip` | **Keep** — **Merge candidate** post-1.0 (`n_low`/`n_high` kwargs on `SigmaClip`) |

### Public helpers (`__all__` but not classes)

| Symbol | Tests | Recommendation |
|---|---|---|
| `safe_arcsinh`, `safe_log` | `TestSafeMath` | **Keep in `__all__`** — document in api-transforms |
| `estimate_background` | `TestEstimateBackground`, typing | **Keep** — shared by normalizers and `AsymmetricSigmaClip` |
| `zscale_limits` | `TestZScaleLimits`, typing | **Keep** — shared by `ZScaleNormalize` |

---

## 2. API consistency

### Compose

- Chains `forward` left-to-right; `inverse` right-to-left. **Consistent and tested.**
- Passes the same `mask` to every child. Typing tests cover all children.
- **Gap:** docs do not show `mask=` usage in pipeline examples.

### Device

- All ops use `x.device` / `x.dtype` for temporaries — no silent CPU promotion.
- **Gap:** no CUDA/device matrix in tests (CPU-only today). Acceptable for 1.0 if documented as “device follows input tensor.”

### Dtype

- Float32 default in tests; `_upcast_for_precision` for arcsinh/log on float16/bfloat16.
- Integer paths in `TNullToNan`, `FITSHeaderNormalize` preserve storage dtypes where appropriate.
- **Consistent** within the file; document half-precision upcast policy in api-transforms.

### Header awareness

| Layer | Header coupling |
|---|---|
| `FITSHeaderScale`, `FITSScaleColumns`, `TNullToNan` | Explicit keywords or `from_header()` factories |
| `FITSHeaderNormalize` | BITPIX/BSCALE/BZERO policy object |
| Stretches / normalizers / spectral | **Tensor-only** — no header reads |

Clear split: FITS metadata transforms vs pure tensor ops. **Keep this boundary.**

### Invertibility contract

| Kind | `inverse()` |
|---|---|
| Stretches, normalizers, FITS scale, continuum divide/subtract, baseline estimators, wavelet | Yes (state cached on instance) |
| `BandMath`, `PhaseFold`, `SigmaClip`, `AsymmetricSigmaClip`, `TNullToNan` | Raises or N/A — **lossy** |

Docs list this per class; **add a summary table** to api-transforms for scanability.

### `FITSTransform` vs `nn.Module`

- **Not** subclasses of `torch.nn.Module`. No `parameters()`, not registered in `nn.Sequential`.
- Compatible with `Dataset.__getitem__` when used as a **callable** (`transform(x)` or `pipeline(x)`).
- Module docstring claims “thread-local” inverse state; implementation uses **per-instance** `_last_*` fields. Safe under `DataLoader(num_workers>0)` when **each worker constructs its own pipeline** (standard pattern). **Fix docs** — not thread-local, instance-local.

### `torch.compile`

- Claimed in api-transforms; **no in-tree test**. Likely works for stateless stretches; data-dependent normalizers cache Python-side state and may graph-break. **Defer compile story to 1.0+** or soften to “stateless transforms are compile-friendly.”

---

## 3. Root namespace policy

**Confirmed done in 0.9.2** (`docs/reviews/release-api-freeze-0.9.2.md`):

- Root `__all__` exposes `torchfits.transforms` as a **lazy namespace only**.
- No transform classes re-exported at `torchfits.*`.
- Examples already use `from torchfits.transforms import …`.

**1.0 rule:** no resurrection of root transform re-exports. New transforms land in `torchfits.transforms` only.

---

## 4. Test coverage map

| Module | Role |
|---|---|
| `tests/test_transforms.py` (~3500 lines) | Unit tests for every class; vectorized-vs-reference parity for heavy spectral/clip paths; mask/NaN edge cases |
| `tests/test_transforms_e2e.py` | Astropy-written FITS files → read → `FITSHeaderScale` / `FITSScaleColumns` / `TNullToNan` / `FITSHeaderNormalize` round-trips |
| `tests/test_transforms_typing.py` | `mask=None` vs `mask=Tensor` for every transform (mypy-strict) |
| `tests/transforms_reference.py` | Slow reference loops for parity (ALS, alpha-shape, continuum, envelope, phase-fold, SG, sigma-clip, wavelet) |

**Well covered:** correctness, batched shapes, inverse round-trips, FITS keyword factories.  
**Gaps:** GPU device, `torch.compile`, autograd/gradients, integration with `torchfits.data` datasets in one e2e test.

---

## 5. Documentation gaps

| Gap | Severity | Action |
|---|---|---|
| `nn.Module`-compatible (false) | High | Fixed in api-transforms.md |
| Root re-export list (stale post-0.9.2) | High | Fixed in api-transforms.md |
| `mask` parameter undocumented in user docs | Medium | Add section + Compose example |
| Public helpers (`estimate_background`, etc.) not in api-transforms | Medium | Add “Utilities” section |
| “25+” class count in `docs/api.md` | Low | Say 27 transform classes (29 including base + Compose) |
| Thread-local vs instance-local state | Medium | Fix module docstring in 1.0 split |
| `torch.compile` unverified | Low | Soften or add smoke test |
| Invertibility summary table | Low | Add to api-transforms |
| Tiering (core vs advanced spectral) | Medium | Reorder api-transforms sections |
| `examples/example_hyperspectral.py` cited but verify exists | Low | Confirm in integrity test |

---

## 6. Blocking for 1.0 vs defer

### Blocking (before `1.0.0rc1`)

1. **File split** — `transforms.py` at 3184 lines blocks safe maintenance (`thermo-nuclear-0.9.2.md`). Split by domain: `_transforms/stretch.py`, `normalize.py`, `fits_meta.py`, `spectral.py`, `continuum.py`, `clip.py`, re-export from `transforms/__init__.py`. Behavior freeze; no API renames.
2. **Docs truth** — namespace-only imports, callable protocol (not `nn.Module`), mask semantics, instance-local inverse state.
3. **Catalog sign-off** — this review committed; no class removals without deprecation cycle.
4. **Integrity** — `tests/test_docs_integrity.py` / public boundary aligned with namespace-only policy.

### Should-fix (1.0 polish)

- Add one `torchfits.data` + `Compose` e2e smoke (image dataset → pipeline → tensor shape).
- Document helper functions in api-transforms.
- Tier “Advanced spectral” section in docs.

### Defer (post-1.0)

- Merge `SigmaClip` + `AsymmetricSigmaClip`.
- `torch.compile` certification and autograd differentiability notes.
- CUDA-specific transform test matrix.
- Remove or demote `BandMath` if scope creep becomes a support burden.
- Gradient-safe training integration (`nn.Module` wrapper or documented pattern).

---

## 7. Recommended 1.0 transforms freeze rule

After split lands:

- **Bugfixes and doc corrections only** on transform math until `1.0.0`.
- New transforms require: class + api-transforms section + `test_transforms.py` minimum + typing mask test.
- No root re-exports. No new public symbols outside `torchfits.transforms.__all__` without changelog entry.

---

## 8. Keep vs cut summary

| Action | Count | Symbols |
|---|---:|---|
| **Keep** | 33 | Full `__all__` as of 0.9.2 |
| **Demote (docs only)** | 4 | `AlphaShapeContinuum`, `AsymmetricLeastSquares`, `PhaseFold`, `BandMath` |
| **Merge (post-1.0)** | 2 → 1 | `SigmaClip` + `AsymmetricSigmaClip` |
| **Remove** | 0 | — |

The catalog is a **keep** for 1.0. Cut **root noise** (already done), not **spectral breadth**.
