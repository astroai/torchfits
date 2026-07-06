1. **Change `Card` from `@dataclass(frozen=True)` to `NamedTuple`**
   - In `src/torchfits/hdu.py`, `Card` is currently implemented as a `@dataclass(frozen=True)`. This has higher initialization overhead than a `NamedTuple`.
   - `NamedTuple` is naturally tuple-compatible (so we can remove the explicit `__iter__`, `__len__`, and `__getitem__` methods currently in the dataclass).
   - This optimization aligns with the memory: "For performance reasons during bulk FITS header parsing, `Card` objects in `src/torchfits/hdu.py` utilize `typing.NamedTuple` instead of `@dataclass` to avoid `__init__` overhead while retaining tuple compatibility."
   - The `@dataclass(frozen=True)` overhead slows down Header parsing.

2. **Run tests**
   - Ensure all tests pass.

3. **Pre commit instructions**
   - Ensure proper testing, verification, review, and reflection are done.

4. **Submit**
   - Commit and submit.
