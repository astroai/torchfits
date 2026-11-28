"""A failed image-meta probe is cached like a successful one (r4a-12).

``get_image_meta`` stores ``None`` when every probe fails, but
``signature_cached_get`` used to return ``None`` for a miss too, so the stored
negative answer was never served and each call re-ran every probe.
"""

from __future__ import annotations

import os
from collections import OrderedDict
from pathlib import Path

from torchfits._io_engine import caches, image_meta


class _FailingCpp:
    def __init__(self) -> None:
        self.calls = 0

    def read_shape(self, path, hdu):  # type: ignore[no-untyped-def]
        self.calls += 1
        raise RuntimeError("not an image")

    def read_header_dict(self, path, hdu):  # type: ignore[no-untyped-def]
        self.calls += 1
        raise RuntimeError("not an image")


def test_negative_image_meta_lookup_is_cached(tmp_path: Path) -> None:
    path = str(tmp_path / "x.fits")
    Path(path).write_bytes(b"\0" * 2880)
    cpp = _FailingCpp()
    caches.image_meta_cache.pop((path, 0), None)

    assert image_meta.get_image_meta(path, 0, cpp_module=cpp) is None
    probes = cpp.calls
    assert probes > 0
    assert image_meta.get_image_meta(path, 0, cpp_module=cpp) is None
    assert cpp.calls == probes  # served from the cache

    # Still signature-validated: a rewrite probes again.
    Path(path).write_bytes(b"\0" * 5760)
    os.utime(path, ns=(1, 1))
    assert image_meta.get_image_meta(path, 0, cpp_module=cpp) is None
    assert cpp.calls == 2 * probes


def test_signature_cached_get_default_tells_miss_from_none(tmp_path: Path) -> None:
    path = str(tmp_path / "y.fits")
    Path(path).write_bytes(b"\0" * 2880)
    cache: OrderedDict = OrderedDict()
    key = (path, 0)
    assert caches.signature_cached_get(cache, key, caches.MISSING) is caches.MISSING
    caches.signature_cached_set(cache, key, None, 8)
    assert caches.signature_cached_get(cache, key, caches.MISSING) is None
    assert caches.signature_cached_get(cache, key) is None
