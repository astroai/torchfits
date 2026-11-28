"""A warm read must not pay per-call path resolution or filesystem probes.

Two release-candidate regressions made every small read slower than 1.3.3:
keying the shared metadata by ``realpath()`` resolved the path twice per read
(one ``readlink()`` per path component, 14 syscalls for a file six
directories deep, and ``_C.read_num_hdus`` went from 0.24 to 7.4 us), and the
learned-HDU-offset cache ran ``fstat()`` + ``fstatfs()`` on every open, though
a handle that never reaches a third HDU has nothing to publish.

An ``LD_PRELOAD`` shim counts the libc calls torchfits makes directly
(``realpath``/``__realpath_chk``, ``statfs``/``fstatfs``) over a loop of warm
reads. Name resolution is remembered for the shared-metadata validation
window (set to a minute here, so the count does not depend on timing), and a
single-HDU read never asks which filesystem it is on.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest

pytestmark = pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="LD_PRELOAD shim is Linux-only"
)

_SHIM = r"""
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stddef.h>
#include <sys/vfs.h>

static long counts[2];
long tf_count(int which) { return counts[which]; }
void tf_reset(void) { counts[0] = counts[1] = 0; }

#define NEXT(name) dlsym(RTLD_NEXT, name)

char* realpath(const char* p, char* out) {
    static char* (*real)(const char*, char*) = 0;
    if (!real) real = (char* (*)(const char*, char*))NEXT("realpath");
    counts[0]++;
    return real(p, out);
}
char* __realpath_chk(const char* p, char* out, size_t n) {
    static char* (*real)(const char*, char*, size_t) = 0;
    if (!real) real = (char* (*)(const char*, char*, size_t))NEXT("__realpath_chk");
    counts[0]++;
    return real(p, out, n);
}
int statfs(const char* p, struct statfs* b) {
    static int (*real)(const char*, struct statfs*) = 0;
    if (!real) real = (int (*)(const char*, struct statfs*))NEXT("statfs");
    counts[1]++;
    return real(p, b);
}
int fstatfs(int fd, struct statfs* b) {
    static int (*real)(int, struct statfs*) = 0;
    if (!real) real = (int (*)(int, struct statfs*))NEXT("fstatfs");
    counts[1]++;
    return real(fd, b);
}
"""

_SCRIPT = textwrap.dedent(
    """
    import ctypes, os, sys
    import numpy as np
    import torch
    import torchfits
    from torchfits import _C

    shim = ctypes.CDLL(sys.argv[1])
    root = sys.argv[2]
    deep = os.path.join(root, "a", "b", "c", "d", "e", "f")
    os.makedirs(deep)
    image = os.path.join(deep, "img.fits")
    table = os.path.join(deep, "tab.fits")
    torchfits.write(image, torch.ones(10, 10), overwrite=True)
    torchfits.write(table, {"x": np.arange(10, dtype=np.int32)}, overwrite=True)
    os.chdir(deep)

    def warm():
        torchfits.read(image)
        torchfits.read("img.fits")
        _C.read_num_hdus(image)
        torchfits.table.read(table, hdu=1)

    for _ in range(5):
        warm()
    shim.tf_reset()
    for _ in range(100):
        warm()
    resolutions, fs_probes = shim.tf_count(0), shim.tf_count(1)
    print("resolutions", resolutions, "fs_probes", fs_probes)
    assert resolutions <= 4, resolutions
    assert fs_probes == 0, fs_probes
    print("ok")
    """
)


def test_warm_reads_do_not_resolve_or_probe_the_path(tmp_path):
    src = tmp_path / "count_shim.c"
    src.write_text(_SHIM)
    shim = tmp_path / "count_shim.so"
    try:
        built = subprocess.run(
            ["cc", "-shared", "-fPIC", "-O1", str(src), "-o", str(shim), "-ldl"],
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        pytest.skip(f"no C compiler: {exc}")
    if built.returncode != 0:
        pytest.skip(f"cannot build the counting shim: {built.stderr[-300:]}")
    work = tmp_path / "work"
    work.mkdir()
    env = dict(
        os.environ,
        LD_PRELOAD=str(shim),
        TORCHFITS_SHARED_META_VALIDATE_INTERVAL_MS="60000",
        KMP_DUPLICATE_LIB_OK="TRUE",
    )
    proc = subprocess.run(
        [sys.executable, "-c", _SCRIPT, str(shim), str(work)],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert proc.stdout.strip().endswith("ok")
