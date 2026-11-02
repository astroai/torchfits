#!/usr/bin/env bash
# Build and run every standalone C++ self-check in tests/cpp.
#
# Why a script rather than copy-pasteable commands in each file: three of these
# link the torch-free core library and CFITSIO's static archive, which are not
# installed into the pixi environment -- they exist only in the build tree
# (build/<wheel-tag>/) that `pixi run dev` writes. The runner finds them there,
# writes the FITS fixture with torchfits, and needs nothing else.
#
# Usage:  tests/cpp/run_all.sh          (after `pixi run dev`)
#         CXX=clang++ tests/cpp/run_all.sh   (override the compiler)
# Exit:   0 if every self-check passed, 1 otherwise.
#
# Every check is run under a timeout, which is not a nicety: the regression
# test_parallel_for_nesting guards is a *deadlock*, so a regression makes that
# check hang forever rather than fail (measured: with the worker-inline branch in
# core/parallel.cpp removed, the check never returns at 2 or 4 threads). A runner
# that waits forever cannot report that, so the timeout is part of the contract.
# Override it with TORCHFITS_CPP_CHECK_TIMEOUT=<seconds>.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"

CPP_SRC="src/torchfits/cpp_src"
CXX="${CXX:-clang++}"
# CXX may carry flags as well as a program name ("ccache c++", "c++ -pipe"), the
# same way the two pytest drivers that compile these same files read it. eval
# respects quoting inside the value, so a toolchain path containing a space
# still works.
# shellcheck disable=SC2206
eval "CXX_CMD=($CXX)"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/torchfits-cpp-tests.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

# --- locate the built artifacts ------------------------------------------
# `pixi run dev` builds into build/<wheel-tag>/: the torch-free core library
# and CFITSIO's static archive (cfitsio_build/libcfitsio.a). CFITSIO's headers
# are the vendored tree. Same discovery as
# tests/test_core_library.py::_core_build_artifacts; nothing is hardcoded.
# Each find result is checked before dirname, since dirname "" is "." and would
# defeat the missing-build guard below.
CFITSIO_DIR="extern/cfitsio"
CORE_LIB="$(find build -maxdepth 2 \( -name 'libtorchfits_core.dylib' -o -name 'libtorchfits_core.so' \) 2>/dev/null | head -1)"
CFITSIO_ARCHIVE=""
[[ -n "$CORE_LIB" ]] && CFITSIO_ARCHIVE="$(find "$(dirname "$CORE_LIB")/cfitsio_build" -maxdepth 1 -name 'libcfitsio.a' 2>/dev/null | head -1)"

if [[ ! -f "$CFITSIO_DIR/fitsio.h" || -z "$CORE_LIB" || -z "$CFITSIO_ARCHIVE" ]]; then
  echo "error: could not find the built core library and CFITSIO under build/" >&2
  echo "       run 'pixi run dev' first." >&2
  exit 1
fi

# CFITSIO's archive needs zlib, bzip2 and libcurl, which come from the pixi env.
ENV_PREFIX="${CONDA_PREFIX:-$REPO/.pixi/envs/default}"
INCLUDES=(-I "$CPP_SRC" -I "$CFITSIO_DIR")
CFITSIO_LINK=("$CFITSIO_ARCHIVE" -L "$ENV_PREFIX/lib" -lbz2 -lz -lcurl)
RPATHS=(-Wl,-rpath,"$REPO/$(dirname "$CORE_LIB")" -Wl,-rpath,"$ENV_PREFIX/lib")

# --- fixture for the checks that need a FITS file on disk ------------------
# test_fitsreader_threads requires >= 2 usable HDUs; test_open_for_write_conflict
# and test_raw_fd_retry require any readable FITS file. One multi-extension MEF
# satisfies all three, written with torchfits itself.
FIXTURE="$WORK/mef.fits"
if ! pixi run -q python - "$FIXTURE" <<'PY'
import sys

import numpy as np
import torch

from torchfits import Header, HDUList, TableHDU, TensorHDU

path = sys.argv[1]
HDUList(
    [
        TensorHDU(torch.arange(64, dtype=torch.int16).reshape(8, 8)),
        TensorHDU(torch.ones(16, 16), header=Header({"EXTNAME": "SCI1"})),
        TableHDU(
            {"A": np.arange(10, dtype=np.int32)}, header=Header({"EXTNAME": "TAB"})
        ),
        TensorHDU(
            torch.zeros(4, 4, dtype=torch.int32), header=Header({"EXTNAME": "SCI2"})
        ),
    ]
).write(path, overwrite=True)
print(f"  fixture: {path}")
PY
then
  echo "error: could not build the FITS fixture" >&2
  exit 1
fi

# name | needs_cfitsio(yes/no) | extra sources/flags for the COMPILE step | args for the RUN step
#
# test_bswap_helpers is compiled with the same SIMD baseline the project builds
# with (CMakeLists.txt adds -mssse3 for x86_64). Without it a default x86_64
# compile only defines SSE2, the header takes neither #elif branch, and the
# check silently degrades to the scalar tail -- the byte-swaps would go
# untested on the one platform whose users get the SSSE3 path.
HOST_ARCH="$(uname -m)"
X86_BASELINE=""
case "$HOST_ARCH" in
  x86_64|amd64|X86_64) X86_BASELINE="-mssse3" ;;
esac

CASES=(
  "test_bracket_detection|no||"
  "test_bswap_helpers|no|$X86_BASELINE||"
  "test_parallel_for_nesting|no|$CPP_SRC/core/parallel.cpp -lpthread|4"
  "test_fork_after_pool|no|$CPP_SRC/core/parallel.cpp -lpthread||"
  "test_open_for_write_conflict|yes||@FIXTURE"
  "test_raw_fd_retry|yes||@FIXTURE"
  "test_fitsreader_threads|yes||@FIXTURE"
)

# --- run one check under a timeout -----------------------------------------
# 124 is timeout(1)'s "timed out" status; the python wrapper uses the same value
# so the message below is the same either way. Output is inherited, so the
# check's own stdout still shows.
TIMEOUT_S="${TORCHFITS_CPP_CHECK_TIMEOUT:-180}"
run_check() {  # run_check <binary> [args...]
  local bin="$1"; shift
  # shellcheck disable=SC2086
  pixi run -q python - "$TIMEOUT_S" "$bin" "$@" <<'PY'
import subprocess
import sys

limit = float(sys.argv[1])
cmd = sys.argv[2:]
try:
    sys.exit(subprocess.run(cmd, timeout=limit).returncode)
except subprocess.TimeoutExpired:
    print(
        f"  TIMEOUT after {limit:g}s -- a self-check that never returns is a "
        f"failure, not a hang to wait out (test_parallel_for_nesting "
        f"deadlocks this way if the worker-inline branch in "
        f"core/parallel.cpp regresses)"
    )
    sys.exit(124)
PY
}

status=0
echo
for entry in "${CASES[@]}"; do
  IFS='|' read -r name needs_cf extra run_args <<<"$entry"
  bin="$WORK/$name"

  if [[ "$needs_cf" == "yes" ]]; then
    cmd=("${CXX_CMD[@]}" -std=c++17 "${INCLUDES[@]}")
  else
    cmd=("${CXX_CMD[@]}" -std=c++17 -I "$CPP_SRC")
  fi
  # extra holds COMPILE-time inputs only.
  [[ -n "$extra" ]] && cmd+=($extra)

  if [[ "$needs_cf" == "yes" ]]; then
    cmd+=("tests/cpp/$name.cpp" "$CORE_LIB" "${CFITSIO_LINK[@]}" -lpthread)
    cmd+=("${RPATHS[@]}")
  else
    cmd+=("tests/cpp/$name.cpp")
  fi
  cmd+=(-o "$bin")

  # run_args are RUN-time arguments; @FIXTURE is expanded to the temp path.
  run_args="${run_args//@FIXTURE/$FIXTURE}"

  printf '=== %s\n' "$name"
  if ! "${cmd[@]}" 2>"$WORK/$name.build"; then
    echo "  BUILD FAILED"
    sed 's/^/    /' "$WORK/$name.build" | head -12
    status=1
    continue
  fi
  # shellcheck disable=SC2086
  if run_check "$bin" $run_args; then
    echo "  PASS"
  else
    echo "  FAIL (exit $?)"
    status=1
  fi
  echo
done

# --- the SSSE3 branch, executed on Apple Silicon via Rosetta ----------------
# internal_utils.h's byte-swaps have three #if branches and a native build
# compiles exactly one of them. On arm64 that is NEON, so the SSSE3 branch --
# which CMakeLists.txt compiles into every x86_64 build -- is never executed
# here, and a wrong mask byte would be silent for every x86 user of the mmap
# cutout path. Rosetta can run an x86_64 build, so build one and run it too.
# Measured: with an SSSE3 mask deliberately broken, the native check passes
# (0 failing cases) and this one reports 98.
#
# Skipped with a printed note when there is no x86 runtime, or when
# TORCHFITS_CPP_X86=0. The AVX2 branch is not attempted: no shipped build
# enables it (the build deliberately avoids -march=native), and Rosetta on this
# host dies with SIGILL on a three-line program that uses one AVX2 intrinsic,
# so it is covered by tests/test_simd_shuffle_masks.py instead.
x86_status=0
echo
if [[ "${TORCHFITS_CPP_X86:-auto}" == "0" ]]; then
  echo "=== test_bswap_helpers (x86_64/SSSE3): skipped (TORCHFITS_CPP_X86=0)"
elif [[ "$(uname -s)" == "Darwin" && "$HOST_ARCH" == "arm64" ]] \
     && command -v clang++ >/dev/null 2>&1 \
     && arch -x86_64 /usr/bin/true >/dev/null 2>&1; then
  x86_bin="$WORK/test_bswap_helpers_ssse3"
  echo "=== test_bswap_helpers (x86_64/SSSE3, via Rosetta)"
  if clang++ -target x86_64-apple-macos11 -mssse3 -std=c++17 -O2 \
       -I "$CPP_SRC" tests/cpp/test_bswap_helpers.cpp -o "$x86_bin" \
       2>"$WORK/x86.build"; then
    if run_check "$x86_bin"; then
      echo "  PASS"
    else
      echo "  FAIL (exit $?)"
      x86_status=1
    fi
  else
    echo "  BUILD FAILED"
    sed 's/^/    /' "$WORK/x86.build" | head -12
    x86_status=1
  fi
else
  echo "=== test_bswap_helpers (x86_64/SSSE3): skipped (no x86 runtime here;" \
       "covered statically by tests/test_simd_shuffle_masks.py)"
fi
status=$((status + x86_status))

if [[ $status -eq 0 ]]; then
  echo "all tests/cpp self-checks passed"
else
  echo "one or more tests/cpp self-checks FAILED" >&2
fi
exit $status
