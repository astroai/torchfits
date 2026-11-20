#include <nanobind/nanobind.h>

namespace nb = nanobind;

#ifndef TORCHFITS_CORE_BUILD_ID
#define TORCHFITS_CORE_BUILD_ID "unknown"
#endif

void bind_fits(nb::module_& m);
void bind_table(nb::module_& m);

// The extension includes and links no libtorch: tensors leave it as DLPack
// capsules adopted by torch's Python API (native_py.h). Importing it therefore
// never loads torch; the first tensor result does.
NB_MODULE(_C, m) {
#ifdef TORCHFITS_HAVE_BZIP2
    m.attr("HAS_BZIP2") = true;
#else
    m.attr("HAS_BZIP2") = false;
#endif
#ifdef TORCHFITS_HAVE_CURL
    m.attr("HAS_CURL") = true;
#else
    m.attr("HAS_CURL") = false;
#endif
    // libtorchfits_core owns CFITSIO and the metadata cache; _C calls into it.
    // torchfits/__init__.py compares this against torchfits._core.__build_id__
    // so a stale core next to a fresh extension fails at import, not later.
    m.attr("__core_build_id__") = TORCHFITS_CORE_BUILD_ID;
    bind_fits(m);
    bind_table(m);
}
