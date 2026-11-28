#pragma once

#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <sys/stat.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/function.h>
#include <fitsio.h>
#include <unistd.h>

#include "native_tensor.h"
#include "fits_file.h"
#include "security.h"

namespace torchfits {

struct HDUInfo {
    int index;
    std::string type;
    std::vector<std::tuple<std::string, std::string, std::string>> header;
};

native::Tensor read_full_cached(const std::string& path, int hdu_num, bool use_mmap);
native::Tensor read_full_unmapped_raw(const std::string& path, int hdu_num);
native::Tensor read_full_nocache(const std::string& path, int hdu_num, bool use_mmap);
int resolve_hdu_name_cached(const std::string& filename, const std::string& hdu_name);
std::vector<native::Tensor> read_images_batch(const std::vector<std::string>& paths, int hdu_num, bool use_mmap = true);
std::vector<native::Tensor> read_hdus_batch(const std::string& path, const std::vector<int>& hdus, bool use_mmap);
std::pair<FITSFile*, std::vector<HDUInfo>> open_and_read_headers(const std::string& path, int mode);
// Write one string per table row to string column `colnum`, starting at
// `first_row`. Each row should already be padded to the column's full width.
// A binary column with TFORM rAw (w < r) holds r/w sub-strings per row, and
// CFITSIO counts TSTRING elements in sub-strings, so passing one element per
// row put the rows into the first row's sub-strings. Split each row into its
// w-character pieces in that case.
inline void write_string_rows(fitsfile* fptr, int colnum, LONGLONG first_row,
                              const std::vector<std::string>& rows, int* status) {
    if (*status != 0 || rows.empty()) return;
    int typecode = 0;
    LONGLONG repeat = 0;
    LONGLONG width = 0;
    int type_status = 0;
    fits_get_coltypell(fptr, colnum, &typecode, &repeat, &width, &type_status);
    std::vector<std::string> pieces;
    const std::vector<std::string>* out = &rows;
    if (type_status == 0 && width > 0 && repeat > width) {
        const LONGLONG nsub = repeat / width;
        pieces.reserve(rows.size() * static_cast<size_t>(nsub));
        for (const auto& row : rows) {
            for (LONGLONG k = 0; k < nsub; ++k) {
                const size_t off = static_cast<size_t>(k * width);
                // CFITSIO blank-pads a short element to the sub-string width.
                pieces.push_back(off < row.size()
                                     ? row.substr(off, static_cast<size_t>(width))
                                     : std::string());
            }
        }
        out = &pieces;
    }
    std::vector<const char*> ptrs;
    ptrs.reserve(out->size());
    for (const auto& s : *out) {
        ptrs.push_back(s.c_str());
    }
    fits_write_col(fptr, TSTRING, colnum, first_row, 1, static_cast<LONGLONG>(out->size()),
                   const_cast<char**>(ptrs.data()), status);
}

// Reject header values the card writers refuse, before anything is written.
void validate_header_values(nb::dict header);
void validate_hdu_list_headers(nb::list hdus);
// Write one keyword value card (not HISTORY/COMMENT): bool, str (long values
// as LONGSTRN CONTINUE chains), int of any size, float, complex, or None for
// an undefined value -- every value read_header produces. Throws for any
// other type; CFITSIO failures land in *status.
void update_header_value(fitsfile* fptr, const std::string& key, nb::handle value,
                         const char* comment, int* status);
// release_gil: run the table creation and data writes without the GIL (the
// caller must hold it on entry and own fptr exclusively -- no FITSFile lock).
void write_table_hdu(fitsfile* fptr, nb::dict tensor_dict, nb::dict header, nb::object schema_obj, bool is_ascii,
                     bool release_gil = false);
void write_table_hdu(fitsfile* fptr, nb::dict tensor_dict, nb::dict header);
// Shared owner of a Python FITSFile's fitsfile*. Throws if the handle is
// closed; keeps it open while held (see FITSFile::share_fptr).
std::shared_ptr<fitsfile> share_fptr_from_python_object(nanobind::object obj);

// Run fn(fitsfile*) on a Python FITSFile's handle with the GIL released and the
// handle's I/O lock held for the whole call. CFITSIO keeps one current-HDU
// cursor per fitsfile, so a borrower that drives it (a TableReader moving to
// its HDU and reading rows) must not interleave with another thread's use of
// the same handle. fn must not touch Python objects; the caller's `obj`
// reference keeps the FITSFile alive, and the lock defers a concurrent close().
//
// The borrower is a TableReader, which also reads the file by *name* (pread,
// mmap, extent checks). Once the path has been replaced it would read the new
// file's bytes with the old handle's geometry and return garbage, so the call
// is refused -- before fn, and after it in case the replace landed mid-read.
template <class F>
void with_locked_fptr_from_python_object(nanobind::object obj, F&& fn) {
    FITSFile& file = nanobind::cast<FITSFile&>(obj);
    nanobind::gil_scoped_release release;
    file.with_open_fptr([&](fitsfile* fptr) {
        static const char* kReplaced =
            "FITS file was replaced on disk after this handle was opened; "
            "reopen it to read the current file";
        if (!file.path_names_open_file()) throw std::runtime_error(kReplaced);
        fn(fptr);
        if (!file.path_names_open_file()) throw std::runtime_error(kReplaced);
    });
}

void invalidate_shared_meta(const std::string& filename);
void clear_shared_read_meta_cache();

// Drop every thread's cached TableReader for `filename` (defined in
// table_bindings.cpp). Writers call this before opening a file READWRITE:
//
// a cached reader holds a CFITSIO handle open READONLY, and CFITSIO refuses to
// reopen a file READWRITE while such a handle is still registered in its
// FptrTable. Measured: the second open returns FILE_NOT_OPENED (104), and
// returns 0 again once the READONLY handle is closed.
//
// The status is raised inside CFITSIO's fits_already_open() (vendored
// extern/cfitsio/cfileio.c), which is the *function* that performs the
// same-file lookup -- not a status code. The status it sets is the generic
// FILE_NOT_OPENED, whose own name ("could not open the named file") points away
// from the cache-conflict meaning. That mismatch is why this retry used to carry
// a bare 104 and a comment naming a constant that does not exist.
void evict_cached_reader(const std::string& filename);
// Drop every thread's cached TableReaders (clear_shared_read_meta_cache).
void evict_all_cached_readers();

// Open `path` for writing, retrying if a cached READONLY handle for it was
// registered in-process. The retry drops that handle and re-opens, so
// read-then-mutate sequences work without the caller knowing about the cache.
// A read in flight on another thread holds its handle until the read ends, so
// for a writable regular file the evict-and-retry repeats with a short backoff
// for up to kWriteOpenRetryBudget (see open_fits_for_write below).
//
// Known limitation, measured: CFITSIO reuses FILE_NOT_OPENED for the
// READONLY-blocks-READWRITE refusal, so this retry also fires on any *ordinary*
// failure to open -- a missing directory, a path that is a directory, a
// permissions error. When the retry cannot fix the conflict, the two are
// indistinguishable by status alone, so the failure is reported with both
// explanations rather than CFITSIO's bare "could not open the named file",
// which names neither. Measured: the most common cause is a *live* reader from
// torchfits.open_table_reader(), which owns its own CFITSIO handle; eviction
// only drops the cache, so the retry cannot release it and the write must wait
// for the caller to close the reader.
//
// Throws std::runtime_error in that one case, so every caller's message improves
// at once. Callers' `if (status != 0)` branches still handle every other status;
// this is the only path that throws from here, and it is documented as such.
// fits_create_file, except that a name needs_literal_disk_name() flags is
// created with fits_create_diskfile. CFITSIO's leading '!' (replace an existing
// file) is honoured for those names too, by removing the file first.
inline int create_fits_file(fitsfile** fptr, const std::string& path) {
    int status = 0;
    *fptr = nullptr;
    size_t i = path.find_first_not_of(' ');
    const bool clobber = i != std::string::npos && path[i] == '!';
    const std::string name = clobber ? path.substr(i + 1) : path;
    if (needs_literal_disk_name(name)) {
        if (clobber) ::unlink(name.c_str());  // ENOENT is fine, as in CFITSIO
        fits_create_diskfile(fptr, name.c_str(), &status);
    } else {
        fits_create_file(fptr, path.c_str(), &status);
    }
    return status;
}

inline void open_fits_readwrite_raw(fitsfile** fptr, const std::string& path, int* status) {
    if (needs_literal_disk_name(path)) {
        fits_open_diskfile(fptr, path.c_str(), 1 /* READWRITE */, status);
    } else {
        fits_open_file(fptr, path.c_str(), 1 /* READWRITE */, status);
    }
}

// A failed CFITSIO open can leave *fptr set (e.g. a compression spec that
// fails to parse after the file was opened); close it so no caller leaks the
// handle or reads a half-initialized one (r9b-06).
inline void drop_failed_open(fitsfile** fptr, int status) {
    if (status != 0 && *fptr != nullptr) {
        int close_status = 0;
        fits_close_file(*fptr, &close_status);
        *fptr = nullptr;
    }
}

// Whether `path` names an existing regular file this process may write: the
// only case in which FILE_NOT_OPENED can be a READONLY handle that is still
// open in another thread, and so worth waiting out.
inline bool is_writable_regular_file(const std::string& path) {
    struct stat st;
    return ::stat(path.c_str(), &st) == 0 && S_ISREG(st.st_mode) &&
           ::access(path.c_str(), W_OK) == 0;
}

// A read on another thread holds its cached reader's READONLY handle for the
// length of the read, out of reach of the eviction. Waiting out such a read
// costs at most this long before the conflict is reported.
constexpr std::chrono::milliseconds kWriteOpenRetryBudget{2000};

// ``wait_for_readers`` lets the retry sleep, so only a caller that runs
// without the GIL may pass true: a reader thread needs the GIL to finish its
// read and hand its handle back, so waiting while holding it only delays the
// failure.
inline int open_fits_for_write(fitsfile** fptr, const std::string& path,
                               bool wait_for_readers = false) {
    int status = 0;
    *fptr = nullptr;
    open_fits_readwrite_raw(fptr, path, &status);
    drop_failed_open(fptr, status);
    if (status == FILE_NOT_OPENED) {
        // Evict and retry. A reader busy on another thread re-caches its
        // handle when its read ends, so keep evicting and retrying with a
        // short backoff while the path is a writable file (any other
        // FILE_NOT_OPENED cannot clear by waiting, and fails after one retry).
        const bool may_wait = wait_for_readers && is_writable_regular_file(path);
        const auto deadline = std::chrono::steady_clock::now() + kWriteOpenRetryBudget;
        std::chrono::milliseconds delay{1};
        for (;;) {
            evict_cached_reader(path);
            status = 0;
            *fptr = nullptr;
            open_fits_readwrite_raw(fptr, path, &status);
            drop_failed_open(fptr, status);
            if (status != FILE_NOT_OPENED || !may_wait ||
                std::chrono::steady_clock::now() >= deadline) {
                break;
            }
            std::this_thread::sleep_for(delay);
            delay = std::min(delay * 2, std::chrono::milliseconds{50});
        }
        if (status == FILE_NOT_OPENED) {
            throw std::runtime_error(
                "Failed to open '" + path + "' for writing: CFITSIO still reports "
                "the file as already open after every cached reader was dropped. "
                "The usual cause is a reader that is still open -- one obtained "
                "from torchfits.open_table_reader() holds its "
                "own CFITSIO handle, which cache eviction cannot release, and a "
                "read running on another thread for the whole retry window holds "
                "one too; close it (or drop the reference) and retry. Otherwise "
                "the path is probably not writable: a missing parent directory, "
                "or insufficient permissions.");
        }
    }
    return status;
}

} // namespace torchfits
