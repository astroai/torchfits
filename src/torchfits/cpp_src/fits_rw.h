#pragma once

#include <stdexcept>
#include <string>
#include <vector>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/function.h>
#include <ATen/ATen.h>
#include <fitsio.h>
#include <unistd.h>

#include "torchfits_torch.h"
#include "fits_file.h"
#include "security.h"

namespace torchfits {

struct HDUInfo {
    int index;
    std::string type;
    std::vector<std::tuple<std::string, std::string, std::string>> header;
};

torch::Tensor read_full_cached(const std::string& path, int hdu_num, bool use_mmap);
torch::Tensor read_full_unmapped_raw(const std::string& path, int hdu_num);
torch::Tensor read_full_nocache(const std::string& path, int hdu_num, bool use_mmap);
int resolve_hdu_name_cached(const std::string& filename, const std::string& hdu_name);
std::vector<torch::Tensor> read_images_batch(const std::vector<std::string>& paths, int hdu_num, bool use_mmap = true);
std::vector<torch::Tensor> read_hdus_batch(const std::string& path, const std::vector<int>& hdus, bool use_mmap);
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
void write_table_hdu(fitsfile* fptr, nb::dict tensor_dict, nb::dict header, nb::object schema_obj, bool is_ascii);
void write_table_hdu(fitsfile* fptr, nb::dict tensor_dict, nb::dict header);
// Shared owner of a Python FITSFile's fitsfile*. Throws if the handle is
// closed; keeps it open while held (see FITSFile::share_fptr).
std::shared_ptr<fitsfile> share_fptr_from_python_object(nanobind::object obj);

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

// Open `path` for writing, retrying once if a cached READONLY handle for it
// was registered in-process. The retry drops that handle and re-opens, so
// read-then-mutate sequences work without the caller knowing about the cache.
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

inline int open_fits_for_write(fitsfile** fptr, const std::string& path) {
    int status = 0;
    *fptr = nullptr;
    open_fits_readwrite_raw(fptr, path, &status);
    if (status == FILE_NOT_OPENED) {
        evict_cached_reader(path);
        status = 0;
        *fptr = nullptr;
        open_fits_readwrite_raw(fptr, path, &status);
        if (status == FILE_NOT_OPENED) {
            throw std::runtime_error(
                "Failed to open '" + path + "' for writing: CFITSIO still reports "
                "the file as already open after every cached reader was dropped. "
                "The usual cause is a reader that is still open -- one obtained "
                "from torchfits.open_table_reader() holds its own CFITSIO handle, "
                "which cache eviction cannot release; close it (or drop the "
                "reference) and retry. Otherwise the path is probably not "
                "writable: a missing parent directory, or insufficient "
                "permissions.");
        }
    }
    return status;
}

} // namespace torchfits
