/**
 * Multi-level caching system header
 *
 * Implements L1 (memory) and L2 (disk) caching for FITS data
 */

#pragma once

#include <string>
#include <memory>
#include <fitsio.h>

namespace torchfits {

// Cache initialization/configuration
void configure_cache(size_t max_files, size_t max_memory_mb);
void clear_file_cache();
void invalidate_file_cache(const std::string& filepath);
size_t get_cache_size();
fitsfile* get_or_open_cached(const std::string& filepath);
void release_cached(const std::string& filepath);
void invalidate_cached(const std::string& filepath);

// Contract: every successful get_or_open_cached() must be paired with
// release_cached() (directly or via FitsHandleGuard{cached=true}). A missing
// release permanently pins the handle in the LRU (refcount never returns to 0).
//
// NOTE (CFITSIO §4 Option A): get_or_open_cached() is NO LONGER used on the
// concurrent read hot paths. Sharing one cached fitsfile* across threads is not
// thread-safe (each handle has a single current-HDU cursor and internal I/O
// buffer). Read paths (read_full_cached, resolve_hdu_name_cached, FITSFile
// read-mode ctor, TableReader) now open a private per-call handle and close it
// on exit. The shared caches are SharedReadMeta (image/scale/compression/name
// metadata) and the shared raw fd — see fits_detail.h. This API is retained for
// any residual callers and for invalidate_cached()/clear_file_cache().

// RAII guard for fitsfile* handles.  Two modes:
//   cached=false (default) — calls fits_close_file on destruction
//   cached=true           — calls release_cached(path) on destruction
struct FitsHandleGuard {
    fitsfile* fptr = nullptr;
    std::string path;
    bool cached = false;

    ~FitsHandleGuard() {
        if (!fptr) return;
        if (cached) {
            release_cached(path);
        } else {
            int status = 0;
            fits_close_file(fptr, &status);
        }
    }
};

} // namespace torchfits
