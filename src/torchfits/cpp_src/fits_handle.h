/**
 * RAII guard for a privately owned CFITSIO handle.
 *
 * Every read opens its own ``fitsfile*`` (CFITSIO Option A): sharing one handle
 * across threads would share a single CHDU cursor. This guard makes ownership
 * explicit at the call sites. It is move-only, because a copied guard would
 * close the same handle twice.
 *
 * This header previously also declared a shared-handle LRU cache
 * (configure_cache/clear_file_cache/invalidate_cached/...). Those were no-ops
 * once shared handles were removed; the live shared state is SharedReadMeta
 * (fits_detail.h) plus the per-thread table-reader cache.
 */

#pragma once

#include <fitsio.h>

#include <string>

#include "core/fits_core.h"

namespace torchfits {

struct FitsHandleGuard {
    fitsfile* fptr = nullptr;
    // Path a read-only handle was opened from: on close its learned HDU
    // offsets are published for the next open (detail::remember_hdu_offsets).
    std::string remember_path;

    FitsHandleGuard() = default;
    FitsHandleGuard(const FitsHandleGuard&) = delete;
    FitsHandleGuard& operator=(const FitsHandleGuard&) = delete;

    FitsHandleGuard(FitsHandleGuard&& other) noexcept
        : fptr(other.fptr), remember_path(std::move(other.remember_path)) {
        other.fptr = nullptr;
    }
    FitsHandleGuard& operator=(FitsHandleGuard&& other) noexcept {
        if (this != &other) {
            release();
            fptr = other.fptr;
            remember_path = std::move(other.remember_path);
            other.fptr = nullptr;
        }
        return *this;
    }

    ~FitsHandleGuard() { release(); }

    void release() {
        if (!fptr) return;
        if (!remember_path.empty()) detail::remember_hdu_offsets(fptr, remember_path);
        int status = 0;
        fits_close_file(fptr, &status);
        fptr = nullptr;
    }
};

}  // namespace torchfits
