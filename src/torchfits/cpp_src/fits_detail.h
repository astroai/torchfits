#pragma once

#include <string>
#include <algorithm>
#include <cctype>
#include <vector>
#include <unordered_map>
#include <array>
#include <tuple>
#include <cmath>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <atomic>
#include <limits>
#include <type_traits>
#include <stdexcept>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#if defined(__APPLE__) || defined(__linux__)
#include <dlfcn.h>
#endif
#include <fitsio.h>

// The torch-free primitives this header used to define inline (shared read
// metadata cache, BSCALE detection, FITS text handling, sign-bit XOR) now live
// in libtorchfits_core. They are declared in the same namespace, so every call
// site below is unchanged.
#include "core/fits_core.h"
#include "internal_utils.h"
#include "hardware.h"
#include "security.h"

namespace torchfits {
namespace detail {

// FITS 4.0 (§5.3) unsigned 64-bit convention: 64-bit integer storage with
// BZERO/TZERO = 2^63 and scale 1. 2^63 is exact in a double, so compare
// exactly; the next representable doubles are 1024/2048 away.
inline bool is_unsigned_longlong_offset(double offset) {
    return offset == 9223372036854775808.0;
}

// fits_update_key(TDOUBLE) prints 15 significant digits, which does not
// round-trip every double: BZERO = 2^63 became 9.22337203685478E+18, so the
// uint64 convention was lost on disk. Keep the familiar 15-digit spelling
// when it reads back exactly, otherwise write 17 digits (always exact).
inline void update_key_double(fitsfile* fptr, const char* key, double value,
                              const char* comment, int* status) {
    char buf[40];
    std::snprintf(buf, sizeof(buf), "%.15G", value);
    const int decimals = (std::strtod(buf, nullptr) == value) ? -15 : -17;
    fits_update_key_dbl(fptr, key, value, decimals, comment, status);
}

// BZERO/TZERO for the uint64 convention: the exact integer 2^63 (FITS 4.0
// §5.3), which does not fit a long long card and would otherwise be spelled
// as a float (9.2233720368547758E+18).
inline void update_key_uint64_offset(fitsfile* fptr, const char* key,
                                     const char* comment, int* status) {
    char literal[] = "9223372036854775808";
    char card[FLEN_CARD];
    fits_make_key(key, literal, comment, card, status);
    fits_update_card(fptr, key, card, status);
}

// Adding 2^63 modulo 2^64 is flipping the sign bit.
inline void flip_sign_bit_u64(uint64_t* data, size_t n) {
    for (size_t i = 0; i < n; ++i) data[i] ^= 0x8000000000000000ULL;
}

// Checked NAXISn product — unsigned multiply with overflow detection so a
// pathological header cannot wrap LONGLONG and under-allocate a read buffer.
inline LONGLONG checked_nelements_product(const LONGLONG* naxes, int naxis) {
    if (naxis <= 0) {
        return 0;
    }
    unsigned long long product = 1ULL;
    for (int i = 0; i < naxis; ++i) {
        const LONGLONG dim = naxes[i];
        if (dim < 0) {
            throw std::runtime_error("NAXIS product overflow: negative axis length");
        }
        if (dim == 0) {
            return 0;
        }
        const unsigned long long udim = static_cast<unsigned long long>(dim);
        if (product > std::numeric_limits<unsigned long long>::max() / udim) {
            throw std::runtime_error("NAXIS product overflow");
        }
        product *= udim;
        // Leave headroom for element size up to 16 bytes (complex128).
        if (product > static_cast<unsigned long long>(std::numeric_limits<LONGLONG>::max() / 16)) {
            throw std::runtime_error("NAXIS product overflow");
        }
    }
    return static_cast<LONGLONG>(product);
}

template <typename DimT>
inline LONGLONG checked_nelements_product(const std::vector<DimT>& naxes) {
    if (naxes.empty()) {
        return 0;
    }
    std::array<LONGLONG, 9> buf{};
    if (naxes.size() > 9) {
        throw std::runtime_error("NAXIS product overflow: too many axes");
    }
    for (size_t i = 0; i < naxes.size(); ++i) {
        buf[i] = static_cast<LONGLONG>(naxes[i]);
    }
    return checked_nelements_product(buf.data(), static_cast<int>(naxes.size()));
}


// Refuse an uncompressed image whose header declares more data than the file
// holds before a buffer of that size is allocated: a corrupt or truncated
// header claiming a huge NAXIS product on a small file raised a bare
// MemoryError (std::bad_alloc) instead of saying the file is truncated, and a
// cutout of it failed with CFITSIO's "tried to move past end of file". Like
// the table readers, the whole declared data unit (without the final block
// padding) must be present. `fptr` must be positioned on the image HDU.
// Tile-compressed images are not checked here: their pixel count does not
// bound the stored bytes.
inline void ensure_image_data_within_file(fitsfile* fptr, int bitpix, LONGLONG nelements) {
    if (fptr == nullptr || fptr->Fptr == nullptr || nelements <= 0) {
        return;
    }
    int status = 0;
    if (fits_is_compressed_image(fptr, &status) != 0 || status != 0) {
        return;
    }
    LONGLONG headstart = 0;
    LONGLONG datastart = 0;
    LONGLONG dataend = 0;
    fits_get_hduaddrll(fptr, &headstart, &datastart, &dataend, &status);
    if (status != 0 || datastart < 0) {
        return;
    }
    const unsigned long long elem = static_cast<unsigned long long>(bitpix < 0 ? -bitpix : bitpix) / 8ULL;
    const unsigned long long n = static_cast<unsigned long long>(nelements);
    const unsigned long long room =
        static_cast<unsigned long long>(std::numeric_limits<LONGLONG>::max()) -
        static_cast<unsigned long long>(datastart);
    const LONGLONG filesize = fptr->Fptr->logfilesize;
    if (elem == 0 || (n <= room / elem && static_cast<LONGLONG>(n * elem) + datastart <= filesize)) {
        return;
    }
    const bool overflow = n > room / elem;
    throw std::runtime_error(
        "FITS image is truncated: header declares " +
        (overflow ? std::string("more data than can be addressed")
                  : std::to_string(n * elem) + " data bytes through offset " +
                        std::to_string(static_cast<long long>(n * elem) + datastart)) +
        " but the file holds only " + std::to_string(static_cast<long long>(filesize)) +
        " bytes; the copy is incomplete or the header is corrupt");
}



// "status=<n> msg=<text>" for a failed image (or cutout) read. CFITSIO has
// no text for NO_COMPRESSED_TILE (415: "unknown error status") and only a
// bare "error uncompressing image" for DATA_DECOMPRESSION_ERR (414), which
// is what a truncated or corrupt tile reports (vendored bounds patches).
// The Rice decoder reports a short or corrupt stream as status 1 ("non-CFITSIO
// program error"); on a tile-compressed HDU (`fptr` positioned on it) that
// status can only come from a decoder, so it gets the same note.
inline std::string image_read_status(int status, fitsfile* fptr = nullptr) {
    char err_text[FLEN_STATUS];
    fits_get_errstatus(status, err_text);
    std::string msg = err_text;
    const bool compressed_hdu =
        fptr != nullptr && fptr->Fptr != nullptr && fptr->Fptr->compressimg != 0;
    if (status == NO_COMPRESSED_TILE) {
        msg = "compressed tile data is truncated or corrupt (tile missing)";
    } else if (status == DATA_DECOMPRESSION_ERR || (status == 1 && compressed_hdu)) {
        msg += ": compressed tile data is truncated or corrupt";
    }
    return "status=" + std::to_string(status) + " msg=" + msg;
}

// ---------------------------------------------------------------------------
// ResolvedFITSMeta — flattened metadata for canonical image read
// ---------------------------------------------------------------------------
struct ResolvedFITSMeta {
    int bitpix = 0;
    int naxis = 0;
    std::array<LONGLONG, 9> naxes_ll{};
    bool scaled = false;
    double bscale = 1.0;
    double bzero = 0.0;
    bool compressed = false;
};


inline size_t datatype_elem_size(int datatype) {
    switch (datatype) {
        case TBYTE:
        case TSBYTE:    return sizeof(uint8_t);
        case TSHORT:
        case TUSHORT:   return sizeof(uint16_t);
        case TINT:
        case TUINT:     return sizeof(uint32_t);
        case TLONGLONG:
        case TULONGLONG: return sizeof(uint64_t);
        case TFLOAT:    return sizeof(float);
        case TDOUBLE:   return sizeof(double);
        default:        return 0;
    }
}

// CFITSIO fnan() (fitsio2.h FNANMASK) treats Inf *and* exponent-zero
// (signed zero, subnormals) as undefined whenever a nulval pointer is
// non-NULL. Float/double *storage* holds IEEE values — also inside
// compressed tiles (GZIP copies bits verbatim; quantized codecs restore
// special values), as astropy reads them — so those reads must pass nullptr
// for Inf / -0 / NaN to survive. A NaN nulval is only for integer storage
// read as float (BLANK promotion / arbitrary scale), where it marks
// undefined pixels. The storage BITPIX decides (r9b-01), with one
// exception: a tile-compressed float HDU that declares ZBLANK (a column or
// keyword; fpack and quantized writes with NaN pixels) stores NaN as that
// integer code in its quantized tiles, and CFITSIO only maps it back to NaN
// when a nulval is passed. Without ZBLANK, CFITSIO never null-checks.
// `compressed` is accepted for call-site symmetry: CFITSIO's own state for
// the current HDU (`fptr`) decides whether it is a compressed image.
inline void* cfitsio_float_nulval_ptr(
    int bitpix, bool /*compressed*/, int datatype, float* fnull, double* dnull,
    fitsfile* fptr = nullptr
) {
    if (datatype != TFLOAT && datatype != TDOUBLE) {
        return nullptr;
    }
    if (bitpix == FLOAT_IMG || bitpix == DOUBLE_IMG) {
        const bool declares_zblank = fptr != nullptr && fptr->Fptr != nullptr &&
            fptr->Fptr->compressimg && fptr->Fptr->cn_zblank != 0;
        if (!declares_zblank) return nullptr;
    }
    return (datatype == TFLOAT) ? static_cast<void*>(fnull)
                                : static_cast<void*>(dnull);
}

// A ZBLANK float HDU can also hold tiles CFITSIO could not quantize (Inf
// pixels, constant tiles): those are stored losslessly as raw IEEE values in
// a GZIP_COMPRESSED_DATA (or legacy UNCOMPRESSED_DATA) column, which the
// writer only adds once such a tile exists. Under the NaN nulval that the
// quantized tiles need, CFITSIO null-checks the raw tiles with fnan() too,
// so Inf read back as NaN and -0.0 / subnormals as +0.0 (astropy keeps them).
inline bool compressed_float_mixes_lossless_tiles(fitsfile* fptr) {
    if (fptr == nullptr || fptr->Fptr == nullptr) return false;
    const FITSfile* f = fptr->Fptr;
    return f->compressimg && f->cn_zblank != 0 &&
        (f->cn_gzip_data >= 1 || f->cn_uncompressed >= 1);
}

// CFITSIO caches the last decompressed tile per tile column without
// recording the null handling it was decoded under, so a read that changes
// the nulval must not be served from it.
inline void drop_compressed_tile_cache(fitsfile* fptr) {
    FITSfile* f = fptr->Fptr;
    if (f->tilerow == nullptr || f->tilesize[0] <= 0) return;
    const long ntilebins = static_cast<long>((f->znaxis[0] - 1) / f->tilesize[0]) + 1;
    for (long k = 0; k < ntilebins; ++k) f->tilerow[k] = 0;
}

// Read `n` image pixels of `datatype` into `dst` through `read(nulval, out,
// &status)` (one fits_read_img / fits_read_subset call), with the nulval
// cfitsio_float_nulval_ptr chooses. For a float HDU that mixes quantized
// ZBLANK tiles with lossless ones, read twice -- with the NaN nulval and
// without one -- and keep the exact pixel unless only the null-checked read
// marks it undefined: a ZBLANK code decodes to a finite number without a
// nulval, while a raw tile's NaN / Inf are non-finite in both reads.
template <typename ReadFn>
inline void read_float_image_pixels(
    fitsfile* fptr, int bitpix, int datatype, void* dst, LONGLONG n,
    ReadFn&& read, int* status
) {
    float fnull = NAN;
    double dnull = NAN;
    void* nulval = cfitsio_float_nulval_ptr(bitpix, false, datatype, &fnull, &dnull, fptr);
    const bool two_pass = nulval != nullptr && n > 0 &&
        (bitpix == FLOAT_IMG || bitpix == DOUBLE_IMG) &&
        compressed_float_mixes_lossless_tiles(fptr);
    if (!two_pass) {
        read(nulval, dst, status);
        return;
    }
    auto merge = [&](auto* out) {
        using T = std::remove_pointer_t<decltype(out)>;
        std::vector<T> exact(static_cast<size_t>(n));
        drop_compressed_tile_cache(fptr);
        read(nulval, dst, status);
        drop_compressed_tile_cache(fptr);
        if (*status != 0) return;
        read(nullptr, exact.data(), status);
        drop_compressed_tile_cache(fptr);
        if (*status != 0) return;
        for (size_t i = 0; i < exact.size(); ++i) {
            if (!(std::isnan(out[i]) && std::isfinite(exact[i]))) out[i] = exact[i];
        }
    };
    if (datatype == TFLOAT) {
        merge(static_cast<float*>(dst));
    } else {
        merge(static_cast<double*>(dst));
    }
}

// ---------------------------------------------------------------------------
// read_tensor_canonical — shared core for all image read paths
// ---------------------------------------------------------------------------
inline native::Tensor read_tensor_canonical(
    fitsfile* fptr,
    const std::string& path,
    const ResolvedFITSMeta& meta,
    bool use_mmap,
    int raw_fd,
    bool use_chunking = false
) {
    const int bitpix = meta.bitpix;
    const int naxis = meta.naxis;
    const std::array<LONGLONG, 9>& naxes_ll = meta.naxes_ll;
    const bool scaled = meta.scaled;
    const double bscale = meta.bscale;
    const double bzero = meta.bzero;
    const bool compressed = meta.compressed;

    validate_image_naxis(naxis);
    if (naxis == 0) {
        native::ScalarType dtype = native::kUInt8;
        switch (bitpix) {
            case BYTE_IMG: dtype = native::kUInt8; break;
            case SHORT_IMG: dtype = native::kInt16; break;
            case LONG_IMG: dtype = native::kInt32; break;
            case LONGLONG_IMG: dtype = native::kInt64; break;
            case FLOAT_IMG: dtype = native::kFloat32; break;
            case DOUBLE_IMG: dtype = native::kFloat64; break;
            default: break;
        }
        return native::empty({0}, native::TensorOptions().dtype(dtype));
    }

    LONGLONG nelements = checked_nelements_product(naxes_ll.data(), naxis);
    ensure_image_data_within_file(fptr, bitpix, nelements);

    int64_t torch_shape[9];
    for (int i = 0; i < naxis; ++i)
        torch_shape[i] = static_cast<int64_t>(naxes_ll[naxis - 1 - i]);

    // Unsigned conventions
    const bool unsigned_short = scaled && bitpix == SHORT_IMG && bscale == 1.0 && is_unsigned_short_offset(bzero);
    const bool unsigned_long  = scaled && bitpix == LONG_IMG  && bscale == 1.0 && is_unsigned_long_offset(bzero);
    const bool unsigned_longlong =
        scaled && bitpix == LONGLONG_IMG && bscale == 1.0 && is_unsigned_longlong_offset(bzero);

    native::ScalarType dtype;
    int datatype;
    if (scaled) {
        if (bitpix == BYTE_IMG && bscale == 1.0 && bzero == -128.0) {
            dtype = native::kChar; datatype = TSBYTE;
        } else if (unsigned_short) {
            dtype = native::kUInt16; datatype = TUSHORT;
        } else if (unsigned_long) {
            dtype = native::kUInt32; datatype = TUINT;
        } else if (unsigned_longlong) {
            dtype = native::kUInt64; datatype = TULONGLONG;
        } else if (bitpix == BYTE_IMG || bitpix == SHORT_IMG) {
            // int8/int16 codes are exact in float32. int32/int64 are not.
            dtype = native::kFloat32; datatype = TFLOAT;
        } else {
            dtype = native::kFloat64; datatype = TDOUBLE;
        }
    } else {
        switch (bitpix) {
            case BYTE_IMG: dtype = native::kUInt8; datatype = TBYTE; break;
            case SHORT_IMG: dtype = native::kInt16; datatype = TSHORT; break;
            case LONG_IMG: dtype = native::kInt32; datatype = TINT; break;
            case LONGLONG_IMG: dtype = native::kInt64; datatype = TLONGLONG; break;
            case FLOAT_IMG: dtype = native::kFloat32; datatype = TFLOAT; break;
            case DOUBLE_IMG: dtype = native::kFloat64; datatype = TDOUBLE; break;
            default: throw std::runtime_error("Unsupported BITPIX");
        }
    }

    auto tensor = native::empty(torch_shape, static_cast<size_t>(naxis), native::TensorOptions().dtype(dtype));

    // BYTE_IMG direct pread — works for mmap on/off (pread is buffered I/O,
    // not a secret mmap). Beats CFITSIO fits_read_img on large int8 payloads.
    const bool signed_byte_scaled = scaled && bitpix == BYTE_IMG && bscale == 1.0 && bzero == -128.0;
    if (!compressed && bitpix == BYTE_IMG && (!scaled || signed_byte_scaled)) {
        int status = 0;
        LONGLONG headstart = 0, data_offset = 0, dataend = 0;
        fits_get_hduaddrll(fptr, &headstart, &data_offset, &dataend, &status);
        if (status == 0 && data_offset > 0) {
            const size_t nbytes = static_cast<size_t>(nelements);
            const int fd = raw_fd;
            if (fd != -1 && read_region_via_fd(fd, static_cast<off_t>(data_offset), tensor.data_ptr(), nbytes)) {
                if (signed_byte_scaled)
                    xor_sign_bit_u8(static_cast<uint8_t*>(tensor.data_ptr()), nbytes);
                return tensor;
            }
        }
    }

    // Multi-byte mmap fast path — SIMD endian convert while copying (all sizes).
    const bool multi_byte_mmap_ok =
        use_mmap && !compressed &&
        (!scaled || unsigned_short || unsigned_long || unsigned_longlong) &&
        !has_cfitsio_extended_filename_syntax(path);
    if (multi_byte_mmap_ok) {
        size_t elem_size = 0;
        switch (bitpix) {
            case SHORT_IMG: elem_size = sizeof(uint16_t); break;
            // LONG_IMG: enable for both plain int32 and unsigned_long.
            // The unsigned offset (+2147483648u) is applied only when
            // unsigned_long is true — see the bswap dispatch below.
            case LONG_IMG: elem_size = sizeof(uint32_t); break;
            case LONGLONG_IMG: elem_size = sizeof(uint64_t); break;
            // FLOAT_IMG: bswap_32 is identical to int32 — the raw bits are
            // the same 4-byte big-endian pattern regardless of interpretation.
            case FLOAT_IMG: elem_size = sizeof(uint32_t); break;
            case DOUBLE_IMG: elem_size = sizeof(uint64_t); break;
            default: break;
        }
        if (elem_size > 0) {
            int status = 0;
            LONGLONG headstart = 0, data_offset = 0, dataend = 0;
            fits_get_hduaddrll(fptr, &headstart, &data_offset, &dataend, &status);
            if (status == 0 && data_offset > 0 && raw_fd != -1) {
                const size_t nbytes = static_cast<size_t>(nelements) * elem_size;
                struct stat sb {};
                if (nbytes > 0 && fstat(raw_fd, &sb) == 0 &&
                    static_cast<size_t>(sb.st_size) >= static_cast<size_t>(data_offset) + nbytes) {
                    static const long kPageSize = sysconf(_SC_PAGESIZE);
                    const off_t page_mask = kPageSize > 0 ? static_cast<off_t>(kPageSize - 1) : 0;
                    const off_t page_offset = data_offset & ~page_mask;
                    const size_t map_len = static_cast<size_t>(nbytes + (data_offset - page_offset));
                    void* map_ptr = mmap(nullptr, map_len, PROT_READ, MAP_SHARED, raw_fd, page_offset);
                    if (map_ptr != MAP_FAILED) {
#if defined(MADV_SEQUENTIAL)
                        madvise(map_ptr, map_len, MADV_SEQUENTIAL);
#endif
#if defined(MADV_WILLNEED)
                        madvise(map_ptr, map_len, MADV_WILLNEED);
#endif
                        const size_t src_offset = static_cast<size_t>(data_offset - page_offset);
                        if (host_is_little_endian()) {
                            if (elem_size == sizeof(uint16_t)) {
                                const auto* src = reinterpret_cast<const uint16_t*>(
                                    static_cast<const uint8_t*>(map_ptr) + src_offset);
                                auto* dst = static_cast<uint16_t*>(tensor.data_ptr());
                                if (unsigned_short) {
                                    internal::bswap16_copy_u16_offset(
                                        src, dst, static_cast<size_t>(nelements),
                                        static_cast<uint16_t>(32768));
                                } else {
                                    internal::bswap16_copy(
                                        src, dst, static_cast<size_t>(nelements));
                                }
                            } else if (elem_size == sizeof(uint32_t)) {
                                const auto* src = reinterpret_cast<const uint32_t*>(
                                    static_cast<const uint8_t*>(map_ptr) + src_offset);
                                auto* dst = static_cast<uint32_t*>(tensor.data_ptr());
                                if (unsigned_long) {
                                    internal::bswap32_copy_u32_offset(
                                        src, dst, static_cast<size_t>(nelements),
                                        2147483648u);
                                } else {
                                    internal::bswap32_copy(
                                        src, dst, static_cast<size_t>(nelements));
                                }
                            } else if (elem_size == sizeof(uint64_t)) {
                                const auto* src = reinterpret_cast<const uint64_t*>(
                                    static_cast<const uint8_t*>(map_ptr) + src_offset);
                                auto* dst = static_cast<uint64_t*>(tensor.data_ptr());
                                internal::bswap64_copy(
                                    src, dst, static_cast<size_t>(nelements));
                            }
                        } else {
                            std::memcpy(tensor.data_ptr(), static_cast<const uint8_t*>(map_ptr) + src_offset, nbytes);
                        }
                        if (unsigned_longlong) {
                            flip_sign_bit_u64(static_cast<uint64_t*>(tensor.data_ptr()),
                                              static_cast<size_t>(nelements));
                        }
                        munmap(map_ptr, map_len);
                        return tensor;
                    }
                }
            }
        }
    }

    // CFITSIO fallback. nulval=NaN only for ZBLANK compressed tiles or
    // integer storage read as float (BLANK). Native IEEE must pass nullptr —
    // fnan() would turn Inf and signed zero into NaN / +0.
    auto read_pixels = [&](LONGLONG first, LONGLONG count, void* dst, int* st) {
        read_float_image_pixels(
            fptr, bitpix, datatype, dst, count,
            [&](void* nulval, void* out, int* s) {
                int anynul = 0;
                fits_read_img(fptr, datatype, first, count, nulval, out, &anynul, s);
            },
            st);
    };

    int status = 0;
    if (!use_chunking) {
        read_pixels(1, nelements, tensor.data_ptr(), &status);
    } else {
        static const size_t kChunkSizeBytes = 128 * 1024 * 1024;
        const size_t pixel_size = datatype_elem_size(datatype);
        const size_t effective_pixel_size = pixel_size > 0 ? pixel_size : 1;
        const LONGLONG chunk_pixels = static_cast<LONGLONG>(kChunkSizeBytes / effective_pixel_size);

        if (nelements <= chunk_pixels) {
            read_pixels(1, nelements, tensor.data_ptr(), &status);
        } else {
            LONGLONG remain = nelements;
            LONGLONG offset = 0;
            char* dst_ptr = static_cast<char*>(tensor.data_ptr());
            while (remain > 0 && status == 0) {
                LONGLONG n_read = (remain > chunk_pixels) ? chunk_pixels : remain;
                read_pixels(1 + offset, n_read,
                            static_cast<void*>(dst_ptr + (offset * effective_pixel_size)), &status);
                offset += n_read;
                remain -= n_read;
            }
        }
    }

    if (status != 0) {
        throw std::runtime_error("Error reading image data: " + image_read_status(status, fptr));
    }

    return tensor;
}

// ---------------------------------------------------------------------------

// FITS text is restricted to printable ASCII (codes 32..126). The lenient
// sanitizer above is for *reading* raw CFITSIO output, where dropping stray
// bytes keeps sloppy files readable. For text the caller supplies we must not
// silently rewrite it: writing the value "l-cold" with a leading Greek lambda
// used to store "-cold", a different string with no error. Reject it loudly
// instead, the way astropy does.
inline std::string require_fits_ascii(const std::string& input, const char* what) {
    for (size_t i = 0; i < input.size(); ++i) {
        const unsigned char c = static_cast<unsigned char>(input[i]);
        if (c < 32 || c > 126) {
            throw std::invalid_argument(
                std::string(what) +
                " must contain only printable ASCII characters (32..126); "
                "unexpected byte " + std::to_string(static_cast<unsigned>(c)) +
                " at index " + std::to_string(i));
        }
    }
    return input;
}


inline std::string sanitize_fits_key(const std::string& input) {
    // The normalization loop below drops every character it does not
    // recognize, so a keyword containing non-ASCII bytes would silently
    // collapse into a different keyword. Reject it instead.
    require_fits_ascii(input, "FITS keyword");
    // Preserve long / hierarchical keywords (spaces allowed). Short FITS
    // keywords stay [A-Z0-9_-] only, uppercased.
    std::string raw = input;
    while (!raw.empty() && (raw.front() == ' ' || raw.front() == '\t')) {
        raw.erase(raw.begin());
    }
    while (!raw.empty() && (raw.back() == ' ' || raw.back() == '\t')) {
        raw.pop_back();
    }
    std::string upper;
    upper.reserve(raw.size());
    for (unsigned char c : raw) {
        upper.push_back(static_cast<char>(std::toupper(c)));
    }
    // CFITSIO adds the HIERARCH token itself for long keys — strip a leading
    // "HIERARCH " so we do not double-prefix.
    if (upper.rfind("HIERARCH ", 0) == 0) {
        raw = raw.substr(9);
        while (!raw.empty() && raw.front() == ' ') {
            raw.erase(raw.begin());
        }
        upper = upper.substr(9);
        while (!upper.empty() && upper.front() == ' ') {
            upper.erase(upper.begin());
        }
    }

    const bool allow_spaces = raw.size() > 8 || raw.find(' ') != std::string::npos;
    std::string output;
    output.reserve(raw.size());
    for (unsigned char c : raw) {
        if (std::isalnum(c) || c == '_' || c == '-' || c == '.' ||
            (allow_spaces && c == ' ')) {
            output.push_back(
                allow_spaces ? static_cast<char>(c)
                             : static_cast<char>(std::toupper(c)));
        }
    }
    if (allow_spaces && !output.empty()) {
        // Uppercase alphanumeric runs but keep spaces (ESO-style keys).
        for (char& c : output) {
            if (std::isalpha(static_cast<unsigned char>(c))) {
                c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
            }
        }
    }
    return output.empty() ? "UNKNOWN" : output;
}

} // namespace detail
} // namespace torchfits
