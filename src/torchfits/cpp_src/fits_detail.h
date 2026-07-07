#pragma once

#include <string>
#include <algorithm>
#include <cctype>
#include <vector>
#include <unordered_map>
#include <thread>
#include <array>
#include <tuple>
#include <cmath>
#include <memory>
#include <mutex>
#include <atomic>
#include <limits>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#if defined(__APPLE__) || defined(__linux__)
#include <dlfcn.h>
#endif
#include <ATen/Parallel.h>
#include <fitsio.h>

#include "internal_utils.h"

namespace torchfits {
namespace detail {

// ---------------------------------------------------------------------------
// Sign-bit XOR for signed-byte encoding
// ---------------------------------------------------------------------------
inline void _xor_sign_bit_u8(uint8_t* p, size_t nbytes) {
    if (!p || nbytes == 0) return;
    static const size_t kParallelMinBytes = []() -> size_t {
        constexpr int64_t kDefault = 1 << 18;
        int64_t parsed = kDefault;
        if (const char* v = std::getenv("TORCHFITS_XOR_PARALLEL_MIN_BYTES")) {
            try { parsed = std::stoll(std::string(v)); } catch (...) { parsed = kDefault; }
        }
        return parsed <= 0 ? 1 : static_cast<size_t>(parsed);
    }();

    auto xor_block = [](uint8_t* ptr, size_t len) {
        if (!ptr || len == 0) return;
        constexpr uint64_t kMask64 = 0x8080808080808080ULL;
        size_t i = 0;
        while (i < len && ((reinterpret_cast<uintptr_t>(ptr + i) & 7u) != 0u)) {
            ptr[i] ^= 0x80;
            ++i;
        }
        uint64_t* p64 = reinterpret_cast<uint64_t*>(ptr + i);
        const size_t n64 = (len - i) / sizeof(uint64_t);
        for (size_t j = 0; j < n64; ++j) p64[j] ^= kMask64;
        i += n64 * sizeof(uint64_t);
        while (i < len) { ptr[i] ^= 0x80; ++i; }
    };

    if (nbytes < kParallelMinBytes) {
        xor_block(p, nbytes);
        return;
    }
    at::parallel_for(0, static_cast<int64_t>(nbytes), 1 << 20, [&](int64_t begin, int64_t end) {
        xor_block(p + begin, static_cast<size_t>(end - begin));
    });
}

// ---------------------------------------------------------------------------
// Scale detection
// ---------------------------------------------------------------------------
struct ScaleDetectionResult {
    bool scaled = false;
    bool trusted = true;
    double bscale = 1.0;
    double bzero = 0.0;
};

inline ScaleDetectionResult detect_scale_info_fast(fitsfile* fptr, int bitpix) {
    ScaleDetectionResult out;
    if (!fptr || bitpix == FLOAT_IMG || bitpix == DOUBLE_IMG) return out;
    int equiv_status = 0;
    int equiv_type = bitpix;
    fits_get_img_equivtype(fptr, &equiv_type, &equiv_status);
    if (equiv_status == 0) {
        if (equiv_type == bitpix) return out;
        if (bitpix == BYTE_IMG && equiv_type == SBYTE_IMG) {
            out.scaled = true; out.bscale = 1.0; out.bzero = -128.0;
            return out;
        }
    }
    double bscale = 1.0;
    double bzero = 0.0;
    int s1 = 0;
    fits_read_key(fptr, TDOUBLE, "BSCALE", &bscale, nullptr, &s1);
    if (s1 == 0) {
        out.bscale = bscale;
        if (bscale != 1.0) out.scaled = true;
    } else if (s1 != KEY_NO_EXIST) {
        out.scaled = true; out.trusted = false;
    }
    int s2 = 0;
    fits_read_key(fptr, TDOUBLE, "BZERO", &bzero, nullptr, &s2);
    if (s2 == 0) {
        out.bzero = bzero;
        if (bzero != 0.0) out.scaled = true;
    } else if (s2 != KEY_NO_EXIST) {
        out.scaled = true; out.trusted = false;
    }
    if (equiv_status == 0 && equiv_type != bitpix) out.scaled = true;
    return out;
}

// ---------------------------------------------------------------------------
// Shared read metadata cache
// ---------------------------------------------------------------------------
struct SharedReadMeta {
    uint64_t uid = 0;
    std::unordered_map<int, std::tuple<int, int, std::array<LONGLONG, 9>>> image_info_cache;
    std::unordered_map<int, bool> compressed_cache;
    std::unordered_map<int, bool> compressed_parallel_cache;
    std::unordered_map<int, bool> compressed_nulls_cache;
    std::unordered_map<int, std::tuple<bool, bool, double, double>> scale_cache;
    std::unordered_map<std::string, int> hdu_name_cache;
    bool has_stat = false;
    off_t size = 0;
    int64_t mtime_ns = 0;
    ino_t inode = 0;
    int64_t last_stat_check_ns = 0;
    int raw_fd = -1;
    std::mutex mutex;

    ~SharedReadMeta() {
        if (raw_fd != -1) { ::close(raw_fd); raw_fd = -1; }
    }
};

inline std::mutex g_shared_meta_mutex;
inline std::unordered_map<std::string, std::shared_ptr<SharedReadMeta>> g_shared_meta;
inline std::atomic<uint64_t> g_shared_meta_uid{1};

inline const bool kValidateSharedMeta = []() {
    return torchfits::internal::env_flag_default_true("TORCHFITS_SHARED_META_VALIDATE");
}();

inline const int64_t kSharedMetaValidateIntervalNs = []() {
    constexpr int64_t kDefaultMs = 1000;
    return torchfits::internal::env_nonnegative_int(
        "TORCHFITS_SHARED_META_VALIDATE_INTERVAL_MS", kDefaultMs) * 1000000LL;
}();

inline std::shared_ptr<SharedReadMeta> get_shared_meta_for_path(const std::string& filename) {
    bool can_stat = kValidateSharedMeta && filename.find('[') == std::string::npos;
    std::shared_ptr<SharedReadMeta> meta;
    {
        std::lock_guard<std::mutex> lock(g_shared_meta_mutex);
        auto it = g_shared_meta.find(filename);
        if (it == g_shared_meta.end()) {
            meta = std::make_shared<SharedReadMeta>();
            meta->uid = g_shared_meta_uid.fetch_add(1, std::memory_order_relaxed);
            g_shared_meta.emplace(filename, meta);
        } else {
            meta = it->second;
        }
    }
    if (!can_stat) return meta;
    const int64_t now_ns = torchfits::internal::monotonic_now_ns();
    std::lock_guard<std::mutex> meta_lock(meta->mutex);
    if (kSharedMetaValidateIntervalNs > 0 && meta->last_stat_check_ns != 0 &&
        (now_ns - meta->last_stat_check_ns) < kSharedMetaValidateIntervalNs) {
        return meta;
    }
    meta->last_stat_check_ns = now_ns;
    struct stat st {};
    if (stat(filename.c_str(), &st) == 0) {
        int64_t cur_mtime_ns = torchfits::internal::mtime_ns_from_stat(st);
        if (!meta->has_stat || meta->size != st.st_size ||
            meta->mtime_ns != cur_mtime_ns || meta->inode != st.st_ino) {
            if (meta->raw_fd != -1) { ::close(meta->raw_fd); meta->raw_fd = -1; }
            meta->image_info_cache.clear();
            meta->compressed_cache.clear();
            meta->compressed_parallel_cache.clear();
            meta->compressed_nulls_cache.clear();
            meta->scale_cache.clear();
            meta->has_stat = true;
            meta->size = st.st_size;
            meta->mtime_ns = cur_mtime_ns;
            meta->inode = st.st_ino;
        }
    }
    return meta;
}

inline int open_readonly_fd(const std::string& filename) {
#ifdef O_CLOEXEC
    int fd = ::open(filename.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd != -1) return fd;
#endif
    return ::open(filename.c_str(), O_RDONLY);
}

inline int get_shared_raw_fd(const std::shared_ptr<SharedReadMeta>& meta, const std::string& filename) {
    if (!meta || filename.find('[') != std::string::npos) return -1;
    std::lock_guard<std::mutex> lock(meta->mutex);
    if (meta->raw_fd != -1) return meta->raw_fd;
    meta->raw_fd = open_readonly_fd(filename);
    return meta->raw_fd;
}

inline bool read_region_via_fd(int fd, off_t offset, void* dst_void, size_t nbytes) {
    if (fd == -1 || !dst_void || nbytes == 0) return false;
    uint8_t* dst = static_cast<uint8_t*>(dst_void);
    size_t remaining = nbytes;
    off_t off = offset;
    while (remaining > 0) {
        ssize_t got = ::pread(fd, dst, remaining, off);
        if (got < 0) { if (errno == EINTR) continue; break; }
        if (got == 0) break;
        dst += static_cast<size_t>(got);
        off += static_cast<off_t>(got);
        remaining -= static_cast<size_t>(got);
    }
    if (remaining == 0) return true;
    struct stat sb {};
    if (fstat(fd, &sb) != 0) return false;
    if (offset < 0) return false;
    if (static_cast<size_t>(sb.st_size) < static_cast<size_t>(offset) + nbytes) return false;
    static const long kPageSize = sysconf(_SC_PAGESIZE);
    const off_t page_mask = kPageSize > 0 ? static_cast<off_t>(kPageSize - 1) : 0;
    const off_t page_offset = offset & ~page_mask;
    const size_t map_len = static_cast<size_t>(nbytes + (offset - page_offset));
    void* map_ptr = mmap(nullptr, map_len, PROT_READ, MAP_SHARED, fd, page_offset);
    if (map_ptr == MAP_FAILED) return false;
    const uint8_t* src = static_cast<const uint8_t*>(map_ptr) + (offset - page_offset);
    std::memcpy(dst_void, src, nbytes);
    munmap(map_ptr, map_len);
    return true;
}

inline void invalidate_shared_meta_for_path(const std::string& filename) {
    std::lock_guard<std::mutex> lock(g_shared_meta_mutex);
    g_shared_meta.erase(filename);
}

inline void clear_shared_meta_cache() {
    std::lock_guard<std::mutex> lock(g_shared_meta_mutex);
    g_shared_meta.clear();
}

using fits_is_compressed_with_nulls_fn = int (*)(fitsfile*);

inline bool has_compressed_nulls(fitsfile* fptr) {
#if defined(__APPLE__) || defined(__linux__)
    static fits_is_compressed_with_nulls_fn fn = []() -> fits_is_compressed_with_nulls_fn {
        void* sym = dlsym(RTLD_DEFAULT, "fits_is_compressed_with_nulls");
        return sym ? reinterpret_cast<fits_is_compressed_with_nulls_fn>(sym) : nullptr;
    }();
    if (fn) return fn(fptr) != 0;
#endif
    return false;
}

// ---------------------------------------------------------------------------
// RAII guard for fitsfile* handles (local version, not the cache.h one)
// ---------------------------------------------------------------------------
class FitsHandleGuard {
    fitsfile* handle_{nullptr};
public:
    FitsHandleGuard() noexcept = default;
    explicit FitsHandleGuard(fitsfile* h) noexcept : handle_(h) {}
    ~FitsHandleGuard() noexcept { close(); }
    FitsHandleGuard(const FitsHandleGuard&) = delete;
    FitsHandleGuard& operator=(const FitsHandleGuard&) = delete;
    FitsHandleGuard(FitsHandleGuard&& other) noexcept : handle_(other.handle_) { other.handle_ = nullptr; }
    FitsHandleGuard& operator=(FitsHandleGuard&& other) noexcept {
        if (this != &other) { close(); handle_ = other.handle_; other.handle_ = nullptr; }
        return *this;
    }
    fitsfile* get() const noexcept { return handle_; }
    fitsfile* release() noexcept { fitsfile* h = handle_; handle_ = nullptr; return h; }
    explicit operator bool() const noexcept { return handle_ != nullptr; }
private:
    void close() noexcept {
        if (handle_) { int status = 0; fits_close_file(handle_, &status); }
    }
};

// ---------------------------------------------------------------------------
// Byte-swap helpers (big-endian FITS → host)
// ---------------------------------------------------------------------------
template <typename T>
inline T load_bswap(const void* src);

template <>
inline uint16_t load_bswap<uint16_t>(const void* src) {
    uint16_t v; std::memcpy(&v, src, sizeof(v));
    return torchfits::internal::bswap_16(v);
}

template <>
inline uint32_t load_bswap<uint32_t>(const void* src) {
    uint32_t v; std::memcpy(&v, src, sizeof(v));
    return torchfits::internal::bswap_32(v);
}

template <>
inline uint64_t load_bswap<uint64_t>(const void* src) {
    uint64_t v; std::memcpy(&v, src, sizeof(v));
    return torchfits::internal::bswap_64(v);
}

// ---------------------------------------------------------------------------
// Compressed parallel-read helpers
// ---------------------------------------------------------------------------
inline bool compressed_parallel_enabled() {
    return torchfits::internal::env_flag_default_true("TORCHFITS_COMPRESSED_PARALLEL");
}

inline int64_t compressed_parallel_min_pixels() {
    constexpr int64_t kDefault = 1024LL * 1024LL;
    return torchfits::internal::env_nonnegative_int("TORCHFITS_COMPRESSED_PARALLEL_MIN_PIXELS", kDefault);
}

inline int64_t compressed_parallel_min_rows_per_thread() {
    constexpr int64_t kDefault = 256;
    int64_t v = torchfits::internal::env_nonnegative_int(
        "TORCHFITS_COMPRESSED_PARALLEL_MIN_ROWS_PER_THREAD", kDefault);
    return v > 0 ? v : 1;
}

inline int64_t compressed_parallel_max_threads() {
    constexpr int64_t kDefault = 2;
    int64_t v = torchfits::internal::env_nonnegative_int(
        "TORCHFITS_COMPRESSED_PARALLEL_MAX_THREADS", kDefault);
    return v > 0 ? v : 1;
}

inline bool compressed_parallel_hcompress_enabled() {
    return torchfits::internal::env_flag_default_true("TORCHFITS_COMPRESSED_PARALLEL_HCOMPRESS");
}

inline size_t datatype_elem_size(int datatype) {
    switch (datatype) {
        case TBYTE:
        case TSBYTE:    return sizeof(uint8_t);
        case TSHORT:    return sizeof(uint16_t);
        case TINT:      return sizeof(uint32_t);
        case TLONGLONG: return sizeof(uint64_t);
        case TFLOAT:    return sizeof(float);
        case TDOUBLE:   return sizeof(double);
        default:        return 0;
    }
}

inline bool try_read_compressed_rows_parallel(
    fitsfile* fptr, const std::string& path, int target_hdu,
    int naxis, const std::array<LONGLONG, 9>& naxes_ll,
    LONGLONG nelements, int datatype, bool allow_float, void* dst
) {
    if (!compressed_parallel_enabled() || !fptr || !dst) return false;
    if (path.find('[') != std::string::npos) return false;
    if (naxis != 2) return false;
    if (nelements < compressed_parallel_min_pixels()) return false;
    const size_t elem_size = datatype_elem_size(datatype);
    if (elem_size == 0) return false;
    if ((datatype == TFLOAT || datatype == TDOUBLE) && !allow_float) return false;

    const LONGLONG width_ll = naxes_ll[0];
    const LONGLONG rows_ll = naxes_ll[1];
    if (width_ll <= 0 || rows_ll <= 1) return false;
    if (width_ll > static_cast<LONGLONG>(std::numeric_limits<long>::max()) ||
        rows_ll > static_cast<LONGLONG>(std::numeric_limits<long>::max())) return false;

    long tile_dims[2] = {0, 0};
    int status = 0;
    fits_get_tile_dim(fptr, 2, tile_dims, &status);
    if (status != 0 || tile_dims[0] <= 0 || tile_dims[1] <= 0) return false;
    const LONGLONG tile_h_ll = static_cast<LONGLONG>(tile_dims[1]);
    if (tile_h_ll <= 0) return false;
    const LONGLONG tile_rows = (rows_ll + tile_h_ll - 1) / tile_h_ll;
    if (tile_rows <= 1) return false;

    const int64_t hw_threads = std::max<int64_t>(
        1, static_cast<int64_t>(std::thread::hardware_concurrency()));
    const int64_t max_threads = std::min<int64_t>(
        std::max<int64_t>(1, compressed_parallel_max_threads()), hw_threads);
    const int64_t min_rows_per_thread = compressed_parallel_min_rows_per_thread();
    const int64_t min_tile_rows_per_thread =
        std::max<int64_t>(1, (min_rows_per_thread + tile_h_ll - 1) / tile_h_ll);
    const int64_t by_tile_rows = tile_rows / min_tile_rows_per_thread;
    const int64_t nthreads = std::min<int64_t>(max_threads, by_tile_rows);
    if (nthreads < 2) return false;

    auto* dst_u8 = static_cast<uint8_t*>(dst);
    std::atomic<int> first_status{0};
    std::vector<std::thread> workers;
    workers.reserve(static_cast<size_t>(nthreads));
    std::vector<FitsHandleGuard> local_handles(static_cast<size_t>(nthreads));

    for (int64_t t = 0; t < nthreads; ++t) {
        fitsfile* raw = nullptr;
        int st = 0;
        ffreopen(fptr, &raw, &st);
        if (st != 0 || !raw) return false;
        FitsHandleGuard local(raw);
        fits_movabs_hdu(local.get(), target_hdu, nullptr, &st);
        if (st != 0) return false;
        local_handles[static_cast<size_t>(t)] = std::move(local);
    }

    for (int64_t t = 0; t < nthreads; ++t) {
        const int64_t tile_row_begin = (tile_rows * t) / nthreads;
        const int64_t tile_row_end = (tile_rows * (t + 1)) / nthreads;
        const int64_t row_begin = std::min<int64_t>(rows_ll, tile_row_begin * tile_h_ll);
        const int64_t row_end = std::min<int64_t>(rows_ll, tile_row_end * tile_h_ll);
        if (row_end <= row_begin) continue;

        fitsfile* local_ptr = local_handles[static_cast<size_t>(t)].get();
        workers.emplace_back([=, &first_status]() {
            if (!local_ptr) return;
            if (first_status.load(std::memory_order_relaxed) != 0) return;
            int st = 0;
            std::array<long, 2> fpixel{1L, static_cast<long>(row_begin + 1)};
            std::array<long, 2> lpixel{static_cast<long>(width_ll), static_cast<long>(row_end)};
            std::array<long, 2> inc{1L, 1L};
            int anynul = 0;
            size_t elem_offset = static_cast<size_t>(row_begin) * static_cast<size_t>(width_ll);
            void* chunk_ptr = static_cast<void*>(dst_u8 + (elem_offset * elem_size));
            fits_read_subset(local_ptr, datatype, fpixel.data(), lpixel.data(), inc.data(),
                             nullptr, chunk_ptr, &anynul, &st);
            if (st != 0) { int expected = 0; first_status.compare_exchange_strong(expected, st); }
        });
    }

    for (auto& worker : workers) worker.join();
    return first_status.load(std::memory_order_relaxed) == 0;
}

// ---------------------------------------------------------------------------
// FITS string sanitization
// ---------------------------------------------------------------------------
inline std::string sanitize_fits_string(const std::string& input) {
    std::string output = input;
    output.erase(std::remove_if(output.begin(), output.end(), [](unsigned char c) {
        return c < 32 || c > 126;
    }), output.end());
    return output;
}

inline std::string sanitize_fits_key(const std::string& input) {
    std::string output;
    output.reserve(input.length());
    for (char c : input) {
        if (std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '-') {
            output.push_back(std::toupper(static_cast<unsigned char>(c)));
        }
    }
    return output.empty() ? "UNKNOWN" : output;
}

} // namespace detail
} // namespace torchfits
