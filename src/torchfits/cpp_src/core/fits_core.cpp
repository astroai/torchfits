// Implementation of the torch-free FITS primitives declared in fits_core.h.
//
// The bodies were moved verbatim out of fits_detail.h (which still includes
// this header) so the metadata path and the tensor path cannot drift apart.
// Only two things changed mechanically: `inline` is gone (these are now
// out-of-line exports of libtorchfits_core), and the sign-bit XOR uses the
// core thread pool instead of at::parallel_for, which libtorchfits_core does
// not link.

#include "core/fits_core.h"

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <climits>
#include <fcntl.h>
#include <list>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/stat.h>
#if defined(__APPLE__)
#include <sys/mount.h>
#include <sys/param.h>
#else
#include <sys/vfs.h>
#endif
#include <unistd.h>

#include "core/parallel.h"
#include "internal_utils.h"
#include "security.h"

namespace torchfits {
namespace detail {

#ifndef TORCHFITS_CORE_BUILD_ID
#define TORCHFITS_CORE_BUILD_ID "unknown"
#endif

namespace {

// CFITSIO makes its global lock (Fitsio_Lock) recursive lazily: the first
// fits_open_* / fits_create_* runs fits_init_cfitsio(), which calls
// pthread_mutex_init on it. Every other entry point -- fits_clear_errmsg (which
// open_fits_readonly calls first), the fptr-table update in fits_close_file --
// takes that same lock before then as a plain zero-initialised mutex. When a
// process's first reads start on several threads at once, one thread can
// re-initialise the mutex while another holds it; the holder's unlock then
// leaves it locked for good and every later CFITSIO call hangs (all threads
// parked in ffxmsg / fits_clear_Fptr). Initialise it while this library loads,
// which is single-threaded and happens before anything can call into it.
[[maybe_unused]] const int g_cfitsio_init_status = fits_init_cfitsio();

}  // namespace

const char* core_library_build_id() {
    return TORCHFITS_CORE_BUILD_ID;
}

RawFdHolder::~RawFdHolder() {
    if (fd != -1) ::close(fd);
}

// FITS stores BZERO/TZERO as ASCII decimals; writers may serialize the
// canonical unsigned offsets with rounding error (e.g. 32767.99999 after a
// float32 round-trip). Match them within the same tight tolerance the table
// reader uses, so image and table paths agree on the unsigned conventions.
bool is_unsigned_short_offset(double offset) {
    return std::abs(offset - 32768.0) < 1e-5;
}
bool is_unsigned_long_offset(double offset) {
    return std::abs(offset - 2147483648.0) < 1e-5;
}

void validate_image_naxis(int naxis) {
    if (naxis < 0 || naxis > 9) {
        throw std::runtime_error(
            "torchfits supports FITS images with at most 9 axes; got NAXIS=" +
            std::to_string(naxis));
    }
}

void read_image_params_9d(
    fitsfile* fptr,
    int* bitpix,
    int* naxis,
    std::array<LONGLONG, 9>& naxes,
    int* status
) {
    // Random Groups data (GROUPS=T with PCOUNT/GCOUNT) is not supported:
    // decoding it as an ordinary image would silently return wrong-shaped
    // values. Fail loudly instead of guessing. Compressed-image HDUs are
    // unaffected — cfitsio virtualizes them as plain images and they never
    // carry a GROUPS keyword.
    {
        int groups_status = 0;
        int groups_val = 0;
        fits_read_key(fptr, TLOGICAL, "GROUPS", &groups_val, nullptr, &groups_status);
        if (groups_status == 0 && groups_val != 0) {
            throw std::runtime_error(
                "Random Groups FITS images (GROUPS=T) are not supported");
        }
    }
    int declared_naxis = 0;
    fits_read_key(fptr, TINT, "NAXIS", &declared_naxis, nullptr, status);
    if (*status != 0) return;
    validate_image_naxis(declared_naxis);
    fits_get_img_paramll(fptr, 9, bitpix, naxis, naxes.data(), status);
}

// ---------------------------------------------------------------------------
// Sign-bit XOR for signed-byte encoding
// ---------------------------------------------------------------------------
void xor_sign_bit_u8(uint8_t* p, size_t nbytes) {
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
    core::parallel_for(0, static_cast<int64_t>(nbytes), 1 << 20, [=](int64_t begin, int64_t end) {
        xor_block(p + begin, static_cast<size_t>(end - begin));
    });
}

ScaleDetectionResult detect_scale_info_fast(fitsfile* fptr, int bitpix) {
    ScaleDetectionResult out;
    if (!fptr || bitpix == FLOAT_IMG || bitpix == DOUBLE_IMG) return out;

    // BLANK means undefined pixels. Identity BSCALE/BZERO is otherwise the
    // unscaled-int16 fast path, which cannot hold NaN — promote to scaled
    // float so nulval=NaN applies (quantize + integer images with BLANK).
    int blank_status = 0;
    long blank_val = 0;
    fits_read_key(fptr, TLONG, "BLANK", &blank_val, nullptr, &blank_status);
    const bool has_blank = (blank_status == 0);

    int equiv_status = 0;
    int equiv_type = bitpix;
    fits_get_img_equivtype(fptr, &equiv_type, &equiv_status);
    if (equiv_status == 0) {
        // CFITSIO has no unsigned 64-bit convention: fits_get_img_equivtype
        // reports TLONGLONG for BITPIX=64 regardless of BZERO, so the shortcut
        // would hide a uint64 convention (BZERO=2^63). Always read BSCALE/BZERO
        // for LONGLONG_IMG. BLANK also skips the shortcut so we still load
        // BSCALE/BZERO before marking scaled.
        if (equiv_type == bitpix && bitpix != LONGLONG_IMG && !has_blank) return out;
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
    if (has_blank) out.scaled = true;
    return out;
}

std::mutex g_shared_meta_mutex;
std::unordered_map<std::string, std::shared_ptr<SharedReadMeta>> g_shared_meta;
std::atomic<uint64_t> g_shared_meta_uid{1};

namespace {
// Bumped by every invalidate_shared_meta_for_path() and
// clear_shared_meta_cache(); see OpenedVersion.
std::atomic<uint64_t> g_shared_meta_epoch{0};
}  // namespace

namespace {

// g_shared_meta gained one entry per distinct path and never shed any, so a
// process scanning a large tree held every path it had ever touched. Cap it and
// drop the least recently used path. Eviction only unregisters: a caller still
// holding the shared_ptr keeps a valid entry, and the next lookup of that path
// starts a fresh one with a new uid, so uid-keyed caches cannot mix the two.
constexpr size_t kMaxSharedMetaEntries = 4096;

// Most recently used path at the back. Guarded by g_shared_meta_mutex.
std::list<std::string>& shared_meta_lru() {
    static std::list<std::string> order;
    return order;
}

// One registry key per file, whatever spelling the caller used. './x.fits',
// 'x.fits', '/abs/x.fits' and a symlink to it all name one file, and a writer
// invalidates by whichever spelling *it* was given: keyed by the raw string, a
// reader's entry under another spelling kept answering read_nrows/read_shape
// from the old file until its validation window ran out. The local file is
// resolved with realpath(); a name that does not resolve (not created yet,
// dangling link) resolves its directory instead, so an invalidation before a
// create and a lookup after it agree. Anything else is kept as given.
//
// canonical_path() is realpath() for an existing name. glibc's realpath()
// readlink()s every path component, so a file twelve directories deep costs
// twelve syscalls each time a name is first seen (or after clear_file_cache).
// On Linux the kernel has already resolved the name when it opens it, and
// /proc/self/fd/<n> reads back that canonical path: three syscalls at any
// depth. Anything unusual (no /proc, an unreachable or deleted path, a name
// too long) falls back to realpath().
bool canonical_path(const std::string& path, std::string* out) {
#if defined(__linux__) && defined(O_PATH)
    const int fd = ::open(path.c_str(), O_PATH | O_CLOEXEC);
    if (fd >= 0) {
        char link[48];
        std::snprintf(link, sizeof(link), "/proc/self/fd/%d", fd);
        char buf[PATH_MAX];
        const ssize_t n = ::readlink(link, buf, sizeof(buf));
        ::close(fd);
        static constexpr char kDeleted[] = " (deleted)";
        constexpr size_t kDeletedLen = sizeof(kDeleted) - 1;
        if (n > 0 && static_cast<size_t>(n) < sizeof(buf) && buf[0] == '/' &&
            !(static_cast<size_t>(n) >= kDeletedLen &&
              std::memcmp(buf + n - kDeletedLen, kDeleted, kDeletedLen) == 0)) {
            out->assign(buf, static_cast<size_t>(n));
            return true;
        }
    } else if (errno == ENOENT || errno == ENOTDIR) {
        return false;  // realpath() would fail the same way
    }
#endif
    char buf[PATH_MAX];
    if (::realpath(path.c_str(), buf) == nullptr) return false;
    out->assign(buf);
    return true;
}

std::string resolve_local_path(const std::string& path) {
    std::string resolved;
    if (canonical_path(path, &resolved)) return resolved;
    const size_t slash = path.find_last_of('/');
    const std::string leaf = slash == std::string::npos ? path : path.substr(slash + 1);
    if (leaf.empty() || leaf == "." || leaf == "..") return path;
    const std::string dir = slash == std::string::npos
        ? std::string(".")
        : (slash == 0 ? std::string("/") : path.substr(0, slash));
    std::string out;
    if (!canonical_path(dir, &out)) return path;
    if (out.empty() || out.back() != '/') out.push_back('/');
    return out + leaf;
}

}  // namespace

const bool kValidateSharedMeta = []() {
    return torchfits::internal::env_flag_default_true("TORCHFITS_SHARED_META_VALIDATE");
}();

const int64_t kSharedMetaValidateIntervalNs = []() {
    constexpr int64_t kDefaultMs = 1000;
    return torchfits::internal::env_nonnegative_int(
        "TORCHFITS_SHARED_META_VALIDATE_INTERVAL_MS", kDefaultMs) * 1000000LL;
}();

namespace {

// resolve_local_path() is glibc realpath(): one readlink() per path component,
// so a read of a file eight directories deep paid ~16 syscalls just to find
// its registry entry (twice per read: lookup and open), and the cached probes
// (read_num_hdus, read_shape) went from a hash lookup to ~7 us. Remember each
// resolution for the same window SharedReadMeta trusts its stat snapshot:
// what a name resolves to can change behind the process's back (a symlink
// retargeted, a directory renamed) exactly as the file's contents can, and
// both are re-checked on the same schedule. An in-process write invalidates
// through invalidate_shared_meta_for_path(), which drops every memo, so this
// process's own writes are seen at once, as before.
//
// Keyed by the absolute spelling: a relative name is prefixed with getcwd(),
// one cheap syscall, so a chdir can never pair a name with another
// directory's file. A validation interval of 0 ("always re-check") resolves
// every time.
struct ResolvedName {
    std::string base;      // resolve_local_path() of the name
    bool exists = false;   // realpath() of the full name succeeded
    int64_t resolved_ns = 0;
};

constexpr size_t kMaxResolvedNames = 4096;

// Guarded by g_shared_meta_mutex, like the registry it serves.
std::unordered_map<std::string, ResolvedName>& resolved_names() {
    static std::unordered_map<std::string, ResolvedName> names;
    return names;
}

// Bumped by every invalidation, so a resolution that raced one is not stored.
uint64_t g_resolved_names_generation = 0;

std::string absolute_spelling(const std::string& path) {
    if (!path.empty() && path[0] == '/') return path;
    char cwd[PATH_MAX];
    if (::getcwd(cwd, sizeof(cwd)) == nullptr) return std::string();
    std::string out(cwd);
    if (out.empty() || out.back() != '/') out.push_back('/');
    return out + path;
}

ResolvedName resolve_name_uncached(const std::string& path, int64_t now_ns) {
    ResolvedName out;
    if (canonical_path(path, &out.base)) {
        out.exists = true;
    } else {
        out.base = resolve_local_path(path);
    }
    out.resolved_ns = now_ns;
    return out;
}

ResolvedName resolve_name(const std::string& path, int64_t now_ns) {
    if (kSharedMetaValidateIntervalNs <= 0 || path.empty()) {
        return resolve_name_uncached(path, now_ns);
    }
    const std::string spelled = absolute_spelling(path);
    if (spelled.empty()) return resolve_name_uncached(path, now_ns);
    uint64_t generation = 0;
    {
        std::lock_guard<std::mutex> lock(g_shared_meta_mutex);
        auto& names = resolved_names();
        auto it = names.find(spelled);
        if (it != names.end() && now_ns - it->second.resolved_ns < kSharedMetaValidateIntervalNs) {
            return it->second;
        }
        generation = g_resolved_names_generation;
    }
    ResolvedName resolved = resolve_name_uncached(spelled, now_ns);
    {
        std::lock_guard<std::mutex> lock(g_shared_meta_mutex);
        if (generation == g_resolved_names_generation) {
            auto& names = resolved_names();
            if (names.size() >= kMaxResolvedNames) names.clear();
            names[spelled] = resolved;
        }
    }
    return resolved;
}

// Called with g_shared_meta_mutex held.
void forget_resolved_names() {
    resolved_names().clear();
    ++g_resolved_names_generation;
}

}  // namespace

bool resolve_existing_local_path(const std::string& path, std::string* out) {
    if (path.empty()) return false;
    ResolvedName resolved = resolve_name(path, torchfits::internal::monotonic_now_ns());
    if (!resolved.exists) return false;
    *out = std::move(resolved.base);
    return true;
}

namespace {

// Drop everything cached about a path and rotate its identity. Called with
// meta->mutex held. In-flight readers hold their own shared_ptr, so clearing
// raw_fd defers the close until they release it -- a pread/mmap can never
// race a close of the same fd.
void reset_cached_metadata(const std::shared_ptr<SharedReadMeta>& meta) {
    meta->raw_fd.reset();
    meta->image_info_cache.clear();
    meta->compressed_cache.clear();
    meta->scale_cache.clear();
    // hdu_name_cache is keyed by EXTNAME alone, so it must be dropped
    // with the rest: a rewrite can move a name to a different index
    // (or reuse it for another HDU), and a surviving entry would make
    // resolution silently read the wrong extension.
    meta->hdu_name_cache.clear();
    meta->nrows_cache.clear();
    meta->colnames_cache.clear();
    meta->hdu_type_cache.clear();
    meta->num_hdus = -1;
    meta->hdu_offsets.reset();
    // Rotate identity so per-thread caches keyed by {uid, hdu} cannot
    // pair stale shape/dtype/scale metadata with the replaced file
    // (R7-CPP2). In-flight readers keep their own shared_ptr and the
    // data they already decoded from the previous generation.
    meta->uid.store(g_shared_meta_uid.fetch_add(1, std::memory_order_relaxed),
                    std::memory_order_relaxed);
}

}  // namespace

std::shared_ptr<SharedReadMeta> get_shared_meta_for_path(const std::string& filename) {
    // Validation stats the file the name reads from: the base file of an
    // extended-syntax name, so "f.fits[1]" notices a rewrite of f.fits like
    // "f.fits" does. A URL has nothing to stat, so nothing about it can ever be
    // re-validated: hand out a private, unregistered entry instead of caching
    // path-level answers that would go stale for the life of the process.
    const std::string stat_path = cfitsio_stat_path(filename);
    if (stat_path.empty()) {
        auto meta = std::make_shared<SharedReadMeta>();
        meta->uid.store(g_shared_meta_uid.fetch_add(1, std::memory_order_relaxed),
                        std::memory_order_relaxed);
        return meta;
    }
    const int64_t now_ns = torchfits::internal::monotonic_now_ns();
    // Registry key: the canonical base file followed by any extended-syntax
    // suffix, so "f.fits[1]" keys as "/abs/f.fits[1]" and an invalidation of
    // the base can find it.
    const std::string base = resolve_name(stat_path, now_ns).base;
    const std::string key = base + filename.substr(stat_path.size());
    std::shared_ptr<SharedReadMeta> meta;
    {
        std::lock_guard<std::mutex> lock(g_shared_meta_mutex);
        auto& order = shared_meta_lru();
        auto it = g_shared_meta.find(key);
        if (it == g_shared_meta.end()) {
            meta = std::make_shared<SharedReadMeta>();
            meta->uid.store(g_shared_meta_uid.fetch_add(1, std::memory_order_relaxed),
                            std::memory_order_relaxed);
            meta->lru_pos = order.insert(order.end(), key);
            g_shared_meta.emplace(key, meta);
            while (g_shared_meta.size() > kMaxSharedMetaEntries) {
                g_shared_meta.erase(order.front());
                order.pop_front();
            }
        } else {
            meta = it->second;
            order.splice(order.end(), order, meta->lru_pos);
        }
    }
    if (!kValidateSharedMeta) return meta;
    std::unique_lock<std::shared_mutex> meta_lock(meta->mutex);
    if (kSharedMetaValidateIntervalNs > 0 && meta->last_stat_check_ns != 0 &&
        (now_ns - meta->last_stat_check_ns) < kSharedMetaValidateIntervalNs) {
        return meta;
    }
    meta->last_stat_check_ns = now_ns;
    // Stat the file the key names, not the caller's spelling: with the name
    // resolution remembered, a symlink retargeted inside the window must not
    // load the new target's identity into the old target's entry.
    struct stat st {};
    if (stat(base.c_str(), &st) != 0) {
        // A stat that fails for a path which previously had one means the file
        // is gone -- unlinked, renamed away, or unreachable through its parent
        // directory. This branch is what that looks like: has_stat was false
        // and stayed false, so every later probe found "no change" and went on
        // answering from shape/nrows/colnames/hdu_type/num_hdus caches about a
        // file that no longer existed. read_shape, read_hdu_type, read_colnames,
        // read_nrows and read_num_hdus all returned a plausible value for an
        // unlinked path instead of the error their payload-reading siblings
        // (read, read_header) correctly raised.
        //
        // Exempt the never-had-one case, which is not "gone": a path that has
        // never been stat-able has nothing cached that a later stat could
        // contradict.
        if (meta->has_stat) {
            reset_cached_metadata(meta);
            meta->has_stat = false;
        }
        return meta;
    }
    {
        int64_t cur_mtime_ns = torchfits::internal::mtime_ns_from_stat(st);
        if (!meta->has_stat || meta->size != st.st_size ||
            meta->mtime_ns != cur_mtime_ns || meta->inode != st.st_ino) {
            reset_cached_metadata(meta);
            meta->has_stat = true;
            meta->size = st.st_size;
            meta->mtime_ns = cur_mtime_ns;
            meta->inode = st.st_ino;
        }
    }
    return meta;
}

int open_readonly_fd(const std::string& filename) {
#ifdef O_CLOEXEC
    int fd = ::open(filename.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd != -1) return fd;
#endif
    return ::open(filename.c_str(), O_RDONLY);
}

// ---------------------------------------------------------------------------
// Learned HDU offsets
// ---------------------------------------------------------------------------
//
// CFITSIO reaches HDU n by reading every header in front of it unless the
// handle already knows where HDU n starts (FITSfile::headstart[]). A read-only
// handle publishes the offsets it learned when it closes, and the next
// open_fits_readonly() of the *same file version* is handed them, which is the
// state it would have reached by walking those headers itself.
//
// "Same file version" is the whole correctness argument, because a wrong
// offset silently reads a different HDU: a same-size rewrite can move HDU n to
// where HDU n-1 used to start. A version is (st_dev, st_ino, size, mtime,
// ctime). ctime is what makes it a content version: every write, truncate,
// rename or link change sets it to the current time and userspace cannot set
// it back (unlike mtime: cp -p, touch -d). It is still only as fine as the
// filesystem's timestamps -- jiffies on most local filesystems, 1 s on some,
// 2 s on FAT -- so two writes in one tick can leave it unchanged. Hence the
// age rule: a version is only recorded when its ctime and mtime are at least
// kHduOffsetMinAgeNs in the past at the moment it is captured. Any write after
// that moment gets a timestamp in a later tick and so a different version;
// a version captured while the file is still being written is never recorded.
//
// The protocol:
//   * open: the version is fstat() of the descriptor CFITSIO actually reads
//     through (fits_get_diskfile_fd, a vendored-CFITSIO patch). Built against
//     a CFITSIO without it, the path is stat()ed before and after
//     fits_open_diskfile() instead and must agree: swapping the file under
//     the name and back would change its ctime (link count or rename), and an
//     inode cannot be reused while it is open -- the one gap is swapping a
//     symlink or a parent directory away and back within the open call. Seed
//     only from a table recorded for exactly this version, never on the
//     strength of SharedReadMeta's validation window.
//   * the version and the SharedReadMeta generation seen at open are kept per
//     handle (open_handles()) when the version is old enough;
//   * close: publish only if the opened file still has that version (nothing
//     changed what the handle read), the path still names it, and the
//     generation is still current (no in-process writer invalidated it).
//
// Two mitigations below narrow what the version can miss: files on FUSE,
// FAT and exFAT mounts are never cached (timestamps_trustworthy), and a
// backwards wall-clock step seen by the process turns the cache off
// (trusted_wall_clock_now).
//
// Residual cases the version cannot see, all needing the writer to defeat
// ctime: a filesystem that does not maintain ctime at all (other than the
// FUSE/FAT mounts recognised above, e.g. some network filesystems);
// a backwards clock step before the process started, on a coarse-timestamp
// filesystem;
// a network filesystem whose server clock runs behind the client's by more
// than the age margin; a single write() call, or writes through a shared
// mapping, that keep modifying the file for longer than the margin while a
// reader opens it. TORCHFITS_HDU_OFFSET_CACHE_MAX_BYTES=0 turns the cache off.
//
// A read-only open never shares CFITSIO's FITSfile with another open
// (fits_already_open() returns early for READONLY, and a READWRITE open of a
// file open READONLY is refused), so the handle's headstart[] is private to it
// and needs no lock beyond the caller's own use of the handle.

namespace {

// How old a file version must be before its offsets are recorded: above the 2 s mtime granularity of FAT, the coarsest filesystem
// in common use, plus a clock tick.
const int64_t kHduOffsetMinAgeNs = torchfits::internal::env_nonnegative_int(
    "TORCHFITS_HDU_OFFSET_CACHE_MIN_AGE_MS", 3000) * 1000000LL;

// Process-wide budget for published offset tables (8 bytes per HDU). Up to
// kMaxSharedMetaEntries paths can each hold one; without a cap a scan over
// thousands of many-HDU files kept every table alive. 0 disables the cache.
const size_t kHduOffsetCacheMaxBytes = static_cast<size_t>(torchfits::internal::env_nonnegative_int(
    "TORCHFITS_HDU_OFFSET_CACHE_MAX_BYTES", 64LL << 20));

std::atomic<size_t> g_hdu_offset_bytes{0};

// The age rule trusts CLOCK_REALTIME to move forward. A backwards step (an
// NTP step, a manual date change) can bring a rewrite back into the very
// tick of a recorded version on a coarse-timestamp filesystem, reproducing
// its (dev, ino, size, mtime, ctime). The process watches the offset between
// CLOCK_REALTIME and an elapsed-time clock that never steps: forward steps
// only raise it, NTP slewing moves it by at most ~0.5 ms/s, so a drop of more
// than kWallClockStepToleranceNs below the largest offset seen means the wall
// clock went back, and the offset cache stays off for the rest of the
// process (offsets already learned are not used either). A step before the
// process started is not visible; see docs/architecture.md.
#if defined(__linux__)
constexpr clockid_t kElapsedClock = CLOCK_BOOTTIME;
#else
constexpr clockid_t kElapsedClock = CLOCK_MONOTONIC;
#endif
constexpr int64_t kWallClockStepToleranceNs = 1000000000LL;
std::atomic<int64_t> g_max_wall_offset_ns{INT64_MIN};
std::atomic<bool> g_wall_clock_stepped_back{false};

// CLOCK_REALTIME now, or false once (or as soon as) a backwards step is seen.
bool trusted_wall_clock_now(int64_t* now_ns) {
    if (g_wall_clock_stepped_back.load(std::memory_order_relaxed)) return false;
    struct timespec real {};
    struct timespec elapsed {};
    if (clock_gettime(CLOCK_REALTIME, &real) != 0 ||
        clock_gettime(kElapsedClock, &elapsed) != 0) {
        return false;
    }
    const int64_t real_ns =
        static_cast<int64_t>(real.tv_sec) * 1000000000LL + static_cast<int64_t>(real.tv_nsec);
    const int64_t offset = real_ns - (static_cast<int64_t>(elapsed.tv_sec) * 1000000000LL +
                                      static_cast<int64_t>(elapsed.tv_nsec));
    int64_t seen = g_max_wall_offset_ns.load(std::memory_order_relaxed);
    while (offset > seen &&
           !g_max_wall_offset_ns.compare_exchange_weak(seen, offset, std::memory_order_relaxed)) {
    }
    if (seen != INT64_MIN && offset < seen - kWallClockStepToleranceNs) {
        g_wall_clock_stepped_back.store(true, std::memory_order_relaxed);
        return false;
    }
    if (now_ns != nullptr) *now_ns = real_ns;
    return true;
}

// Take the first reading when the library loads, so a step any time after
// that is caught even before the first open.
[[maybe_unused]] const bool g_wall_clock_baseline = trusted_wall_clock_now(nullptr);

// Filesystems whose timestamps cannot carry the version argument: FUSE mounts
// (sshfs and others report ctime = mtime, so `rsync --inplace -t` can restore
// a version exactly) and FAT/exFAT (no ctime at all, 2 s mtime). Their files
// are read with the plain header walk.
bool timestamps_trustworthy(int fd, const std::string& path) {
#if defined(__APPLE__)
    struct statfs fs {};
    const int rc = fd >= 0 ? ::fstatfs(fd, &fs) : ::statfs(path.c_str(), &fs);
    if (rc != 0) return false;
    const std::string type(fs.f_fstypename);
    return !(type == "msdos" || type == "exfat" || type.find("fuse") != std::string::npos);
#else
    struct statfs fs {};
    const int rc = fd >= 0 ? ::fstatfs(fd, &fs) : ::statfs(path.c_str(), &fs);
    if (rc != 0) return false;
    const auto type = static_cast<unsigned long>(fs.f_type);
    constexpr unsigned long kFuseMagic = 0x65735546UL;
    constexpr unsigned long kMsdosMagic = 0x4d44UL;
    constexpr unsigned long kExfatMagic = 0x2011BAB0UL;
    return type != kFuseMagic && type != kMsdosMagic && type != kExfatMagic;
#endif
}

}  // namespace

struct FileVersion {
    dev_t dev = 0;
    ino_t ino = 0;
    off_t size = 0;
    int64_t mtime_ns = 0;
    int64_t ctime_ns = 0;

    static FileVersion of(const struct stat& st) {
        FileVersion v;
        v.dev = st.st_dev;
        v.ino = st.st_ino;
        v.size = st.st_size;
        v.mtime_ns = torchfits::internal::mtime_ns_from_stat(st);
#if defined(__APPLE__)
        v.ctime_ns = static_cast<int64_t>(st.st_ctimespec.tv_sec) * 1000000000LL +
                     static_cast<int64_t>(st.st_ctimespec.tv_nsec);
#else
        v.ctime_ns = static_cast<int64_t>(st.st_ctim.tv_sec) * 1000000000LL +
                     static_cast<int64_t>(st.st_ctim.tv_nsec);
#endif
        return v;
    }

    bool operator==(const FileVersion& o) const {
        return dev == o.dev && ino == o.ino && size == o.size && mtime_ns == o.mtime_ns &&
               ctime_ns == o.ctime_ns;
    }

    // See the age rule above. A timestamp in the future (clock skew) is not
    // old enough either.
    bool settled() const {
        int64_t now_ns = 0;
        if (!trusted_wall_clock_now(&now_ns)) return false;
        return now_ns - ctime_ns >= kHduOffsetMinAgeNs && now_ns - mtime_ns >= kHduOffsetMinAgeNs;
    }
};

// headstart[0 .. maxhdu + 1] of one file version. Charged against
// kHduOffsetCacheMaxBytes for as long as it lives.
struct HduOffsetTable {
    FileVersion version;
    std::vector<LONGLONG> headstart;
    size_t charged = 0;

    HduOffsetTable() = default;
    HduOffsetTable(const HduOffsetTable&) = delete;
    HduOffsetTable& operator=(const HduOffsetTable&) = delete;
    ~HduOffsetTable() {
        if (charged != 0) g_hdu_offset_bytes.fetch_sub(charged, std::memory_order_relaxed);
    }
};

namespace {

// Only plain local paths: a URL has no stat identity to validate offsets
// against, and an extended-syntax name may open a filtered or virtual file.
bool hdu_offsets_cacheable(const std::string& path) {
    return kHduOffsetCacheMaxBytes != 0 && !has_cfitsio_extended_filename_syntax(path) &&
           path.find("://") == std::string::npos;
}

// What a handle saw when it was opened; see the protocol above. With the
// descriptor accessor the version is not read at open (`version_known` false):
// most handles never see three HDUs and so never publish, and the fstat() and
// fstatfs() that capturing it cost were paid on every small read. The close
// reads it from the same descriptor instead and requires it to have been
// settled at `open_wall_ns` -- the age rule applied at the moment of the open,
// which is exactly the version an open-time capture would have recorded: any
// write after the open stamps a ctime later than open_wall_ns - min_age.
//
// `meta_uid` is the generation of the path's SharedReadMeta at open, or 0 when
// none was registered; then `epoch` (bumped by every invalidation and clear)
// must be unchanged at close instead.
struct OpenedVersion {
    FileVersion version;
    bool version_known = false;
    int64_t open_wall_ns = 0;
    uint64_t meta_uid = 0;
    uint64_t epoch = 0;
};

// Keyed by handle address. Every open_fits_readonly() of a cacheable path
// overwrites or erases its handle's entry and remember_hdu_offsets() erases
// it, so an entry left behind by a handle closed some other way can only be
// read by a later handle at the same address that was not opened through
// open_fits_readonly(); that entry's version predates the later handle, so it
// still only publishes when nothing changed the file since. The map is capped;
// losing entries only loses publications.
constexpr size_t kMaxTrackedHandles = 1 << 16;

std::mutex& open_handles_mutex() {
    static std::mutex m;
    return m;
}

std::unordered_map<const fitsfile*, OpenedVersion>& open_handles() {
    static std::unordered_map<const fitsfile*, OpenedVersion> handles;
    return handles;
}

void track_open_handle(const fitsfile* fptr, const OpenedVersion* opened) {
    std::lock_guard<std::mutex> lock(open_handles_mutex());
    auto& handles = open_handles();
    if (opened == nullptr) {
        handles.erase(fptr);
        return;
    }
    if (handles.size() >= kMaxTrackedHandles) handles.clear();
    handles[fptr] = *opened;
}

bool take_open_handle(const fitsfile* fptr, OpenedVersion* out) {
    std::lock_guard<std::mutex> lock(open_handles_mutex());
    auto& handles = open_handles();
    auto it = handles.find(fptr);
    if (it == handles.end()) return false;
    *out = it->second;
    handles.erase(it);
    return true;
}

std::shared_ptr<const HduOffsetTable> make_offset_table(
    const FileVersion& version, const LONGLONG* headstart, size_t count) {
    auto table = std::make_shared<HduOffsetTable>();
    table->version = version;
    table->headstart.assign(headstart, headstart + count);
    const size_t bytes = count * sizeof(LONGLONG);
    size_t used = g_hdu_offset_bytes.load(std::memory_order_relaxed);
    do {
        if (bytes > kHduOffsetCacheMaxBytes || used > kHduOffsetCacheMaxBytes - bytes) {
            return nullptr;
        }
    } while (!g_hdu_offset_bytes.compare_exchange_weak(used, used + bytes,
                                                       std::memory_order_relaxed));
    table->charged = bytes;
    return table;
}

// Hand a freshly opened read-only handle the offsets recorded for the exact
// version it opened. CFITSIO's fits_movabs_hdu() jumps straight to any HDU
// whose headstart[] entry is known and otherwise reads every header in between;
// seeding headstart[]/maxhdu is exactly the state the handle would have
// reached by walking those headers itself.
void seed_hdu_offsets(fitsfile* fptr, const HduOffsetTable& table) {
    FITSfile* file = fptr->Fptr;
    const std::vector<LONGLONG>& known = table.headstart;
    if (file->writemode != READONLY || file->logfilesize != table.version.size ||
        known.size() < 3 || file->maxhdu < 0) {
        return;
    }
    // The handle has read its primary header: it must agree with the table on
    // where HDU 1 starts, or this is not the file the table describes.
    if (known[0] != file->headstart[0] || known[1] != file->headstart[1]) return;
    // headstart[maxhdu + 1] (the start of the next HDU) is known whenever HDU
    // maxhdu's header has been read, so `known` covers maxhdu = size - 2.
    const int known_max = static_cast<int>(known.size()) - 2;
    if (known_max <= file->maxhdu) return;
    if (known.size() > static_cast<size_t>(file->MAXHDU) + 1) {
        // Same growth rule as ffmahd(): MAXHDU + 1 entries are allocated.
        const size_t entries = known.size() + 1000;
        auto* grown = static_cast<LONGLONG*>(
            realloc(file->headstart, entries * sizeof(LONGLONG)));
        if (grown == nullptr) return;
        file->headstart = grown;
        file->MAXHDU = static_cast<int>(entries) - 1;
    }
    // Entries 0 .. maxhdu + 1 are the handle's own; only extend past them.
    for (size_t i = static_cast<size_t>(file->maxhdu) + 2; i < known.size(); ++i) {
        file->headstart[i] = known[i];
    }
    file->maxhdu = known_max;
}

#if defined(TORCHFITS_CFITSIO_HAS_DISKFILE_FD)
constexpr bool kVersionFromDescriptor = true;
#else
constexpr bool kVersionFromDescriptor = false;
#endif

// Version of the regular file an open handle reads, from its own descriptor.
bool descriptor_version(fitsfile* fptr, FileVersion* out) {
#if defined(TORCHFITS_CFITSIO_HAS_DISKFILE_FD)
    const int fd = fits_get_diskfile_fd(fptr);
    struct stat st {};
    if (fd < 0 || ::fstat(fd, &st) != 0 || !S_ISREG(st.st_mode)) return false;
    *out = FileVersion::of(st);
    return true;
#else
    (void)fptr;
    (void)out;
    return false;
#endif
}

// The version a just-opened handle reads (see the protocol above). `before`
// is the pre-open stat of the path when there is no descriptor accessor.
bool opened_version(fitsfile* fptr, const std::string& path, const struct stat* before,
                    FileVersion* out) {
    if (kVersionFromDescriptor) return descriptor_version(fptr, out);
    struct stat after {};
    if (before == nullptr || ::stat(path.c_str(), &after) != 0) return false;
    const FileVersion version = FileVersion::of(after);
    if (!(version == FileVersion::of(*before))) return false;
    *out = version;
    return true;
}

// The registered entry for `path`, or null: no creation, no validation stat.
// An open only needs an existing entry's offset table (seeding is checked
// against the descriptor's exact version anyway); creating and stat-ing one
// for every open cost a header-only read (read_header after
// clear_file_cache) a name resolution and a stat it never used.
std::shared_ptr<SharedReadMeta> peek_shared_meta(const std::string& path) {
    {
        std::lock_guard<std::mutex> lock(g_shared_meta_mutex);
        if (g_shared_meta.empty()) return nullptr;
    }
    const std::string base = resolve_name(path, torchfits::internal::monotonic_now_ns()).base;
    std::lock_guard<std::mutex> lock(g_shared_meta_mutex);
    auto it = g_shared_meta.find(base);
    return it == g_shared_meta.end() ? nullptr : it->second;
}

// After a successful open of a cacheable path: record what the handle opened
// and seed it from a table recorded for that version.
void after_cacheable_open(fitsfile* fptr, const std::string& path, const struct stat* before) {
    track_open_handle(fptr, nullptr);
    int64_t open_wall_ns = 0;
    if (!trusted_wall_clock_now(&open_wall_ns)) return;
    if (fptr->Fptr == nullptr) return;
    OpenedVersion opened;
    opened.open_wall_ns = open_wall_ns;
    opened.epoch = g_shared_meta_epoch.load(std::memory_order_acquire);
    std::shared_ptr<const HduOffsetTable> table;
    if (auto meta = peek_shared_meta(path)) {
        std::shared_lock<std::shared_mutex> lock(meta->mutex);
        opened.meta_uid = meta->uid.load(std::memory_order_relaxed);
        table = meta->hdu_offsets;
    }
    if (kVersionFromDescriptor) {
        // Seed only from a table recorded for exactly the version this
        // descriptor reads. A table is only ever recorded for a file on a
        // filesystem whose timestamps passed timestamps_trustworthy(), and its
        // version names that filesystem (st_dev), so a match needs no statfs.
        FileVersion version;
        if (table && descriptor_version(fptr, &version) && table->version == version) {
            seed_hdu_offsets(fptr, *table);
        }
        track_open_handle(fptr, &opened);
        return;
    }
    FileVersion version;
    if (!opened_version(fptr, path, before, &version)) return;
    if (!timestamps_trustworthy(-1, path)) return;
    opened.version = version;
    opened.version_known = true;
    if (table && table->version == version) seed_hdu_offsets(fptr, *table);
    if (version.settled()) track_open_handle(fptr, &opened);
}

}  // namespace

// Prefer fits_open_diskfile for plain local paths (skips CFITSIO extended-syntax parse).
int open_fits_readonly(fitsfile** fptr, const std::string& path) {
    int status = 0;
    // Drop stale messages so open_failure_message() reports this open only.
    fits_clear_errmsg();
    if (has_cfitsio_extended_filename_syntax(path) || path.find("://") != std::string::npos) {
        fits_open_file(fptr, path.c_str(), 0 /* READONLY */, &status);
        return status;
    }
    const bool cacheable = hdu_offsets_cacheable(path);
    struct stat before {};
    const bool have_before = cacheable && !kVersionFromDescriptor &&
                             ::stat(path.c_str(), &before) == 0 && S_ISREG(before.st_mode);
    fits_open_diskfile(fptr, path.c_str(), 0 /* READONLY */, &status);
    if (status == 0 && cacheable && *fptr != nullptr) {
        try {
            after_cacheable_open(*fptr, path, have_before ? &before : nullptr);
        } catch (...) {
            // A cache miss is never worth failing an open over; make sure the
            // handle is not left with a record it did not earn.
            try {
                track_open_handle(*fptr, nullptr);
            } catch (...) {
            }
        }
    }
    return status;
}

void remember_hdu_offsets(fitsfile* fptr, const std::string& path) noexcept {
    try {
        if (fptr == nullptr || fptr->Fptr == nullptr) return;
        if (!hdu_offsets_cacheable(path)) return;
        OpenedVersion opened;
        if (!take_open_handle(fptr, &opened)) return;
        if (!trusted_wall_clock_now(nullptr)) return;
        FITSfile* file = fptr->Fptr;
        // Fewer than three known HDUs is nothing a fresh open does not
        // already learn from its first header.
        if (file->writemode != READONLY || file->maxhdu < 2 ||
            (opened.version_known && file->logfilesize != opened.version.size)) {
            return;
        }
        // Nothing changed the file the handle read (a write moves its ctime)...
        FileVersion read_now;
        if (kVersionFromDescriptor) {
            if (!descriptor_version(fptr, &read_now)) return;
            if (opened.version_known) {
                if (!(read_now == opened.version)) return;
            } else {
                // ...which, read only now, must already have been settled
                // when the handle was opened (see OpenedVersion).
                if (opened.open_wall_ns - read_now.ctime_ns < kHduOffsetMinAgeNs ||
                    opened.open_wall_ns - read_now.mtime_ns < kHduOffsetMinAgeNs ||
                    file->logfilesize != read_now.size) {
                    return;
                }
                opened.version = read_now;
                opened.version_known = true;
            }
        }
        if (!opened.version_known) return;
        const size_t count = static_cast<size_t>(file->maxhdu) + 2;
        auto meta = get_shared_meta_for_path(path);
        if (!meta) return;
        auto already_known = [&]() {
            return meta->hdu_offsets && meta->hdu_offsets->version == opened.version &&
                   meta->hdu_offsets->headstart.size() >= count;
        };
        // Same generation the handle started in: an in-process writer's
        // invalidation (or a validation that saw the file change) rotates it.
        // With no entry at open, no invalidation or clear may have happened
        // since; the entry created just now is then the one to fill.
        const bool had_entry = opened.meta_uid != 0;
        if (!had_entry) {
            if (g_shared_meta_epoch.load(std::memory_order_acquire) != opened.epoch) return;
            std::shared_lock<std::shared_mutex> lock(meta->mutex);
            opened.meta_uid = meta->uid.load(std::memory_order_relaxed);
        }
        {
            std::shared_lock<std::shared_mutex> lock(meta->mutex);
            if (meta->uid.load(std::memory_order_relaxed) != opened.meta_uid || already_known()) {
                return;
            }
        }
#if defined(TORCHFITS_CFITSIO_HAS_DISKFILE_FD)
        if (!timestamps_trustworthy(fits_get_diskfile_fd(fptr), path)) return;
#endif
        // ...and the path still names it (a replacement has another inode).
        struct stat now {};
        if (::stat(path.c_str(), &now) != 0 || !(FileVersion::of(now) == opened.version)) return;
        auto table = make_offset_table(opened.version, file->headstart, count);
        if (!table) return;
        std::unique_lock<std::shared_mutex> lock(meta->mutex);
        if (!had_entry && g_shared_meta_epoch.load(std::memory_order_acquire) != opened.epoch) {
            return;
        }
        if (meta->uid.load(std::memory_order_relaxed) != opened.meta_uid || already_known()) {
            return;
        }
        meta->hdu_offsets = std::move(table);
    } catch (...) {
        // A cache fill is never worth failing a close over.
    }
}

std::string open_failure_message(const std::string& path, int status) {
    std::string details;
    char msg[FLEN_ERRMSG];
    while (fits_read_errmsg(msg) != 0) {
        std::string line(msg);
        while (!line.empty() &&
               (line.back() == ' ' || line.back() == '\n' || line.back() == '\r')) {
            line.pop_back();
        }
        // CFITSIO truncates messages at a byte boundary (splitting UTF-8) and
        // may echo raw server bytes; keep printable ASCII so the exception
        // text always decodes.
        for (char& c : line) {
            const unsigned char u = static_cast<unsigned char>(c);
            if (u < 0x20 || u > 0x7e) {
                c = '?';
            }
        }
        if (line.empty()) {
            continue;
        }
        if (!details.empty()) {
            details += "; ";
        }
        details += line;
    }
    if (details.empty() && status != 0) {
        char text[FLEN_STATUS];
        fits_get_errstatus(status, text);
        details = text;
    }
    std::string out = "Could not open FITS file: " + path;
    if (!details.empty()) {
        out += " (" + details + ")";
    }
    return out;
}

// ---------------------------------------------------------------------------
// Bounded raw-fd cache
// ---------------------------------------------------------------------------
// get_shared_raw_fd() memoises the descriptor so a repeat read skips an
// open(), and nothing bounded that memo: a loop over N distinct paths left N
// descriptors open for the life of the process. Measured on macOS with
// RLIMIT_NOFILE=64, after 90 reads the process held 64 of 64 descriptors and
// 199 of 200 further opens failed with EMFILE -- the fd table is shared, so
// what torchfits keeps is what astropy, h5py or matplotlib in the same process
// can no longer open.
//
// Cap the retained descriptors and release the least recently used. A reader
// borrows its own shared_ptr for the duration of its pread/mmap, so a release
// is deferred until the last holder drops it and eviction can never close a
// descriptor out from under an in-flight read. TORCHFITS_MAX_CACHED_FDS=0
// restores the unbounded behaviour.
//
// The default is small on purpose: the memo buys one open() syscall, which is
// not measurable. Measured over 300 repeat reads of a 2 MiB HDU with a warm
// page cache, the best of five runs per setting: 719 us/read at cap 64, 691
// at 8, 688 at 1 and 680 unbounded -- all within run-to-run spread, so the
// hot path loses nothing by keeping the working set small.
const size_t kMaxCachedRawFds = []() -> size_t {
    constexpr int64_t kDefault = 32;
    return static_cast<size_t>(
        torchfits::internal::env_nonnegative_int("TORCHFITS_MAX_CACHED_FDS", kDefault));
}();

namespace {

// Most recently used at the back. weak_ptr, so a meta erased by
// clear_shared_meta_cache() or invalidate_shared_meta_for_path() drops out on
// its own instead of needing a matching removal here.
std::mutex& raw_fd_lru_mutex() {
    static std::mutex m;
    return m;
}

std::list<std::weak_ptr<SharedReadMeta>>& raw_fd_lru() {
    static std::list<std::weak_ptr<SharedReadMeta>> order;
    return order;
}

// Called with meta->mutex NOT held. It takes the LRU lock and then the victims'
// meta locks, so calling it under a meta lock would invert the order against
// another thread trimming a different path.
//
// A repeat read only moves its own entry to the back (O(1), via the position
// the meta keeps). Walking the whole list on every read -- locking every
// entry's weak_ptr -- cost a cache miss per cached descriptor, which showed
// as several microseconds per small read once other work had evicted them.
// Entries whose meta has died are swept only when the list is over its cap.
void touch_raw_fd_cache(const std::shared_ptr<SharedReadMeta>& meta) {
    if (kMaxCachedRawFds == 0) return;
    // Victims are released after the LRU lock is dropped: the last reference
    // to a meta must not be destroyed while holding it.
    std::vector<std::shared_ptr<SharedReadMeta>> victims;
    {
        std::lock_guard<std::mutex> guard(raw_fd_lru_mutex());
        auto& order = raw_fd_lru();
        if (meta->raw_fd_lru_listed) {
            order.splice(order.end(), order, meta->raw_fd_lru_pos);
            return;
        }
        meta->raw_fd_lru_pos = order.insert(order.end(), meta);
        meta->raw_fd_lru_listed = true;
        if (order.size() <= kMaxCachedRawFds) return;
        for (auto it = order.begin(); it != order.end();) {
            if (it->expired()) {
                it = order.erase(it);
            } else {
                ++it;
            }
        }
        while (order.size() > kMaxCachedRawFds) {
            std::shared_ptr<SharedReadMeta> held = order.front().lock();
            order.pop_front();
            if (held) {
                held->raw_fd_lru_listed = false;
                victims.push_back(std::move(held));
            }
        }
    }
    for (const auto& held : victims) {
        std::unique_lock<std::shared_mutex> lock(held->mutex);
        held->raw_fd.reset();
    }
}

}  // namespace

std::shared_ptr<RawFdHolder> get_shared_raw_fd(
    const std::shared_ptr<SharedReadMeta>& meta, const std::string& filename) {
    if (!meta || has_cfitsio_extended_filename_syntax(filename)) return nullptr;
    std::shared_ptr<RawFdHolder> holder;
    {
        std::unique_lock<std::shared_mutex> lock(meta->mutex);
        if (!meta->raw_fd || meta->raw_fd->fd == -1) {
            // A failed open must not be memoised. An fd of -1 is a dead cache
            // entry that nothing clears: the stat identity that would clear it
            // (inode, size, mtime) does not change when the cause does, so a
            // fixed permission -- or a descriptor limit that eased -- left the
            // path on the slow CFITSIO path for the rest of the process.
            meta->raw_fd.reset();
            const int fd = open_readonly_fd(filename);
            if (fd == -1) return nullptr;  // callers fall back to their own handle
            meta->raw_fd = std::make_shared<RawFdHolder>(fd);
        }
        // Open-time snapshot (r9b-06): the fd must describe the same file
        // generation the shared stat snapshot does. A replacement that has not
        // been observed yet (or a lazy open after one) would otherwise hand
        // another file's bytes to readers holding old-generation metadata.
        // Callers fall back to their own CFITSIO handle when this returns null.
        if (meta->raw_fd->fd != -1 && meta->has_stat) {
            struct stat st {};
            if (fstat(meta->raw_fd->fd, &st) != 0 ||
                st.st_ino != meta->inode ||
                static_cast<off_t>(st.st_size) != meta->size ||
                torchfits::internal::mtime_ns_from_stat(st) != meta->mtime_ns) {
                return nullptr;
            }
        }
        holder = meta->raw_fd;
    }
    touch_raw_fd_cache(meta);
    return holder;
}

bool read_region_via_fd(int fd, off_t offset, void* dst_void, size_t nbytes) {
    if (fd == -1 || !dst_void || nbytes == 0) return false;
    // Hint the kernel to start async page-in before the synchronous pread loop.
    // This overlaps I/O with any preceding computation on the calling thread.
#if defined(__linux__)
    (void)::posix_fadvise(fd, offset, static_cast<off_t>(nbytes), POSIX_FADV_WILLNEED);
#endif
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

// Drops every entry that reads the same local file as `filename`, under any
// spelling: the canonical base and its extended-syntax names ("f.fits[1]",
// "f.fits(out.fits)[2]"), since a write to f.fits changes what they all read.
void invalidate_shared_meta_for_path(const std::string& filename) {
    const std::string stat_path = cfitsio_stat_path(filename);
    if (stat_path.empty()) return;  // URLs are never registered
    const std::string spelled = absolute_spelling(stat_path);
    const std::string base = resolve_local_path(spelled.empty() ? stat_path : spelled);
    std::lock_guard<std::mutex> lock(g_shared_meta_mutex);
    forget_resolved_names();
    g_shared_meta_epoch.fetch_add(1, std::memory_order_acq_rel);
    auto& order = shared_meta_lru();
    for (auto it = g_shared_meta.begin(); it != g_shared_meta.end();) {
        const std::string& key = it->first;
        const bool same_file =
            key.compare(0, base.size(), base) == 0 &&
            (key.size() == base.size() || key[base.size()] == '[' || key[base.size()] == '(');
        if (same_file) {
            order.erase(it->second->lru_pos);
            it = g_shared_meta.erase(it);
        } else {
            ++it;
        }
    }
}

void clear_shared_meta_cache() {
    std::lock_guard<std::mutex> lock(g_shared_meta_mutex);
    g_shared_meta.clear();
    shared_meta_lru().clear();
    forget_resolved_names();
    g_shared_meta_epoch.fetch_add(1, std::memory_order_acq_rel);
}

std::string sanitize_fits_string(const std::string& input) {
    std::string output = input;
    output.erase(std::remove_if(output.begin(), output.end(), [](unsigned char c) {
        return c < 32 || c > 126;
    }), output.end());
    return output;
}

// Drop only the bytes that cannot appear in UTF-8 (>= 0x80), leaving every
// card's structure intact. nanobind converts a std::string to Python `str` by
// UTF-8 decoding, so one stray high-bit byte in a header made
// read_header_string() raise; the caller then quietly fell back to the raw
// string dict, where every value loses its type (17 -> '17', T -> 'T'). Card
// text is written in printable ASCII, so a byte >= 0x80 is always corruption,
// and the read side already drops exactly such bytes from header values via
// sanitize_fits_string().
std::string drop_non_ascii_bytes(const std::string& input) {
    if (std::none_of(input.begin(), input.end(), [](unsigned char c) {
            return c >= 0x80;
        })) {
        return input;
    }
    std::string output;
    output.reserve(input.size());
    for (unsigned char c : input) {
        if (c < 0x80) output.push_back(static_cast<char>(c));
    }
    return output;
}

}  // namespace detail
}  // namespace torchfits
