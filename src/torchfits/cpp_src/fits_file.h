#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <mutex>
#include <array>
#include <tuple>
#include <cstdint>
#include <sys/stat.h>
#include <sys/types.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <fitsio.h>

#include "native_tensor.h"

namespace torchfits {
namespace detail {
struct SharedReadMeta;
struct RawFdHolder;
}

class FITSFile {
public:
    struct ScaleInfo {
        bool scaled = false;
        bool trusted = true;
        double bscale = 1.0;
        double bzero = 0.0;
    };

    FITSFile(const char* filename, int mode);
    ~FITSFile();
    // Closes (or, with borrowers, flushes) the file. For a write handle a
    // failed final flush/close throws std::runtime_error: the file on disk is
    // incomplete. Read handles never throw here.
    void close();

    fitsfile* get_fptr() const { return fptr_; }

    // Take io_mutex_ from a thread that holds the GIL, waiting for it with the
    // GIL released. The write paths hold io_mutex_ while they give the GIL up
    // for the CFITSIO writes and take it back for the header cards, so a
    // thread blocking on the lock with the GIL held would deadlock against
    // them. Callers without the GIL lock io_mutex_ directly.
    std::unique_lock<std::recursive_mutex> lock_io_holding_gil() const {
        std::unique_lock<std::recursive_mutex> lock(io_mutex_, std::try_to_lock);
        if (!lock.owns_lock()) {
            nanobind::gil_scoped_release release;
            lock.lock();
        }
        return lock;
    }
    int get_start_hdu() const { return start_hdu_; }

    // For code that drives CFITSIO on this handle directly (bindings, table
    // readers). Throws on a closed handle instead of handing out nullptr. The
    // returned owner keeps the fitsfile open until the borrower drops it, so a
    // close() while a TableReader still uses the handle is deferred, not a
    // use-after-free.
    // Called with the GIL held (share_fptr_from_python_object).
    std::shared_ptr<fitsfile> share_fptr() {
        auto lock = lock_io_holding_gil();
        if (!fptr_) throw std::runtime_error("FITSFile is closed");
        return fptr_owner_;
    }

    // Run fn(fitsfile*) under the handle lock; throws on a closed handle.
    template <class F>
    auto with_open_fptr(F&& fn) {
        std::lock_guard<std::recursive_mutex> lock(io_mutex_);
        if (!fptr_) throw std::runtime_error("FITSFile is closed");
        return fn(fptr_);
    }

    // Run fn() under the handle's I/O lock whether or not the handle is still
    // open. A persistent TableReader keeps the fitsfile alive past close()
    // (share_fptr) and must keep serializing its cursor use with every other
    // borrower of the same handle.
    template <class F>
    auto with_io_lock(F&& fn) {
        std::lock_guard<std::recursive_mutex> lock(io_mutex_);
        return fn();
    }

    // True while the shared per-path metadata (and its raw fd) still describes
    // the file this handle opened. False after an out-of-band replace: the
    // shared fd then reads the new file, so callers must use the handle.
    bool shares_current_file() const;

    // False when this read-only handle's path now names a different file than
    // the one it opened (an out-of-band replace: new inode, size or mtime).
    // Borrowers that also read the path directly -- a TableReader's pread/mmap
    // paths reopen it by name -- would then mix the replacement's bytes with
    // this handle's geometry. True for write handles, names whose identity was
    // not captured, and a path that no longer exists (nothing to mix in).
    bool path_names_open_file() const;

    // The same test for an already-resolved file: false when st (an fstat of
    // a descriptor opened on this handle's path) is not the file this
    // read-only handle opened. A TableReader on this handle vets the
    // descriptor it pins for its raw reads with it.
    bool names_open_file(const struct stat& st) const;

    void ensure_hdu(int hdu_num, int* status);
    // Same move, but throws naming the query that needed it. Used by the
    // HDU-keyed cache lookups below, which read the CFITSIO current-HDU cursor:
    // their caches are keyed by hdu_num, so they must position the cursor
    // themselves rather than trust the caller to have done it first.
    void ensure_hdu_checked(int hdu_num, const char* what);

    const ScaleInfo& get_scale_info(int hdu_num, int bitpix);
    ScaleInfo get_scale_info_for_hdu(int hdu_num);
    bool is_compressed_image_cached(int hdu_num);
    const std::tuple<int, int, std::array<LONGLONG, 9>>& get_image_info(int hdu_num);

    native::Tensor read_tensor(int hdu_num, bool use_mmap = true);
    native::Tensor read_image_raw(int hdu_num, bool use_mmap = true);
    bool write_image(nb::ndarray<> tensor);
    std::vector<std::tuple<std::string, std::string, std::string>> get_header(int hdu_num);
    std::vector<long> get_shape(int hdu_num);
    int get_dtype(int hdu_num);
    native::Tensor read_subset(int hdu_num, long x1, long y1, long x2, long y2);
    int get_num_hdus();
    std::string get_hdu_type(int hdu_num);
    bool write_hdus(nb::list hdus, bool overwrite);
    bool write_hdus_compressed_images(nb::list hdus, int compression_type);
    std::string read_header_to_string(int hdu_num);

private:
    // Write paths: the open handle's owner, or throw "FITSFile is closed".
    // io_mutex_ is recursive and the writers run Python code with the GIL
    // (to_tensor(), header values, GC finalizers), so that code can close()
    // this handle on the writing thread itself. The writer holds the returned
    // owner (close() then only flushes and the fitsfile stays valid) and
    // calls this again after every point where Python could have run.
    std::shared_ptr<fitsfile> require_open_for_write() const {
        if (!fptr_ || !fptr_owner_) throw std::runtime_error("FITSFile is closed");
        return fptr_owner_;
    }
    bool ensure_raw_fd(size_t required_end);
    void close_raw_fd();
    // Identity of the file this handle opened, captured in the constructor.
    // A handle is allowed to outlive the file it opened (open_subset_reader and
    // a reused FITSFile both do), and the SharedReadMeta it publishes into is
    // shared with every *other* reader, whose handles describe the file as it
    // is now. Comparing the two is what keeps a stale handle from writing a
    // replaced file's header into the slot the replacement owns.
    void capture_open_identity();
    // True when this handle still describes the generation shared_meta_ names.
    // The caller holds shared_meta_->mutex.
    bool owns_current_generation_locked() const;
    void publish_image_info(
        int hdu_num, const std::tuple<int, int, std::array<LONGLONG, 9>>& info) const;
    void publish_scale(int hdu_num, const ScaleInfo& info) const;
    void publish_compressed(int hdu_num, bool compressed) const;
    std::string filename_;
    int mode_;
    fitsfile* fptr_ = nullptr;
    // Owns fptr_; its deleter closes the file. Shared with borrowers.
    std::shared_ptr<fitsfile> fptr_owner_;
    // Write handles: the deleter's fits_close_file status. CFITSIO flushes
    // its last buffered blocks there, so a full disk, quota, EIO or
    // RLIMIT_FSIZE failure shows up only in this status (close() throws).
    std::shared_ptr<int> write_close_status_;
    int start_hdu_ = 1;
    int current_hdu_ = 1;
    int raw_fd_ = -1;
    off_t raw_file_size_ = 0;
    bool raw_fd_ready_ = false;
    bool open_identity_valid_ = false;
    ino_t open_inode_ = 0;
    off_t open_size_ = 0;
    int64_t open_mtime_ns_ = 0;
    std::unordered_map<int, ScaleInfo> scale_cache_;
    std::unordered_map<int, bool> compressed_cache_;
    std::unordered_map<int, std::tuple<int, int, std::array<LONGLONG, 9>>> image_info_cache_;
    std::shared_ptr<detail::SharedReadMeta> shared_meta_;
    // CFITSIO exposes one mutable current-HDU cursor per fitsfile handle.
    // Persistent FITSFile objects are shared by Python threads, so every
    // public operation and close() must serialize over that cursor/cache.
    mutable std::recursive_mutex io_mutex_;
};

class SubsetReader {
public:
    SubsetReader(const std::string& filename, int hdu_num = 0);
    ~SubsetReader();
    native::Tensor read(long x1, long y1, long x2, long y2);
    void close();
    long width() const { return max_x_; }
    long height() const { return max_y_; }
    int hdu() const { return hdu_num_; }
private:
    void init_from_hdu();
    bool ensure_data_mmap();
    bool try_read_via_mmap(long x1, long y1, long x2, long y2, native::Tensor& out);
    void release_data_mmap();
    FITSFile file_;
    std::string filename_;
    int hdu_num_ = 0;
    long max_x_ = 0;
    long max_y_ = 0;
    int naxis_ = 0;
    std::vector<long> naxes_;
    native::ScalarType dtype_ = native::kFloat32;
    int datatype_ = TFLOAT;
    int bitpix_ = FLOAT_IMG;
    LONGLONG data_offset_ = 0;
    size_t elem_bytes_ = 0;
    bool raw_fast_ok_ = false;
    bool closed_ = false;
    // FITS integer conventions applied after mmap endian swap (0 = none).
    // signed-byte: XOR 0x80; uint16: +32768; uint32: +2147483648.
    enum class MmapConv : uint8_t { None, SignedByte, UInt16, UInt32 };
    MmapConv mmap_conv_ = MmapConv::None;
    // Uncompressed 2D mosaic fast path: map data once, slice+bswap per cutout.
    void* map_ptr_ = nullptr;
    size_t map_len_ = 0;
    off_t map_page_offset_ = 0;
    // Keep the metadata generation captured at construction. A lazy mmap must
    // never reacquire a descriptor for a replacement at the same path.
    std::shared_ptr<detail::SharedReadMeta> shared_meta_;
    const uint8_t* pixel_base_ = nullptr;
    // Refcounted fd keeps the backing file descriptor alive while mapped and
    // guards against invalidation closing it mid-mmap.
    std::shared_ptr<detail::RawFdHolder> raw_fd_holder_;
    // Serializes read()/close() when one persistent reader is shared across
    // Python threads (mirrors TableReader::io_mutex_): the CFITSIO cursor,
    // the lazy mmap init, and the closed_ flag are not thread-safe.
    mutable std::mutex io_mutex_;
};

} // namespace torchfits
