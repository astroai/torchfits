// TableReader: opening a table HDU, its TFORM/TTYPE column layout, lazy
// TSCAL/TZERO/TNULL resolution, and the checks that keep header-derived
// extents inside the file on disk.

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <unordered_set>
#include <fcntl.h>
#include <unistd.h>

#include "table_reader.h"
#include "checked_arithmetic.h"
#include "fits_detail.h"
#include "security.h"

namespace torchfits {

TableReader::TableReader(const std::string& filename, int hdu_num)
    : filename_(filename), hdu_num_(hdu_num), owns_fptr_(true) {
    torchfits::check_fits_filename_security(filename);
    target_hdu_ = hdu_num + 1;  // CFITSIO 1-based absolute HDU
    // Private per-instance handle (CFITSIO §4 Option A): never share one
    // fitsfile* across threads for reads.
    int status = torchfits::detail::open_fits_readonly(&fptr_, filename);
    if (status != 0 || !fptr_) {
        throw std::runtime_error("Failed to open FITS file");
    }
    // If HDU move or analysis fails, the destructor will not run (the object
    // is not fully constructed), so close the handle (and descriptor) we
    // opened before rethrowing to avoid leaking them.
    try {
        open_data_fd(nullptr);
        // Capture the file identity for stale-reader detection: a cached
        // reader may outlive a writer on another thread replacing the file
        // (thread-local caches can only be evicted by their own thread).
        // With a data fd, the identity is that of the file it pins.
        if (data_fd_ >= 0) {
            stat_valid_ = (::fstat(data_fd_, &file_stat_) == 0);
        } else {
            stat_valid_ = (::stat(filename_.c_str(), &file_stat_) == 0);
        }
        status = 0;
        fits_movabs_hdu(fptr_, target_hdu_, nullptr, &status);
        if (status != 0) {
            char err_msg[81];
            fits_get_errstatus(status, err_msg);
            throw std::runtime_error(
                "Invalid table geometry for HDU " + std::to_string(hdu_num_) +
                ": " + std::string(err_msg));
        }
        analyze_table();
    } catch (...) {
        close_data_fd();
        torchfits::detail::remember_hdu_offsets(fptr_, filename_);
        int close_status = 0;
        fits_close_file(fptr_, &close_status);
        fptr_ = nullptr;
        throw;
    }
}

TableReader::TableReader(fitsfile* fptr, int hdu_num, const FileIdentityCheck& same_file)
    : fptr_(fptr), hdu_num_(hdu_num), owns_fptr_(false) {
    int status = 0;
    target_hdu_ = hdu_num + 1;
    fits_movabs_hdu(fptr_, target_hdu_, nullptr, &status);
    if (status != 0) {
        char err_msg[81];
        fits_get_errstatus(status, err_msg);
        throw std::runtime_error(
            "Invalid table geometry for HDU " + std::to_string(hdu_num_) +
            ": " + std::string(err_msg));
    }
    // Several paths below (extent validation, buffered pread, VLA heap
    // pread) read the file on disk, not CFITSIO's buffers. A borrowed
    // handle may have just written this HDU, so flush first; this is a
    // no-op for a read-only handle.
    {
        int flush_status = 0;
        fits_flush_buffer(fptr_, 0, &flush_status);
    }
    // The name is resolved once, here: the data fd opened from it is what
    // every later raw read uses, so a persistent reader never re-resolves the
    // path per read.
    char fname[FLEN_FILENAME] = {0};
    int name_status = 0;
    fits_file_name(fptr_, fname, &name_status);
    if (name_status == 0 && fname[0] != '\0') {
        filename_ = fname;
    }
    open_data_fd(same_file ? &same_file : nullptr);
    try {
        analyze_table();
    } catch (...) {
        close_data_fd();
        throw;
    }
}

// Open the descriptor every raw read of this reader uses (buffered pread,
// mmap, filtered scans, the VLA heap pread, the physical-extent check). It is
// opened once, so the bytes those paths read come from the file this reader
// opened even if the path is later replaced; CFITSIO's own handle keeps that
// file open too, so both views stay consistent. Only plain local files get
// one (direct_io_ok); a non-regular file is left to CFITSIO. same_file, when
// given, must accept the descriptor's fstat -- a borrowed handle passes its
// own open identity, so a replace that landed between the handle's open and
// this one is refused instead of mixing two files.
void TableReader::open_data_fd(const FileIdentityCheck* same_file) {
    if (!direct_io_ok()) {
        return;
    }
    // O_NONBLOCK: a FIFO must not block the open; it is a no-op for the
    // regular files kept below.
    const int fd = ::open(filename_.c_str(), O_RDONLY | O_CLOEXEC | O_NONBLOCK);
    if (fd < 0) {
        return;
    }
    struct stat st {};
    if (::fstat(fd, &st) != 0 || !S_ISREG(st.st_mode)) {
        ::close(fd);
        return;
    }
    if (same_file != nullptr && !(*same_file)(st)) {
        ::close(fd);
        throw std::runtime_error(
            "FITS file was replaced on disk after this handle was opened; "
            "reopen it to read the current file");
    }
    data_fd_ = fd;
}

void TableReader::close_data_fd() noexcept {
    if (data_fd_ >= 0) {
        ::close(data_fd_);
        data_fd_ = -1;
    }
}

TableReader::~TableReader() {
    close_data_fd();
    if (fptr_ && owns_fptr_) {
        torchfits::detail::remember_hdu_offsets(fptr_, filename_);
        int status = 0;
        fits_close_file(fptr_, &status);
    }
}

// True if the file at filename_ still matches the identity captured at
// construction. Path-based readers cached in the thread-local LRU call
// this on acquire; a file replaced by another thread's writer (new inode,
// size or mtime) makes the cached handle and pread fd stale, so the cache
// must drop it and open fresh. Compares mtime at nanosecond resolution:
// second-granularity st_mtime misses in-place rewrites within the same
// second when size and inode are unchanged.
bool TableReader::file_unchanged() const {
    if (!stat_valid_) {
        return false;
    }
    struct stat st;
    if (::stat(filename_.c_str(), &st) != 0) {
        return false;
    }
    if (st.st_dev != file_stat_.st_dev || st.st_ino != file_stat_.st_ino ||
        st.st_size != file_stat_.st_size) {
        return false;
    }
#if defined(__APPLE__)
    return st.st_mtimespec.tv_sec == file_stat_.st_mtimespec.tv_sec &&
           st.st_mtimespec.tv_nsec == file_stat_.st_mtimespec.tv_nsec;
#else
    return st.st_mtim.tv_sec == file_stat_.st_mtim.tv_sec &&
           st.st_mtim.tv_nsec == file_stat_.st_mtim.tv_nsec;
#endif
}

void TableReader::analyze_table() {
    int status = 0;

    // Check HDU type
    int hdutype;
    fits_get_hdu_type(fptr_, &hdutype, &status);
    is_ascii_ = (hdutype == ASCII_TBL);

    // Get table dimensions
    fits_get_num_rows(fptr_, &nrows_, &status);
    fits_get_num_cols(fptr_, &ncols_, &status);

    if (status != 0) {
        throw std::runtime_error("Failed to get table dimensions");
    }

    // Analyze columns
    columns_.clear();
    columns_.reserve(ncols_);
    // Names already given to earlier columns: a repeated TTYPE is named
    // COL<i> like a missing one (fits_schema._reader_column_name), since
    // every result is keyed by name and the repeat used to be dropped.
    std::unordered_set<std::string> seen_names;

    for (int i = 1; i <= ncols_; i++) {
        ColumnInfo col;

        char ttype[FLEN_VALUE], tform[FLEN_VALUE], tunit[FLEN_VALUE];

        // Initialize arrays to avoid garbage values
        memset(ttype, 0, FLEN_VALUE);
        memset(tform, 0, FLEN_VALUE);
        memset(tunit, 0, FLEN_VALUE);

        // Get column name
        int col_status = 0;

        // We use fits_read_key to read TTYPEn.
        char keyname[FLEN_KEYWORD];
        snprintf(keyname, FLEN_KEYWORD, "TTYPE%d", i);
        fits_read_key(fptr_, TSTRING, keyname, ttype, nullptr, &col_status);

        if (col_status != 0 || ttype[0] == '\0' || seen_names.count(ttype) != 0) {
            // TTYPE is optional: a column without one (or with an empty
            // or repeated one) is named COL<i>, as
            // fits_schema._reader_column_name.
            col_status = 0; // Reset status
            snprintf(ttype, FLEN_VALUE, "COL%d", i);
        }
        seen_names.insert(ttype);

        snprintf(keyname, FLEN_KEYWORD, "TFORM%d", i);
        fits_read_key(fptr_, TSTRING, keyname, tform, nullptr, &col_status);
        if (col_status != 0) {
            col_status = 0;
            tform[0] = '\0';
        }

        int typecode;
        long repeat_long, width_long;
        fits_get_coltype(fptr_, i, &typecode, &repeat_long, &width_long, &col_status);

        if (col_status != 0) {
             char err_msg[81];
             fits_get_errstatus(col_status, err_msg);
             throw std::runtime_error(
                 "Invalid table geometry for column " + std::string(ttype) +
                 ": " + std::string(err_msg));
        }              // A repeat of 0 is legal in a TFORM and astropy reads such a column
          // as zero-width (shape (nrows, 0)). It must not reach the read
          // paths: col.repeat drives the element count, so a zero-repeat
          // column reads a cell count of 0 from a row that does contain
          // neighbouring columns' bytes, and torchfits then returns another
          // column's data under this column's name. extract_column_data
          // already rejects repeat <= 0; rejecting it here too makes the
          // two agree and fails at open time rather than mid-read.
          if (repeat_long <= 0 ||
              repeat_long > static_cast<long>(std::numeric_limits<int>::max())) {
              throw std::runtime_error(
                  std::string("TFORM repeat count must be positive for column ") +
                  ttype + " (got " + std::to_string(repeat_long) + ")");
          }
        col.repeat = static_cast<int>(repeat_long);
        col.name = std::string(ttype);
        col.width = 1;  // Will be set based on type
        col.fits_typecode = typecode;

        // Defer TSCAL/TZERO until a column is first read (ensure_column_scale).
        col.tscale = 1.0;
        col.tzero = 0.0;
        col.scaled = false;
        col.scale_resolved = false;
        col.is_unsigned_int = false;
        col.unsigned_offset = 0;

        #ifdef DEBUG_TABLE
        fprintf(stderr, "Column %d: name='%s', typecode=%d, repeat=%d\n", i, ttype, typecode, col.repeat);
        #endif

        if (typecode < 0) {
            // Variable length array
            col.type = FITSColumnType::VARIABLE;
            int abs_type = -typecode;
            switch (abs_type) {
                case TLOGICAL: col.torch_type = native::kBool; break;
                case TBYTE: col.torch_type = native::kUInt8; break;
                case TSBYTE: col.torch_type = native::kInt8; break;
                case TSHORT: col.torch_type = native::kInt16; break;
                case TINT: col.torch_type = native::kInt32; break;
                case TLONG: col.torch_type = native::kInt32; break;
                case TLONGLONG: col.torch_type = native::kInt64; break;
                case TFLOAT: col.torch_type = native::kFloat32; break;
                case TDOUBLE: col.torch_type = native::kFloat64; break;
                // 'PA': raw bytes, like fixed-width string columns.
                case TSTRING: col.torch_type = native::kUInt8; break;
                // 'PC'/'PM'/'PX' are rejected when read (see
                // read_vla_column_flat); record the true element type
                // instead of the old silent float32 fallback.
                case TCOMPLEX: col.torch_type = native::kComplexFloat; break;
                case TDBLCOMPLEX: col.torch_type = native::kComplexDouble; break;
                default: col.torch_type = native::kFloat32;
            }
            col.width = 8;
            // P descriptors occupy 8 bytes and Q descriptors occupy 16 bytes,
            // regardless of the maximum element repeat reported by CFITSIO.
            const std::string format(tform);
            col.storage_bytes =
                (format.find('Q') != std::string::npos ||
                 format.find('q') != std::string::npos)
                ? 16
                : 8;
        } else {
            switch (typecode) {
                case TLOGICAL:
                    col.type = FITSColumnType::LOGICAL;
                    col.torch_type = native::kBool;
                    col.width = 1;
                    break;
                case TBYTE:
                    col.type = FITSColumnType::BYTE;
                    col.torch_type = native::kUInt8;
                    col.width = 1;
                    break;
                case TSBYTE:
                    // Signed bytes must stay signed: kUInt8 would turn
                    // 128..255 into -128..-1 on every read path.
                    col.type = FITSColumnType::BYTE;
                    col.torch_type = native::kInt8;
                    col.width = 1;
                    break;
                case TBIT:
                    col.type = FITSColumnType::BIT;
                    col.torch_type = native::kUInt8;
                    // Expose bit arrays as uint8[repeat] values (0/1).
                    col.width = 1;
                    break;
                case TSHORT:
                case TUSHORT:
                    col.type = FITSColumnType::SHORT;
                    col.torch_type = native::kInt16;
                    col.width = 2;
                    break;
                case TINT:
                case TUINT:
                    col.type = FITSColumnType::INT;
                    col.torch_type = native::kInt32;
                    col.width = 4;
                    break;
                case TSTRING:
                    col.type = FITSColumnType::STRING;
                    col.torch_type = native::kUInt8;
                    if (is_ascii_) {
                         col.repeat = 1; // One string per row
                         // width_long derives from the TFORM/field width and
                         // must fit int: truncation would corrupt cell sizes
                         // and offsets for absurd TFORM widths (same class as
                         // the repeat guard below).
                         if (width_long < 0 || width_long > 0x7fffffffL) {
                             throw std::runtime_error(
                                 "Column width exceeds supported range for column " + std::string(ttype));
                         }
                         col.width = (int)width_long;
                    } else {
                         // Binary table
                         col.width = 1;
                         // For binary tables, repeat_long is often the string length,
                         // but some FITS writers may populate width_long instead.
                         if (repeat_long > 1) {
        // repeat_long comes from TFORMn and must fit int: truncation would
        // corrupt cell sizes/offsets for absurd TFORM repeats.
        if (repeat_long > 0x7fffffffL) {
            throw std::runtime_error(
                "Column repeat exceeds supported range for column " + std::string(ttype));
        }
        col.repeat = (int)repeat_long;
                         } else if (width_long > 0) {
                             if (width_long > 0x7fffffffL) {
                                 throw std::runtime_error(
                                     "Column width exceeds supported range for column " + std::string(ttype));
                             }
                             col.repeat = (int)width_long;
                         }
                    }
                    break;
                case TLONG:
                    // CFITSIO reports TLONG as FITS 32-bit integer (same code as TINT32BIT).
                    col.type = FITSColumnType::INT;
                    col.torch_type = native::kInt32;
                    col.width = 4;
                    break;
                case TULONG:
                    if (sizeof(long) == 8) {
                        col.type = FITSColumnType::LONG;
                        col.torch_type = native::kInt64;
                        col.width = 8;
                    } else {
                        col.type = FITSColumnType::INT;
                        col.torch_type = native::kInt32;
                        col.width = 4;
                    }
                    break;
                case TLONGLONG:
                    col.type = FITSColumnType::LONG;
                    col.torch_type = native::kInt64;
                    col.width = 8;
                    break;
                case TFLOAT:
                    col.type = FITSColumnType::FLOAT;
                    col.torch_type = native::kFloat32;
                    col.width = 4;
                    break;
                case TDOUBLE:
                    col.type = FITSColumnType::DOUBLE;
                    col.torch_type = native::kFloat64;
                    col.width = 8;
                    break;
#ifdef TCOMPLEX
                case TCOMPLEX:
                    col.type = FITSColumnType::COMPLEX_FLOAT;
                    col.torch_type = native::kComplexFloat;
                    col.width = 8; // two float32 values
                    break;
#endif
#ifdef TDBLCOMPLEX
                case TDBLCOMPLEX:
                    col.type = FITSColumnType::COMPLEX_DOUBLE;
                    col.torch_type = native::kComplexDouble;
                    col.width = 16; // two float64 values
                    break;
#endif
                default:
                    throw std::runtime_error(
                        "Unsupported FITS column typecode " + std::to_string(typecode) +
                        " for column " + std::string(ttype)
                    );
            }
            const std::string geometry =
                "Invalid table geometry for column " + col.name;
            if (is_ascii_) {
                col.storage_bytes = internal::checked_nonnegative_cast<long>(
                    width_long, geometry + " width");
            } else if (col.type == FITSColumnType::BIT) {
                col.storage_bytes = internal::checked_nonnegative_add<long>(
                    static_cast<long>(col.repeat), 7L, geometry + " storage") / 8L;
            } else {
                col.storage_bytes = internal::checked_nonnegative_multiply<long>(
                    static_cast<long>(col.width), static_cast<long>(col.repeat),
                    geometry + " storage");
            }
            if (col.width <= 0 || col.repeat <= 0 || col.storage_bytes <= 0) {
                throw std::runtime_error(geometry + ": non-positive width or repeat");
            }
        }


        columns_.push_back(col);
    }

    #ifdef DEBUG_TABLE
    fprintf(stderr, "analyze_table complete: found %zu columns out of %d expected\n", columns_.size(), ncols_);
    #endif

    // Verify we have the expected number of columns
    if ((int)columns_.size() != ncols_) {
        // This is a critical error - the table metadata is inconsistent
        throw std::runtime_error("Column count mismatch: expected " + std::to_string(ncols_) +
                                ", found " + std::to_string(columns_.size()));
    }

    // Calculate offsets
    long current_offset = 0;
    for (size_t i = 0; i < columns_.size(); ++i) {
        auto& col = columns_[i];
        if (is_ascii_) {
            long tbcol = 0;
            int tbcol_status = 0;
            char tbcol_key[FLEN_KEYWORD];
            snprintf(tbcol_key, FLEN_KEYWORD, "TBCOL%zu", i + 1);
            fits_read_key(
                fptr_, TLONG, tbcol_key, &tbcol, nullptr, &tbcol_status);
            if (tbcol_status != 0 || tbcol <= 0) {
                throw std::runtime_error(
                    "Invalid table geometry for column " + col.name + ": TBCOL");
            }
            col.byte_offset = tbcol - 1;
            current_offset = std::max(
                current_offset,
                internal::checked_nonnegative_add<long>(
                    col.byte_offset, col.storage_bytes,
                    "Invalid table geometry for column " + col.name + " byte offset"));
        } else {
            col.byte_offset = current_offset;
            current_offset = internal::checked_nonnegative_add<long>(
                current_offset, col.storage_bytes,
                "Invalid table geometry for column " + col.name + " byte offset");
        }
    }
    long declared_row_width = 0;
    int row_width_status = 0;
    fits_read_key(
        fptr_,
        TLONG,
        const_cast<char*>("NAXIS1"),
        &declared_row_width,
        nullptr,
        &row_width_status
    );
    if (row_width_status != 0 || declared_row_width < 0 ||
        (declared_row_width == 0 && ncols_ != 0)) {
        throw std::runtime_error("Invalid table geometry: NAXIS1");
    }
    if (current_offset > declared_row_width) {
        throw std::runtime_error(
            "Invalid table geometry: aggregate column storage " +
            std::to_string(current_offset) + " exceeds NAXIS1 " +
            std::to_string(declared_row_width));
    }
    row_width_bytes_ = declared_row_width;
    validate_physical_table_extent();
}

void TableReader::ensure_column_scale(int col_idx) {
    auto& col = columns_[col_idx];
    if (col.scale_resolved) {
        return;
    }
    col.scale_resolved = true;
    int scale_status = 0;
    char scale_key[FLEN_KEYWORD];
    const int i = col_idx + 1;
    snprintf(scale_key, FLEN_KEYWORD, "TSCAL%d", i);
    fits_read_key(fptr_, TDOUBLE, scale_key, &col.tscale, nullptr, &scale_status);
    const bool has_tscal = (scale_status == 0);
    if (scale_status != 0) {
        scale_status = 0;
        col.tscale = 1.0;
    }
    snprintf(scale_key, FLEN_KEYWORD, "TZERO%d", i);
    fits_read_key(fptr_, TDOUBLE, scale_key, &col.tzero, nullptr, &scale_status);
    const bool has_tzero = (scale_status == 0);
    if (scale_status != 0) {
        scale_status = 0;
        col.tzero = 0.0;
    }
    snprintf(scale_key, FLEN_KEYWORD, "TNULL%d", i);
    long long tnull_val = 0;
    int tnull_status = 0;
    fits_read_key(fptr_, TLONGLONG, scale_key, &tnull_val, nullptr, &tnull_status);
    if (tnull_status == 0) {
        col.has_tnull = true;
        col.tnull = tnull_val;
    }
    col.scaled = (col.tscale != 1.0 || col.tzero != 0.0);
    const int typecode = col.fits_typecode;
    if (col.tscale == 1.0) {
        if ((typecode == TSHORT || typecode == TUSHORT) && detail::is_unsigned_short_offset(col.tzero)) {
            col.is_unsigned_int = true;
            col.unsigned_offset = 32768;
            col.unsigned_target_type = native::kUInt16;
            col.scaled = false;
        } else if (
            // CFITSIO reports FITS `J` as TLONG/TINT32BIT (41), not TINT (31).
            (typecode == TINT || typecode == TUINT || typecode == TLONG ||
             typecode == TINT32BIT) &&
            detail::is_unsigned_long_offset(col.tzero)) {
            col.is_unsigned_int = true;
            col.unsigned_offset = 2147483648;
            col.unsigned_target_type = native::kUInt32;
            col.scaled = false;
        } else if (typecode == TBYTE && col.tzero == -128.0) {
            // Signed-byte convention (TFORM B, TZERO = -128): storage is
            // value XOR 0x80, read back as int8 like the image path.
            col.is_unsigned_int = true;
            col.unsigned_offset = -128;
            col.unsigned_target_type = native::kInt8;
            col.scaled = false;
        } else if (typecode == TLONGLONG && detail::is_unsigned_longlong_offset(col.tzero)) {
            // uint64 convention (TZERO = 2^63). The offset does not fit
            // int64, so unsigned_offset stays 0 and every consumer keys
            // on unsigned_target_type == kUInt64 (sign-bit flip).
            col.is_unsigned_int = true;
            col.unsigned_offset = 0;
            col.unsigned_target_type = native::kUInt64;
            col.scaled = false;
        }
    }
    // Quantize writes TSCAL/TZERO even when they are identity; TNULL on
    // that path must become NaN after the linear map (same as BLANK images).
    if (col.has_tnull && (has_tscal || has_tzero) && !col.is_unsigned_int) {
        col.scaled = true;
    }
}

// A truncated file must raise a clean error instead of SIGBUS on first
// touch: header-derived extents are validated against the real size.
void TableReader::ensure_extent_within_file(off_t file_size, LONGLONG data_offset,
                                            long long rows_before, long long nrows) const {
    const LONGLONG row_bytes = static_cast<LONGLONG>(row_width_bytes_);
    const LONGLONG needed = data_offset
        + static_cast<LONGLONG>(rows_before) * row_bytes
        + static_cast<LONGLONG>(nrows) * row_bytes;
    if (static_cast<LONGLONG>(file_size) < needed) {
        throw std::runtime_error(
            "FITS table is truncated: header claims table bytes through "
            "offset " + std::to_string(static_cast<long long>(needed))
            + " but the file holds only "
            + std::to_string(static_cast<long long>(file_size))
            + "; the copy is incomplete or corrupt");
    }
}

void TableReader::validate_physical_table_extent() const {
    if (!direct_io_ok()) {
        return;
    }
    // The size of the file the raw paths will read (open_data_fd keeps
    // regular files only); without a descriptor nothing reads the file raw.
    struct stat st;
    if (data_fd_ < 0 || ::fstat(data_fd_, &st) != 0) {
        return;
    }
    LONGLONG data_offset = 0;
    LONGLONG data_end = 0;
    int status = 0;
    fits_get_hduaddrll(fptr_, nullptr, &data_offset, &data_end, &status);
    if (status != 0 || data_offset < 0 || data_end < data_offset) {
        throw std::runtime_error("Invalid table geometry: HDU extent");
    }
    LONGLONG pcount = 0;
    status = 0;
    fits_read_key(
        fptr_, TLONGLONG, const_cast<char*>("PCOUNT"), &pcount, nullptr, &status);
    if (status != 0 || pcount < 0) {
        throw std::runtime_error("Invalid table geometry: PCOUNT");
    }
    const LONGLONG table_bytes =
        internal::checked_nonnegative_multiply<LONGLONG>(
            internal::checked_nonnegative_cast<LONGLONG>(
                row_width_bytes_, "Invalid table geometry: row width"),
            internal::checked_nonnegative_cast<LONGLONG>(
                nrows_, "Invalid table geometry: row count"),
            "Invalid table geometry: table extent");
    const LONGLONG required = internal::checked_nonnegative_add<LONGLONG>(
        data_offset,
        internal::checked_nonnegative_add<LONGLONG>(
            table_bytes, pcount, "Invalid table geometry: HDU data extent"),
        "Invalid table geometry: HDU data extent");
    const LONGLONG file_size = internal::checked_nonnegative_cast<LONGLONG>(
        st.st_size, "Invalid table geometry: physical file size");
    if (required > file_size) {
        throw std::runtime_error(
            "Invalid table geometry: declared HDU data extent " +
            std::to_string(required) + " exceeds physical file size " +
            std::to_string(file_size));
    }
}

std::vector<std::string> TableReader::get_column_names() const {
    std::vector<std::string> names;
    for (const auto& col : columns_) {
        names.push_back(col.name);
    }
    return names;
}

// True when raw fd/pread/mmap access to filename_ is byte-compatible
// with the FITS layout: plain local, uncompressed files only. Whole-file
// compressed inputs (.bz2) decompress inside CFITSIO, so their on-disk
// size/layout do NOT match header-derived offsets — direct-I/O paths
// must route through CFITSIO instead (native .bz2 support).
bool TableReader::direct_io_ok() const {
    if (filename_.empty()) return false;
    auto ends_with_ci = [](const std::string& s, const char* suf) {
        const size_t n = std::strlen(suf);
        if (s.size() < n) return false;
        for (size_t i = 0; i < n; ++i) {
            const unsigned char c =
                static_cast<unsigned char>(s[s.size() - n + i]);
            if (std::tolower(c) != static_cast<unsigned char>(suf[i])) {
                return false;
            }
        }
        return true;
    };
    if (ends_with_ci(filename_, ".bz2") || ends_with_ci(filename_, ".gz") ||
        ends_with_ci(filename_, ".zip")) {
        return false;
    }
    if (filename_.size() >= 2 && filename_[filename_.size() - 2] == '.' &&
        (filename_.back() == 'Z' || filename_.back() == 'z')) {
        return false;
    }
    if (filename_.find("://") != std::string::npos) return false;
    return !has_cfitsio_extended_filename_syntax(filename_);
}

// Move to the target table HDU when the handle is shared via the cache.
void TableReader::ensure_table_hdu() {
    if (!fptr_) return;
    int cur = 0;
    fits_get_hdu_num(fptr_, &cur);
    if (cur != target_hdu_) {
        int status = 0;
        fits_movabs_hdu(fptr_, target_hdu_, nullptr, &status);
        if (status != 0) {
            throw std::runtime_error("Failed to move to table HDU");
        }
    }
}

}  // namespace torchfits
