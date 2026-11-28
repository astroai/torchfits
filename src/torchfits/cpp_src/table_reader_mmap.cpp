// TableReader mmap paths: whole-window reads from a read-only mapping and
// in-place cell updates through a writable one.

#include <algorithm>
#include <cerrno>
#include <complex>
#include <cstring>
#include <stdexcept>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>

#include "table_reader.h"
#include "fits_detail.h"
#include "hardware.h"
#include "table_gather.h"

namespace torchfits {

namespace {

// The dtype policy update_rows_mmap accepts, in one place so the up-front
// validation and the write dispatch cannot disagree: each case below mirrors
// the guard in the matching arm of the write loop.
bool mmap_update_dtype_accepted(
    const ColumnInfo& col, const nb::dlpack::dtype& dt) {
    using code = nb::dlpack::dtype_code;
    const auto is = [&](code c, int bits) {
        return dt.code == static_cast<uint8_t>(c) && dt.bits == static_cast<uint8_t>(bits);
    };
    // A column that has no arm in the write loop's switch cannot be written
    // at all; rejecting it here turns what would be a late "unsupported
    // column type" throw into the same up-front decision as every other
    // rejection.
    switch (col.type) {
        case FITSColumnType::BYTE:
            if (col.is_unsigned_int && col.unsigned_target_type == native::kInt8) {
                return is(code::Int, 8);
            }
            return col.fits_typecode == TSBYTE ? is(code::Int, 8) : is(code::UInt, 8);
        case FITSColumnType::LOGICAL:
        case FITSColumnType::BIT:
            return is(code::Bool, 8) || is(code::UInt, 8);
        // Unsigned-convention columns also take their physical unsigned
        // dtype (the write loop flips the sign bit).
        case FITSColumnType::SHORT:
            return is(code::Int, 16) || (col.is_unsigned_int && is(code::UInt, 16));
        case FITSColumnType::INT:
            return is(code::Int, 32) || (col.is_unsigned_int && is(code::UInt, 32));
        case FITSColumnType::LONG:
            return is(code::Int, 64) || (col.is_unsigned_int && is(code::UInt, 64));
        case FITSColumnType::FLOAT: return is(code::Float, 32);
        case FITSColumnType::DOUBLE: return is(code::Float, 64);
        case FITSColumnType::STRING: return is(code::UInt, 8);
        case FITSColumnType::COMPLEX_FLOAT: return is(code::Complex, 64);
        case FITSColumnType::COMPLEX_DOUBLE: return is(code::Complex, 128);
        default: return false;
    }
}

}  // namespace

// Memory-mapped column reading
// Returns a dict of column name to native::Tensor (or numpy array for strings)
// in file/request column order (see read_columns signature comment).
std::vector<std::pair<std::string, native::Tensor>> TableReader::read_columns_mmap(
    const std::vector<std::string>& column_names,
    long start_row, long num_rows) {

    ensure_table_hdu();
    if (num_rows == -1) {
        num_rows = nrows_;
    }

    // Resolve requested columns first so unknown names fail loudly even
    // for empty tables, and a repeated name resolves once (the result
    // assembly moves each entry out; a duplicated index would hand back a
    // moved-from tensor).
    std::vector<int> col_indices;
    if (column_names.empty()) {
        for (int i = 0; i < ncols_; i++) col_indices.push_back(i);
    } else {
        for (const auto& name : column_names) {
            bool found = false;
            for (int i = 0; i < ncols_; i++) {
                if (columns_[i].name == name) {
                    if (std::find(col_indices.begin(), col_indices.end(), i) == col_indices.end()) {
                        col_indices.push_back(i);
                    }
                    found = true;
                    break;
                }
            }
            if (!found) throw std::runtime_error("Column not found: " + name);
        }
    }

    if (nrows_ == 0 || num_rows == 0) {
        // Typed empty columns come from the CFITSIO reader (same dtype
        // post-processing as a non-empty read; no bytes to map).
        auto rows = read_columns(column_names, start_row, 0, false);
        std::vector<std::pair<std::string, native::Tensor>> out;
        out.reserve(rows.size());
        for (auto& kv : rows) {
            if (kv.second.is_vla) {
                throw std::runtime_error("VLA columns not supported for mmap");
            }
            out.emplace_back(kv.first, std::move(kv.second.fixed_data));
        }
        return out;
    }
    if (num_rows < 0) {
        throw std::runtime_error("Invalid table row count: " + std::to_string(num_rows));
    }

    // Validate rows
    if (start_row < 1 || start_row > nrows_) {
        throw std::runtime_error(
            "Invalid start row: " + std::to_string(start_row) +
            ", nrows: " + std::to_string(nrows_));
    }
    // Reordered comparison: start_row + num_rows - 1 can overflow long for
    // adversarial inputs; nrows_ - start_row + 1 cannot (start_row in range).
    if (num_rows > nrows_ - start_row + 1) {
        num_rows = nrows_ - start_row + 1;
    }

    for (int col_idx : col_indices) {
        ensure_column_scale(col_idx);
        const auto& col = columns_[col_idx];
        if (is_ascii_) {
            // ASCII tables store numeric fields as text; the mmap layout
            // (binary cells, byte-swap) does not apply. Route to the
            // buffered path, which uses fits_read_col and handles ASCII.
            throw std::runtime_error("ASCII tables are not supported for mmap reads");
        }
        if (col.type == FITSColumnType::VARIABLE) {
            throw std::runtime_error("VLA columns not supported for mmap");
        }
        if (col.type == FITSColumnType::BIT) {
            // BIT columns are read as uint8 from mmap; converted to bool below.
        }
        if (col.scaled && !col.is_unsigned_int) {
            throw std::runtime_error("Scaled columns not supported for mmap");
        }
    }

    // Get offset to the start of the table data
    LONGLONG headstart, data_offset, dataend;
    int status = 0;
    fits_get_hduaddrll(fptr_, &headstart, &data_offset, &dataend, &status);
    if (status != 0) {
         char err_msg[81];
         fits_get_errstatus(status, err_msg);
         throw std::runtime_error("Failed to get HDU data offset: " + std::string(err_msg));
    }

    // Whole-file compressed inputs (.bz2): no direct byte layout on
    // disk matches the header extents, so decode through CFITSIO and
    // return the same fixed-column tensor map the mmap path produces.
    if (!direct_io_ok()) {
        auto rows = read_columns(column_names, start_row, num_rows, false);
        std::vector<std::pair<std::string, native::Tensor>> out;
        out.reserve(rows.size());
        for (auto& kv : rows) {
            out.emplace_back(kv.first, std::move(kv.second.fixed_data));
        }
        return out;
    }

    // Map the reader's own descriptor (open_data_fd), never the path: a
    // persistent reader must not map a file that replaced the one its
    // geometry came from.
    const int fd = data_fd_;
    if (fd == -1) {
        throw std::runtime_error("Failed to open file for mmap");
    }

    // Get file size
    struct stat sb;
    if (fstat(fd, &sb) == -1) {
        throw std::runtime_error("Failed to stat file");
    }
    ensure_extent_within_file(sb.st_size, data_offset, start_row - 1, num_rows);

    // Map the whole file
    void* map_ptr = mmap(nullptr, sb.st_size, PROT_READ, MAP_SHARED, fd, 0);
    if (map_ptr == MAP_FAILED) {
        throw std::runtime_error("Failed to mmap file");
    }

    // RAII mmap guard — tensors are copies (not views) so the mmap only
    // needs to survive for the duration of this function. It unmaps only;
    // the descriptor belongs to the reader.
    // size must match the original mmap() length (full file) passed to munmap().
    MMapHandle mmap_guard(map_ptr, sb.st_size, -1);

    std::unordered_map<int, native::Tensor> result;
    const uint8_t* base_ptr = static_cast<const uint8_t*>(map_ptr) + data_offset;

    // Calculate start offset based on start_row (0-based offset).
    // Validated against the file size by ensure_extent_within_file above.
    size_t row_start_offset = static_cast<size_t>(start_row - 1) * static_cast<size_t>(row_width_bytes_);

#if defined(POSIX_MADV_SEQUENTIAL)
    // Hint only: saturate instead of wrapping on absurd headers.
    const uint64_t hint_len = static_cast<uint64_t>(num_rows) * static_cast<uint64_t>(row_width_bytes_);
    const size_t byte_len = hint_len > SIZE_MAX ? SIZE_MAX : static_cast<size_t>(hint_len);
    posix_madvise(const_cast<uint8_t*>(base_ptr + row_start_offset), byte_len, POSIX_MADV_SEQUENTIAL);
#endif

    // One output per fixed-width column, in request order. Their dtypes:
    // BIT unpacks to bool here (read_columns reads it as uint8 and coerces
    // after scaling); everything else is the column's own dtype.
    auto mmap_dtype = [](const ColumnInfo& col) -> native::ScalarType {
        switch (col.type) {
            case FITSColumnType::FLOAT: return native::kFloat32;
            case FITSColumnType::DOUBLE: return native::kFloat64;
            case FITSColumnType::INT: return native::kInt32;
            case FITSColumnType::SHORT: return native::kInt16;
            case FITSColumnType::LONG: return native::kInt64;
            case FITSColumnType::BYTE: return col.torch_type;  // kUInt8, or kInt8 for TSBYTE
            case FITSColumnType::BIT: return native::kBool;
            case FITSColumnType::LOGICAL: return native::kBool;
            case FITSColumnType::STRING: return native::kUInt8;
            case FITSColumnType::COMPLEX_FLOAT: return native::kComplexFloat;
            case FITSColumnType::COMPLEX_DOUBLE: return native::kComplexDouble;
            default: return native::kFloat32;
        }
    };
    struct MmapCopy {
        int col_idx;
        const ColumnInfo* col;
        const uint8_t* col_ptr;  // the column's cell in the window's first row
        long repeat;
        bool gathered;           // copied by table_gather (else BIT / complex below)
        table_gather::ColumnGather gather;
        native::Tensor tensor;
    };
    std::vector<MmapCopy> copies;
    std::vector<std::pair<std::vector<int64_t>, native::ScalarType>> specs;
    copies.reserve(col_indices.size());
    specs.reserve(col_indices.size());
    for (int col_idx : col_indices) {
        const auto& col = columns_[col_idx];
        if (col.type == FITSColumnType::VARIABLE) {
            continue;
        }
        // Determine shape
        std::vector<int64_t> shape;
        shape.push_back(num_rows);
        if (col.type == FITSColumnType::STRING) {
            // For strings, we return a ByteTensor of shape (num_rows, width)
            // width is the string length (col.width)
            shape.push_back(is_ascii_ ? col.width : col.repeat);
        } else if (col.repeat > 1) {
            shape.push_back(col.repeat);
        }
        MmapCopy copy;
        copy.col_idx = col_idx;
        copy.col = &col;
        // Pointer to start of column data for the first requested row
        copy.col_ptr = base_ptr + row_start_offset + col.byte_offset;
        copy.repeat = (col.repeat > 1) ? col.repeat : 1;
        if (col.type == FITSColumnType::STRING) {
            copy.repeat = is_ascii_ ? col.width : col.repeat; // String length
        }
        copy.gathered = col.type != FITSColumnType::BIT &&
                        col.type != FITSColumnType::COMPLEX_FLOAT &&
                        col.type != FITSColumnType::COMPLEX_DOUBLE;
        if (copy.gathered) {
            try {
                copy.gather = table_gather::plan_column_gather(col, row_width_bytes_);
            } catch (const std::exception& e) {
                throw std::runtime_error("Failed to mmap column " + col.name + ": " + e.what());
            }
        }
        copies.push_back(std::move(copy));
        specs.emplace_back(std::move(shape), mmap_dtype(col));
    }
    {
        std::vector<native::Tensor> tensors;
        try {
            tensors = table_gather::allocate_outputs(specs);
        } catch (const std::exception& e) {
            throw std::runtime_error(
                "Failed to mmap column " + copies.front().col->name + ": " + e.what());
        }
        for (size_t k = 0; k < copies.size(); ++k) {
            copies[k].tensor = std::move(tensors[k]);
        }
    }

    // Copy and byte-swap every column one block of rows at a time: the
    // block's mapped pages are read once for all columns (a pass per column
    // re-read the whole window), and the blocks spread across the pool.
    const uint8_t* window = base_ptr + row_start_offset;
    const long rows_per_block = row_width_bytes_ > 0
        ? std::max<long>(
              2048, static_cast<long>((size_t{64} << 10) / static_cast<size_t>(row_width_bytes_)))
        : num_rows;
    native::parallel_for(0, copies.empty() ? 0 : num_rows, rows_per_block, [&](long start, long end) {
        for (const MmapCopy& copy : copies) {
            const ColumnInfo& col = *copy.col;
            if (copy.gathered) {
                table_gather::run_column_gather(
                    copy.gather, window, start, end,
                    static_cast<uint8_t*>(copy.tensor.data_ptr()));
                continue;
            }
            const uint8_t* col_ptr = copy.col_ptr;
            const long repeat = copy.repeat;
            if (col.type == FITSColumnType::BIT) {
                bool* out = static_cast<bool*>(copy.tensor.data_ptr());
                for (long i = start; i < end; i++) {
                    const uint8_t* row_in =
                        col_ptr + i * row_width_bytes_;
                    bool* row_out = out + i * repeat;
                    for (long j = 0; j < repeat; j++) {
                        const uint8_t packed = row_in[j / 8];
                        row_out[j] =
                            ((packed >> (7 - (j % 8))) & 0x1U) != 0;
                    }
                }
            } else if (col.type == FITSColumnType::COMPLEX_FLOAT) {
                auto* out = static_cast<std::complex<float>*>(copy.tensor.data_ptr());
                for (long i = start; i < end; i++) {
                    const uint8_t* row_in =
                        col_ptr + i * row_width_bytes_;
                    std::complex<float>* row_out = out + i * repeat;
                    for (long j = 0; j < repeat; j++) {
                        int32_t re_bits, im_bits;
                        std::memcpy(
                            &re_bits,
                            row_in + j * 2 * sizeof(int32_t),
                            sizeof(int32_t)
                        );
                        std::memcpy(
                            &im_bits,
                            row_in + (j * 2 + 1) * sizeof(int32_t),
                            sizeof(int32_t)
                        );
                        float re, im;
                        re_bits = bswap_32(re_bits);
                        im_bits = bswap_32(im_bits);
                        std::memcpy(&re, &re_bits, sizeof(float));
                        std::memcpy(&im, &im_bits, sizeof(float));
                        row_out[j] = std::complex<float>(re, im);
                    }
                }
            } else if (col.type == FITSColumnType::COMPLEX_DOUBLE) {
                auto* out = static_cast<std::complex<double>*>(copy.tensor.data_ptr());
                for (long i = start; i < end; i++) {
                    const uint8_t* row_in =
                        col_ptr + i * row_width_bytes_;
                    std::complex<double>* row_out = out + i * repeat;
                    for (long j = 0; j < repeat; j++) {
                        int64_t re_bits, im_bits;
                        std::memcpy(
                            &re_bits,
                            row_in + j * 2 * sizeof(int64_t),
                            sizeof(int64_t)
                        );
                        std::memcpy(
                            &im_bits,
                            row_in + (j * 2 + 1) * sizeof(int64_t),
                            sizeof(int64_t)
                        );
                        double re, im;
                        re_bits = bswap_64(re_bits);
                        im_bits = bswap_64(im_bits);
                        std::memcpy(&re, &re_bits, sizeof(double));
                        std::memcpy(&im, &im_bits, sizeof(double));
                        row_out[j] = std::complex<double>(re, im);
                    }
                }
            }
        }
    });

    for (MmapCopy& copy : copies) {
        const ColumnInfo& col = *copy.col;
        native::Tensor tensor = std::move(copy.tensor);
        try {
            if (col.scaled &&
                (col.type == FITSColumnType::COMPLEX_FLOAT ||
                 col.type == FITSColumnType::COMPLEX_DOUBLE)) {
                scale_complex_inplace(tensor, col.tscale, col.tzero);
            }
            // Apply unsigned-int coercion before storing.
            if (col.is_unsigned_int) {
                tensor = apply_unsigned_offset(
                    std::move(tensor), col.unsigned_target_type);
            }
        } catch (const std::exception& e) {
            // Never swallow: a failed column must surface as an error, not
            // as a silently missing column in the result dict.
            throw std::runtime_error("Failed to mmap column " + col.name + ": " + e.what());
        }
        result[copy.col_idx] = std::move(tensor);
    }

    // Order by file/request column index (see signature comment).
    std::vector<std::pair<std::string, native::Tensor>> ordered;
    ordered.reserve(col_indices.size());
    for (int col_idx : col_indices) {
        auto it = result.find(col_idx);
        if (it != result.end()) {
            ordered.emplace_back(columns_[col_idx].name, std::move(it->second));
        }
    }
    return ordered;
}

void TableReader::update_rows_mmap(nb::dict tensor_dict, long start_row, long num_rows) {
    if (num_rows == -1) {
        num_rows = nrows_ - start_row + 1;
    }
    if (num_rows <= 0) {
        return;
    }
    ensure_table_hdu();
    if (is_ascii_) {
        // Writing binary cells into ASCII text fields would corrupt the
        // table; update_rows (fits_write_col) handles ASCII tables.
        throw std::runtime_error("ASCII tables are not supported for mmap updates");
    }
    if (start_row < 1 || start_row > nrows_) {
        throw std::runtime_error("Invalid start row");
    }
    // Reordered comparison (cf. read_columns): start_row + num_rows - 1
    // can overflow long for adversarial inputs.
    if (num_rows > nrows_ - start_row + 1) {
        throw std::runtime_error("Row range exceeds table length");
    }

    // Build column index map. Duplicate TTYPE cards resolve to the first
    // match — the same resolution the read paths use for name lookups — so
    // an update can never land on a different column than a read of the
    // same name returns.
    std::unordered_map<std::string, const ColumnInfo*> column_map;
    column_map.reserve(columns_.size());
    for (const auto& col : columns_) {
        column_map.emplace(col.name, &col);
    }

    // Validate columns and types
    for (auto item : tensor_dict) {
        std::string name = nb::cast<std::string>(item.first);
        auto it = column_map.find(name);
        if (it == column_map.end()) {
            throw std::runtime_error("Column not found: " + name);
        }
        const ColumnInfo* col = it->second;
        // Resolve TSCAL/TZERO first: on a fresh reader the scale flags are
        // lazily populated, so the scaled guard below would be a no-op and
        // let a scaled column take the raw-bit write path — silently
        // storing physical values as raw cells.
        ensure_column_scale(static_cast<int>(col - columns_.data()));
        if (col->type == FITSColumnType::VARIABLE) {
            throw std::runtime_error("VLA columns not supported for mmap updates");
        }
        if (col->scaled) {
            throw std::runtime_error("Scaled columns not supported for mmap updates");
        }
        // BIT, STRING, COMPLEX_FLOAT, COMPLEX_DOUBLE, and LOGICAL are
        // accepted and handled by the per-element dispatches below. Other
        // column types fall through to the `default` branch in the write loop.
    }

    // Get offset to the start of the table data
    LONGLONG headstart, data_offset, dataend;
    int status = 0;
    fits_get_hduaddrll(fptr_, &headstart, &data_offset, &dataend, &status);
    if (status != 0) {
        char err_msg[81];
        fits_get_errstatus(status, err_msg);
        throw std::runtime_error("Failed to get HDU data offset: " + std::string(err_msg));
    }

    if (!direct_io_ok()) {
        throw std::runtime_error(
            "in-place mmap updates require an uncompressed local file "
            "(whole-file compressed inputs like .bz2 are read-only)");
    }

    // Validate every column BEFORE the file is mapped writable.
    //
    // The write loop below validates each column as it reaches it, and its
    // bail-outs only munmap/close -- which does not undo what is already in
    // the mapping. MAP_SHARED dirty pages are written back by the kernel
    // whether or not msync is ever called, so a call that raised on the
    // third column still left the first two rewritten on disk: measured, a
    // two-column update whose second column had the wrong dtype raised
    // "update_rows mmap dtype mismatch" and left the first column holding
    // the new values. The exception is the only signal the caller has, and
    // it said the update failed.
    //
    // Nothing here needs the mapping -- every check reads the tensor and the
    // column descriptor -- so the whole operation is decided before the
    // first byte can be touched. The per-column checks in the write loop
    // stay as a second line of defence.
    for (auto item : tensor_dict) {
        const std::string name = nb::cast<std::string>(item.first);
        const ColumnInfo* col = column_map.at(name);
        const nb::ndarray<> tensor = nb::cast<nb::ndarray<>>(item.second);
        long rows = 1;
        long repeat = 1;
        if (tensor.ndim() == 1) {
            rows = static_cast<long>(tensor.shape(0));
        } else if (tensor.ndim() == 2) {
            rows = static_cast<long>(tensor.shape(0));
            repeat = static_cast<long>(tensor.shape(1));
        } else if (tensor.ndim() != 0) {
            throw std::runtime_error(
                "update_rows mmap only supports 1D/2D columns for " + name);
        }
        const long expected_repeat = (col->repeat > 0) ? col->repeat : 1;
        if (col->type == FITSColumnType::STRING) {
            if (repeat == 0 || repeat > expected_repeat) {
                throw std::runtime_error(
                    "update_rows mmap string width must be 1.." +
                    std::to_string(expected_repeat) + " for " + name);
            }
        } else if (repeat != expected_repeat) {
            throw std::runtime_error("update_rows mmap repeat mismatch for " + name);
        }
        if (rows != num_rows) {
            throw std::runtime_error("update_rows mmap row count mismatch for " + name);
        }
        if (!mmap_update_dtype_accepted(*col, tensor.dtype())) {
            throw std::runtime_error("update_rows mmap dtype mismatch for " + name);
        }
    }

    int fd = open(filename_.c_str(), O_RDWR);
    if (fd == -1) {
        throw std::runtime_error("Failed to open file for mmap update");
    }
    // Adopt the descriptor before the extent check, which rejects truncated
    // files: unwinding does not close a raw int (R2-011).
    MMapHandle fd_owner(nullptr, 0, fd);

    struct stat sb;
    if (fstat(fd, &sb) == -1) {
        throw std::runtime_error("Failed to stat file for mmap update");
    }
    // The writable descriptor has to be resolved by name (data_fd_ is
    // read-only); it must still be the file this reader's geometry describes.
    struct stat pinned;
    if (data_fd_ >= 0 && fstat(data_fd_, &pinned) == 0 &&
        (pinned.st_dev != sb.st_dev || pinned.st_ino != sb.st_ino)) {
        throw std::runtime_error(
            "FITS file was replaced on disk while it was being updated");
    }
    ensure_extent_within_file(sb.st_size, data_offset, start_row - 1, num_rows);

    void* map_ptr = mmap(nullptr, sb.st_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (map_ptr == MAP_FAILED) {
        throw std::runtime_error("Failed to mmap file for update");
    }
    // From here on every bail-out arm in the loop below closes fd itself,
    // so take the descriptor back out of the guard before entering that
    // regime; `fd` keeps its value and the arms stay correct.
    fd_owner.detach_fd();

    uint8_t* base_ptr = static_cast<uint8_t*>(map_ptr) + data_offset;
    size_t row_start_offset = static_cast<size_t>(start_row - 1) * row_width_bytes_;

    for (auto item : tensor_dict) {
        std::string name = nb::cast<std::string>(item.first);
        const ColumnInfo* col = column_map.at(name);

        nb::ndarray<> tensor = nb::cast<nb::ndarray<>>(item.second);
        int ndim = tensor.ndim();
        long rows = 1;
        long repeat = 1;
        if (ndim == 0) {
            rows = 1;
            repeat = 1;
        } else if (ndim == 1) {
            rows = static_cast<long>(tensor.shape(0));
            repeat = 1;
        } else if (ndim == 2) {
            rows = static_cast<long>(tensor.shape(0));
            repeat = static_cast<long>(tensor.shape(1));
        } else {
            munmap(map_ptr, sb.st_size);
            close(fd);
            throw std::runtime_error("update_rows mmap only supports 1D/2D columns for " + name);
        }

        long expected_repeat = (col->repeat > 0) ? col->repeat : 1;
        // STRING columns allow shorter user-provided widths; trailing bytes
        // are padded with ASCII spaces to match the FITS CHAR contract.
        // BIT and COMPLEX columns require an exact repeat match.
        long user_repeat = repeat;
        if (col->type == FITSColumnType::STRING) {
            if (user_repeat == 0 || user_repeat > expected_repeat) {
                munmap(map_ptr, sb.st_size);
                close(fd);
                throw std::runtime_error(
                    "update_rows mmap string width must be 1.." + std::to_string(expected_repeat) +
                    " for " + name
                );
            }
            repeat = expected_repeat;
        } else if (repeat != expected_repeat) {
            munmap(map_ptr, sb.st_size);
            close(fd);
            throw std::runtime_error("update_rows mmap repeat mismatch for " + name);
        }
        if (rows != num_rows) {
            munmap(map_ptr, sb.st_size);
            close(fd);
            throw std::runtime_error("update_rows mmap row count mismatch for " + name);
        }

        nb::dlpack::dtype dt = tensor.dtype();

        const uint8_t* src_u8 = static_cast<const uint8_t*>(tensor.data());
        const bool* src_bool = static_cast<const bool*>(tensor.data());
        const int16_t* src_i16 = static_cast<const int16_t*>(tensor.data());
        const int32_t* src_i32 = static_cast<const int32_t*>(tensor.data());
        const int64_t* src_i64 = static_cast<const int64_t*>(tensor.data());
        const float* src_f32 = static_cast<const float*>(tensor.data());
        const double* src_f64 = static_cast<const double*>(tensor.data());

        // Element offsets honoring DLPack strides (nanobind reports
        // strides in elements), so non-contiguous / negative-stride
        // views write the same values as the buffered
        // (fits_write_col) path instead of reading out of order.
        // A 0-dim payload carries no stride array at all (DLPack strides
        // are NULL) — treat it as one element instead of reading
        // stride(0) out of bounds.
        const long stride0 = (ndim >= 1) ? tensor.stride(0) : 1;
        const long stride1 = (ndim >= 2) ? tensor.stride(1) : 1;

        for (long i = 0; i < num_rows; i++) {
            uint8_t* dest_row = base_ptr + row_start_offset + i * row_width_bytes_ + col->byte_offset;
            for (long j = 0; j < repeat; j++) {
                uint8_t* dest = dest_row + j * col->width;
                const long idx = i * stride0 + j * stride1;

                switch (col->type) {
                    case FITSColumnType::BYTE: {
                        const bool offset_sbyte = col->is_unsigned_int &&
                                                  col->unsigned_target_type == native::kInt8;
                        const bool is_sbyte = col->fits_typecode == TSBYTE || offset_sbyte;
                        if (is_sbyte) {
                            if (!(dt.code == (uint8_t)nb::dlpack::dtype_code::Int && dt.bits == 8)) {
                                munmap(map_ptr, sb.st_size);
                                close(fd);
                                throw std::runtime_error("update_rows mmap dtype mismatch for " + name);
                            }
                        } else if (!(dt.code == (uint8_t)nb::dlpack::dtype_code::UInt && dt.bits == 8)) {
                            munmap(map_ptr, sb.st_size);
                            close(fd);
                            throw std::runtime_error("update_rows mmap dtype mismatch for " + name);
                        }
                        // Byte-for-byte identical bit pattern for signed/unsigned 8-bit;
                        // the TZERO = -128 convention stores value XOR 0x80.
                        *dest = offset_sbyte ? static_cast<uint8_t>(src_u8[idx] ^ 0x80u)
                                             : src_u8[idx];
                        break;
                    }
                    case FITSColumnType::LOGICAL: {
                        bool val = false;
                        if (dt.code == (uint8_t)nb::dlpack::dtype_code::Bool && dt.bits == 8) {
                            val = src_bool[idx];
                        } else                            if (dt.code == (uint8_t)nb::dlpack::dtype_code::UInt && dt.bits == 8) {
                            val = src_u8[idx] != 0;
                        } else {
                            munmap(map_ptr, sb.st_size);
                            close(fd);
                            throw std::runtime_error("update_rows mmap dtype mismatch for " + name);
                        }
                        *dest = val ? 'T' : 'F';
                        break;
                    }
                    case FITSColumnType::SHORT: {
                        // An unsigned column also takes its physical
                        // uint16 values; storage is value XOR sign bit.
                        const bool u16_payload =
                            col->is_unsigned_int &&
                            dt.code == (uint8_t)nb::dlpack::dtype_code::UInt && dt.bits == 16;
                        if (!u16_payload &&
                            !(dt.code == (uint8_t)nb::dlpack::dtype_code::Int && dt.bits == 16)) {
                            munmap(map_ptr, sb.st_size);
                            close(fd);
                            throw std::runtime_error("update_rows mmap dtype mismatch for " + name);
                        }
                        uint16_t v;
                        if (u16_payload) {
                            std::memcpy(&v, &src_i16[idx], sizeof(uint16_t));
                            v ^= 0x8000u;
                        } else {
                            int32_t phys = src_i16[idx];
                            if (col->is_unsigned_int) {
                                phys -= static_cast<int32_t>(col->unsigned_offset);
                            }
                            const int16_t stored = static_cast<int16_t>(phys);
                            std::memcpy(&v, &stored, sizeof(uint16_t));
                        }
                        v = bswap_16(v);
                        std::memcpy(dest, &v, sizeof(uint16_t));
                        break;
                    }
                    case FITSColumnType::INT: {
                        const bool u32_payload =
                            col->is_unsigned_int &&
                            dt.code == (uint8_t)nb::dlpack::dtype_code::UInt && dt.bits == 32;
                        if (!u32_payload &&
                            !(dt.code == (uint8_t)nb::dlpack::dtype_code::Int && dt.bits == 32)) {
                            munmap(map_ptr, sb.st_size);
                            close(fd);
                            throw std::runtime_error("update_rows mmap dtype mismatch for " + name);
                        }
                        uint32_t v;
                        if (u32_payload) {
                            std::memcpy(&v, &src_i32[idx], sizeof(uint32_t));
                            v ^= 0x80000000u;
                        } else {
                            int64_t phys = src_i32[idx];
                            if (col->is_unsigned_int) {
                                phys -= col->unsigned_offset;
                            }
                            const int32_t stored = static_cast<int32_t>(phys);
                            std::memcpy(&v, &stored, sizeof(uint32_t));
                        }
                        v = bswap_32(v);
                        std::memcpy(dest, &v, sizeof(uint32_t));
                        break;
                    }
                    case FITSColumnType::LONG: {
                        const bool u64_col = col->is_unsigned_int &&
                                             col->unsigned_target_type == native::kUInt64;
                        const bool dtype_ok =
                            dt.bits == 64 &&
                            (dt.code == (uint8_t)nb::dlpack::dtype_code::Int ||
                             (u64_col && dt.code == (uint8_t)nb::dlpack::dtype_code::UInt));
                        if (!dtype_ok) {
                            munmap(map_ptr, sb.st_size);
                            close(fd);
                            throw std::runtime_error("update_rows mmap dtype mismatch for " + name);
                        }
                        uint64_t v;
                        std::memcpy(&v, &src_i64[idx], sizeof(uint64_t));
                        if (u64_col) {
                            // Physical value -> stored int64 (TZERO = 2**63),
                            // for uint64 and int64 payloads alike, matching
                            // the SHORT/INT arms' physical semantics.
                            v ^= 0x8000000000000000ULL;
                        }
                        v = bswap_64(v);
                        std::memcpy(dest, &v, sizeof(uint64_t));
                        break;
                    }
                    case FITSColumnType::FLOAT: {
                        if (!(dt.code == (uint8_t)nb::dlpack::dtype_code::Float && dt.bits == 32)) {
                            munmap(map_ptr, sb.st_size);
                            close(fd);
                            throw std::runtime_error("update_rows mmap dtype mismatch for " + name);
                        }
                        uint32_t v;
                        std::memcpy(&v, &src_f32[idx], sizeof(uint32_t));
                        v = bswap_32(v);
                        std::memcpy(dest, &v, sizeof(uint32_t));
                        break;
                    }
                    case FITSColumnType::DOUBLE: {
                        if (!(dt.code == (uint8_t)nb::dlpack::dtype_code::Float && dt.bits == 64)) {
                            munmap(map_ptr, sb.st_size);
                            close(fd);
                            throw std::runtime_error("update_rows mmap dtype mismatch for " + name);
                        }
                        uint64_t v;
                        std::memcpy(&v, &src_f64[idx], sizeof(uint64_t));
                        v = bswap_64(v);
                        std::memcpy(dest, &v, sizeof(uint64_t));
                        break;
                    }
                    case FITSColumnType::BIT: {
                        // Extract a bool from a packed BIT column (MSB-first).
                        bool val = false;
                        if (dt.code == (uint8_t)nb::dlpack::dtype_code::Bool && dt.bits == 8) {
                            val = src_bool[idx];
                        } else                            if (dt.code == (uint8_t)nb::dlpack::dtype_code::UInt && dt.bits == 8) {
                            val = src_u8[idx] != 0;
                        } else {
                            munmap(map_ptr, sb.st_size);
                            close(fd);
                            throw std::runtime_error("update_rows mmap dtype mismatch for " + name);
                        }
                        // FITS BIT columns are MSB-first within each byte.
                        // The default dest stride (j * col->width) is meaningless
                        // for BIT (col->width == 1 but storage is bit-packed), so
                        // we operate directly on dest_row instead.
                        uint8_t* target_byte = dest_row + (j / 8);
                        uint8_t bit_position = static_cast<uint8_t>(7 - (j % 8));
                        if (j % 8 == 0) {
                            *target_byte = 0;
                        }
                        if (val) {
                            *target_byte |= static_cast<uint8_t>(1U << bit_position);
                        }
                        break;
                    }
                    case FITSColumnType::STRING: {
                        if (!(dt.code == (uint8_t)nb::dlpack::dtype_code::UInt && dt.bits == 8)) {
                            munmap(map_ptr, sb.st_size);
                            close(fd);
                            throw std::runtime_error("update_rows mmap dtype mismatch for " + name);
                        }
                        // STRING: dest stride is 1 byte. Pad trailing bytes with
                        // ASCII spaces when the user-provided width is shorter
                        // than the FITS column width.
                        if (j < user_repeat) {
                            // idx honors DLPack strides so nanobind
                            // strided views (e.g. S8 over UCS-4 with stride(1)==4)
                            // land the right byte instead of sweeping NUL padding.
                            dest[0] = src_u8[idx];
                        } else {
                            dest[0] = 0x20; // ASCII space; matches FITS CHAR convention
                        }
                        break;
                    }
                    case FITSColumnType::COMPLEX_FLOAT: {
                        // DLPack `Complex` with bits == 64 corresponds to complex64
                        // (re/im float32). Total bit width is 64 (2 * 32-bit floats).
                        if (!(dt.code == (uint8_t)nb::dlpack::dtype_code::Complex && dt.bits == 64)) {
                            munmap(map_ptr, sb.st_size);
                            close(fd);
                            throw std::runtime_error("update_rows mmap dtype mismatch for " + name);
                        }
                        const auto* src_complex32 =
                            static_cast<const std::complex<float>*>(tensor.data());
                        int32_t re_bits = 0;
                        int32_t im_bits = 0;
                        float re = src_complex32[idx].real();
                        float im = src_complex32[idx].imag();
                        std::memcpy(&re_bits, &re, sizeof(int32_t));
                        std::memcpy(&im_bits, &im, sizeof(int32_t));
                        re_bits = bswap_32(re_bits);
                        im_bits = bswap_32(im_bits);
                        std::memcpy(dest, &re_bits, sizeof(int32_t));
                        std::memcpy(dest + sizeof(int32_t), &im_bits, sizeof(int32_t));
                        break;
                    }
                    case FITSColumnType::COMPLEX_DOUBLE: {
                        // DLPack `Complex` with bits == 128 corresponds to complex128
                        // (re/im float64). Total bit width is 128 (2 * 64-bit floats).
                        if (!(dt.code == (uint8_t)nb::dlpack::dtype_code::Complex && dt.bits == 128)) {
                            munmap(map_ptr, sb.st_size);
                            close(fd);
                            throw std::runtime_error("update_rows mmap dtype mismatch for " + name);
                        }
                        const auto* src_complex64 =
                            static_cast<const std::complex<double>*>(tensor.data());
                        int64_t re_bits = 0;
                        int64_t im_bits = 0;
                        double re = src_complex64[idx].real();
                        double im = src_complex64[idx].imag();
                        std::memcpy(&re_bits, &re, sizeof(int64_t));
                        std::memcpy(&im_bits, &im, sizeof(int64_t));
                        re_bits = bswap_64(re_bits);
                        im_bits = bswap_64(im_bits);
                        std::memcpy(dest, &re_bits, sizeof(int64_t));
                        std::memcpy(dest + sizeof(int64_t), &im_bits, sizeof(int64_t));
                        break;
                    }
                    default:
                        munmap(map_ptr, sb.st_size);
                        close(fd);
                        throw std::runtime_error("update_rows mmap unsupported column type");
                }
            }
        }
    }
    if (msync(map_ptr, sb.st_size, MS_SYNC) != 0) {
        const int sync_errno = errno;
        munmap(map_ptr, sb.st_size);
        close(fd);
        throw std::runtime_error(
            "mmap update msync failed: " + std::string(strerror(sync_errno)));
    }
    munmap(map_ptr, sb.st_size);
    close(fd);
}

}  // namespace torchfits
