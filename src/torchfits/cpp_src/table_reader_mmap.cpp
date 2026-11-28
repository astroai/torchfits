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

namespace torchfits {

namespace {

// Template helper for reading typed columns from mmap with byte-swapping.
// OutT is the output tensor element type; RawT is the integer type used for
// raw reads and byte-swapping (same size as OutT, verified by static_assert).
// bswap_fn is the byte-swap function (bswap_16 / bswap_32 / bswap_64).
template <typename OutT, typename RawT, RawT (*bswap_fn)(RawT)>
void read_typed_mmap_column(
    const uint8_t* col_ptr,
    OutT* out,
    long num_rows,
    long repeat,
    long row_width_bytes)
{
    constexpr size_t elem_size = sizeof(RawT);
    static_assert(sizeof(OutT) == sizeof(RawT));

    // Contiguous layout (single-column table or tightly packed repeat):
    // one SIMD bulk byte-swap instead of per-element memcpy + scalar
    // swap through a function pointer (which blocks vectorization).
    if (num_rows > 0 && repeat > 0 &&
        static_cast<uint64_t>(num_rows) <= SIZE_MAX / static_cast<uint64_t>(repeat)) {
        const size_t total = static_cast<size_t>(num_rows) * static_cast<size_t>(repeat);
        const bool contiguous =
            (repeat == 1 && row_width_bytes == static_cast<long>(elem_size)) ||
            (repeat > 1 && row_width_bytes == static_cast<long>(elem_size * static_cast<size_t>(repeat)));
        if (contiguous && total > 0) {
            const auto* src = reinterpret_cast<const RawT*>(col_ptr);
            auto* dst = reinterpret_cast<RawT*>(out);
            if constexpr (sizeof(RawT) == 2) {
                internal::bswap16_copy(reinterpret_cast<const uint16_t*>(src),
                                       reinterpret_cast<uint16_t*>(dst), total);
            } else if constexpr (sizeof(RawT) == 4) {
                internal::bswap32_copy(reinterpret_cast<const uint32_t*>(src),
                                       reinterpret_cast<uint32_t*>(dst), total);
            } else if constexpr (sizeof(RawT) == 8) {
                internal::bswap64_copy(reinterpret_cast<const uint64_t*>(src),
                                       reinterpret_cast<uint64_t*>(dst), total);
            } else {
                for (size_t k = 0; k < total; ++k) dst[k] = bswap_fn(src[k]);
            }
            return;
        }
    }

    if (repeat == 1) {
        native::parallel_for(0, num_rows, 2048, [&](long start, long end) {
            for (long i = start; i < end; i++) {
                RawT raw_val;
                memcpy(&raw_val, col_ptr + i * row_width_bytes, elem_size);
                RawT val = bswap_fn(raw_val);
                memcpy(&out[i], &val, elem_size);
            }
        });
    } else {
        native::parallel_for(0, num_rows, 2048, [&](long start, long end) {
            for (long i = start; i < end; i++) {
                OutT* row_out = out + i * repeat;
                for (long j = 0; j < repeat; j++) {
                    RawT raw_val;
                    memcpy(&raw_val, col_ptr + i * row_width_bytes + j * elem_size, elem_size);
                    RawT val = bswap_fn(raw_val);
                    memcpy(&row_out[j], &val, elem_size);
                }
            }
        });
    }
}

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

    // Open file with mmap
    int fd = open(filename_.c_str(), O_RDONLY);
    if (fd == -1) {
        throw std::runtime_error("Failed to open file for mmap");
    }
    // Adopt the descriptor immediately: the two statements below can both
    // throw, and a raw int on the stack is not closed by unwinding. A
    // truncated table used to leak one descriptor per rejected call
    // (R2-011) -- exactly the case this check exists to catch.
    MMapHandle fd_owner(nullptr, 0, fd);

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
    // needs to survive for the duration of this function.
    // size must match the original mmap() length (full file) passed to munmap().
    MMapHandle mmap_guard(map_ptr, sb.st_size, fd_owner.detach_fd());

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

    for (int col_idx : col_indices) {
        const auto& col = columns_[col_idx];

        if (col.type == FITSColumnType::VARIABLE) {
            continue;
        }
        // Complex columns are supported for mmap reads below (byte-swap path).

        // Pointer to start of column data for the first requested row
        const uint8_t* col_ptr = base_ptr + row_start_offset + col.byte_offset;

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

        try {
            // Create Torch Tensor and copy/swap
            native::ScalarType dtype;
            switch (col.type) {
                case FITSColumnType::FLOAT: dtype = native::kFloat32; break;
                case FITSColumnType::DOUBLE: dtype = native::kFloat64; break;
                case FITSColumnType::INT: dtype = native::kInt32; break;
                case FITSColumnType::SHORT: dtype = native::kInt16; break;
                case FITSColumnType::LONG: dtype = native::kInt64; break;
                case FITSColumnType::BYTE: dtype = col.torch_type; break;  // kUInt8, or kInt8 for TSBYTE
                case FITSColumnType::BIT: dtype = native::kBool; break;
                case FITSColumnType::LOGICAL: dtype = native::kBool; break;
                case FITSColumnType::STRING: dtype = native::kUInt8; break;
                case FITSColumnType::COMPLEX_FLOAT: dtype = native::kComplexFloat; break;
                case FITSColumnType::COMPLEX_DOUBLE: dtype = native::kComplexDouble; break;
                default: dtype = native::kFloat32;
            }

            auto options = native::TensorOptions().dtype(dtype);
            native::Tensor tensor = native::empty(shape, options);

            // Parallel copy and swap
            long repeat = (col.repeat > 1) ? col.repeat : 1;
            if (col.type == FITSColumnType::STRING) {
                repeat = is_ascii_ ? col.width : col.repeat; // String length
            }

            if (col.type == FITSColumnType::FLOAT) {
                read_typed_mmap_column<float, int32_t, bswap_32>(
                    col_ptr, tensor.data_ptr<float>(), num_rows, repeat, row_width_bytes_);
            } else if (col.type == FITSColumnType::DOUBLE) {
                read_typed_mmap_column<double, int64_t, bswap_64>(
                    col_ptr, tensor.data_ptr<double>(), num_rows, repeat, row_width_bytes_);
            } else if (col.type == FITSColumnType::INT) {
                read_typed_mmap_column<int32_t, int32_t, bswap_32>(
                    col_ptr, tensor.data_ptr<int32_t>(), num_rows, repeat, row_width_bytes_);
            } else if (col.type == FITSColumnType::SHORT) {
                read_typed_mmap_column<int16_t, int16_t, bswap_16>(
                    col_ptr, tensor.data_ptr<int16_t>(), num_rows, repeat, row_width_bytes_);
            } else if (col.type == FITSColumnType::LONG) {
                read_typed_mmap_column<int64_t, int64_t, bswap_64>(
                    col_ptr, tensor.data_ptr<int64_t>(), num_rows, repeat, row_width_bytes_);
            } else if (col.type == FITSColumnType::BYTE || col.type == FITSColumnType::STRING) {
                uint8_t* out = tensor.data_ptr<uint8_t>();
                native::parallel_for(0, num_rows, 2048, [&](long start, long end) {
                    for (long i = start; i < end; i++) {
                        const uint8_t* in = (const uint8_t*)(col_ptr + i * row_width_bytes_);
                        uint8_t* row_out = out + i * repeat;
                        memcpy(row_out, in, repeat);
                    }
                });
                if (col.type == FITSColumnType::BYTE && col.fits_typecode == TSBYTE) {
                    detail::xor_sign_bit_u8(
                        out,
                        static_cast<size_t>(num_rows) * static_cast<size_t>(repeat));
                }
            } else if (col.type == FITSColumnType::BIT) {
                bool* out = tensor.data_ptr<bool>();
                native::parallel_for(0, num_rows, 2048, [&](long start, long end) {
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
                });
            } else if (col.type == FITSColumnType::COMPLEX_FLOAT) {
                std::complex<float>* out = tensor.data_ptr<std::complex<float>>();
                native::parallel_for(0, num_rows, 2048, [&](long start, long end) {
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
                });
            } else if (col.type == FITSColumnType::COMPLEX_DOUBLE) {
                std::complex<double>* out = tensor.data_ptr<std::complex<double>>();
                native::parallel_for(0, num_rows, 2048, [&](long start, long end) {
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
                });
            } else if (col.type == FITSColumnType::LOGICAL) {
                bool* out = tensor.data_ptr<bool>();
                native::parallel_for(0, num_rows, 2048, [&](long start, long end) {
                    for (long i = start; i < end; i++) {
                        const char* in = (const char*)(col_ptr + i * row_width_bytes_);
                        bool* row_out = out + i * repeat; // repeat is usually 1 for logical
                        for (long j = 0; j < repeat; j++) {
                            row_out[j] = (in[j] == 'T' || in[j] == '1' || in[j] == 1);
                        }
                    }
                });
            }

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
            result[col_idx] = std::move(tensor);

        } catch (const std::exception& e) {
            // Never swallow: a failed column must surface as an error, not
            // as a silently missing column in the result dict.
            throw std::runtime_error("Failed to mmap column " + col.name + ": " + e.what());
        }
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
