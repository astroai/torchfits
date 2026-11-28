// TableReader::read_columns: column resolution, dispatch to the buffered or
// CFITSIO column path, and the post-pass that applies TSCAL/TZERO, TNULL and
// the unsigned-integer conventions shared with the mmap and VLA readers.

#include <algorithm>
#include <complex>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>

#include "table_reader.h"
#include "checked_arithmetic.h"

namespace torchfits {

// A real scalar added to a complex tensor touches only the real part.
// TZERO has to move both components.
void TableReader::scale_complex_inplace(native::Tensor& tensor, double tscale, double tzero) {
    if (tscale != 1.0) tensor.mul_(tscale);
    if (tzero == 0.0) return;
    if (tensor.scalar_type() == native::kComplexFloat) {
        tensor.add_(std::complex<float>(
            static_cast<float>(tzero), static_cast<float>(tzero)));
    } else {
        tensor.add_(std::complex<double>(tzero, tzero));
    }
}

// FITS TSCAL/TZERO applied in-memory (float64) with TNULL mapped to NaN
// after the linear map. Shared by the fixed-width post-process and the VLA
// decode so both obey one convention: raw integer TNULL without
// TSCAL/TZERO stays a sentinel (tnull-read-torch).
native::Tensor TableReader::apply_scale_and_nulls(const ColumnInfo& col, native::Tensor raw) {
    native::Tensor null_mask;
    if (col.has_tnull && native::is_integral(raw.scalar_type(), /*includeBool=*/false)) {
        const long long null_ll = col.tnull;
        bool compared = false;
        switch (raw.scalar_type()) {
            case native::kInt8:
                if (null_ll >= std::numeric_limits<int8_t>::min() &&
                    null_ll <= std::numeric_limits<int8_t>::max()) {
                    null_mask = raw.eq(static_cast<int8_t>(null_ll));
                    compared = true;
                }
                break;
            case native::kUInt8:
                if (null_ll >= 0 &&
                    null_ll <= static_cast<long long>(std::numeric_limits<uint8_t>::max())) {
                    null_mask = raw.eq(static_cast<uint8_t>(null_ll));
                    compared = true;
                }
                break;
            case native::kInt16:
                if (null_ll >= std::numeric_limits<int16_t>::min() &&
                    null_ll <= std::numeric_limits<int16_t>::max()) {
                    null_mask = raw.eq(static_cast<int16_t>(null_ll));
                    compared = true;
                }
                break;
            case native::kInt32:
                if (null_ll >= std::numeric_limits<int32_t>::min() &&
                    null_ll <= std::numeric_limits<int32_t>::max()) {
                    null_mask = raw.eq(static_cast<int32_t>(null_ll));
                    compared = true;
                }
                break;
            default:
                break;
        }
        if (!compared) {
            null_mask = raw.to(native::kInt64).eq(static_cast<int64_t>(null_ll));
        }
    }
    native::Tensor scaled = raw.to(native::kFloat64);
    if (col.tscale != 1.0) {
        scaled.mul_(col.tscale);
    }
    if (col.tzero != 0.0) {
        scaled.add_(col.tzero);
    }
    if (null_mask.defined()) {
        scaled.masked_fill_(null_mask, std::numeric_limits<double>::quiet_NaN());
    }
    return scaled;
}

// FITS stores uint16/uint32 columns as the signed value plus a 2**(b-1)
// offset (TZERO). Adding the offset modulo 2**b is exactly flipping the
// sign bit, so reinterpret the signed buffer as unsigned and XOR in place:
// no int64 upcast and no second allocation. The input is a freshly decoded
// column owned by the caller, so mutating it is safe.
native::Tensor TableReader::apply_unsigned_offset(native::Tensor raw, native::ScalarType target) {
    if (!raw.is_contiguous()) {
        raw = raw.contiguous();
    }
    if (target == native::kInt8) {
        raw.bitwise_xor_(0x80);
        return raw.view(native::kInt8);
    }
    if (target == native::kUInt16) {
        raw.bitwise_xor_(std::numeric_limits<int16_t>::min());
        return raw.view(native::kUInt16);
    }
    if (target == native::kUInt32) {
        raw.bitwise_xor_(std::numeric_limits<int32_t>::min());
        return raw.view(native::kUInt32);
    }
    if (target == native::kUInt64) {
        raw.bitwise_xor_(std::numeric_limits<int64_t>::min());
        return raw.view(native::kUInt64);
    }
    // ensure_column_scale only ever selects the four targets above.
    throw std::logic_error(
        std::string("unsigned-offset convention has no target ") +
        native::dtype_name(target));
}

// Read columns from the table
// Returns column data in file order (request order when column_names is
// given), so callers that pair columns against TFORM/TTYPE cards (e.g. the
// table rewrite path) never see an arbitrary hash iteration order.
std::vector<std::pair<std::string, TableReader::ColumnData>> TableReader::read_columns(
    const std::vector<std::string>& column_names,
    long start_row, long num_rows, bool vla_flat) {

    ensure_table_hdu();
    if (num_rows == -1) {
        num_rows = nrows_;
    }

    // Resolve requested columns first so unknown names fail loudly even
    // for empty tables (an early `{}` silently dropped them). A repeated
    // name resolves once: the assembly loop moves each entry out, so a
    // duplicated index would hand back a moved-from (null) tensor.
    std::vector<int> col_indices;
    if (column_names.empty()) {
        // Read all columns
        for (int i = 0; i < ncols_; i++) {
            col_indices.push_back(i);
        }
    } else {
        // Read specified columns
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
            if (!found) {
                std::string available;
                for (int k = 0; k < ncols_; k++) {
                    if (k > 0) available += ", ";
                    available += columns_[k].name;
                }
                throw std::runtime_error(
                    "Column not found: " + name + ". Available columns: " + available);
            }
        }
    }

    // An empty window (0-row table, or num_rows == 0) still returns every
    // requested column, as an empty tensor of the dtype a non-empty read
    // gives, so the result's keys and dtypes do not depend on row count.
    if (num_rows < 0) {
        throw std::runtime_error("Invalid table row count: " + std::to_string(num_rows));
    }
    if (nrows_ == 0) {
        num_rows = 0;
    }
    if (num_rows > 0) {
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
    } else if (start_row < 1 || start_row > nrows_ + 1) {
        throw std::runtime_error(
            "Invalid start row: " + std::to_string(start_row) +
            ", nrows: " + std::to_string(nrows_));
    }
    for (int col_idx : col_indices) {
        ensure_column_scale(col_idx);
    }

    int status = 0;
    // Keyed by column INDEX, not name: FITS permits duplicate TTYPE cards,
    // and a name-keyed map silently overwrote the earlier column. The
    // assembly loop below then moved the same entry out twice, so the
    // surviving column came back as a null tensor (None in Python).
    std::unordered_map<int, ColumnData> result;

    auto cfitsio_read_datatype = [](const ColumnInfo& col) -> int {
        switch (col.type) {
            case FITSColumnType::LOGICAL: return TLOGICAL;
            case FITSColumnType::BIT: return TBIT;
            case FITSColumnType::BYTE: return TBYTE;
            case FITSColumnType::SHORT: return TSHORT;
            case FITSColumnType::INT: return TINT;
            case FITSColumnType::LONG: return TLONGLONG;
            case FITSColumnType::FLOAT: return TFLOAT;
            case FITSColumnType::DOUBLE: return TDOUBLE;
#ifdef TCOMPLEX
            case FITSColumnType::COMPLEX_FLOAT: return TCOMPLEX;
#endif
#ifdef TDBLCOMPLEX
            case FITSColumnType::COMPLEX_DOUBLE: return TDBLCOMPLEX;
#endif
            case FITSColumnType::STRING: return TBYTE; // Read strings as bytes
            default: return TFLOAT;
        }
    };


    // Allocate tensors for all requested columns (except VLA)
    for (int col_idx : col_indices) {
        const auto& col = columns_[col_idx];


        if (col.type == FITSColumnType::VARIABLE) {
            // VLA columns will be handled separately
            continue;
        }

        std::vector<int64_t> shape;
        shape.push_back(num_rows);
        // Handle multi-dimensional columns AND strings
        if (col.type == FITSColumnType::STRING) {
            if (is_ascii_) {
                shape.push_back(col.width);
            } else {
                shape.push_back(col.repeat);
            }
        } else if (col.repeat > 1) {
             shape.push_back(col.repeat);
        }

        const int64_t row_elements =
            col.type == FITSColumnType::STRING && is_ascii_
            ? static_cast<int64_t>(col.width)
            : static_cast<int64_t>(col.repeat);
        const int64_t allocation_elements =
            internal::checked_nonnegative_multiply<int64_t>(
            static_cast<int64_t>(num_rows), row_elements,
            "Invalid table geometry for column " + col.name + " allocation");
        const size_t allocation_element_bytes =
            col.type == FITSColumnType::BIT || col.type == FITSColumnType::LOGICAL ||
                col.type == FITSColumnType::BYTE || col.type == FITSColumnType::STRING
            ? 1
            : internal::checked_nonnegative_cast<size_t>(
                  col.width,
                  "Invalid table geometry for column " + col.name + " element width");
        internal::checked_nonnegative_multiply<size_t>(
            internal::checked_nonnegative_cast<size_t>(
                allocation_elements,
                "Invalid table geometry for column " + col.name + " allocation"),
            allocation_element_bytes,
            "Invalid table geometry for column " + col.name + " allocation");

        // Create tensor
        native::Tensor tensor = native::empty(shape, native::TensorOptions().dtype(col.torch_type));

        // Store in result map as ColumnData

        result[col_idx] = ColumnData(tensor);
    }

    // Row-buffered reads (tiled pread, else fits_read_tblbytes) for every
    // fixed-width binary projection, single columns included: CFITSIO's
    // fits_read_col walks the same row span block by block, so a 1-column
    // read of a 20-column table cost more than the 2-column buffered read.
    // VLA / BIT / complex stay on fits_read_col.
    long requested_bytes = 0;
    bool has_vla = false;
    bool has_bit = false;
    bool has_complex = false;
    for (int col_idx : col_indices) {
        const auto& col = columns_[col_idx];
        if (col.type == FITSColumnType::VARIABLE) {
            has_vla = true;
        } else if (col.type == FITSColumnType::BIT) {
            has_bit = true;
        } else if (col.type == FITSColumnType::COMPLEX_FLOAT || col.type == FITSColumnType::COMPLEX_DOUBLE) {
            has_complex = true;
        } else {
            // long math: width/repeat are int and their product can
            // overflow int for absurd TFORM repeats before it lands in
            // this long accumulator.
            requested_bytes = internal::checked_nonnegative_add<long>(
                requested_bytes, col.storage_bytes,
                "Invalid table geometry for projected columns");
        }
    }

    bool use_buffered = false;
    if (table_buffered_read_enabled() &&
        !is_ascii_ && !has_vla && !has_bit && !has_complex && row_width_bytes_ > 0) {
        use_buffered = requested_bytes > 0;
    }

    auto read_column_by_column = [&]() {
        // Read column by column
        for (int col_idx : col_indices) {
            const auto& col = columns_[col_idx];

            if (col.type == FITSColumnType::VARIABLE) {
                // Read VLA column
                if (vla_flat) {
                    auto flat = read_vla_column_flat(col_idx, start_row, num_rows, col);
                    result[col_idx] = ColumnData(std::move(flat.first), std::move(flat.second), true);
                } else {
                    result[col_idx] = ColumnData(read_vla_column(col_idx, start_row, num_rows, col));
                }
            } else {
                // Read fixed width column
                native::Tensor tensor = result[col_idx].fixed_data;

                int status = 0;
                // Use fits_read_col to read directly into tensor memory
                // Note: fits_read_col handles byte swapping automatically!

                int datatype = cfitsio_read_datatype(col);

                long firstelem = 1;
                long nelements = internal::checked_nonnegative_multiply<long>(
                    num_rows, static_cast<long>(col.repeat),
                    "Invalid table geometry for column " + col.name + " element count");

                if (col.type == FITSColumnType::STRING) {
                     if (is_ascii_) {
                         // ASCII table: read as strings
                          const size_t pointer_count =
                              internal::checked_nonnegative_cast<size_t>(
                                  nelements,
                                  "Invalid ASCII table geometry for column " + col.name);
                          const size_t string_stride =
                              internal::checked_nonnegative_add<size_t>(
                                  internal::checked_nonnegative_cast<size_t>(
                                      col.width,
                                      "Invalid ASCII table geometry for column " + col.name),
                                  1,
                                  "Invalid ASCII table geometry for column " + col.name);
                          const size_t buffer_size =
                              internal::checked_nonnegative_multiply<size_t>(
                                  pointer_count, string_stride,
                                  "Invalid ASCII table geometry for column " + col.name);
                          std::vector<char*> pointers(pointer_count);
                          std::vector<char> buffer(buffer_size);

                         for (long i = 0; i < nelements; i++) {
                              pointers[static_cast<size_t>(i)] =
                                  &buffer[static_cast<size_t>(i) * string_stride];
                         }

                         fits_read_col(fptr_, TSTRING, col_idx + 1, start_row, firstelem, nelements,
                                      nullptr, pointers.data(), nullptr, &status);

                         // Copy to tensor (row-major)
                         uint8_t* tensor_data = (uint8_t*)tensor.data_ptr();
                         for (long i = 0; i < nelements; i++) {
                             // Copy string to tensor, padding with spaces or nulls?
                             // TorchFits convention: raw bytes.
                             // cfitsio returns null-terminated string.
                             // We copy up to col.width.
                             const char* src = pointers[i];
                             size_t len = strlen(src);
                             for (int j = 0; j < col.width; j++) {
                                 if (j < len) {
                                     tensor_data[i * col.width + j] = (uint8_t)src[j];
                                 } else {
                                     tensor_data[i * col.width + j] = ' '; // Pad with spaces for ASCII?
                                 }
                             }
                         }
                     } else {
                         // Binary table: read as raw bytes
                         fits_read_col(fptr_, datatype, col_idx + 1, start_row, firstelem, nelements,
                                      nullptr, tensor.data_ptr(), nullptr, &status);
                     }
                } else if (col.type == FITSColumnType::LOGICAL) {
                     fits_read_col(fptr_, TBYTE, col_idx + 1, start_row, firstelem, nelements,
                                  nullptr, tensor.data_ptr(), nullptr, &status);

                     // Convert logical byte to 1/0. CFITSIO converts logical
                     // columns to the integer values 1/0 when read as TBYTE;
                     // raw bytes may also be 'T'/'F' or '1'/'0'.
                     uint8_t* data = (uint8_t*)tensor.data_ptr();
                     for (long i = 0; i < nelements; i++) {
                         data[i] = (data[i] == 'T' || data[i] == '1' || data[i] == 1) ? 1 : 0;
                     }
                } else {
                    #ifdef DEBUG_TABLE
                    fprintf(stderr, "Reading col %d (%s), type %d, datatype %d, rows %ld\n",
                            col_idx+1, col.name.c_str(), (int)col.type, datatype, num_rows);
                    #endif
                    // Raw storage codes: apply TSCAL/TZERO in-memory below.
                    // Leaving CFITSIO auto-scale on overflows int destinations
                    // when physical values exceed the storage type range.
                    {
                        int scale_status = 0;
                        fits_set_tscale(fptr_, col_idx + 1, 1.0, 0.0, &scale_status);
                    }
                    if (col.type == FITSColumnType::BIT && col.repeat % 8 != 0) {
                        // CFITSIO's TBIT reader walks every byte of each
                        // cell, pad bits included, before the next row, so
                        // a multi-row read is misaligned unless the repeat
                        // fills whole bytes. Read one row at a time.
                        auto* dst = static_cast<char*>(tensor.data_ptr());
                        for (long r = 0; r < num_rows && status == 0; ++r) {
                            fits_read_col(fptr_, datatype, col_idx + 1, start_row + r, firstelem,
                                          static_cast<long>(col.repeat), nullptr,
                                          dst + r * static_cast<long>(col.repeat), nullptr, &status);
                        }
                    } else {
                        fits_read_col(fptr_, datatype, col_idx + 1, start_row, firstelem, nelements,
                                      nullptr, tensor.data_ptr(), nullptr, &status);
                    }
                }

                if (status != 0) {
                     char err_msg[81];
                     fits_get_errstatus(status, err_msg);
                     throw std::runtime_error("Failed to read column " + col.name + ": " + std::string(err_msg));
                }
            }
        }
    };

    if (num_rows == 0) {
        // Nothing to read: the fixed-width tensors above are already
        // (0, ...) and go through the same dtype post-processing below.
        for (int col_idx : col_indices) {
            const auto& col = columns_[col_idx];
            if (col.type != FITSColumnType::VARIABLE) continue;
            if (vla_flat) {
                native::Tensor values = native::empty(
                    {0}, native::TensorOptions().dtype(col.torch_type));
                native::Tensor offsets = native::zeros(
                    {1}, native::TensorOptions().dtype(native::kInt64));
                result[col_idx] = ColumnData(values, offsets, true);
            } else {
                result[col_idx] = ColumnData(std::vector<native::Tensor>{});
            }
        }
    } else if (use_buffered) {
        try {
            read_columns_buffered(col_indices, start_row, num_rows, result);
        } catch (const std::exception&) {
            // Fallback for CFITSIO edge cases where tblbytes reads fail.
            read_column_by_column();
        }
    } else {
        read_column_by_column();
    }

    // Post-process with one map lookup per column. The per-column order
    // is scale -> BIT-coerce -> unsigned-offset, exactly matching the
    // three legacy passes (columns are independent, so fusing the outer
    // loop changes nothing observable). In particular a TSCAL/TZERO BIT
    // column scales first and coerces second, as before.
    // Null masks compare in the raw dtype when TNULL fits, avoiding a
    // full int64 upcast copy of the column just for the comparison.
    for (int col_idx : col_indices) {
        const auto& col = columns_[col_idx];
        auto it = result.find(col_idx);
        if (it == result.end() || !it->second.fixed_data.defined()) {
            continue;
        }
        // (1) FITS TSCAL/TZERO in-memory (float64). FLOAT/DOUBLE columns
        // are scaled too — only complex/string/logical/VLA are exempt
        // (BIT is NOT exempt: it scales here, coerces below).
        if (col.scaled &&
            (col.type == FITSColumnType::COMPLEX_FLOAT ||
             col.type == FITSColumnType::COMPLEX_DOUBLE)) {
            // apply_scale_and_nulls casts to float64 and would drop the
            // imaginary part. Scale both components in the complex dtype.
            scale_complex_inplace(it->second.fixed_data, col.tscale, col.tzero);
        } else if (col.scaled &&
            col.type != FITSColumnType::STRING &&
            col.type != FITSColumnType::LOGICAL &&
            col.type != FITSColumnType::VARIABLE) {
            it->second.fixed_data = apply_scale_and_nulls(col, it->second.fixed_data);
        }
        // (2) BIT->bool coercion.
        if (col.type == FITSColumnType::BIT) {
            it->second.fixed_data = it->second.fixed_data.to(native::kBool);
        }
        // (3) Unsigned integer offset for the uint16/uint32 convention.
        // Unsigned columns never enter block (1) (scaled == false), so
        // this only runs on unscaled integer storage, as before.
        if (col.is_unsigned_int) {
            it->second.fixed_data = apply_unsigned_offset(
                std::move(it->second.fixed_data), col.unsigned_target_type);
        }
    }

    // Order by file/request column index (see signature comment).
    std::vector<std::pair<std::string, ColumnData>> ordered;
    ordered.reserve(col_indices.size());
    for (int col_idx : col_indices) {
        auto it = result.find(col_idx);
        if (it != result.end()) {
            ordered.emplace_back(columns_[col_idx].name, std::move(it->second));
        }
    }
    return ordered;

}

}  // namespace torchfits
