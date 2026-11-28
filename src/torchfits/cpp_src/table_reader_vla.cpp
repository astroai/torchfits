// TableReader variable-length array (P/Q) columns: descriptors are read and
// validated against the heap extent, then the heap is decoded per row through
// CFITSIO (or, opt-in, with one contiguous pread).

#include <cerrno>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <fcntl.h>
#include <unistd.h>

#include "table_reader.h"
#include "checked_arithmetic.h"
#include "hardware.h"
#include "internal_utils.h"

namespace torchfits {

// Read a Variable Length Array column
// Returns a list of tensors (one per row) as views into one flat buffer.
std::vector<native::Tensor> TableReader::read_vla_column(int col_idx, long start_row, long num_rows, const ColumnInfo& col) {
    auto flat = read_vla_column_flat(col_idx, start_row, num_rows, col);
    native::Tensor& values = flat.first;
    native::Tensor& offs = flat.second;
    std::vector<native::Tensor> column_data;
    column_data.reserve(static_cast<size_t>(num_rows));
    const int64_t* op = offs.data_ptr<int64_t>();
    for (long i = 0; i < num_rows; i++) {
        column_data.push_back(values.slice(0, op[i], op[i + 1]));
    }
    return column_data;
}

// Read a VLA column as flat values + row offsets for fast Arrow ListArray construction.
std::pair<native::Tensor, native::Tensor> TableReader::read_vla_column_flat(
    int col_idx, long start_row, long num_rows, const ColumnInfo& col
) {
    {
        // Complex and bit heap elements need a decode this reader does not
        // have (two swapped halves; packed bits counted in bits). Reading
        // them as float32 returned truncated, mistyped data.
        const int vla_type = -col.fits_typecode;
        if (vla_type == TCOMPLEX || vla_type == TDBLCOMPLEX || vla_type == TBIT) {
            throw std::runtime_error(
                "Variable-length complex or bit columns ('PC', 'PM', 'PX') are not "
                "supported: " + col.name);
        }
    }
    const size_t descriptor_count = internal::checked_nonnegative_cast<size_t>(
        num_rows, "Invalid VLA geometry for column " + col.name + " row count");
    std::vector<LONGLONG> repeats(descriptor_count, 0);
    std::vector<LONGLONG> heap_offsets(descriptor_count, 0);
    int status = 0;
    fits_read_descriptsll(
        fptr_, col_idx + 1, start_row, num_rows, repeats.data(), heap_offsets.data(), &status
    );
    if (status != 0) {
        status = 0;
        for (long i = 0; i < num_rows; i++) {
            long row = start_row + i;
            fits_read_descriptll(
                fptr_, col_idx + 1, row, &repeats[i], &heap_offsets[i], &status);
            if (status != 0) {
                char err_msg[81];
                fits_get_errstatus(status, err_msg);
                throw std::runtime_error(
                    "Failed to read VLA descriptor for column " + col.name +
                    " row " + std::to_string(row) + ": " + std::string(err_msg));
            }
        }
    }

    size_t elem_bytes = 0;
    switch (col.torch_type) {
        case native::kBool:
        case native::kUInt8:
        case native::kInt8: elem_bytes = 1; break;
        case native::kInt16: elem_bytes = 2; break;
        case native::kInt32:
        case native::kFloat32: elem_bytes = 4; break;
        case native::kInt64:
        case native::kFloat64: elem_bytes = 8; break;
        default:
            throw std::runtime_error("Unsupported VLA scalar type for column " + col.name);
    }

    const size_t offsets_count = internal::checked_nonnegative_add<size_t>(
        descriptor_count, 1, "Invalid VLA geometry for column " + col.name + " offsets");
    std::vector<int64_t> offsets(offsets_count, 0);
    int64_t total = 0;
    for (long i = 0; i < num_rows; i++) {
        const long row = start_row + i;
        if (repeats[i] < 0) {
            throw std::runtime_error(
                "Invalid VLA geometry for column " + col.name + " row " +
                std::to_string(row) + ": negative repeat " +
                std::to_string(repeats[i]));
        }
        if (heap_offsets[i] < 0) {
            throw std::runtime_error(
                "Invalid VLA geometry for column " + col.name + " row " +
                std::to_string(row) + ": negative heap offset " +
                std::to_string(heap_offsets[i]));
        }
        const int64_t rep = internal::checked_nonnegative_cast<int64_t>(
            repeats[i], "Invalid VLA geometry for column " + col.name +
                            " row " + std::to_string(row) + " repeat");
        total = internal::checked_nonnegative_add<int64_t>(
            total, rep, "Invalid VLA geometry for column " + col.name +
                            " row " + std::to_string(row) + " aggregate repeat");
        offsets[i + 1] = total;
    }

    const size_t total_size = internal::checked_nonnegative_cast<size_t>(
        total, "Invalid VLA geometry for column " + col.name + " allocation");
    internal::checked_nonnegative_multiply<size_t>(
        total_size, elem_bytes,
        "Invalid VLA geometry for column " + col.name + " allocation");

    LONGLONG datastart = 0, dataend = 0;
    status = 0;
    fits_get_hduaddrll(fptr_, nullptr, &datastart, &dataend, &status);
    if (status != 0 || datastart < 0 || dataend < datastart) {
        throw std::runtime_error("Invalid VLA geometry for column " + col.name + ": HDU extent");
    }
    const LONGLONG table_bytes = internal::checked_nonnegative_multiply<LONGLONG>(
        internal::checked_nonnegative_cast<LONGLONG>(
            row_width_bytes_, "Invalid VLA geometry for column " + col.name + " row width"),
        internal::checked_nonnegative_cast<LONGLONG>(
            nrows_, "Invalid VLA geometry for column " + col.name + " row count"),
        "Invalid VLA geometry for column " + col.name + " table bytes");
    LONGLONG pcount = 0;
    int pcount_status = 0;
    fits_read_key(
        fptr_, TLONGLONG, const_cast<char*>("PCOUNT"), &pcount, nullptr, &pcount_status);
    if (pcount_status != 0 || pcount < 0) {
        throw std::runtime_error("Invalid VLA geometry for column " + col.name + ": PCOUNT");
    }
    LONGLONG theap = table_bytes;
    int theap_status = 0;
    fits_read_key(
        fptr_, TLONGLONG, const_cast<char*>("THEAP"), &theap, nullptr, &theap_status);
    if (theap_status != 0) {
        theap = table_bytes;
    }
    if (theap < table_bytes) {
        throw std::runtime_error("Invalid VLA geometry for column " + col.name + ": THEAP");
    }
    const LONGLONG data_extent = internal::checked_nonnegative_add<LONGLONG>(
        datastart,
        internal::checked_nonnegative_add<LONGLONG>(
            table_bytes, pcount,
            "Invalid VLA geometry for column " + col.name + " data extent"),
        "Invalid VLA geometry for column " + col.name + " data extent");
    const LONGLONG heap_base = internal::checked_nonnegative_add<LONGLONG>(
        datastart, theap, "Invalid VLA geometry for column " + col.name + " heap base");
    if (heap_base > data_extent || data_extent > dataend) {
        throw std::runtime_error("Invalid VLA geometry for column " + col.name + ": heap extent");
    }
    for (long i = 0; i < num_rows; i++) {
        const long row = start_row + i;
        const LONGLONG length = internal::checked_nonnegative_multiply<LONGLONG>(
            repeats[i], internal::checked_nonnegative_cast<LONGLONG>(
                            elem_bytes, "Invalid VLA geometry for column " + col.name),
            "Invalid VLA geometry for column " + col.name + " row " +
                std::to_string(row) + " heap length");
        const LONGLONG end = internal::checked_nonnegative_add<LONGLONG>(
            heap_offsets[i], length,
            "Invalid VLA geometry for column " + col.name + " row " +
                std::to_string(row) + " heap range");
        const LONGLONG absolute_end = internal::checked_nonnegative_add<LONGLONG>(
            heap_base, end,
            "Invalid VLA geometry for column " + col.name + " row " +
                std::to_string(row) + " file range");
        if (absolute_end > data_extent) {
            throw std::runtime_error(
                "Invalid VLA geometry for column " + col.name + " row " +
                std::to_string(row) + ": heap range exceeds HDU extent");
        }
    }

    native::Tensor values = native::empty(
        {total}, native::TensorOptions().dtype(col.torch_type)
    );
    native::Tensor offs = native::copy_of(
        offsets.data(), static_cast<int64_t>(offsets.size()), native::kInt64);

    // TSCAL/TZERO/TNULL follow the fixed-width convention: raw cells in,
    // float64 physical values (NaN at TNULL) out. Both heap paths read raw
    // cells (the pread below copies heap bytes; the per-row fits_read_col
    // path resets CFITSIO auto-scaling first), so the map is applied once
    // here.
    auto finish = [&]() {
        if (col.scaled && !col.is_unsigned_int && values.scalar_type() != native::kBool) {
            values = apply_scale_and_nulls(col, values);
        }
        return std::make_pair(values, offs);
    };

    int type_code = 0;
    switch (col.torch_type) {
        case native::kFloat32: type_code = TFLOAT; break;
        case native::kFloat64: type_code = TDOUBLE; break;
        case native::kInt8: type_code = TSBYTE; break;
        case native::kInt32: type_code = TINT; break;
        case native::kInt16: type_code = TSHORT; break;
        case native::kInt64: type_code = TLONGLONG; break;
        case native::kUInt8: type_code = TBYTE; break;
        case native::kBool: type_code = TLOGICAL; break;
        default: type_code = TFLOAT;
    }
    const bool heap_element_supported = values.scalar_type() != native::kBool;

    // Contiguous heap → one pread + endian convert (skips N× fits_read_col).
    // Default off until THEAP/offset edge cases are fully proven vs CFITSIO.
    // env_flag_default_false rather than a local check, so the opt-in
    // vocabulary matches every other flag ("1"/"true"/"yes"/"on", any
    // casing). The previous first-character test accepted 't'/'y' but not
    // 'o', so TORCHFITS_VLA_HEAP_PREAD=on silently stayed off.
    static const bool heap_pread_enabled =
        internal::env_flag_default_false("TORCHFITS_VLA_HEAP_PREAD");
    bool heap_contiguous =
        heap_pread_enabled && direct_io_ok() &&
        (heap_element_supported && total > 0);
    // uint64 math: long offsets × repeats from hostile headers must not
    // wrap and fake contiguity (wrong bytes out, silently). Negative
    // heap offsets disqualify the fast path outright.
    uint64_t expect_off = 0;
    uint64_t first_heap = 0;
    bool have_first = false;
    for (long i = 0; i < num_rows && heap_contiguous; i++) {
        if (repeats[i] <= 0) {
            continue;
        }
        if (heap_offsets[i] < 0) {
            heap_contiguous = false;
            break;
        }
        const uint64_t off = static_cast<uint64_t>(heap_offsets[i]);
        const uint64_t eb = static_cast<uint64_t>(elem_bytes);
        if (eb == 0 || static_cast<uint64_t>(repeats[i]) > UINT64_MAX / eb) {
            heap_contiguous = false;
            break;
        }
        const uint64_t len = static_cast<uint64_t>(repeats[i]) * eb;
        if (off > UINT64_MAX - len) {
            heap_contiguous = false;
            break;
        }
        if (!have_first) {
            first_heap = off;
            expect_off = off + len;
            have_first = true;
        } else if (off != expect_off) {
            heap_contiguous = false;
        } else {
            expect_off = off + len;
        }
    }

    if (heap_contiguous && have_first &&
        first_heap <= static_cast<uint64_t>(std::numeric_limits<LONGLONG>::max())) {
        const LONGLONG absolute_offset = internal::checked_nonnegative_add<LONGLONG>(
            heap_base, static_cast<LONGLONG>(first_heap),
            "Invalid VLA geometry for column " + col.name + " pread offset");
        const off_t file_off = internal::checked_nonnegative_cast<off_t>(
            absolute_offset,
            "Invalid VLA geometry for column " + col.name + " pread offset");
        const size_t nbytes = internal::checked_nonnegative_multiply<size_t>(
            total_size, elem_bytes,
            "Invalid VLA geometry for column " + col.name + " pread size");
        if (internal::checked_nonnegative_add<LONGLONG>(
                absolute_offset,
                internal::checked_nonnegative_cast<LONGLONG>(
                    nbytes, "Invalid VLA geometry for column " + col.name + " pread size"),
                "Invalid VLA geometry for column " + col.name + " pread range") <= data_extent) {
            // The reader's own descriptor (open_data_fd), never the path.
            const int fd = data_fd_;
            if (fd >= 0) {
                struct stat file_stat;
                if (::fstat(fd, &file_stat) != 0 || file_stat.st_size < 0 ||
                    internal::checked_nonnegative_cast<LONGLONG>(
                        file_stat.st_size,
                        "Invalid VLA geometry for column " + col.name + " file size") <
                        internal::checked_nonnegative_add<LONGLONG>(
                            absolute_offset,
                            internal::checked_nonnegative_cast<LONGLONG>(
                                nbytes,
                                "Invalid VLA geometry for column " + col.name + " pread size"),
                            "Invalid VLA geometry for column " + col.name + " pread range")) {
                    throw std::runtime_error(
                        "Invalid VLA geometry for column " + col.name +
                        ": pread range exceeds file extent");
                }
                std::vector<uint8_t> raw(nbytes);
                size_t got = 0;
                bool ok = true;
                while (got < nbytes) {
                    const ssize_t n = ::pread(
                        fd, raw.data() + got, nbytes - got,
                        internal::checked_nonnegative_add<off_t>(
                            file_off,
                            internal::checked_nonnegative_cast<off_t>(
                                got,
                                "Invalid VLA geometry for column " + col.name +
                                    " pread progress"),
                            "Invalid VLA geometry for column " + col.name +
                                " pread progress"));
                    if (n < 0) {
                        if (errno == EINTR) continue;
                        ok = false;
                        break;
                    }
                    if (n == 0) {
                        ok = false;
                        break;
                    }
                    got += static_cast<size_t>(n);
                }
                if (ok) {
                    uint8_t* dst = static_cast<uint8_t*>(values.data_ptr());
                    if (elem_bytes == 1 || !host_is_little_endian()) {
                        std::memcpy(dst, raw.data(), nbytes);
                    } else if (elem_bytes == 2) {
                        for (size_t i = 0; i < static_cast<size_t>(total); i++) {
                            uint16_t v;
                            std::memcpy(&v, raw.data() + i * 2, 2);
                            v = bswap_16(v);
                            std::memcpy(dst + i * 2, &v, 2);
                        }
                    } else if (elem_bytes == 4) {
                        for (size_t i = 0; i < static_cast<size_t>(total); i++) {
                            uint32_t v;
                            std::memcpy(&v, raw.data() + i * 4, 4);
                            v = bswap_32(v);
                            std::memcpy(dst + i * 4, &v, 4);
                        }
                    } else {
                        for (size_t i = 0; i < static_cast<size_t>(total); i++) {
                            uint64_t v;
                            std::memcpy(&v, raw.data() + i * 8, 8);
                            v = bswap_64(v);
                            std::memcpy(dst + i * 8, &v, 8);
                        }
                    }
                    return finish();
                }
            }
        }
    }

    // Raw storage codes here as well: leaving CFITSIO auto-scaling on
    // truncates scaled physical values into the raw integer destination
    // (silent data loss), and the pread path above always reads raw.
    {
        int scale_status = 0;
        fits_set_tscale(fptr_, col_idx + 1, 1.0, 0.0, &scale_status);
    }

    int64_t cursor = 0;
    for (long i = 0; i < num_rows; i++) {
        LONGLONG rep = repeats[i];
        if (rep <= 0) {
            continue;
        }

        long row = start_row + i;
        int anynul = 0;
        status = 0;

        void* dst = nullptr;
        switch (values.scalar_type()) {
            case native::kBool:
                dst = static_cast<void*>(values.data_ptr<bool>() + cursor);
                break;
            case native::kUInt8:
                dst = static_cast<void*>(values.data_ptr<uint8_t>() + cursor);
                break;
            case native::kInt8:
                dst = static_cast<void*>(values.data_ptr<int8_t>() + cursor);
                break;
            case native::kInt16:
                dst = static_cast<void*>(values.data_ptr<int16_t>() + cursor);
                break;
            case native::kInt32:
                dst = static_cast<void*>(values.data_ptr<int32_t>() + cursor);
                break;
            case native::kInt64:
                dst = static_cast<void*>(values.data_ptr<int64_t>() + cursor);
                break;
            case native::kFloat32:
                dst = static_cast<void*>(values.data_ptr<float>() + cursor);
                break;
            case native::kFloat64:
                dst = static_cast<void*>(values.data_ptr<double>() + cursor);
                break;
            default:
                throw std::runtime_error("Unsupported VLA scalar type");
        }

        fits_read_col(
            fptr_, type_code, col_idx + 1, row, 1, rep, nullptr, dst, &anynul, &status
        );
        if (status != 0) {
            char err_msg[81];
            fits_get_errstatus(status, err_msg);
            throw std::runtime_error("Failed to read VLA data: " + std::string(err_msg));
        }
        cursor = internal::checked_nonnegative_add<int64_t>(
            cursor, internal::checked_nonnegative_cast<int64_t>(
                        rep, "Invalid VLA geometry for column " + col.name +
                                 " row " + std::to_string(row) + " repeat"),
            "Invalid VLA geometry for column " + col.name + " cursor");
    }

    return finish();
}

}  // namespace torchfits
