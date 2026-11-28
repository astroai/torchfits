#pragma once
// Row-to-column gathers shared by the TableReader paths that copy fixed-width
// cells out of whole big-endian rows (tiled pread, fits_read_tblbytes chunks,
// mmap), and the allocation of their output columns.

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "checked_arithmetic.h"
#include "fits_detail.h"
#include "hardware.h"
#include "internal_utils.h"
#include "native_tensor.h"
#include "table_types.h"

namespace torchfits {
namespace table_gather {

// How one fixed-width column is copied out of whole big-endian rows into its
// dense destination. Resolved (and its geometry checked) once per read, so
// the per-tile and per-chunk gathers below are only the copy loops.
struct ColumnGather {
    enum class Kind {
        Logical,  // 'T'/'F' (or '1'/'0') bytes -> bool
        Bytes,    // raw bytes: B, A, signed bytes (sign bit flipped after)
        Swap2,    // big-endian 2/4/8-byte elements
        Swap4,
        Swap8,
    };
    Kind kind = Kind::Bytes;
    size_t offset = 0;      // column offset within a row
    size_t row_stride = 0;  // bytes per row of the source
    size_t repeat = 0;      // elements per cell
    size_t cell_bytes = 0;  // bytes per cell (source and destination)
    bool flip_sign_bit = false;
};

inline ColumnGather plan_column_gather(const ColumnInfo& col, long row_width_bytes) {
    // int->size_t: width/repeat come from TFORM cards and must be positive;
    // a negative would wrap huge and drive wild copy lengths below.
    if (col.width <= 0 || col.repeat <= 0 || row_width_bytes <= 0) {
        throw std::runtime_error("extract_column_data: invalid column geometry");
    }
    ColumnGather g;
    const size_t col_width = static_cast<size_t>(col.width);  // bytes per element
    g.cell_bytes = internal::checked_nonnegative_cast<size_t>(
        col.storage_bytes, "extract_column_data: invalid column storage");
    g.row_stride = static_cast<size_t>(row_width_bytes);
    g.offset = internal::checked_nonnegative_cast<size_t>(
        col.byte_offset, "extract_column_data: invalid column offset");
    g.repeat = static_cast<size_t>(col.repeat);
    if (g.offset > g.row_stride || g.cell_bytes > g.row_stride - g.offset) {
        throw std::runtime_error("extract_column_data: column exceeds the row");
    }
    if (col.type == FITSColumnType::LOGICAL) {
        g.kind = ColumnGather::Kind::Logical;
    } else if (col.type == FITSColumnType::STRING || col.type == FITSColumnType::BYTE) {
        g.kind = ColumnGather::Kind::Bytes;
        g.flip_sign_bit = col.type == FITSColumnType::BYTE && col.fits_typecode == TSBYTE;
    } else if (!host_is_little_endian()) {
        // FITS is big-endian: a big-endian host copies the cells as they are.
        g.kind = ColumnGather::Kind::Bytes;
    } else if (col_width == 2 && g.cell_bytes == 2 * g.repeat) {
        g.kind = ColumnGather::Kind::Swap2;
    } else if (col_width == 4 && g.cell_bytes == 4 * g.repeat) {
        g.kind = ColumnGather::Kind::Swap4;
    } else if (col_width == 8 && g.cell_bytes == 8 * g.repeat) {
        g.kind = ColumnGather::Kind::Swap8;
    } else {
        // Not reached for the types the buffered path takes; copy verbatim.
        g.kind = ColumnGather::Kind::Bytes;
    }
    return g;
}

template <size_t W> struct UIntOf;
template <> struct UIntOf<1> { using type = uint8_t; };
template <> struct UIntOf<2> { using type = uint16_t; };
template <> struct UIntOf<4> { using type = uint32_t; };
template <> struct UIntOf<8> { using type = uint64_t; };

template <size_t W>
inline typename UIntOf<W>::type swap_bytes(typename UIntOf<W>::type v) {
    if constexpr (W == 2) {
        return __builtin_bswap16(v);
    } else if constexpr (W == 4) {
        return __builtin_bswap32(v);
    } else if constexpr (W == 8) {
        return __builtin_bswap64(v);
    } else {
        return v;
    }
}

// Copy W-byte elements of rows [begin, end) out of the strided source cells
// into the dense destination, byte-swapping them when W > 1. W is a
// compile-time constant so every load and store is a single move; the old
// per-row memcpy with a run-time length was a libc call per cell (a 1-byte
// column of 1M rows cost ~3.5 ms on one core).
template <size_t W>
inline void gather_elements(const uint8_t* src, size_t row_stride, size_t repeat,
                     uint8_t* dst, long begin, long end) {
    using U = typename UIntOf<W>::type;
    const size_t cell = W * repeat;
    const size_t b = static_cast<size_t>(begin);
    const size_t n = static_cast<size_t>(end - begin);
    const uint8_t* s = src + b * row_stride;
    uint8_t* d = dst + b * cell;
    if (row_stride == cell) {
        // The rows hold nothing but this column: one dense block.
        if constexpr (W == 1) {
            std::memcpy(d, s, n * cell);
        } else if constexpr (W == 2) {
            internal::bswap16_copy(s, d, n * repeat);
        } else if constexpr (W == 4) {
            internal::bswap32_copy(s, d, n * repeat);
        } else {
            internal::bswap64_copy(s, d, n * repeat);
        }
        return;
    }
    if (repeat == 1) {
        for (size_t i = 0; i < n; ++i) {
            U v;
            std::memcpy(&v, s + i * row_stride, W);
            v = swap_bytes<W>(v);
            std::memcpy(d + i * W, &v, W);
        }
        return;
    }
    if constexpr (W == 1) {
        // Strings and byte vectors: one block copy per cell.
        for (size_t i = 0; i < n; ++i) {
            std::memcpy(d + i * cell, s + i * row_stride, cell);
        }
    } else {
        for (size_t i = 0; i < n; ++i) {
            const uint8_t* sc = s + i * row_stride;
            uint8_t* dc = d + i * cell;
            for (size_t j = 0; j < repeat; ++j) {
                U v;
                std::memcpy(&v, sc + j * W, W);
                v = swap_bytes<W>(v);
                std::memcpy(dc + j * W, &v, W);
            }
        }
    }
}

// Gather rows [begin, end) of one column out of `rows` (whole rows, the
// first at row 0) into dest, whose row 0 is the window's row 0.
inline void run_column_gather(const ColumnGather& g, const uint8_t* rows, long begin, long end,
                       uint8_t* dest) {
    if (end <= begin) {
        return;
    }
    const uint8_t* src = rows + g.offset;
    switch (g.kind) {
        case ColumnGather::Kind::Logical: {
            bool* out = reinterpret_cast<bool*>(dest);
            for (long i = begin; i < end; i++) {
                const uint8_t* src_cell = src + static_cast<size_t>(i) * g.row_stride;
                bool* out_cell = out + static_cast<size_t>(i) * g.repeat;
                for (size_t j = 0; j < g.repeat; j++) {
                    const uint8_t v = src_cell[j];
                    out_cell[j] = (v == 'T' || v == '1' || v == 1);
                }
            }
            return;
        }
        case ColumnGather::Kind::Bytes:
            // cell_bytes, not repeat: a big-endian host copies wider cells
            // verbatim through this branch too.
            gather_elements<1>(src, g.row_stride, g.cell_bytes, dest, begin, end);
            if (g.flip_sign_bit) {
                detail::xor_sign_bit_u8(
                    dest + static_cast<size_t>(begin) * g.cell_bytes,
                    static_cast<size_t>(end - begin) * g.cell_bytes);
            }
            return;
        case ColumnGather::Kind::Swap2:
            gather_elements<2>(src, g.row_stride, g.repeat, dest, begin, end);
            return;
        case ColumnGather::Kind::Swap4:
            gather_elements<4>(src, g.row_stride, g.repeat, dest, begin, end);
            return;
        case ColumnGather::Kind::Swap8:
            gather_elements<8>(src, g.row_stride, g.repeat, dest, begin, end);
            return;
    }
}

// Allocate output tensors for one read. When they are at most
// kSharedOutputBytes together they are carved out of a single allocation
// (64-byte aligned slices, each tensor keeping the block alive). Separate
// buffers of a few MiB each are handed back to the OS on free -- glibc trims
// a free heap top larger than twice its largest recent mmap'd chunk -- so
// every read refaults them: ~2k page faults, about half the time of a
// full read of a 1M-row, 13-byte-row table. One block of the total size
// stays reusable, as one record array (fitsio) does. Larger outputs are
// mmap'd and faulted fresh either way, and get one buffer per tensor.
inline std::vector<native::Tensor> allocate_outputs(
    const std::vector<std::pair<std::vector<int64_t>, native::ScalarType>>& specs) {
    constexpr size_t kSharedOutputBytes = size_t{32} << 20;
    constexpr size_t kAlign = 64;
    std::vector<native::Tensor> out;
    out.reserve(specs.size());
    std::vector<size_t> offsets;
    size_t total = 0;
    if (specs.size() > 1) {
        offsets.reserve(specs.size());
        for (const auto& [shape, dtype] : specs) {
            size_t nbytes = native::element_size(dtype);
            bool fits = true;
            for (int64_t d : shape) {
                if (d < 0 || (d != 0 && nbytes > kSharedOutputBytes / static_cast<size_t>(d))) {
                    fits = false;
                    break;
                }
                nbytes *= static_cast<size_t>(d);
            }
            if (!fits || nbytes > kSharedOutputBytes - total) {
                break;
            }
            offsets.push_back(total);
            total += (nbytes + kAlign - 1) / kAlign * kAlign;
        }
    }
    if (offsets.size() == specs.size() && total > 0 && total <= kSharedOutputBytes) {
        std::shared_ptr<void> block = native::detail::allocate(total);
        auto* base = static_cast<std::byte*>(block.get());
        for (size_t k = 0; k < specs.size(); ++k) {
            out.push_back(native::Tensor::view_of(
                block, base + offsets[k], specs[k].first, specs[k].second));
        }
        return out;
    }
    for (const auto& [shape, dtype] : specs) {
        out.push_back(native::empty(shape, native::TensorOptions().dtype(dtype)));
    }
    return out;
}

}  // namespace table_gather
}  // namespace torchfits
