// TableReader::read_columns_mmap_filtered: windowed WHERE pushdown. The
// predicate is evaluated on raw big-endian cells of the mapped rows, and the
// matching rows of each output column are gathered with their byte swap.

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include "table_reader.h"
#include "hardware.h"

namespace torchfits {

// Filtered reading over the row window [start_row, start_row + num_rows)
// (1-based; num_rows = -1 means through the last row). The predicate is
// evaluated only inside the window, so a windowed scan maps and compares
// just its own rows; a window past the end clamps to the table.
std::vector<std::pair<std::string, native::Tensor>> TableReader::read_columns_mmap_filtered(
    const std::vector<std::string>& column_names,
    const std::vector<TableFilter>& filters,
    long start_row, long num_rows) {

    ensure_table_hdu();
    if (filters.empty()) {
        // Need to convert read_columns_mmap result (nb::dict) to map<string, Tensor>?
        // Or just throw error -> filters shouldn't be empty here if called correctly.
        // But for robustness:
         throw std::runtime_error("Filters cannot be empty in read_columns_mmap_filtered");
    }
    if (is_ascii_) {
        // ASCII tables store numeric fields as text; the mmap layout does
        // not apply. Route to the buffered/mask paths instead.
        throw std::runtime_error("ASCII tables are not supported for mmap filtered reads");
    }

    if (!direct_io_ok()) {
        throw std::runtime_error(
            "filtered mmap scans require an uncompressed local file; "
            "the caller falls back to the Arrow predicate engine");
    }
    // Map the reader's own descriptor (open_data_fd), never the path.
    const int fd = data_fd_;
    if (fd == -1) throw std::runtime_error("Failed to open file");

    struct stat sb;
    if (fstat(fd, &sb) == -1) { throw std::runtime_error("Failed to stat file"); }

    void* map_ptr = mmap(nullptr, sb.st_size, PROT_READ, MAP_SHARED, fd, 0);
    if (map_ptr == MAP_FAILED) { throw std::runtime_error("Failed to mmap"); }

    // Unmaps only; the descriptor belongs to the reader.
    MMapHandle guard(map_ptr, (size_t)sb.st_size, -1);

    uint8_t* base_ptr = static_cast<uint8_t*>(map_ptr);

    // fast-forward to data
    int status = 0;
    LONGLONG headstart, data_start, data_end;
    fits_get_hduaddrll(fptr_, &headstart, &data_start, &data_end, &status);
    if (status) throw std::runtime_error("Failed to get HDU address");

    uint8_t* data_ptr = base_ptr + data_start;
    if (start_row < 1) {
        throw std::runtime_error(
            "Invalid start row: " + std::to_string(start_row));
    }
    if (num_rows < -1) {
        throw std::runtime_error(
            "Invalid table row count: " + std::to_string(num_rows));
    }
    // Absolute 0-based row range [row_begin, row_end) of the window.
    const long row_begin = std::min<long>(start_row - 1, nrows_);
    const long row_end = (num_rows == -1 || num_rows > nrows_ - row_begin)
        ? nrows_
        : row_begin + num_rows;
    ensure_extent_within_file(sb.st_size, data_start, row_begin, row_end - row_begin);

    // Resolve filter columns
    struct FilterContext {
        const TableFilter* filter;
        const ColumnInfo* col_info;
        size_t offset;
        int col_idx = -1;
        // Cached types for fast switch
        bool is_float = false;
        bool is_double = false;
        bool is_int = false;
        bool is_long = false;
        bool is_short = false;
        bool is_byte = false;
        // Effective integer comparison target: the raw literal, offset by
        // TZERO for the uint16/uint32 FITS convention. Comparing against
        // this (in int64) yields the same predicate as the decoded-physical
        // value comparison without truncating the literal to the column's
        // storage width.
        int64_t target_i = 0;
        // True when target_i is representable in the column's integer
        // storage width; EQ/NE raw-byte compare is exact only then.
        bool target_fits_storage = true;
        // Effective operator: equal to filter->op except when a
        // non-integral float literal meets an integer column (see below).
        FilterOp op = FilterOp::EQ;
    };

    std::vector<FilterContext> ctxs;
    for (const auto& f : filters) {
        bool found = false;
        for (int ci = 0; ci < static_cast<int>(columns_.size()); ++ci) {
            const auto& c = columns_[ci];
            if (c.name == f.col_name) {
                FilterContext ctx;
                ctx.filter = &f;
                ctx.col_info = &c;
                ctx.offset = c.byte_offset;

                if (c.type == FITSColumnType::FLOAT) ctx.is_float = true;
                else if (c.type == FITSColumnType::DOUBLE) ctx.is_double = true;
                else if (c.type == FITSColumnType::INT) ctx.is_int = true;
                else if (c.type == FITSColumnType::LONG) ctx.is_long = true;
                else if (c.type == FITSColumnType::SHORT) ctx.is_short = true;
                else if (c.type == FITSColumnType::BYTE) ctx.is_byte = true;

                // Resolve TSCAL/TZERO so the scaled/unsigned state is
                // accurate, then reject what the raw-byte scan cannot do.
                ensure_column_scale(ci);
                if (c.scaled) {
                    throw std::runtime_error(
                        "Filtering on scaled columns is not supported for mmap filtered reads: " +
                        c.name);
                }
                if (!ctx.is_float && !ctx.is_double && !ctx.is_int &&
                    !ctx.is_long && !ctx.is_short && !ctx.is_byte) {
                    // LOGICAL/STRING/BIT/VLA predicates would silently match
                    // nothing — fail loudly instead of returning empty data.
                    throw std::runtime_error(
                        "Filtering on column type is not supported: " + c.name);
                }

                // Apply the uint16/uint32 FITS TZERO convention to the
                // literal so the raw signed storage bytes compare against
                // the physical value (see the scan below). EQ/NE raw-byte
                // compare is exact only when the effective target fits the
                // column's integer storage width; ordering comparisons are
                // promoted to int64 and never truncate.
                ctx.op = f.op;
                ctx.target_i = f.val_i;
                bool constant_predicate = false;
                bool u64_target_mapped = false;
                const bool is_u64_col =
                    c.is_unsigned_int && c.unsigned_target_type == native::kUInt64;
                if (f.out_of_int64 != 0 && !ctx.is_float && !ctx.is_double) {
                    // A literal outside int64 on an integer column: val_i is
                    // a clamped proxy (X < 2^63 + 0.5 must keep INT64_MAX).
                    if (f.has_u64 && is_u64_col) {
                        // Exact: raw int64 = physical XOR 2^63.
                        ctx.target_i = static_cast<int64_t>(
                            f.val_u ^ (uint64_t{1} << 63));
                        u64_target_mapped = true;
                    } else {
                        // Above (or below) every value the column holds.
                        const bool above = f.out_of_int64 > 0;
                        bool always = false;
                        switch (f.op) {
                            case FilterOp::GT: case FilterOp::GE: always = !above; break;
                            case FilterOp::LT: case FilterOp::LE: always = above; break;
                            case FilterOp::EQ: always = false; break;
                            case FilterOp::NE: always = true; break;
                        }
                        ctx.op = always ? FilterOp::GE : FilterOp::LT;
                        ctx.target_i = std::numeric_limits<int64_t>::min();
                        constant_predicate = true;
                    }
                } else if (f.type_idx == 0 && !ctx.is_float && !ctx.is_double &&
                    std::floor(f.val_d) != f.val_d) {
                    // A non-integral float literal on an integer column:
                    // val_i is the literal truncated toward zero, which
                    // makes `X >= 2.5` match 2. Rewrite to exact integer
                    // bounds (|val_d| < 2^52 here, so floor fits int64).
                    const auto lo = static_cast<int64_t>(std::floor(f.val_d));
                    switch (f.op) {
                        case FilterOp::GT:
                        case FilterOp::GE:
                            ctx.op = FilterOp::GT;  // X > 2.5  <=>  X > 2
                            ctx.target_i = lo;
                            break;
                        case FilterOp::LT:
                        case FilterOp::LE:
                            ctx.op = FilterOp::LT;  // X < 2.5  <=>  X < 3
                            ctx.target_i = lo + 1;
                            break;
                        case FilterOp::EQ:  // never true
                            ctx.op = FilterOp::LT;
                            ctx.target_i = std::numeric_limits<int64_t>::min();
                            constant_predicate = true;
                            break;
                        case FilterOp::NE:  // always true
                            ctx.op = FilterOp::GE;
                            ctx.target_i = std::numeric_limits<int64_t>::min();
                            constant_predicate = true;
                            break;
                    }
                }
                if (u64_target_mapped) {
                    // Already the raw int64 target.
                } else if (is_u64_col && !constant_predicate) {
                    // Raw int64 = physical XOR 2^63, which preserves order.
                    // Physical values are >= 0, so a negative literal makes
                    // the predicate constant; otherwise map it the same way.
                    if (ctx.target_i < 0) {
                        const bool always = ctx.op == FilterOp::GT ||
                                            ctx.op == FilterOp::GE ||
                                            ctx.op == FilterOp::NE;
                        ctx.op = always ? FilterOp::GE : FilterOp::LT;
                        ctx.target_i = std::numeric_limits<int64_t>::min();
                    } else {
                        ctx.target_i += std::numeric_limits<int64_t>::min();
                    }
                } else if (c.is_unsigned_int && !constant_predicate) {
                    ctx.target_i -= c.unsigned_offset;
                }
                if (ctx.is_short) {
                    ctx.target_fits_storage =
                        ctx.target_i >= -32768 && ctx.target_i <= 32767;
                } else if (ctx.is_int) {
                    ctx.target_fits_storage =
                        ctx.target_i >= -2147483648LL &&
                        ctx.target_i <= 2147483647LL;
                }

                ctx.col_idx = ci;
                ctxs.push_back(ctx);
                found = true;
                break;
            }
        }
        if (!found) throw std::runtime_error("Filter column not found: " + f.col_name);
    }

    // Resolve the output columns before scanning, so an unsupported one
    // fails before the predicate pass rather than after it.
    std::vector<int> out_col_indices;
    if (column_names.empty()) {
        for(int i=0; i<ncols_; ++i) out_col_indices.push_back(i);
    } else {
         for(const auto& name : column_names) {
             bool found = false;
             for(int i=0; i<ncols_; ++i) {
                 if(columns_[i].name == name) {
                     // Repeats resolve once: the assembly loop moves each
                     // entry out (moved-from tensor on a duplicated index).
                     if (std::find(out_col_indices.begin(), out_col_indices.end(), i) == out_col_indices.end()) {
                         out_col_indices.push_back(i);
                     }
                     found = true;
                     break;
                 }
             }
             // Unknown output columns fail loudly here too — a silent
             // skip returns wrong-shaped data for a non-empty request.
             if (!found) throw std::runtime_error("Column not found: " + name);
         }
    }

    // Resolve TSCAL/TZERO for gathered columns: the unsigned-int post-pass
    // below relies on unsigned_offset being populated, and scaled (non-
    // unsigned) columns cannot be gathered from raw storage bytes.
    for (int col_idx : out_col_indices) {
        ensure_column_scale(col_idx);
        const auto& col = columns_[col_idx];
        if (col.scaled && !col.is_unsigned_int) {
            throw std::runtime_error(
                "Scaled columns not supported for mmap filtered reads: " + col.name);
        }
        // A VLA column has no fixed cell to gather; dropping it from the
        // result would hand back fewer columns than were asked for.
        if (col.type == FITSColumnType::VARIABLE) {
            throw std::runtime_error(
                "VLA columns not supported for mmap filtered reads: " + col.name);
        }
    }

    // Hints: sequential access pattern for kernel prefetch
#if defined(POSIX_MADV_SEQUENTIAL)
    if (row_end > row_begin) {
        const uint64_t hint_len = static_cast<uint64_t>(row_end - row_begin) *
                                  static_cast<uint64_t>(row_width_bytes_);
        const size_t byte_len = hint_len > SIZE_MAX ? SIZE_MAX : static_cast<size_t>(hint_len);
        posix_madvise(
            const_cast<uint8_t*>(
                data_ptr + static_cast<size_t>(row_begin) *
                               static_cast<size_t>(row_width_bytes_)),
            byte_len, POSIX_MADV_SEQUENTIAL);
    }
#endif

    // Scan rows
    std::vector<long> valid_indices;

    if (ctxs.size() == 1) {
        const auto& ctx = ctxs[0];
        const size_t offset = ctx.offset;
        const FilterOp op = ctx.op;

        // Pre-byte-swap the target so we can compare raw FITS bytes directly,
        // eliminating per-row bswap instructions for EQ/NE on integers.
        // Uses the offset-adjusted target (ctx.target_i), which for unsigned
        // columns is the physical literal minus TZERO — the value the raw
        // signed storage bytes actually encode.
        auto pre_swapped_target_int = [&ctx]() -> int64_t {
            int64_t t = ctx.target_i;
            // The raw FITS bytes are big-endian; pre-swap the target to match.
            // For 2-byte values, we need the 16-bit-swapped value in the low 16 bits.
            if (ctx.is_short) {
                // FITS short is big-endian int16; pre-swap to host endianness once.
                return (int64_t)(int16_t)bswap_16((uint16_t)(int16_t)t);
            }
            if (ctx.is_int) {
                return (int64_t)(int32_t)bswap_32((uint32_t)(int32_t)t);
            }
            if (ctx.is_long) {
                return (int64_t)bswap_64((uint64_t)t);
            }
            // byte/float/double — not pre-swapped (compared as decoded values below)
            return t;
        }();

        valid_indices.reserve(row_end - row_begin);

        // Scan body. Called exactly once, sequentially, over the whole row
        // range -- see the call site below for why a parallel chunk + merge
        // was measured and dropped. `out` therefore receives indices in
        // ascending order, which the result assembly relies on; a parallel
        // dispatch would have to restore that order itself.
        auto scan_chunk = [&](long start, long end, std::vector<long>& out) {
            if (ctx.is_int) {
                if (op == FilterOp::EQ || op == FilterOp::NE) {
                    // Raw-byte EQ/NE is exact only when the literal fits
                    // int32 storage; an out-of-range literal can never equal
                    // (EQ) / always differs from (NE) any int32 cell.
                    if (!ctx.target_fits_storage) {
                        if (op == FilterOp::NE) {
                            for (long i = start; i < end; i++) out.push_back(i);
                        }
                    } else {
                        const int32_t pre_swapped = (int32_t)pre_swapped_target_int;
                        if (op == FilterOp::EQ) {
                            for (long i = start; i < end; i++) {
                                const uint8_t* val_ptr = data_ptr + i * row_width_bytes_ + offset;
                                uint32_t raw; std::memcpy(&raw, val_ptr, 4);
                                if (raw == (uint32_t)pre_swapped) out.push_back(i);
                            }
                        } else {
                            for (long i = start; i < end; i++) {
                                const uint8_t* val_ptr = data_ptr + i * row_width_bytes_ + offset;
                                uint32_t raw; std::memcpy(&raw, val_ptr, 4);
                                if (raw != (uint32_t)pre_swapped) out.push_back(i);
                            }
                        }
                    }
                } else {
                    // Ordering: promote both sides to int64 so an out-of-range
                    // literal and unsigned offsets compare exactly (no int32
                    // truncation of the filter literal).
                    const int64_t target = ctx.target_i;
                    for (long i = start; i < end; i++) {
                        const uint8_t* val_ptr = data_ptr + i * row_width_bytes_ + offset;
                        uint32_t tmp; std::memcpy(&tmp, val_ptr, 4);
                        const int64_t val = (int64_t)(int32_t)bswap_32(tmp);
                        bool match = false;
                        switch (op) {
                            case FilterOp::GT: match = (val > target); break;
                            case FilterOp::LT: match = (val < target); break;
                            case FilterOp::GE: match = (val >= target); break;
                            case FilterOp::LE: match = (val <= target); break;
                            default: break;
                        }
                        if (match) out.push_back(i);
                    }
                }
            } else if (ctx.is_short) {
                if (op == FilterOp::EQ || op == FilterOp::NE) {
                    // See the is_int branch: out-of-range literals make EQ
                    // vacuously false and NE vacuously true.
                    if (!ctx.target_fits_storage) {
                        if (op == FilterOp::NE) {
                            for (long i = start; i < end; i++) out.push_back(i);
                        }
                    } else {
                        const int16_t pre_swapped = (int16_t)pre_swapped_target_int;
                        if (op == FilterOp::EQ) {
                            for (long i = start; i < end; i++) {
                                const uint8_t* val_ptr = data_ptr + i * row_width_bytes_ + offset;
                                uint16_t raw; std::memcpy(&raw, val_ptr, 2);
                                if (raw == (uint16_t)pre_swapped) out.push_back(i);
                            }
                        } else {
                            for (long i = start; i < end; i++) {
                                const uint8_t* val_ptr = data_ptr + i * row_width_bytes_ + offset;
                                uint16_t raw; std::memcpy(&raw, val_ptr, 2);
                                if (raw != (uint16_t)pre_swapped) out.push_back(i);
                            }
                        }
                    }
                } else {
                    const int64_t target = ctx.target_i;
                    for (long i = start; i < end; i++) {
                        const uint8_t* val_ptr = data_ptr + i * row_width_bytes_ + offset;
                        uint16_t tmp; std::memcpy(&tmp, val_ptr, 2);
                        const int64_t val = (int64_t)(int16_t)bswap_16(tmp);
                        bool match = false;
                        switch (op) {
                            case FilterOp::GT: match = (val > target); break;
                            case FilterOp::LT: match = (val < target); break;
                            case FilterOp::GE: match = (val >= target); break;
                            case FilterOp::LE: match = (val <= target); break;
                            default: break;
                        }
                        if (match) out.push_back(i);
                    }
                }
            } else if (ctx.is_long) {
                // int64 storage always fits the literal (val_i is int64); the
                // uint16/uint32 TZERO convention never applies to LONG.
                const int64_t pre_swapped = pre_swapped_target_int;
                if (op == FilterOp::EQ) {
                    for (long i = start; i < end; i++) {
                        const uint8_t* val_ptr = data_ptr + i * row_width_bytes_ + offset;
                        uint64_t raw; std::memcpy(&raw, val_ptr, 8);
                        if (raw == (uint64_t)pre_swapped) out.push_back(i);
                    }
                } else if (op == FilterOp::NE) {
                    for (long i = start; i < end; i++) {
                        const uint8_t* val_ptr = data_ptr + i * row_width_bytes_ + offset;
                        uint64_t raw; std::memcpy(&raw, val_ptr, 8);
                        if (raw != (uint64_t)pre_swapped) out.push_back(i);
                    }
                } else {
                    const int64_t target = ctx.target_i;
                    for (long i = start; i < end; i++) {
                        const uint8_t* val_ptr = data_ptr + i * row_width_bytes_ + offset;
                        uint64_t tmp; std::memcpy(&tmp, val_ptr, 8);
                        const int64_t val = (int64_t)bswap_64(tmp);
                        bool match = false;
                        switch (op) {
                            case FilterOp::GT: match = (val > target); break;
                            case FilterOp::LT: match = (val < target); break;
                            case FilterOp::GE: match = (val >= target); break;
                            case FilterOp::LE: match = (val <= target); break;
                            default: break;
                        }
                        if (match) out.push_back(i);
                    }
                }
            } else if (ctx.is_float) {
                const float target = (float)ctx.filter->val_d;
                for (long i = start; i < end; i++) {
                    uint8_t* val_ptr = data_ptr + i * row_width_bytes_ + offset;
                    uint32_t tmp; std::memcpy(&tmp, val_ptr, 4);
                    tmp = bswap_32(tmp);
                    float val; std::memcpy(&val, &tmp, 4);
                    bool match = false;
                    switch (op) {
                        case FilterOp::EQ: match = (val == target); break;
                        // IEEE NaN != x is true; SQL leaves that comparison unknown.
                        case FilterOp::NE: match = (val != target) && (val == val); break;
                        case FilterOp::GT: match = (val > target); break;
                        case FilterOp::LT: match = (val < target); break;
                        case FilterOp::GE: match = (val >= target); break;
                        case FilterOp::LE: match = (val <= target); break;
                    }
                    if (match) out.push_back(i);
                }
            } else if (ctx.is_double) {
                const double target = ctx.filter->val_d;
                for (long i = start; i < end; i++) {
                    uint8_t* val_ptr = data_ptr + i * row_width_bytes_ + offset;
                    uint64_t tmp; std::memcpy(&tmp, val_ptr, 8);
                    tmp = bswap_64(tmp);
                    double val; std::memcpy(&val, &tmp, 8);
                    bool match = false;
                    switch (op) {
                        case FilterOp::EQ: match = (val == target); break;
                        case FilterOp::NE: match = (val != target) && (val == val); break;
                        case FilterOp::GT: match = (val > target); break;
                        case FilterOp::LT: match = (val < target); break;
                        case FilterOp::GE: match = (val >= target); break;
                        case FilterOp::LE: match = (val <= target); break;
                    }
                    if (match) out.push_back(i);
                }
            } else {
                // Fallback for byte type
                for (long i = start; i < end; i++) {
                    uint8_t* row_ptr = data_ptr + i * row_width_bytes_;
                    uint8_t* val_ptr = row_ptr + ctx.offset;
                    bool match = false;
                    if (ctx.is_byte) {
                        uint8_t val = *val_ptr;
                        int64_t target = ctx.target_i;
                        switch (op) {
                            case FilterOp::EQ: match = (val == target); break;
                            case FilterOp::NE: match = (val != target); break;
                            case FilterOp::GT: match = (val > target); break;
                            case FilterOp::LT: match = (val < target); break;
                            case FilterOp::GE: match = (val >= target); break;
                            case FilterOp::LE: match = (val <= target); break;
                        }
                    }
                    if (match) out.push_back(i);
                }
            }
        };

        // Sequential scan: parallel chunk + mutex merge + sort loses on
        // typical predicate widths vs a single pass (no N threshold).
        scan_chunk(row_begin, row_end, valid_indices);
    } else {
        // Multi-filter fallback (single-threaded, uncommon case)
        std::vector<long> local;
        local.reserve(row_end - row_begin);
        for (long i = row_begin; i < row_end; i++) {
            uint8_t* row_ptr = data_ptr + i * row_width_bytes_;
            bool row_match = true;
            for (const auto& ctx : ctxs) {
                uint8_t* val_ptr = row_ptr + ctx.offset;
                bool match = false;
                if (ctx.is_double) {
                    uint64_t tmp; std::memcpy(&tmp, val_ptr, 8); tmp = bswap_64(tmp);
                    double val; std::memcpy(&val, &tmp, 8);
                    double target = ctx.filter->val_d;
                    switch (ctx.op) {
                       case FilterOp::EQ: match = (val == target); break;
                       case FilterOp::NE: match = (val != target) && (val == val); break;
                       case FilterOp::GT: match = (val > target); break;
                       case FilterOp::LT: match = (val < target); break;
                       case FilterOp::GE: match = (val >= target); break;
                       case FilterOp::LE: match = (val <= target); break;
                    }
                } else if (ctx.is_float) {
                    uint32_t tmp; std::memcpy(&tmp, val_ptr, 4); tmp = bswap_32(tmp);
                    float val; std::memcpy(&val, &tmp, 4);
                    float target = (float)ctx.filter->val_d;
                    switch (ctx.op) {
                       case FilterOp::EQ: match = (val == target); break;
                       case FilterOp::NE: match = (val != target) && (val == val); break;
                       case FilterOp::GT: match = (val > target); break;
                       case FilterOp::LT: match = (val < target); break;
                       case FilterOp::GE: match = (val >= target); break;
                       case FilterOp::LE: match = (val <= target); break;
                    }
                } else if (ctx.is_long) {
                   uint64_t tmp; memcpy(&tmp, val_ptr, 8);
                   int64_t val = (int64_t)bswap_64(tmp);
                   int64_t target = ctx.target_i;
                   switch (ctx.op) {
                       case FilterOp::EQ: match = (val == target); break;
                       case FilterOp::NE: match = (val != target); break;
                       case FilterOp::GT: match = (val > target); break;
                       case FilterOp::LT: match = (val < target); break;
                       case FilterOp::GE: match = (val >= target); break;
                       case FilterOp::LE: match = (val <= target); break;
                   }
                } else if (ctx.is_int) {
                   uint32_t tmp; memcpy(&tmp, val_ptr, 4);
                   int32_t val = (int32_t)bswap_32(tmp);
                   int64_t target = ctx.target_i;
                   switch (ctx.op) {
                       case FilterOp::EQ: match = (val == target); break;
                       case FilterOp::NE: match = (val != target); break;
                       case FilterOp::GT: match = (val > target); break;
                       case FilterOp::LT: match = (val < target); break;
                       case FilterOp::GE: match = (val >= target); break;
                       case FilterOp::LE: match = (val <= target); break;
                   }
                } else if (ctx.is_short) {
                   uint16_t tmp; memcpy(&tmp, val_ptr, 2);
                   int16_t val = (int16_t)bswap_16(tmp);
                   int64_t target = ctx.target_i;
                   switch (ctx.op) {
                       case FilterOp::EQ: match = (val == target); break;
                       case FilterOp::NE: match = (val != target); break;
                       case FilterOp::GT: match = (val > target); break;
                       case FilterOp::LT: match = (val < target); break;
                       case FilterOp::GE: match = (val >= target); break;
                       case FilterOp::LE: match = (val <= target); break;
                   }
                } else if (ctx.is_byte) {
                   uint8_t val = *val_ptr;
                   int64_t target = ctx.target_i;
                   switch (ctx.op) {
                       case FilterOp::EQ: match = (val == target); break;
                       case FilterOp::NE: match = (val != target); break;
                       case FilterOp::GT: match = (val > target); break;
                       case FilterOp::LT: match = (val < target); break;
                       case FilterOp::GE: match = (val >= target); break;
                       case FilterOp::LE: match = (val <= target); break;
                   }
                }
                if (!match) { row_match = false; break; }
            }
            if (row_match) local.push_back(i);
        }
        valid_indices = std::move(local);
    }

    // Gather results (num_valid==0 → empty tensors, keyed schema preserved)
    std::unordered_map<int, native::Tensor> result;
    long num_valid = valid_indices.size();

    for (int col_idx : out_col_indices) {
        const auto& col = columns_[col_idx];

        // Allocate output tensor
        std::vector<int64_t> shape;
        shape.push_back(num_valid);
         if (col.type == FITSColumnType::STRING) {
            shape.push_back(is_ascii_ ? col.width : col.repeat);
        } else if (col.repeat > 1) {
            shape.push_back(col.repeat);
        }

        auto options = native::TensorOptions().dtype(
            col.type == FITSColumnType::BIT ? native::kBool : col.torch_type);
        native::Tensor out_tensor = native::empty(shape, options);

        if (num_valid == 0) {
            result[col_idx] = out_tensor;
            continue;
        }

        // item_size is the byte-swap unit; a complex cell is two of them
        // (real, imaginary), each swapped on its own.
        int item_size = 0;
        int items_per_value = 1;
        if (col.type == FITSColumnType::DOUBLE || col.type == FITSColumnType::LONG) item_size = 8;
        else if (col.type == FITSColumnType::FLOAT || col.type == FITSColumnType::INT) item_size = 4;
        else if (col.type == FITSColumnType::SHORT) item_size = 2;
        else if (col.type == FITSColumnType::COMPLEX_FLOAT) { item_size = 4; items_per_value = 2; }
        else if (col.type == FITSColumnType::COMPLEX_DOUBLE) { item_size = 8; items_per_value = 2; }
        else item_size = 1;

        // Cell size in storage bytes. BIT columns are bit-packed, so the
        // repeat (bit count) must not be used as a byte count — that would
        // over-read into adjacent columns / past EOF.
        size_t cell_size = 0;
        if (col.type == FITSColumnType::STRING) {
            cell_size = static_cast<size_t>(is_ascii_ ? col.width : col.repeat);
        } else if (col.type == FITSColumnType::BIT) {
            cell_size = (static_cast<size_t>(col.repeat) + 7) / 8;
        } else {
            cell_size = static_cast<size_t>(item_size) * static_cast<size_t>(items_per_value) *
                        static_cast<size_t>(std::max(1, col.repeat));
        }
        uint8_t* out_ptr = (uint8_t*)out_tensor.data_ptr();

        // LOGICAL: storage is ASCII 'T'/'F' — raw memcpy into a bool tensor
        // would decode both as true (both nonzero).
        // Every element of a vector cell (rL), as read_columns_mmap does.
        if (col.type == FITSColumnType::LOGICAL) {
            bool* out_bool = static_cast<bool*>(out_tensor.data_ptr());
            const long repeat = std::max(1L, (long)col.repeat);
            native::parallel_for(0, num_valid, 2048, [&](long start, long end) {
                for (long k = start; k < end; k++) {
                    const uint8_t* src =
                        data_ptr + valid_indices[k] * row_width_bytes_ + col.byte_offset;
                    bool* row_out = out_bool + k * repeat;
                    for (long j = 0; j < repeat; j++) {
                        row_out[j] = (src[j] == 'T' || src[j] == '1' || src[j] == 1);
                    }
                }
            });
            result[col_idx] = out_tensor;
            continue;
        }

        // BIT: unpack MSB-first bits into bools (mirrors read_columns_mmap).
        if (col.type == FITSColumnType::BIT) {
            bool* out_bool = static_cast<bool*>(out_tensor.data_ptr());
            const long repeat = std::max(1L, (long)col.repeat);
            native::parallel_for(0, num_valid, 2048, [&](long start, long end) {
                for (long k = start; k < end; k++) {
                    const uint8_t* row_in =
                        data_ptr + valid_indices[k] * row_width_bytes_ + col.byte_offset;
                    bool* row_out = out_bool + k * repeat;
                    for (long j = 0; j < repeat; j++) {
                        const uint8_t packed = row_in[j / 8];
                        row_out[j] = ((packed >> (7 - (j % 8))) & 0x1U) != 0;
                    }
                }
            });
            result[col_idx] = out_tensor;
            continue;
        }

        // Parallel gathering with contiguous block detection.
        // Each thread processes a disjoint range of valid_indices and writes
        // to output at deterministic offsets, so no synchronization needed.
        native::parallel_for(0, num_valid, 2048, [&](long start, long end) {
            long k = start;
            while (k < end) {
                long run_start_k = k;
                long run_start_row = valid_indices[k];
                // Extend run while rows are consecutive, but stop at the
                // thread-chunk boundary to avoid overlapping with another thread.
                while (k + 1 < end && valid_indices[k+1] == valid_indices[k] + 1) {
                    k++;
                }
                long run_len = k - run_start_k + 1;

                const uint8_t* src_base = data_ptr + run_start_row * row_width_bytes_ + col.byte_offset;
                uint8_t* dst_base = out_ptr + run_start_k * cell_size;

                if (item_size == 1 && row_width_bytes_ == cell_size) {
                    std::memcpy(dst_base, src_base, run_len * cell_size);
                } else {
                    for (long r = 0; r < run_len; ++r) {
                        const uint8_t* src = src_base + r * row_width_bytes_;
                        uint8_t* dst = dst_base + r * cell_size;

                        if (item_size == 1) {
                            std::memcpy(dst, src, cell_size);
                        } else if (item_size == 2) {
                            int n_items = cell_size / 2;
                            uint16_t* d = (uint16_t*)dst;
                            for(int j=0; j<n_items; ++j) {
                                uint16_t val; std::memcpy(&val, src + j*2, 2);
                                d[j] = bswap_16(val);
                            }
                        } else if (item_size == 4) {
                            int n_items = cell_size / 4;
                            uint32_t* d = (uint32_t*)dst;
                            for(int j=0; j<n_items; ++j) {
                                uint32_t val; std::memcpy(&val, src + j*4, 4);
                                d[j] = bswap_32(val);
                            }
                        } else if (item_size == 8) {
                            int n_items = cell_size / 8;
                            uint64_t* d = (uint64_t*)dst;
                            for(int j=0; j<n_items; ++j) {
                                uint64_t val; std::memcpy(&val, src + j*8, 8);
                                d[j] = bswap_64(val);
                            }
                        }
                    }
                }
                k++;
            }
        });

        // Apply unsigned integer offset for uint16/uint32 FITS convention.
        if (col.is_unsigned_int) {
            out_tensor = apply_unsigned_offset(
                std::move(out_tensor), col.unsigned_target_type);
        }
        result[col_idx] = out_tensor;
    }

    // Order by file/request column index (see read_columns signature comment).
    std::vector<std::pair<std::string, native::Tensor>> ordered;
    ordered.reserve(out_col_indices.size());
    for (int col_idx : out_col_indices) {
        auto it = result.find(col_idx);
        if (it != result.end()) {
            ordered.emplace_back(columns_[col_idx].name, std::move(it->second));
        }
    }
    return ordered;
}

}  // namespace torchfits
