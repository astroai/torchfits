// TableReader buffered fixed-width reads: whole rows are read (tiled
// parallel pread on a raw fd, else fits_read_tblbytes chunks) and every
// requested column is gathered out of them with its byte swap.

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstring>
#include <stdexcept>
#include <fcntl.h>
#include <unistd.h>

#include "table_reader.h"
#include "checked_arithmetic.h"
#include "fits_detail.h"
#include "hardware.h"
#include "core/parallel.h"
#include "table_gather.h"

namespace torchfits {

using table_gather::ColumnGather;
using table_gather::plan_column_gather;
using table_gather::run_column_gather;

namespace {

void* get_tensor_data_ptr(native::Tensor& tensor, long offset) {
    switch (tensor.scalar_type()) {
        case native::kBool: return tensor.data_ptr<bool>() + offset;
        case native::kUInt8: return tensor.data_ptr<uint8_t>() + offset;
        case native::kInt8: return tensor.data_ptr<int8_t>() + offset;
        case native::kInt16: return tensor.data_ptr<int16_t>() + offset;
        case native::kInt32: return tensor.data_ptr<int32_t>() + offset;
        case native::kInt64: return tensor.data_ptr<int64_t>() + offset;
        case native::kFloat32: return tensor.data_ptr<float>() + offset;
        case native::kFloat64: return tensor.data_ptr<double>() + offset;
        default: throw std::runtime_error("Unsupported tensor type");
    }
}

}  // namespace

// Buffered reading implementation
void TableReader::read_columns_buffered(
    const std::vector<int>& col_indices,
    long start_row, long num_rows,
    std::unordered_map<int, ColumnData>& result) {

    // Prefer direct pread of the table heap over fits_read_tblbytes when we
    // have a local path (same bytes, less CFITSIO overhead on cold opens).
    // The data offset is cached: fits_get_hduaddrll touches ~12 MiB of
    // anonymous memory on every call (kernel rusage shows ~2930 minor
    // faults per read), which dominates the pread cost on a busy host.
    int data_fd = -1;
    if (data_fd_ >= 0) {
        if (data_offset_ < 0) {
            LONGLONG headstart = 0, dataend = 0;
            int addr_status = 0;
            fits_get_hduaddrll(fptr_, &headstart, &data_offset_, &dataend, &addr_status);
            if (addr_status != 0 || data_offset_ <= 0) {
                data_offset_ = -1;
            }
        }
        if (data_offset_ > 0) {
            data_fd = data_fd_;
        }
    }

    if (data_fd >= 0) {
        read_columns_pread_tiled(col_indices, start_row, num_rows, result, data_fd);
        return;
    }

    // 16MB chunk target, then align with CFITSIO's suggested table row buffer.
    // Clamp to the requested row window so small tables do not allocate a full
    // 16 MiB scratch (visible as rss when mmap/cache are off in scorecard).
    const size_t TARGET_CHUNK_SIZE = 16 * 1024 * 1024;
    long rows_per_chunk = std::max(1L, (long)(TARGET_CHUNK_SIZE / row_width_bytes_));
    {
        int status = 0;
        long cfitsio_rows_per_buf = 0;
        fits_get_rowsize(fptr_, &cfitsio_rows_per_buf, &status);
        if (status == 0 && cfitsio_rows_per_buf > 0) {
            if (rows_per_chunk < cfitsio_rows_per_buf) {
                rows_per_chunk = cfitsio_rows_per_buf;
            } else {
                rows_per_chunk =
                    std::max(1L, (rows_per_chunk / cfitsio_rows_per_buf) * cfitsio_rows_per_buf);
            }
        }
    }
    rows_per_chunk = std::max(1L, std::min(rows_per_chunk, num_rows));

    // Reuse the persistent scratch buffer so its pages stay resident
    // across calls (a fresh per-call allocation would refault ~6k pages
    // per 12 MiB read, ~5-10 ms on a busy host).
    const size_t row_width_size = internal::checked_nonnegative_cast<size_t>(
        row_width_bytes_, "Invalid table geometry for scratch row width");
    const size_t need_bytes = internal::checked_nonnegative_multiply<size_t>(
        internal::checked_nonnegative_cast<size_t>(
            rows_per_chunk, "Invalid table geometry for scratch row count"),
        row_width_size, "Invalid table geometry for scratch buffer");
    if (scratch_buffer_.size() < need_bytes) {
        scratch_buffer_.resize(need_bytes);
    }
    std::vector<uint8_t>& buffer = scratch_buffer_;

    // No raw fd (borrowed handle, compressed or remote input): read whole
    // rows through CFITSIO a chunk at a time and de-interleave them.
    long rows_done = 0;
    while (rows_done < num_rows) {
        const long rows = std::min(rows_per_chunk, num_rows - rows_done);
        int status = 0;
        fits_read_tblbytes(
            fptr_, start_row + rows_done, 1,
            internal::checked_nonnegative_multiply<long>(
                rows, row_width_bytes_,
                "Invalid table geometry for buffered read size"),
            buffer.data(), &status);
        if (status != 0) {
            char err_msg[81];
            fits_get_errstatus(status, err_msg);
            throw std::runtime_error(
                "Failed to read table bytes: " + std::string(err_msg));
        }
        for (int col_idx : col_indices) {
            const auto& col = columns_[col_idx];
            native::Tensor tensor = result[col_idx].fixed_data;
            uint8_t* dest_ptr =
                (uint8_t*)get_tensor_data_ptr(
                    tensor,
                    internal::checked_nonnegative_multiply<long>(
                        rows_done, static_cast<long>(col.repeat),
                        "Invalid table geometry for column " + col.name +
                            " destination offset"));
            extract_column_data(buffer.data(), rows, col, dest_ptr);
        }
        rows_done += rows;
    }
}

// Direct-I/O buffered read: the row window is split into ~512 KiB tiles
// of whole rows, and each pool thread preads one tile into its own
// cache-resident scratch and gathers every requested column out of it
// before claiming the next. The 16 MiB serial scratch it replaces sent
// each column back to DRAM for its gather, so a 1-column projection of a
// 20-column table cost more than fits_read_col, and a full read of an
// 84 MB table ran 5x behind mmap. Several preads in flight also overlap
// I/O with the gather on cold storage, which the old double-buffered
// prefetch thread did for >= 64 MiB reads only.
void TableReader::read_columns_pread_tiled(
    const std::vector<int>& col_indices,
    long start_row, long num_rows,
    std::unordered_map<int, ColumnData>& result,
    int data_fd) {
    const size_t row_width_size = internal::checked_nonnegative_cast<size_t>(
        row_width_bytes_, "Invalid table geometry for scratch row width");
    constexpr size_t kTileBytes = 512u * 1024u;
    const long rows_per_tile = std::max<long>(
        1L, std::min<long>(num_rows, static_cast<long>(kTileBytes / row_width_size)));
    const size_t tile_bytes = internal::checked_nonnegative_multiply<size_t>(
        static_cast<size_t>(rows_per_tile), row_width_size,
        "Invalid table geometry for scratch buffer");
    const long ntiles = (num_rows + rows_per_tile - 1) / rows_per_tile;

    // Destination bases resolved on the calling thread: the workers only
    // see raw pointers, never the result map.
    struct Target {
        ColumnGather gather;
        uint8_t* base;
    };
    std::vector<Target> targets;
    targets.reserve(col_indices.size());
    for (int col_idx : col_indices) {
        const auto& col = columns_[col_idx];
        native::Tensor tensor = result[col_idx].fixed_data;
        Target target{plan_column_gather(col, row_width_bytes_),
                      static_cast<uint8_t*>(get_tensor_data_ptr(tensor, 0))};
        // The destination holds num_rows cells of the source cell's size.
        if (internal::checked_nonnegative_multiply<size_t>(
                static_cast<size_t>(num_rows), target.gather.cell_bytes,
                "Invalid table geometry for column " + col.name + " destination") !=
            tensor.nbytes()) {
            throw std::runtime_error(
                "Invalid table geometry for column " + col.name + " destination");
        }
        targets.push_back(target);
    }
    // Absolute file offset of the window's first row, checked once; each
    // tile's offset below stays inside [first, first + window bytes).
    const off_t window_offset = internal::checked_nonnegative_cast<off_t>(
        internal::checked_nonnegative_add<LONGLONG>(
            data_offset_,
            internal::checked_nonnegative_multiply<LONGLONG>(
                static_cast<LONGLONG>(start_row - 1),
                static_cast<LONGLONG>(row_width_bytes_),
                "Invalid table geometry for pread offset"),
            "Invalid table geometry for pread offset"),
        "Invalid table geometry for pread offset");
    internal::checked_nonnegative_add<off_t>(
        window_offset,
        internal::checked_nonnegative_cast<off_t>(
            internal::checked_nonnegative_multiply<size_t>(
                static_cast<size_t>(num_rows), row_width_size,
                "Invalid table geometry for pread size"),
            "Invalid table geometry for pread size"),
        "Invalid table geometry for pread range");

    std::atomic<bool> failed{false};
    auto run_tiles = [&](int64_t t_begin, int64_t t_end) {
        // Per-thread scratch: pool threads persist, so the pages stay
        // resident across reads instead of refaulting per call.
        thread_local std::vector<uint8_t> tile;
        if (tile.size() < tile_bytes) {
            tile.resize(tile_bytes);
        }
        for (int64_t t = t_begin; t < t_end; ++t) {
            if (failed.load(std::memory_order_relaxed)) return;
            const long first = static_cast<long>(t) * rows_per_tile;
            const long rows = std::min(rows_per_tile, num_rows - first);
            const size_t nbytes = static_cast<size_t>(rows) * row_width_size;
            const off_t off =
                window_offset + static_cast<off_t>(first) * static_cast<off_t>(row_width_size);
            size_t got = 0;
            while (got < nbytes) {
                const ssize_t n = ::pread(data_fd, tile.data() + got, nbytes - got,
                                          off + static_cast<off_t>(got));
                if (n < 0 && errno == EINTR) continue;
                if (n <= 0) {
                    failed.store(true, std::memory_order_relaxed);
                    return;
                }
                got += static_cast<size_t>(n);
            }
            for (const Target& target : targets) {
                run_column_gather(
                    target.gather, tile.data(), 0, rows,
                    target.base + static_cast<size_t>(first) * target.gather.cell_bytes);
            }
        }
    };
    if (ntiles <= 1) {
        run_tiles(0, ntiles);
    } else {
        // Grain 1: tiles are claimed dynamically, so a preempted thread on
        // a busy host takes fewer of them instead of stalling the read.
        core::parallel_for(0, ntiles, 1, run_tiles);
    }
    if (failed.load()) {
        // The descriptor stays open: it is the reader's file, and read_columns
        // retries this window through CFITSIO.
        throw std::runtime_error("Failed to pread table bytes");
    }
}

void TableReader::extract_column_data(const uint8_t* buffer, long num_rows, const ColumnInfo& col, uint8_t* dest,
                                      bool allow_parallel) {
    const ColumnGather gather = plan_column_gather(col, row_width_bytes_);
    internal::checked_nonnegative_multiply<size_t>(
        internal::checked_nonnegative_cast<size_t>(
            num_rows, "extract_column_data: invalid row count"),
        gather.cell_bytes, "extract_column_data: invalid output size");
    if (num_rows <= 0) {
        return;
    }
    // Each row writes to a non-overlapping region of dest, so rows split
    // across threads safely. Tiny windows (and callers already running on a
    // pool thread, such as the tiled pread) stay on the calling thread.
    if (num_rows < 512 || !allow_parallel) {
        run_column_gather(gather, buffer, 0, num_rows, dest);
    } else {
        native::parallel_for(0, num_rows, 2048, [&](long start, long end) {
            run_column_gather(gather, buffer, start, end, dest);
        });
    }
}

}  // namespace torchfits
