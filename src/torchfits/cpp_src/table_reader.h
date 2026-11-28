#pragma once
// TableReader: decodes one FITS table HDU into native tensors.
//
// The class is declared here and implemented by concern, one translation unit
// each, so a change to one read path recompiles only that file:
//
//   table_reader_layout.cpp    open/close, TFORM/TTYPE column layout, lazy
//                              TSCAL/TZERO/TNULL resolution, extent checks
//   table_reader_read.cpp      read_columns dispatch, the CFITSIO column path
//                              (ASCII strings, logicals, bits) and the shared
//                              scale / null / unsigned-offset conventions
//   table_reader_buffered.cpp  fixed-width projections from raw rows: tiled
//                              parallel pread, or fits_read_tblbytes chunks
//   table_reader_mmap.cpp      whole-window mmap reads and in-place updates
//   table_reader_filter.cpp    windowed WHERE scans over the mapped rows
//   table_reader_vla.cpp       variable-length array (P/Q) columns

#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include <sys/stat.h>
#include <sys/types.h>
#include <nanobind/nanobind.h>
#include <fitsio.h>

#include "native_tensor.h"
#include "table_types.h"

namespace nb = nanobind;

namespace torchfits {

class TableReader {
public:
    // Opens a private handle on filename (never shared across threads).
    TableReader(const std::string& filename, int hdu_num = 1);
    // Borrows fptr. The caller keeps it open (hold_fptr_owner) and serializes
    // every use of its CFITSIO cursor (set_io_guard).
    TableReader(fitsfile* fptr, int hdu_num = 1);
    ~TableReader();

    TableReader(const TableReader&) = delete;
    TableReader& operator=(const TableReader&) = delete;

    // A reader built on a borrowed handle keeps that handle's owner alive, so
    // closing the Python FITSFile first cannot free the fitsfile under it.
    void hold_fptr_owner(std::shared_ptr<void> owner) { fptr_owner_ = std::move(owner); }

    // A reader built on a borrowed handle shares that handle's CFITSIO
    // cursor (and reads its file by name) with every other borrower, so each
    // read must run under the handle's own guard: its I/O lock plus the
    // replaced-file check. Path readers own their fitsfile and have none.
    using IoGuard = std::function<void(const std::function<void()>&)>;
    void set_io_guard(IoGuard guard) { io_guard_ = std::move(guard); }

    // Run one read of a persistent reader: under its own io_mutex_ and, for a
    // borrowed handle, under the handle's guard. The GIL must be released.
    template <class F>
    void run_io(F&& fn) {
        std::lock_guard<std::mutex> io_lock(io_mutex_);
        if (io_guard_) {
            io_guard_(std::function<void()>(std::forward<F>(fn)));
        } else {
            fn();
        }
    }

    // True if the file at filename_ still matches the identity captured at
    // construction (see table_reader_layout.cpp).
    bool file_unchanged() const;

    // Helper struct to hold column data (either fixed or VLA)
    struct ColumnData {
        bool is_vla;
        native::Tensor fixed_data;
        std::vector<native::Tensor> vla_data;
        native::Tensor vla_offsets;

        ColumnData() : is_vla(false) {}
        ColumnData(native::Tensor t) : is_vla(false), fixed_data(t) {}
        ColumnData(std::vector<native::Tensor> v) : is_vla(true), vla_data(v) {}
        ColumnData(native::Tensor values, native::Tensor offsets, bool /*flat_vla*/)
            : is_vla(true), fixed_data(values), vla_offsets(offsets) {}
    };

    // Read columns from the table
    // Returns column data in file order (request order when column_names is
    // given), so callers that pair columns against TFORM/TTYPE cards (e.g. the
    // table rewrite path) never see an arbitrary hash iteration order.
    std::vector<std::pair<std::string, ColumnData>> read_columns(
        const std::vector<std::string>& column_names = {},
        long start_row = 1, long num_rows = -1, bool vla_flat = false);

    // Memory-mapped column reading, in file/request column order. Fixed-width
    // binary columns only (no VLA, ASCII or scaled non-unsigned columns).
    std::vector<std::pair<std::string, native::Tensor>> read_columns_mmap(
        const std::vector<std::string>& column_names = {},
        long start_row = 1, long num_rows = -1);

    // Filtered reading over the row window [start_row, start_row + num_rows)
    // (1-based; num_rows = -1 means through the last row).
    std::vector<std::pair<std::string, native::Tensor>> read_columns_mmap_filtered(
        const std::vector<std::string>& column_names,
        const std::vector<TableFilter>& filters,
        long start_row = 1, long num_rows = -1);

    // In-place update of fixed-width binary cells through a writable mapping.
    void update_rows_mmap(nb::dict tensor_dict, long start_row, long num_rows);

    std::vector<std::string> get_column_names() const;
    long get_num_rows() const { return nrows_; }
    int get_num_cols() const { return ncols_; }
    bool is_ascii_table() const { return is_ascii_; }

    // Serializes access to a reader that is genuinely shared across Python
    // threads: the CFITSIO cursor, the cached offsets, and scratch_buffer_ are
    // not safe for unsynchronized concurrent use.
    //
    // Nothing in the TableReader sources acquires it. The lock is taken by the
    // bindings that hand out a shared reader -- read_fits_table_rows_mmap_from_reader
    // and read_fits_table_rows_mmap_from_reader_raw in table_bindings.cpp, both of
    // which hold it across read_columns_mmap() with the GIL released -- and by
    // run_io() above, for the persistent TableReader(file_obj) bindings. It is
    // public so those callers can reach it.
    //
    // Every other path is per-call or per-thread and needs no lock: the bindings
    // taking a filename either build a stack-local TableReader, or take one from
    // g_reader_cache, which is a ThreadLocalReaderCache and so never hands the
    // same reader to two threads. update_rows_mmap() is in the first group --
    // it takes a filename and opens its own reader.
    //
    // INVARIANT: any binding that accepts a TableReader capsule must hold
    // io_mutex_ for the whole call, with the GIL released.
    mutable std::mutex io_mutex_;

private:
    // --- layout (table_reader_layout.cpp) ---
    void analyze_table();
    void ensure_column_scale(int col_idx);
    void ensure_extent_within_file(off_t file_size, LONGLONG data_offset,
                                   long long rows_before, long long nrows) const;
    void validate_physical_table_extent() const;
    bool direct_io_ok() const;
    void ensure_table_hdu();

    // --- value conventions (table_reader_read.cpp) ---
    static void scale_complex_inplace(native::Tensor& tensor, double tscale, double tzero);
    native::Tensor apply_scale_and_nulls(const ColumnInfo& col, native::Tensor raw);
    native::Tensor apply_unsigned_offset(native::Tensor raw, native::ScalarType target);

    // --- buffered fixed-width reads (table_reader_buffered.cpp) ---
    void read_columns_buffered(
        const std::vector<int>& col_indices,
        long start_row, long num_rows,
        std::unordered_map<int, ColumnData>& result);
    void read_columns_pread_tiled(
        const std::vector<int>& col_indices,
        long start_row, long num_rows,
        std::unordered_map<int, ColumnData>& result,
        int data_fd);
    void extract_column_data(const uint8_t* buffer, long num_rows, const ColumnInfo& col, uint8_t* dest,
                             bool allow_parallel = true);

    // --- variable-length arrays (table_reader_vla.cpp) ---
    std::vector<native::Tensor> read_vla_column(int col_idx, long start_row, long num_rows, const ColumnInfo& col);
    std::pair<native::Tensor, native::Tensor> read_vla_column_flat(
        int col_idx, long start_row, long num_rows, const ColumnInfo& col);

    fitsfile* fptr_ = nullptr;
    std::string filename_;
    int hdu_num_;
    int target_hdu_ = 1;
    bool owns_fptr_ = false;
    std::shared_ptr<void> fptr_owner_;
    IoGuard io_guard_;
    long nrows_;
    int ncols_;
    long row_width_bytes_ = 0;
    std::vector<ColumnInfo> columns_;
    bool is_ascii_ = false;
    // Reused row-buffer for buffered table reads. A fresh per-call
    // std::vector would be mmap'd and munmap'd by glibc (large sizes),
    // refaulting every page on every read (~6k minor faults/12 MiB); keeping
    // it on the reader keeps the pages resident across calls.
    std::vector<uint8_t> scratch_buffer_;
    // Cached byte offset of the table data section (fits_get_hduaddrll
    // refaults ~12 MiB of anonymous memory per call on this host).
    LONGLONG data_offset_ = -1;
    // Persistent raw fd for pread of the table heap (avoids re-opening the
    // file on every buffered read).
    int data_fd_cached_ = -1;
    // File identity captured at construction (stat of filename_) for the
    // stale-reader check in file_unchanged().
    struct stat file_stat_ {};
    bool stat_valid_ = false;
};

} // namespace torchfits
