#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <fitsio.h>

void write_fits_table(const char* filename, nb::dict tensor_dict, nb::dict header,
                      bool overwrite, nb::object schema_obj, const std::string& table_type);
long infer_num_rows_from_payload(nb::dict tensor_dict);
void append_rows(const char* filename, int hdu_num, nb::dict tensor_dict);
void insert_rows(const char* filename, int hdu_num, nb::dict tensor_dict, long start_row);
void delete_rows(const char* filename, int hdu_num, long start_row, long num_rows);
void update_rows(const char* filename, int hdu_num, nb::dict tensor_dict,
                 long start_row, long num_rows);
void update_rows_mmap(const char* filename, int hdu_num, nb::dict tensor_dict,
                      long start_row, long num_rows);
// The mutation writers below run their file IO without the GIL: append/insert/
// update convert the payload with the GIL held (callers must hold it) and
// release it around the CFITSIO calls; rename/drop take plain strings and must
// be called with the GIL already released.
void rename_columns(const char* filename, int hdu_num,
                    const std::vector<std::pair<std::string, std::string>>& mapping);
void drop_columns(const char* filename, int hdu_num, const std::vector<std::string>& columns);

// Incremental table writer: appends row batches to an existing table HDU on
// one open handle. Batches are planned with the GIL held and written without
// it; payloads are in stored form (TSCALn/TZEROn are not applied), the
// convention write_fits_table uses for the first batch. Not thread-safe: a
// concurrent call is refused.
class TableStreamWriter {
public:
    TableStreamWriter(const std::string& filename, int hdu_num);
    ~TableStreamWriter();
    TableStreamWriter(const TableStreamWriter&) = delete;
    TableStreamWriter& operator=(const TableStreamWriter&) = delete;

    void write_rows(nb::dict tensor_dict);
    void write_raw_rows(const uint8_t* data, long nrows);
    void close();

    long num_rows() const { return num_rows_; }
    long row_bytes() const { return row_bytes_; }
    bool closed() const { return fptr_ == nullptr; }
    bool busy() const { return busy_; }
    void set_busy(bool value) { busy_ = value; }

private:
    std::string filename_;
    fitsfile* fptr_ = nullptr;
    long num_rows_ = 0;
    long row_bytes_ = 0;
    bool busy_ = false;
};
