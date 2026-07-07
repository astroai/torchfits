#include <string>
#include <vector>
#include <algorithm>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/unordered_map.h>
#undef READONLY
#include <fitsio.h>

#include "torchfits_torch.h"
#include "cache.h"
#include "table_types.h"
#include "table_reader.h"
#include "security.h"
#include "fits_rw.h"

namespace nb = nanobind;

void* open_table_reader(const char* filename, int hdu_num) {
    try {
        return new torchfits::TableReader(filename, hdu_num);
    } catch (...) {
        return nullptr;
    }
}

void* open_table_reader_from_handle(uintptr_t handle, int hdu_num) {
    try {
        // Cast to fitsfile pointer directly since we can't include FITSFile here
        auto* fptr = reinterpret_cast<fitsfile*>(handle);
        return new torchfits::TableReader(fptr, hdu_num);
    } catch (...) {
        return nullptr;
    }
}

void close_table_reader(void* reader_handle) {
    if (reader_handle) {
        delete static_cast<torchfits::TableReader*>(reader_handle);
    }
}

int read_table_columns(void* reader_handle, const char** column_names, int num_columns,
                      long start_row, long num_rows, nb::dict* result_dict) {
    if (!reader_handle) return -1;

    try {
        auto* reader = static_cast<torchfits::TableReader*>(reader_handle);

        std::vector<std::string> cols;
        for (int i = 0; i < num_columns; i++) {
            cols.push_back(std::string(column_names[i]));
        }

        auto result = reader->read_columns(cols, start_row, num_rows);
        *result_dict = nb::cast<nb::dict>(nb::cast(result));
        return 0;
    } catch (...) {
        return -1;
    }
}



void write_fits_table(const char* filename, nb::dict tensor_dict, nb::dict header, bool overwrite, nb::object schema_obj, const std::string& table_type) {
    torchfits::check_fits_filename_security(filename ? filename : "");
    fitsfile* fptr;
    int status = 0;

    if (overwrite) {
        std::string path = filename ? filename : "";
        if (!path.empty() && path[0] != '!') {
            path = "!" + path;
        }
        fits_create_file(&fptr, path.c_str(), &status);
    } else {
        fits_create_file(&fptr, filename, &status);
    }

    if (status != 0) {
        throw std::runtime_error("Failed to open FITS file for writing");
    }

    try {
        bool is_ascii = false;
        std::string kind = table_type;
        for (auto& c : kind) {
            c = std::tolower(static_cast<unsigned char>(c));
        }
        if (kind == "ascii") {
            is_ascii = true;
        }
        torchfits::write_table_hdu(fptr, tensor_dict, header, schema_obj, is_ascii);
    } catch (...) {
        fits_close_file(fptr, &status);
        throw;
    }

    fits_close_file(fptr, &status);
}

long infer_num_rows_from_payload(nb::dict tensor_dict) {
    long num_rows = 0;
    if (tensor_dict.size() <= 0) {
        return 0;
    }

    nb::handle first_obj = (*tensor_dict.begin()).second;
    if (nb::isinstance<nb::list>(first_obj)) {
        nb::list lst = nb::cast<nb::list>(first_obj);
        return static_cast<long>(lst.size());
    }
    if (nb::isinstance<nb::tuple>(first_obj)) {
        nb::tuple tup = nb::cast<nb::tuple>(first_obj);
        return static_cast<long>(tup.size());
    }
    if (nb::isinstance<nb::str>(first_obj) || nb::isinstance<nb::bytes>(first_obj)) {
        return 1;
    }

    nb::ndarray<> first_col = nb::cast<nb::ndarray<>>(first_obj);
    int ndim = first_col.ndim();
    if (ndim == 0) {
        return 1;
    }
    return static_cast<long>(first_col.shape(0));
}

// forward decl: used by insert_rows below
void update_rows(const char* filename, int hdu_num, nb::dict tensor_dict, long start_row, long num_rows);

void append_rows(const char* filename, int hdu_num, nb::dict tensor_dict) {
    fitsfile* fptr;
    int status = 0;

    // Use explicit cfitsio mode value to avoid macro collisions with Python headers.
    constexpr int kFitsReadWrite = 1;
    torchfits::check_fits_filename_security(filename ? filename : "");
    fits_open_file(&fptr, filename, kFitsReadWrite, &status);
    if (status != 0) {
        char err_msg[FLEN_STATUS];
        fits_get_errstatus(status, err_msg);
        throw std::runtime_error(
            std::string("Failed to open FITS file for writing: ") + err_msg
        );
    }

    fits_movabs_hdu(fptr, hdu_num + 1, nullptr, &status);
    if (status != 0) {
        fits_close_file(fptr, &status);
        throw std::runtime_error("Failed to move to table HDU");
    }

    long num_rows = infer_num_rows_from_payload(tensor_dict);

    long start_row;
    fits_get_num_rows(fptr, &start_row, &status);
    start_row++;

    fits_insert_rows(fptr, start_row -1, num_rows, &status);

    for (auto item : tensor_dict) {
        std::string col_name = nb::cast<std::string>(item.first);
        int colnum = 0;
        fits_get_colnum(fptr, CASEINSEN, const_cast<char*>(col_name.c_str()), &colnum, &status);
        if (status != 0) {
            fits_close_file(fptr, &status);
            throw std::runtime_error("Column not found for append_rows: " + col_name);
        }

        int col_status = 0;
        int typecode = 0;
        long repeat = 0;
        long width = 0;
        fits_get_coltype(fptr, colnum, &typecode, &repeat, &width, &col_status);
        if (col_status != 0) {
            fits_close_file(fptr, &status);
            throw std::runtime_error("Failed to get column type for append_rows: " + col_name);
        }

        if (typecode < 0) {
            int base_type = -typecode;
            nb::handle obj = item.second;
            if (!(nb::isinstance<nb::list>(obj) || nb::isinstance<nb::tuple>(obj))) {
                fits_close_file(fptr, &status);
                throw std::runtime_error("append_rows VLA column expects list/tuple for " + col_name);
            }

            nb::sequence seq = nb::cast<nb::sequence>(obj);
            long seq_len = static_cast<long>(nb::len(seq));
            if (seq_len != num_rows) {
                fits_close_file(fptr, &status);
                throw std::runtime_error("append_rows column length mismatch for " + col_name);
            }

            for (long row = 0; row < num_rows; ++row) {
                nb::ndarray<> arr = nb::cast<nb::ndarray<>>(seq[row]);
                if (arr.ndim() > 1) {
                    fits_close_file(fptr, &status);
                    throw std::runtime_error("append_rows VLA rows must be 1D for " + col_name);
                }
                long nelements = static_cast<long>(arr.size());
                void* data_ptr = arr.size() ? arr.data() : nullptr;
                std::vector<unsigned char> logical;

                if (base_type == TLOGICAL && nelements > 0) {
                    nb::dlpack::dtype dt = arr.dtype();
                    logical.resize(static_cast<size_t>(nelements));
                    if (dt.code == (uint8_t)nb::dlpack::dtype_code::Bool && dt.bits == 8) {
                        const bool* src = static_cast<const bool*>(arr.data());
                        for (long idx = 0; idx < nelements; ++idx) {
                            logical[static_cast<size_t>(idx)] = src[idx] ? 1 : 0;
                        }
                    } else {
                        const uint8_t* src = static_cast<const uint8_t*>(arr.data());
                        for (long idx = 0; idx < nelements; ++idx) {
                            logical[static_cast<size_t>(idx)] = src[idx] ? 1 : 0;
                        }
                    }
                    data_ptr = logical.data();
                }

                fits_write_col(fptr, base_type, colnum, start_row + row, 1, nelements, data_ptr, &status);
            }
            continue;
        }

        if (typecode == TSTRING) {
            std::vector<std::string> values;
            nb::handle obj = item.second;
            if (nb::isinstance<nb::list>(obj)) {
                nb::list lst = nb::cast<nb::list>(obj);
                values.reserve(lst.size());
                for (auto v : lst) {
                    values.push_back(nb::cast<std::string>(v));
                }
            } else if (nb::isinstance<nb::tuple>(obj)) {
                nb::tuple tup = nb::cast<nb::tuple>(obj);
                values.reserve(tup.size());
                for (auto v : tup) {
                    values.push_back(nb::cast<std::string>(v));
                }
            } else if (nb::isinstance<nb::str>(obj) || nb::isinstance<nb::bytes>(obj)) {
                values.push_back(nb::cast<std::string>(obj));
            } else {
                fits_close_file(fptr, &status);
                throw std::runtime_error("append_rows string column expects list/tuple/str for " + col_name);
            }

            if (static_cast<long>(values.size()) != num_rows) {
                fits_close_file(fptr, &status);
                throw std::runtime_error("append_rows column length mismatch for " + col_name);
            }

            long width_chars = repeat > 0 ? repeat : 1;
            std::vector<std::string> padded;
            padded.reserve(values.size());
            for (const auto& v : values) {
                std::string s = v;
                if (static_cast<long>(s.size()) > width_chars) {
                    s = s.substr(0, static_cast<size_t>(width_chars));
                } else if (static_cast<long>(s.size()) < width_chars) {
                    s.append(static_cast<size_t>(width_chars - s.size()), ' ');
                }
                padded.push_back(std::move(s));
            }
            std::vector<const char*> ptrs;
            ptrs.reserve(padded.size());
            for (const auto& s : padded) {
                ptrs.push_back(s.c_str());
            }

            fits_write_col(fptr, TSTRING, colnum, start_row, 1, num_rows,
                           const_cast<char**>(ptrs.data()), &status);
            continue;
        }

        nb::ndarray<> tensor = nb::cast<nb::ndarray<>>(item.second);
        int ndim = tensor.ndim();
        long rows = 1;
        long repeat_vals = 1;
        if (ndim == 0) {
            rows = 1;
            repeat_vals = 1;
        } else if (ndim == 1) {
            rows = static_cast<long>(tensor.shape(0));
            repeat_vals = 1;
        } else if (ndim == 2) {
            rows = static_cast<long>(tensor.shape(0));
            repeat_vals = static_cast<long>(tensor.shape(1));
        } else {
            fits_close_file(fptr, &status);
            throw std::runtime_error("append_rows only supports 1D/2D columns for " + col_name);
        }

        if (rows != num_rows) {
            fits_close_file(fptr, &status);
            throw std::runtime_error("append_rows column length mismatch for " + col_name);
        }

        void* data_ptr = tensor.data();
        int fits_type = 0;
        std::vector<unsigned char> logical_buffer;

        nb::dlpack::dtype dt = tensor.dtype();
        if (dt.code == (uint8_t)nb::dlpack::dtype_code::Bool && dt.bits == 8) {
            fits_type = TLOGICAL;
            long nelements = rows * repeat_vals;
            logical_buffer.resize(static_cast<size_t>(nelements));
            const bool* src = static_cast<const bool*>(tensor.data());
            for (long idx = 0; idx < nelements; ++idx) {
                logical_buffer[static_cast<size_t>(idx)] = src[idx] ? 1 : 0;
            }
            data_ptr = logical_buffer.data();
        } else if (dt.code == (uint8_t)nb::dlpack::dtype_code::UInt && dt.bits == 8) {
            fits_type = TBYTE;
        } else if (dt.code == (uint8_t)nb::dlpack::dtype_code::Int && dt.bits == 16) {
            fits_type = TSHORT;
        } else if (dt.code == (uint8_t)nb::dlpack::dtype_code::Int && dt.bits == 32) {
            fits_type = TINT;
        } else if (dt.code == (uint8_t)nb::dlpack::dtype_code::Float && dt.bits == 32) {
            fits_type = TFLOAT;
        } else if (dt.code == (uint8_t)nb::dlpack::dtype_code::Float && dt.bits == 64) {
            fits_type = TDOUBLE;
        } else if (dt.code == (uint8_t)nb::dlpack::dtype_code::Int && dt.bits == 64) {
            fits_type = TLONGLONG;
        } else if (dt.code == (uint8_t)nb::dlpack::dtype_code::Complex && dt.bits == 64) {
            fits_type = TCOMPLEX;
        } else if (dt.code == (uint8_t)nb::dlpack::dtype_code::Complex && dt.bits == 128) {
            fits_type = TDBLCOMPLEX;
        } else {
            fits_close_file(fptr, &status);
            throw std::runtime_error("Unsupported dtype for append_rows");
        }

        long nelements = num_rows * repeat_vals;
        fits_write_col(fptr, fits_type, colnum, start_row, 1, nelements, data_ptr, &status);
    }

    fits_close_file(fptr, &status);

    if (status != 0) {
        throw std::runtime_error("Failed to append rows to FITS table");
    }
}

void insert_rows(const char* filename, int hdu_num, nb::dict tensor_dict, long start_row) {
    long num_rows = infer_num_rows_from_payload(tensor_dict);
    if (num_rows <= 0) {
        return;
    }

    fitsfile* fptr = nullptr;
    int status = 0;

    constexpr int kFitsReadWrite = 1;
    torchfits::check_fits_filename_security(filename ? filename : "");
    fits_open_file(&fptr, filename, kFitsReadWrite, &status);
    if (status != 0) {
        char err_msg[FLEN_STATUS];
        fits_get_errstatus(status, err_msg);
        throw std::runtime_error(
            std::string("Failed to open FITS file for writing: ") + err_msg
        );
    }

    fits_movabs_hdu(fptr, hdu_num + 1, nullptr, &status);
    if (status != 0) {
        fits_close_file(fptr, &status);
        throw std::runtime_error("Failed to move to table HDU");
    }

    long total_rows = 0;
    fits_get_num_rows(fptr, &total_rows, &status);
    if (status != 0) {
        fits_close_file(fptr, &status);
        throw std::runtime_error("Failed to get table row count");
    }

    if (start_row < 1 || start_row > (total_rows + 1)) {
        fits_close_file(fptr, &status);
        throw std::runtime_error("insert_rows start_row out of range");
    }

    fits_insert_rows(fptr, start_row - 1, num_rows, &status);
    fits_close_file(fptr, &status);
    if (status != 0) {
        throw std::runtime_error("Failed to insert rows into FITS table");
    }

    // Reuse the existing typed write path to populate inserted rows.
    update_rows(filename, hdu_num, tensor_dict, start_row, num_rows);
}

void delete_rows(const char* filename, int hdu_num, long start_row, long num_rows) {
    if (num_rows <= 0) {
        return;
    }

    fitsfile* fptr = nullptr;
    int status = 0;

    constexpr int kFitsReadWrite = 1;
    torchfits::check_fits_filename_security(filename ? filename : "");
    fits_open_file(&fptr, filename, kFitsReadWrite, &status);
    if (status != 0) {
        char err_msg[FLEN_STATUS];
        fits_get_errstatus(status, err_msg);
        throw std::runtime_error(
            std::string("Failed to open FITS file for writing: ") + err_msg
        );
    }

    fits_movabs_hdu(fptr, hdu_num + 1, nullptr, &status);
    if (status != 0) {
        fits_close_file(fptr, &status);
        throw std::runtime_error("Failed to move to table HDU");
    }

    long total_rows = 0;
    fits_get_num_rows(fptr, &total_rows, &status);
    if (status != 0) {
        fits_close_file(fptr, &status);
        throw std::runtime_error("Failed to get table row count");
    }

    if (start_row < 1 || start_row > total_rows) {
        fits_close_file(fptr, &status);
        throw std::runtime_error("delete_rows start_row out of range");
    }

    long max_rows = total_rows - start_row + 1;
    long ndelete = std::min(num_rows, max_rows);
    fits_delete_rows(fptr, start_row, ndelete, &status);
    fits_close_file(fptr, &status);

    if (status != 0) {
        throw std::runtime_error("Failed to delete rows from FITS table");
    }
}

void update_rows(const char* filename, int hdu_num, nb::dict tensor_dict, long start_row, long num_rows) {
    if (num_rows <= 0) {
        return;
    }

    fitsfile* fptr;
    int status = 0;

    constexpr int kFitsReadWrite = 1;
    torchfits::check_fits_filename_security(filename ? filename : "");
    fits_open_file(&fptr, filename, kFitsReadWrite, &status);
    if (status != 0) {
        char err_msg[FLEN_STATUS];
        fits_get_errstatus(status, err_msg);
        throw std::runtime_error(
            std::string("Failed to open FITS file for writing: ") + err_msg
        );
    }

    fits_movabs_hdu(fptr, hdu_num + 1, nullptr, &status);
    if (status != 0) {
        fits_close_file(fptr, &status);
        throw std::runtime_error("Failed to move to table HDU");
    }

    for (auto item : tensor_dict) {
        std::string col_name = nb::cast<std::string>(item.first);
        int colnum = 0;
        fits_get_colnum(fptr, CASEINSEN, const_cast<char*>(col_name.c_str()), &colnum, &status);
        if (status != 0) {
            fits_close_file(fptr, &status);
            throw std::runtime_error("Column not found for update_rows: " + col_name);
        }

        int col_status = 0;
        int typecode = 0;
        long repeat = 0;
        long width = 0;
        fits_get_coltype(fptr, colnum, &typecode, &repeat, &width, &col_status);
        if (col_status != 0) {
            fits_close_file(fptr, &status);
            throw std::runtime_error("Failed to get column type for update_rows: " + col_name);
        }

        if (typecode < 0) {
            int base_type = -typecode;
            nb::handle obj = item.second;
            if (!(nb::isinstance<nb::list>(obj) || nb::isinstance<nb::tuple>(obj))) {
                fits_close_file(fptr, &status);
                throw std::runtime_error("update_rows VLA column expects list/tuple for " + col_name);
            }

            nb::sequence seq = nb::cast<nb::sequence>(obj);
            long seq_len = static_cast<long>(nb::len(seq));
            if (seq_len != num_rows) {
                fits_close_file(fptr, &status);
                throw std::runtime_error("update_rows column length mismatch for " + col_name);
            }

            for (long row = 0; row < num_rows; ++row) {
                nb::ndarray<> arr = nb::cast<nb::ndarray<>>(seq[row]);
                if (arr.ndim() > 1) {
                    fits_close_file(fptr, &status);
                    throw std::runtime_error("update_rows VLA rows must be 1D for " + col_name);
                }
                long nelements = static_cast<long>(arr.size());
                void* data_ptr = arr.size() ? arr.data() : nullptr;
                std::vector<unsigned char> logical;

                if (base_type == TLOGICAL && nelements > 0) {
                    nb::dlpack::dtype dt = arr.dtype();
                    logical.resize(static_cast<size_t>(nelements));
                    if (dt.code == (uint8_t)nb::dlpack::dtype_code::Bool && dt.bits == 8) {
                        const bool* src = static_cast<const bool*>(arr.data());
                        for (long idx = 0; idx < nelements; ++idx) {
                            logical[static_cast<size_t>(idx)] = src[idx] ? 1 : 0;
                        }
                    } else {
                        const uint8_t* src = static_cast<const uint8_t*>(arr.data());
                        for (long idx = 0; idx < nelements; ++idx) {
                            logical[static_cast<size_t>(idx)] = src[idx] ? 1 : 0;
                        }
                    }
                    data_ptr = logical.data();
                }

                fits_write_col(fptr, base_type, colnum, start_row + row, 1, nelements, data_ptr, &status);
            }
            continue;
        }

        if (typecode == TSTRING) {
            std::vector<std::string> values;
            nb::handle obj = item.second;
            if (nb::isinstance<nb::list>(obj)) {
                nb::list lst = nb::cast<nb::list>(obj);
                values.reserve(lst.size());
                for (auto v : lst) {
                    values.push_back(nb::cast<std::string>(v));
                }
            } else if (nb::isinstance<nb::tuple>(obj)) {
                nb::tuple tup = nb::cast<nb::tuple>(obj);
                values.reserve(tup.size());
                for (auto v : tup) {
                    values.push_back(nb::cast<std::string>(v));
                }
            } else if (nb::isinstance<nb::str>(obj) || nb::isinstance<nb::bytes>(obj)) {
                values.push_back(nb::cast<std::string>(obj));
            } else if (nb::isinstance<nb::ndarray<>>(obj)) {
                // Python's update_rows materialises fixed-width CHAR columns
                // as a (num_rows, width) uint8 ndarray (see the
                // has_string / dtype / string_widths branch in
                // torchfits.table.update_rows). Mirror the mmap-path's
                // STRING case: copy bytes left-to-right per row and
                // right-pad with ASCII spaces (0x20) so short user
                // payloads land the same bytes as the mmap writer.
                nb::ndarray<> t_str = nb::cast<nb::ndarray<>>(obj);
                nb::dlpack::dtype dt_str = t_str.dtype();
                if (
                    !(dt_str.code == (uint8_t)nb::dlpack::dtype_code::UInt &&
                      dt_str.bits == 8)
                ) {
                    fits_close_file(fptr, &status);
                    throw std::runtime_error(
                        "update_rows string ndarray must be uint8 for " + col_name
                    );
                }
                int ndim_str = t_str.ndim();
                long user_repeat_str = 1;
                long rows_str = 1;
                if (ndim_str == 0) {
                    rows_str = 1;
                    user_repeat_str = 1;
                } else if (ndim_str == 1) {
                    rows_str = static_cast<long>(t_str.shape(0));
                    user_repeat_str = 1;
                } else if (ndim_str == 2) {
                    rows_str = static_cast<long>(t_str.shape(0));
                    user_repeat_str = static_cast<long>(t_str.shape(1));
                } else {
                    fits_close_file(fptr, &status);
                    throw std::runtime_error(
                        "update_rows string ndarray must be 1D/2D for " + col_name
                    );
                }
                if (rows_str != num_rows) {
                    fits_close_file(fptr, &status);
                    throw std::runtime_error(
                        "update_rows column length mismatch for " + col_name
                    );
                }
                long width_chars_str = repeat > 0 ? repeat : 1;
                if (user_repeat_str > width_chars_str) {
                    fits_close_file(fptr, &status);
                    throw std::runtime_error(
                        "update_rows string width " +
                        std::to_string(user_repeat_str) + " exceeds column " +
                        std::to_string(width_chars_str) + " for " + col_name
                    );
                }
                const uint8_t* src_str = static_cast<const uint8_t*>(t_str.data());
                std::vector<std::string> padded_str;
                padded_str.reserve(static_cast<size_t>(num_rows));
                for (long i = 0; i < num_rows; ++i) {
                    std::string row(static_cast<size_t>(width_chars_str), ' ');
                    for (long j = 0; j < user_repeat_str; ++j) {
                        long byte_off_str = (ndim_str == 2)
                            ? i * t_str.stride(0) + j * t_str.stride(1)
                            : i * t_str.stride(0) + j;
                        row[static_cast<size_t>(j)] =
                            static_cast<char>(src_str[byte_off_str]);
                    }
                    padded_str.push_back(std::move(row));
                }
                std::vector<const char*> ptrs_str;
                ptrs_str.reserve(padded_str.size());
                for (const auto& s : padded_str) {
                    ptrs_str.push_back(s.c_str());
                }
                fits_write_col(
                    fptr, TSTRING, colnum, start_row, 1, num_rows,
                    const_cast<char**>(ptrs_str.data()), &status
                );
                continue;
            } else {
                fits_close_file(fptr, &status);
                throw std::runtime_error("update_rows string column expects list/tuple/str for " + col_name);
            }

            if (static_cast<long>(values.size()) != num_rows) {
                fits_close_file(fptr, &status);
                throw std::runtime_error("update_rows column length mismatch for " + col_name);
            }

            long width_chars = repeat > 0 ? repeat : 1;
            std::vector<std::string> padded;
            padded.reserve(values.size());
            for (const auto& v : values) {
                std::string s = v;
                if (static_cast<long>(s.size()) > width_chars) {
                    s = s.substr(0, static_cast<size_t>(width_chars));
                } else if (static_cast<long>(s.size()) < width_chars) {
                    s.append(static_cast<size_t>(width_chars - s.size()), ' ');
                }
                padded.push_back(std::move(s));
            }
            std::vector<const char*> ptrs;
            ptrs.reserve(padded.size());
            for (const auto& s : padded) {
                ptrs.push_back(s.c_str());
            }

            fits_write_col(fptr, TSTRING, colnum, start_row, 1, num_rows,
                           const_cast<char**>(ptrs.data()), &status);
            continue;
        }

        if (typecode == TBIT) {
            // BIT columns: pack booleans MSB-first into (ceil(repeat/8))
            // bytes per row and write via fits_write_col(FT, TBIT, ...).
            // CFITSIO's TBIT descriptor expects bytes containing 8 packed
            // bits each, mirroring the mmap-path's per-byte packing so the
            // two writers stay byte-equivalent.
            nb::ndarray<> t = nb::cast<nb::ndarray<>>(item.second);
            int ndim_t = t.ndim();
            long rows_t = 1;
            long user_repeat = 1;
            if (ndim_t == 0) {
                rows_t = 1;
                user_repeat = 1;
            } else if (ndim_t == 1) {
                rows_t = static_cast<long>(t.shape(0));
                user_repeat = 1;
            } else if (ndim_t == 2) {
                rows_t = static_cast<long>(t.shape(0));
                user_repeat = static_cast<long>(t.shape(1));
            } else {
                fits_close_file(fptr, &status);
                throw std::runtime_error(
                    "update_rows BIT only supports 1D/2D columns for " + col_name
                );
            }
            if (rows_t != num_rows) {
                fits_close_file(fptr, &status);
                throw std::runtime_error(
                    "update_rows column length mismatch for " + col_name
                );
            }
            if (user_repeat <= 0 || user_repeat > repeat) {
                fits_close_file(fptr, &status);
                throw std::runtime_error(
                    "update_rows BIT repeat must be 1.." + std::to_string(repeat) +
                    " for " + col_name
                );
            }

            long packed_bytes_per_row = (repeat + 7) / 8;
            std::vector<unsigned char> packed(
                static_cast<size_t>(num_rows * packed_bytes_per_row), 0
            );

            nb::dlpack::dtype dt_b = t.dtype();
            const bool* src_bool_b = static_cast<const bool*>(t.data());
            const uint8_t* src_u8_b = static_cast<const uint8_t*>(t.data());

            for (long i = 0; i < num_rows; ++i) {
                for (long j = 0; j < user_repeat; ++j) {
                    bool val = false;
                    long byte_off = (ndim_t == 2)
                        ? i * t.stride(0) + j * t.stride(1)
                        : i * t.stride(0) + j;
                    if (
                        dt_b.code == (uint8_t)nb::dlpack::dtype_code::Bool &&
                        dt_b.bits == 8
                    ) {
                        val = src_bool_b[byte_off];
                    } else if (
                        dt_b.code == (uint8_t)nb::dlpack::dtype_code::UInt &&
                        dt_b.bits == 8
                    ) {
                        val = src_u8_b[byte_off] != 0;
                    } else {
                        fits_close_file(fptr, &status);
                        throw std::runtime_error(
                            "update_rows BIT dtype must be bool or uint8 for " +
                            col_name
                        );
                    }
                    if (val) {
                        packed[
                            static_cast<size_t>(i * packed_bytes_per_row + j / 8)
                        ] |= static_cast<unsigned char>(1U << (7 - (j % 8)));
                    }
                }
            }

            fits_write_col(
                fptr, TBIT, colnum, start_row, 1,
                num_rows * packed_bytes_per_row, packed.data(), &status
            );
            continue;
        }

        nb::ndarray<> tensor = nb::cast<nb::ndarray<>>(item.second);
        int ndim = tensor.ndim();
        long rows = 1;
        long repeat_vals = 1;
        if (ndim == 0) {
            rows = 1;
            repeat_vals = 1;
        } else if (ndim == 1) {
            rows = static_cast<long>(tensor.shape(0));
            repeat_vals = 1;
        } else if (ndim == 2) {
            rows = static_cast<long>(tensor.shape(0));
            repeat_vals = static_cast<long>(tensor.shape(1));
        } else {
            fits_close_file(fptr, &status);
            throw std::runtime_error("update_rows only supports 1D/2D columns for " + col_name);
        }

        if (rows != num_rows) {
            fits_close_file(fptr, &status);
            throw std::runtime_error("update_rows column length mismatch for " + col_name);
        }

        void* data_ptr = tensor.data();
        int fits_type = 0;
        std::vector<unsigned char> logical_buffer;

        nb::dlpack::dtype dt = tensor.dtype();
        if (dt.code == (uint8_t)nb::dlpack::dtype_code::Bool && dt.bits == 8) {
            fits_type = TLOGICAL;
            long nelements = rows * repeat_vals;
            logical_buffer.resize(static_cast<size_t>(nelements));
            const bool* src = static_cast<const bool*>(tensor.data());
            for (long idx = 0; idx < nelements; ++idx) {
                logical_buffer[static_cast<size_t>(idx)] = src[idx] ? 1 : 0;
            }
            data_ptr = logical_buffer.data();
        } else if (dt.code == (uint8_t)nb::dlpack::dtype_code::UInt && dt.bits == 8) {
            fits_type = TBYTE;
        } else if (dt.code == (uint8_t)nb::dlpack::dtype_code::Int && dt.bits == 16) {
            fits_type = TSHORT;
        } else if (dt.code == (uint8_t)nb::dlpack::dtype_code::Int && dt.bits == 32) {
            fits_type = TINT;
        } else if (dt.code == (uint8_t)nb::dlpack::dtype_code::Float && dt.bits == 32) {
            fits_type = TFLOAT;
        } else if (dt.code == (uint8_t)nb::dlpack::dtype_code::Float && dt.bits == 64) {
            fits_type = TDOUBLE;
        } else if (dt.code == (uint8_t)nb::dlpack::dtype_code::Int && dt.bits == 64) {
            fits_type = TLONGLONG;
        } else if (dt.code == (uint8_t)nb::dlpack::dtype_code::Complex && dt.bits == 64) {
            fits_type = TCOMPLEX;
        } else if (dt.code == (uint8_t)nb::dlpack::dtype_code::Complex && dt.bits == 128) {
            fits_type = TDBLCOMPLEX;
        } else {
            fits_close_file(fptr, &status);
            throw std::runtime_error("Unsupported dtype for update_rows");
        }

        long nelements = num_rows * repeat_vals;
        fits_write_col(fptr, fits_type, colnum, start_row, 1, nelements, data_ptr, &status);
    }

    fits_close_file(fptr, &status);

    if (status != 0) {
        throw std::runtime_error("Failed to update rows in FITS table");
    }
}

void update_rows_mmap(const char* filename, int hdu_num, nb::dict tensor_dict, long start_row, long num_rows) {
    torchfits::TableReader reader(filename ? filename : "", hdu_num);
    reader.update_rows_mmap(tensor_dict, start_row, num_rows);
}

void rename_columns(const char* filename, int hdu_num, nb::dict mapping) {
    fitsfile* fptr;
    int status = 0;

    constexpr int kFitsReadWrite = 1;
    torchfits::check_fits_filename_security(filename ? filename : "");
    fits_open_file(&fptr, filename, kFitsReadWrite, &status);
    if (status != 0) {
        char err_msg[FLEN_STATUS];
        fits_get_errstatus(status, err_msg);
        throw std::runtime_error(
            std::string("Failed to open FITS file for writing: ") + err_msg
        );
    }

    fits_movabs_hdu(fptr, hdu_num + 1, nullptr, &status);
    if (status != 0) {
        fits_close_file(fptr, &status);
        throw std::runtime_error("Failed to move to table HDU");
    }

    for (auto item : mapping) {
        std::string old_name = nb::cast<std::string>(item.first);
        std::string new_name = nb::cast<std::string>(item.second);
        if (old_name == new_name) {
            continue;
        }

        int colnum = 0;
        fits_get_colnum(fptr, CASEINSEN, const_cast<char*>(old_name.c_str()), &colnum, &status);
        if (status != 0) {
            fits_close_file(fptr, &status);
            throw std::runtime_error("Column not found for rename_columns: " + old_name);
        }

        int check_status = 0;
        int existing = 0;
        fits_get_colnum(fptr, CASEINSEN, const_cast<char*>(new_name.c_str()), &existing, &check_status);
        if (check_status == 0 && existing > 0) {
            fits_close_file(fptr, &status);
            throw std::runtime_error("Target column already exists: " + new_name);
        }

        char keyname[FLEN_KEYWORD];
        fits_make_keyn("TTYPE", colnum, keyname, &status);
        fits_update_key(fptr, TSTRING, keyname, (void*)new_name.c_str(), nullptr, &status);
        if (status != 0) {
            fits_close_file(fptr, &status);
            throw std::runtime_error("Failed to update column name for " + old_name);
        }
    }

    fits_close_file(fptr, &status);

    if (status != 0) {
        throw std::runtime_error("Failed to rename FITS table columns");
    }
}

void drop_columns(const char* filename, int hdu_num, nb::list columns) {
    fitsfile* fptr;
    int status = 0;

    constexpr int kFitsReadWrite = 1;
    torchfits::check_fits_filename_security(filename ? filename : "");
    fits_open_file(&fptr, filename, kFitsReadWrite, &status);
    if (status != 0) {
        char err_msg[FLEN_STATUS];
        fits_get_errstatus(status, err_msg);
        throw std::runtime_error(
            std::string("Failed to open FITS file for writing: ") + err_msg
        );
    }

    fits_movabs_hdu(fptr, hdu_num + 1, nullptr, &status);
    if (status != 0) {
        fits_close_file(fptr, &status);
        throw std::runtime_error("Failed to move to table HDU");
    }

    std::vector<int> colnums;
    colnums.reserve(static_cast<size_t>(columns.size()));
    for (auto name_obj : columns) {
        std::string name = nb::cast<std::string>(name_obj);
        int colnum = 0;
        fits_get_colnum(fptr, CASEINSEN, const_cast<char*>(name.c_str()), &colnum, &status);
        if (status != 0) {
            fits_close_file(fptr, &status);
            throw std::runtime_error("Column not found for drop_columns: " + name);
        }
        colnums.push_back(colnum);
    }

    std::sort(colnums.begin(), colnums.end(), std::greater<int>());
    colnums.erase(std::unique(colnums.begin(), colnums.end()), colnums.end());

    for (int colnum : colnums) {
        fits_delete_col(fptr, colnum, &status);
        if (status != 0) {
            fits_close_file(fptr, &status);
            throw std::runtime_error("Failed to delete column");
        }
    }

    fits_close_file(fptr, &status);

    if (status != 0) {
        throw std::runtime_error("Failed to drop FITS table columns");
    }
}
