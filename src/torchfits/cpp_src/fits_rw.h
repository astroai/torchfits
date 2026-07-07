#pragma once

#include <string>
#include <vector>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/function.h>
#include <ATen/ATen.h>
#include <fitsio.h>

#include "torchfits_torch.h"
#include "fits_file.h"

namespace torchfits {

struct HDUInfo {
    int index;
    std::string type;
    std::vector<std::tuple<std::string, std::string, std::string>> header;
};

torch::Tensor read_full_cached(const std::string& path, int hdu_num, bool use_mmap);
torch::Tensor read_full_unmapped(const std::string& path, int hdu_num);
torch::Tensor read_full_unmapped_raw(const std::string& path, int hdu_num);
torch::Tensor read_full_nocache(const std::string& path, int hdu_num, bool use_mmap);
int resolve_hdu_name_cached(const std::string& filename, const std::string& hdu_name);
std::vector<torch::Tensor> read_images_batch(const std::vector<std::string>& paths, int hdu_num);
std::vector<torch::Tensor> read_hdus_batch(const std::string& path, const std::vector<int>& hdus, bool use_mmap);
torch::Tensor read_hdus_sequence_last(const std::string& path, const std::vector<int>& hdus, bool use_mmap);
std::pair<FITSFile*, std::vector<HDUInfo>> open_and_read_headers(const std::string& path, int mode);
void write_table_hdu(fitsfile* fptr, nb::dict tensor_dict, nb::dict header, nb::object schema_obj, bool is_ascii);
void write_table_hdu(fitsfile* fptr, nb::dict tensor_dict, nb::dict header);
void* get_fptr_from_python_object(nanobind::object obj);

void invalidate_shared_meta(const std::string& filename);
void clear_shared_read_meta_cache();

} // namespace torchfits
