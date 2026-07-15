#undef READONLY
#include <string>
#include <algorithm>
#include <cctype>
#include <vector>
#include <unordered_map>
#include <memory>
#include <mutex>
#include <atomic>
#include <limits>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/tuple.h>
#include <fitsio.h>

#include "torchfits_torch.h"
#include "torch_compat.h"
#include "cache.h"
#include "security.h"
#include "hardware.h"
#include "fits_detail.h"
#include "fits_file.h"
#include "fits_rw.h"

namespace torchfits {

// ---------------------------------------------------------------------------
// BScaleGuard — temporarily disable BSCALE/BZERO for raw read
// ---------------------------------------------------------------------------
struct BScaleGuard {
    fitsfile* fptr = nullptr;
    double bscale = 1.0;
    double bzero = 0.0;
    bool active = false;
    BScaleGuard() = default;
    explicit BScaleGuard(fitsfile* f) : fptr(f) {}
    ~BScaleGuard() {
        if (!active || !fptr) return;
        int status = 0;
        fits_set_bscale(fptr, bscale, bzero, &status);
    }
};

// ===========================================================================
// FITSFile implementation
// ===========================================================================
FITSFile::FITSFile(const char* filename, int mode) : filename_(filename), mode_(mode) {
    check_fits_filename_security(filename_);
    int status = 0;
    if (mode == 0) {
        fptr_ = get_or_open_cached(filename_);
        use_cache_ = true;
        if (!fptr_) status = 1;
    } else {
        fits_create_file(&fptr_, filename, &status);
        use_cache_ = false;
    }
    if (status != 0) throw std::runtime_error("Could not open FITS file: " + filename_);
    cached_ = false;
    if (mode == 0) shared_meta_ = detail::get_shared_meta_for_path(filename_);
    const bool has_extension = filename_.find('[') != std::string::npos;
    if (!has_extension) {
        start_hdu_ = 1;
        current_hdu_ = -1;
        if (shared_meta_) {
            std::lock_guard<std::mutex> lock(shared_meta_->mutex);
            // Shared fptr remembers its HDU across one-shot wrappers.
            current_hdu_ = shared_meta_->current_fits_hdu;
        }
    } else {
        fits_get_hdu_num(fptr_, &start_hdu_);
        current_hdu_ = start_hdu_;
    }
}

FITSFile::~FITSFile() { close(); }

void FITSFile::close() {
    close_raw_fd();
    if (fptr_) {
        if (use_cache_) release_cached(filename_);
        else { int status = 0; fits_close_file(fptr_, &status); }
        fptr_ = nullptr;
    }
}

void FITSFile::ensure_hdu(int hdu_num, int* status) {
    if (!fptr_) throw std::runtime_error("FITSFile is closed");
    int target_hdu = hdu_num + start_hdu_;
    if (current_hdu_ != target_hdu) {
        fits_movabs_hdu(fptr_, target_hdu, nullptr, status);
        if (*status == 0) {
            current_hdu_ = target_hdu;
            if (shared_meta_) {
                std::lock_guard<std::mutex> lock(shared_meta_->mutex);
                shared_meta_->current_fits_hdu = target_hdu;
            }
        }
    }
}

const FITSFile::ScaleInfo& FITSFile::get_scale_info(int hdu_num, int bitpix) {
    auto it = scale_cache_.find(hdu_num);
    if (it != scale_cache_.end()) return it->second;
    if (shared_meta_) {
        std::lock_guard<std::mutex> lock(shared_meta_->mutex);
        auto sit = shared_meta_->scale_cache.find(hdu_num);
        if (sit != shared_meta_->scale_cache.end()) {
            auto [scaled, trusted, bscale, bzero] = sit->second;
            ScaleInfo shared_info;
            shared_info.scaled = scaled; shared_info.trusted = trusted;
            shared_info.bscale = bscale; shared_info.bzero = bzero;
            auto inserted = scale_cache_.emplace(hdu_num, shared_info);
            return inserted.first->second;
        }
    }
    ScaleInfo info;
    const auto detected = detail::detect_scale_info_fast(fptr_, bitpix);
    info.scaled = detected.scaled; info.trusted = detected.trusted;
    info.bscale = detected.bscale; info.bzero = detected.bzero;
    auto inserted = scale_cache_.emplace(hdu_num, info);
    if (shared_meta_) {
        std::lock_guard<std::mutex> lock(shared_meta_->mutex);
        shared_meta_->scale_cache[hdu_num] = std::make_tuple(
            info.scaled, info.trusted, info.bscale, info.bzero);
    }
    return inserted.first->second;
}

FITSFile::ScaleInfo FITSFile::get_scale_info_for_hdu(int hdu_num) {
    const auto& info = get_image_info(hdu_num);
    return get_scale_info(hdu_num, std::get<0>(info));
}

bool FITSFile::is_compressed_image_cached(int hdu_num) {
    auto it = compressed_cache_.find(hdu_num);
    if (it != compressed_cache_.end()) return it->second;
    if (shared_meta_) {
        std::lock_guard<std::mutex> lock(shared_meta_->mutex);
        auto sit = shared_meta_->compressed_cache.find(hdu_num);
        if (sit != shared_meta_->compressed_cache.end()) {
            compressed_cache_[hdu_num] = sit->second;
            return sit->second;
        }
    }
    int status = 0;
    int is_compressed = fits_is_compressed_image(fptr_, &status);
    bool result = (status == 0 && is_compressed);
    compressed_cache_[hdu_num] = result;
    if (shared_meta_) {
        std::lock_guard<std::mutex> lock(shared_meta_->mutex);
        shared_meta_->compressed_cache[hdu_num] = result;
    }
    return result;
}

bool FITSFile::has_compressed_nulls_cached(int hdu_num) {
    auto it = compressed_nulls_cache_.find(hdu_num);
    if (it != compressed_nulls_cache_.end()) return it->second;
    if (shared_meta_) {
        std::lock_guard<std::mutex> lock(shared_meta_->mutex);
        auto sit = shared_meta_->compressed_nulls_cache.find(hdu_num);
        if (sit != shared_meta_->compressed_nulls_cache.end()) {
            compressed_nulls_cache_[hdu_num] = sit->second;
            return sit->second;
        }
    }
    bool result = detail::has_compressed_nulls(fptr_);
    compressed_nulls_cache_[hdu_num] = result;
    if (shared_meta_) {
        std::lock_guard<std::mutex> lock(shared_meta_->mutex);
        shared_meta_->compressed_nulls_cache[hdu_num] = result;
    }
    return result;
}

const std::tuple<int, int, std::array<LONGLONG, 9>>& FITSFile::get_image_info(int hdu_num) {
    auto it = image_info_cache_.find(hdu_num);
    if (it != image_info_cache_.end()) return it->second;
    if (shared_meta_) {
        std::lock_guard<std::mutex> lock(shared_meta_->mutex);
        auto sit = shared_meta_->image_info_cache.find(hdu_num);
        if (sit != shared_meta_->image_info_cache.end()) {
            auto inserted = image_info_cache_.emplace(hdu_num, sit->second);
            return inserted.first->second;
        }
    }
    int status = 0;
    int bitpix = 0;
    int naxis = 0;
    std::array<LONGLONG, 9> naxes_ll{};
    naxes_ll.fill(0);
    fits_get_img_paramll(fptr_, 9, &bitpix, &naxis, naxes_ll.data(), &status);
    if (status != 0) throw std::runtime_error("Could not read image parameters");
    auto inserted = image_info_cache_.emplace(hdu_num, std::make_tuple(bitpix, naxis, naxes_ll));
    if (shared_meta_) {
        std::lock_guard<std::mutex> lock(shared_meta_->mutex);
        shared_meta_->image_info_cache[hdu_num] = inserted.first->second;
    }
    return inserted.first->second;
}

torch::Tensor FITSFile::read_tensor(int hdu_num, bool use_mmap) {
    int status = 0;
    ensure_hdu(hdu_num, &status);
    if (status != 0) throw std::runtime_error("Could not move to HDU");
    int bitpix = 0, naxis = 0;
    std::array<LONGLONG, 9> naxes_ll{};
    naxes_ll.fill(0);
    // Direct paramll (skip shared_meta locks) — local FITSFile caches die with
    // the one-shot wrapper anyway.
    fits_get_img_paramll(fptr_, 9, &bitpix, &naxis, naxes_ll.data(), &status);
    if (status != 0) throw std::runtime_error("Could not read image parameters");
    if (naxis == 0) {
        torch::ScalarType dtype;
        switch (bitpix) {
            case BYTE_IMG:   dtype = torch::kUInt8; break;
            case SHORT_IMG:  dtype = torch::kInt16; break;
            case LONG_IMG:   dtype = torch::kInt32; break;
            case LONGLONG_IMG: dtype = torch::kInt64; break;
            case FLOAT_IMG:  dtype = torch::kFloat32; break;
            case DOUBLE_IMG: dtype = torch::kFloat64; break;
            default:         dtype = torch::kUInt8; break;
        }
        return torch::empty({0}, torch::TensorOptions().dtype(dtype));
    }
    detail::ResolvedFITSMeta meta;
    meta.bitpix = bitpix;
    meta.naxis = naxis;
    meta.naxes_ll = naxes_ll;

    // Float/double thin path when mmap cannot help: either the caller asked
    // mmap-off or the HDU is CompImage (tile decode). Skip scale/null/fd probes.
    const bool float_like = (bitpix == FLOAT_IMG || bitpix == DOUBLE_IMG);
    bool take_thin_float = !use_mmap && float_like;
    if (!take_thin_float && float_like && use_mmap) {
        int st = 0;
        const int is_comp = fits_is_compressed_image(fptr_, &st);
        take_thin_float = (st == 0) && (is_comp != 0);
    }
    if (take_thin_float) {
        meta.scaled = false;
        meta.bscale = 1.0;
        meta.bzero = 0.0;
        meta.compressed = false;
        meta.compressed_nulls = false;
        return detail::read_tensor_canonical(
            fptr_, filename_, meta, /*use_mmap=*/false, /*raw_fd=*/-1, /*use_chunking=*/false);
    }

    image_info_cache_[hdu_num] = std::make_tuple(bitpix, naxis, naxes_ll);
    if (shared_meta_) {
        std::lock_guard<std::mutex> lock(shared_meta_->mutex);
        shared_meta_->image_info_cache[hdu_num] = image_info_cache_[hdu_num];
    }

    const auto& scale_info = get_scale_info(hdu_num, bitpix);
    meta.scaled = scale_info.scaled;
    meta.bscale = scale_info.bscale;
    meta.bzero = scale_info.bzero;
    meta.compressed = is_compressed_image_cached(hdu_num);
    meta.compressed_nulls = meta.compressed ? has_compressed_nulls_cached(hdu_num) : false;

    const int fd = meta.compressed ? -1 : detail::get_shared_raw_fd(shared_meta_, filename_);
    return detail::read_tensor_canonical(fptr_, filename_, meta, use_mmap, fd, /*use_chunking=*/false);
}

torch::Tensor FITSFile::read_image_raw(int hdu_num, bool use_mmap) {
    int status = 0;
    ensure_hdu(hdu_num, &status);
    if (status != 0) throw std::runtime_error("Could not move to HDU");
    const bool want_mmap = use_mmap;
    int bitpix = 0, naxis = 0;
    std::array<LONGLONG, 9> naxes_ll{};
    {
        const auto& info = get_image_info(hdu_num);
        bitpix = std::get<0>(info);
        naxis = std::get<1>(info);
        naxes_ll = std::get<2>(info);
    }
    if (naxis == 0) {
        torch::ScalarType dtype;
        switch (bitpix) {
            case BYTE_IMG:   dtype = torch::kUInt8; break;
            case SHORT_IMG:  dtype = torch::kInt16; break;
            case LONG_IMG:   dtype = torch::kInt32; break;
            case LONGLONG_IMG: dtype = torch::kInt64; break;
            case FLOAT_IMG:  dtype = torch::kFloat32; break;
            case DOUBLE_IMG: dtype = torch::kFloat64; break;
            default:         dtype = torch::kUInt8; break;
        }
        return torch::empty({0}, torch::TensorOptions().dtype(dtype));
    }
    torch::ScalarType dtype;
    int datatype;
    switch (bitpix) {
        case BYTE_IMG:     dtype = torch::kUInt8;  datatype = TBYTE;      break;
        case SHORT_IMG:    dtype = torch::kInt16;  datatype = TSHORT;     break;
        case LONG_IMG:     dtype = torch::kInt32;  datatype = TINT;       break;
        case LONGLONG_IMG: dtype = torch::kInt64;  datatype = TLONGLONG;  break;
        case FLOAT_IMG:    dtype = torch::kFloat32; datatype = TFLOAT;     break;
        case DOUBLE_IMG:   dtype = torch::kFloat64; datatype = TDOUBLE;    break;
        default: throw std::runtime_error("Unsupported BITPIX");
    }
    LONGLONG nelements = 0;
    if (naxis > 0) {
        nelements = 1;
        for (int i = 0; i < naxis; ++i) nelements *= naxes_ll[i];
    }
    int64_t torch_shape[9];
    for (int i = 0; i < naxis; ++i) torch_shape[i] = static_cast<int64_t>(naxes_ll[naxis - 1 - i]);
    auto tensor = torch::empty(at::IntArrayRef(torch_shape, naxis), torch::TensorOptions().dtype(dtype));
    const bool compressed = is_compressed_image_cached(hdu_num);
    if (want_mmap && !compressed && bitpix == BYTE_IMG) {
        if (filename_.find('[') == std::string::npos) {
            LONGLONG headstart = 0, data_offset = 0, dataend = 0;
            status = 0;
            fits_get_hduaddrll(fptr_, &headstart, &data_offset, &dataend, &status);
            if (status == 0 && data_offset > 0) {
                const size_t nbytes = static_cast<size_t>(nelements);
                if (nbytes > 0) {
                    const size_t end_off = static_cast<size_t>(data_offset) + nbytes;
                    if (ensure_raw_fd(end_off)) {
                        uint8_t* dst = static_cast<uint8_t*>(tensor.data_ptr());
                        size_t remaining = nbytes;
                        off_t off = static_cast<off_t>(data_offset);
                        bool ok = true;
                        while (remaining > 0) {
                            ssize_t got = ::pread(raw_fd_, dst, remaining, off);
                            if (got < 0) { if (errno == EINTR) continue; ok = false; break; }
                            if (got == 0) { ok = false; break; }
                            dst += static_cast<size_t>(got);
                            off += static_cast<off_t>(got);
                            remaining -= static_cast<size_t>(got);
                        }
                        if (ok) return tensor;
                        void* map_ptr = mmap(nullptr, static_cast<size_t>(raw_file_size_),
                                             PROT_READ, MAP_SHARED, raw_fd_, 0);
                        if (map_ptr != MAP_FAILED) {
                            const uint8_t* src = static_cast<const uint8_t*>(map_ptr) + data_offset;
                            std::memcpy(tensor.data_ptr(), src, nbytes);
                            munmap(map_ptr, static_cast<size_t>(raw_file_size_));
                            return tensor;
                        }
                    }
                }
            } else { status = 0; }
        }
    }
    BScaleGuard guard(fptr_);
    const bool needs_bscale_guard = (bitpix != FLOAT_IMG && bitpix != DOUBLE_IMG);
    if (needs_bscale_guard) {
        int key_status = 0;
        double bscale = 1.0, bzero = 0.0;
        key_status = 0;
        fits_read_key(fptr_, TDOUBLE, "BSCALE", &bscale, nullptr, &key_status);
        if (key_status != 0 && key_status != KEY_NO_EXIST) bscale = 1.0;
        key_status = 0;
        fits_read_key(fptr_, TDOUBLE, "BZERO", &bzero, nullptr, &key_status);
        if (key_status != 0 && key_status != KEY_NO_EXIST) bzero = 0.0;
        guard.bscale = bscale; guard.bzero = bzero;
        status = 0;
        fits_set_bscale(fptr_, 1.0, 0.0, &status);
        if (status == 0) guard.active = true;
        else status = 0;
    }
    int anynul = 0;
    float fnullval = NAN;
    double dnullval = NAN;
    void* nullval_ptr = nullptr;
    if ((datatype == TFLOAT || datatype == TDOUBLE) && compressed &&
        has_compressed_nulls_cached(hdu_num))
        nullval_ptr = (datatype == TFLOAT) ? (void*)&fnullval : (void*)&dnullval;
    fits_read_img(fptr_, datatype, 1, nelements, nullval_ptr, tensor.data_ptr(), &anynul, &status);
    if (status != 0) {
        char err_text[31];
        fits_get_errstatus(status, err_text);
        throw std::runtime_error("Error reading image data: status=" + std::to_string(status) +
                                 " msg=" + std::string(err_text));
    }
    return tensor;
}

bool FITSFile::write_image(nb::ndarray<> tensor, int hdu_num, double bscale, double bzero) {
    int status = 0;
    int naxis = tensor.ndim();
    std::vector<long> naxes(naxis);
    for (int i = 0; i < naxis; ++i) naxes[i] = tensor.shape(i);
    std::reverse(naxes.begin(), naxes.end());
    int bitpix = FLOAT_IMG;
    int datatype = TFLOAT;
    nb::dlpack::dtype dt = tensor.dtype();
    if (dt.code == (uint8_t)nb::dlpack::dtype_code::UInt && dt.bits == 8) { bitpix = BYTE_IMG; datatype = TBYTE; }
    else if (dt.code == (uint8_t)nb::dlpack::dtype_code::Int && dt.bits == 16) { bitpix = SHORT_IMG; datatype = TSHORT; }
    else if (dt.code == (uint8_t)nb::dlpack::dtype_code::Int && dt.bits == 32) { bitpix = LONG_IMG; datatype = TINT; }
    else if (dt.code == (uint8_t)nb::dlpack::dtype_code::Float && dt.bits == 32) { bitpix = FLOAT_IMG; datatype = TFLOAT; }
    else if (dt.code == (uint8_t)nb::dlpack::dtype_code::Float && dt.bits == 64) { bitpix = DOUBLE_IMG; datatype = TDOUBLE; }
    else throw std::runtime_error("Unsupported tensor dtype");
    long nelements = 1;
    for (long dim : naxes) nelements *= dim;
    fits_create_img(fptr_, bitpix, naxis, naxes.data(), &status);
    if (status != 0) {
        char err_text[31];
        fits_get_errstatus(status, err_text);
        throw std::runtime_error("Error creating image: status=" + std::to_string(status) +
                                 " msg=" + std::string(err_text));
    }
    fits_write_img(fptr_, datatype, 1, nelements, tensor.data(), &status);
    if (status != 0) {
        char err_text[31];
        fits_get_errstatus(status, err_text);
        throw std::runtime_error("Error writing image: status=" + std::to_string(status) +
                                 " msg=" + std::string(err_text));
    }
    return true;
}

std::vector<std::tuple<std::string, std::string, std::string>> FITSFile::get_header(int hdu_num) {
    int status = 0;
    ensure_hdu(hdu_num, &status);
    if (status != 0) throw std::runtime_error("Could not move to HDU");
    int nkeys = 0, morekeys = 0;
    fits_get_hdrspace(fptr_, &nkeys, &morekeys, &status);
    std::vector<std::tuple<std::string, std::string, std::string>> header;
    header.reserve(nkeys);
    char keyname[FLEN_KEYWORD], value[FLEN_VALUE], comment[FLEN_COMMENT];
    for (int i = 1; i <= nkeys; i++) {
        fits_read_keyn(fptr_, i, keyname, value, comment, &status);
        if (status == 0) {
            std::string key_str(keyname), val_str(value), com_str(comment);
            val_str.erase(std::remove_if(val_str.begin(), val_str.end(), [](unsigned char c) {
                return c < 32 || c > 126;
            }), val_str.end());
            if (val_str.length() >= 2 && val_str.front() == '\'') {
                size_t last_quote = val_str.rfind('\'');
                if (last_quote != std::string::npos && last_quote > 0) {
                    val_str = val_str.substr(1, last_quote - 1);
                    size_t last_char = val_str.find_last_not_of(' ');
                    if (last_char != std::string::npos) val_str = val_str.substr(0, last_char + 1);
                    else val_str = "";
                    size_t pos = 0;
                    while ((pos = val_str.find("''", pos)) != std::string::npos) {
                        val_str.replace(pos, 2, "'"); pos += 1;
                    }
                }
            }
            if (key_str == "HISTORY" || key_str == "COMMENT") {
                if (val_str.empty() && !com_str.empty()) {
                    val_str = com_str; com_str = "";
                }
            }
            header.emplace_back(key_str, val_str, com_str);
        } else { status = 0; }
    }
    return header;
}

std::vector<long> FITSFile::get_shape(int hdu_num) {
    int status = 0;
    ensure_hdu(hdu_num, &status);
    int naxis = 0;
    fits_get_img_dim(fptr_, &naxis, &status);
    std::vector<long> naxes(naxis);
    fits_get_img_size(fptr_, naxis, naxes.data(), &status);
    std::reverse(naxes.begin(), naxes.end());
    return naxes;
}

int FITSFile::get_dtype(int hdu_num) {
    int status = 0;
    ensure_hdu(hdu_num, &status);
    int bitpix = 0;
    fits_get_img_type(fptr_, &bitpix, &status);
    return bitpix;
}

torch::Tensor FITSFile::read_subset(int hdu_num, long x1, long y1, long x2, long y2) {
    int status = 0;
    ensure_hdu(hdu_num, &status);
    if (status != 0) throw std::runtime_error("Could not move to HDU");
    int bitpix = 0, naxis = 0;
    long naxes[9] = {0};
    {
        const auto& info = get_image_info(hdu_num);
        bitpix = std::get<0>(info);
        naxis = std::get<1>(info);
        const auto& naxes_ll = std::get<2>(info);
        for (int i = 0; i < 9; ++i) naxes[i] = static_cast<long>(naxes_ll[i]);
    }
    if (naxis < 2) throw std::runtime_error("Subset reading requires at least 2D image");
    long max_x = naxes[0], max_y = naxes[1];
    if (x1 < 0) x1 = 0; if (y1 < 0) y1 = 0;
    if (x2 > max_x) x2 = max_x; if (y2 > max_y) y2 = max_y;
    if (x2 <= x1 || y2 <= y1)
        return torch::empty({0, 0}, torch::TensorOptions().dtype(torch::kFloat32));
    const auto& scale_info = get_scale_info(hdu_num, bitpix);
    bool scaled = scale_info.scaled;
    torch::ScalarType dtype;
    int datatype;
    if (scaled) { dtype = torch::kFloat32; datatype = TFLOAT; }
    else {
        switch (bitpix) {
            case BYTE_IMG:     dtype = torch::kUInt8;  datatype = TBYTE;      break;
            case SHORT_IMG:    dtype = torch::kInt16;  datatype = TSHORT;     break;
            case LONG_IMG:     dtype = torch::kInt32;  datatype = TINT;       break;
            case LONGLONG_IMG: dtype = torch::kInt64;  datatype = TLONGLONG;  break;
            case FLOAT_IMG:    dtype = torch::kFloat32; datatype = TFLOAT;     break;
            case DOUBLE_IMG:   dtype = torch::kFloat64; datatype = TDOUBLE;    break;
            default: throw std::runtime_error("Unsupported BITPIX");
        }
    }
    long width = x2 - x1, height = y2 - y1;
    auto tensor = torch::empty(std::vector<int64_t>{height, width}, torch::TensorOptions().dtype(dtype));
    std::vector<long> fpixel(naxis, 1), lpixel(naxis, 1), inc(naxis, 1);
    fpixel[0] = x1 + 1; fpixel[1] = y1 + 1;
    lpixel[0] = x2; lpixel[1] = y2;
    int anynul = 0;
    fits_read_subset(fptr_, datatype, fpixel.data(), lpixel.data(), inc.data(),
                     nullptr, tensor.data_ptr(), &anynul, &status);
    if (status != 0) {
        char err_text[31];
        fits_get_errstatus(status, err_text);
        throw std::runtime_error("Error reading subset: status=" + std::to_string(status) +
                                 " msg=" + std::string(err_text));
    }
    return tensor;
}

int FITSFile::get_num_hdus() {
    int status = 0, nhdus = 0;
    fits_get_num_hdus(fptr_, &nhdus, &status);
    return nhdus;
}

std::string FITSFile::get_hdu_type(int hdu_num) {
    int status = 0;
    ensure_hdu(hdu_num, &status);
    int hdutype = 0;
    fits_get_hdu_type(fptr_, &hdutype, &status);
    if (hdutype == IMAGE_HDU) return "IMAGE";
    if (hdutype == ASCII_TBL) return "ASCII_TABLE";
    if (hdutype == BINARY_TBL) return "BINARY_TABLE";
    return "UNKNOWN";
}

bool FITSFile::write_hdus(nb::list hdus, bool /*overwrite*/) {
    int hdu_count = 0;
    for (auto handle : hdus) {
        nb::object hdu_obj = nb::cast<nb::object>(handle);
        if (nb::hasattr(hdu_obj, "_raw_data")) {
            nb::dict data_dict = nb::cast<nb::dict>(hdu_obj.attr("_raw_data"));
            nb::dict header_dict;
            if (nb::hasattr(hdu_obj, "header"))
                header_dict = nb::cast<nb::dict>(hdu_obj.attr("header"));
            nb::object schema_obj = nb::none();
            if (nb::hasattr(hdu_obj, "_schema")) {
                schema_obj = hdu_obj.attr("_schema");
                if (schema_obj.is_none()) schema_obj = nb::none();
            }
            write_table_hdu(fptr_, data_dict, header_dict, schema_obj, false);
            hdu_count++;
            continue;
        }
        nb::object data_obj;
        bool has_data = false;
        if (nb::hasattr(hdu_obj, "to_tensor")) {
            data_obj = hdu_obj.attr("to_tensor")();
            has_data = true;
        }
        if (!has_data && nb::hasattr(hdu_obj, "data")) {
            data_obj = hdu_obj.attr("data"); has_data = true;
        } else if (!has_data && nb::isinstance<nb::dict>(hdu_obj)) {
            nb::dict d = nb::cast<nb::dict>(hdu_obj);
            if (d.contains("data")) { data_obj = d["data"]; has_data = true; }
        }
        if (has_data) {
            nb::ndarray<> tensor = nb::cast<nb::ndarray<>>(data_obj);
            write_image(tensor, hdu_count, 1.0, 0.0);
        } else {
            int status = 0;
            long naxes[1] = {0};
            fits_create_img(fptr_, BYTE_IMG, 0, naxes, &status);
            if (status != 0) throw std::runtime_error("Failed to create empty image HDU");
        }
        nb::object header_obj;
        if (nb::hasattr(hdu_obj, "header"))
            header_obj = hdu_obj.attr("header");
        else if (nb::isinstance<nb::dict>(hdu_obj)) {
            nb::dict d = nb::cast<nb::dict>(hdu_obj);
            if (d.contains("header")) header_obj = d["header"];
        }
        if (header_obj.is_valid()) {
            nb::dict header = nb::cast<nb::dict>(header_obj);
            for (auto item : header) {
                std::string key = nb::cast<std::string>(item.first);
                key = detail::sanitize_fits_key(key);
                std::string key_upper = key;
                std::transform(key_upper.begin(), key_upper.end(), key_upper.begin(),
                               [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
                if (key_upper == "END" || key_upper == "SIMPLE" || key_upper == "XTENSION" ||
                    key_upper == "BITPIX" || key_upper == "NAXIS" || key_upper == "EXTEND" ||
                    key_upper == "PCOUNT" || key_upper == "GCOUNT" || key_upper == "TFIELDS" ||
                    key_upper == "THEAP" || key_upper.rfind("NAXIS", 0) == 0) continue;
                int key_status = 0;
                if (nb::isinstance<bool>(item.second)) {
                    int val = nb::cast<bool>(item.second) ? 1 : 0;
                    fits_update_key(fptr_, TLOGICAL, key.c_str(), &val, nullptr, &key_status);
                } else if (nb::isinstance<nb::str>(item.second)) {
                    std::string val = detail::sanitize_fits_string(nb::cast<std::string>(item.second));
                    fits_update_key(fptr_, TSTRING, key.c_str(), (void*)val.c_str(), nullptr, &key_status);
                } else if (nb::isinstance<int>(item.second)) {
                    long long val = nb::cast<long long>(item.second);
                    fits_update_key(fptr_, TLONGLONG, key.c_str(), &val, nullptr, &key_status);
                } else if (nb::isinstance<double>(item.second) || nb::isinstance<float>(item.second)) {
                    double val = nb::cast<double>(item.second);
                    fits_update_key(fptr_, TDOUBLE, key.c_str(), &val, nullptr, &key_status);
                }
                if (key_status != 0) {
                    throw std::runtime_error("Failed to write FITS header keyword: " + key);
                }
            }
        }
        hdu_count++;
    }
    return true;
}

bool FITSFile::write_hdus_compressed_images(nb::list hdus, int compression_type) {
    int status = 0;
    long naxes0[1] = {0};
    fits_create_img(fptr_, BYTE_IMG, 0, naxes0, &status);
    if (status != 0) throw std::runtime_error("Failed to create primary HDU for compressed file");
    auto write_header_dict = [&](nb::object hdu_obj) {
        nb::object header_obj;
        if (nb::hasattr(hdu_obj, "header"))
            header_obj = hdu_obj.attr("header");
        else if (nb::isinstance<nb::dict>(hdu_obj)) {
            nb::dict d = nb::cast<nb::dict>(hdu_obj);
            if (d.contains("header")) header_obj = d["header"];
        }
        if (!header_obj.is_valid()) return;
        nb::dict header = nb::cast<nb::dict>(header_obj);
        for (auto item : header) {
            std::string key = nb::cast<std::string>(item.first);
            key = detail::sanitize_fits_key(key);
            std::string key_upper = key;
            std::transform(key_upper.begin(), key_upper.end(), key_upper.begin(),
                           [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
            if (key_upper == "END" || key_upper == "SIMPLE" || key_upper == "XTENSION" ||
                key_upper == "BITPIX" || key_upper == "NAXIS" || key_upper == "EXTEND" ||
                key_upper == "PCOUNT" || key_upper == "GCOUNT" || key_upper == "TFIELDS" ||
                key_upper == "THEAP" || key_upper == "DATASUM" || key_upper == "CHECKSUM" ||
                key_upper == "ZIMAGE" || key_upper == "ZCMPTYPE" || key_upper == "ZBITPIX" ||
                key_upper == "ZNAXIS" || key_upper == "ZPCOUNT" || key_upper == "ZGCOUNT" ||
                key_upper == "ZHECKSUM" || key_upper == "ZDATASUM" ||
                key_upper.rfind("NAXIS", 0) == 0) continue;
            if (key_upper.rfind("ZNAXIS", 0) == 0 || key_upper.rfind("ZTILE", 0) == 0 ||
                key_upper.rfind("ZNAME", 0) == 0 || key_upper.rfind("ZVAL", 0) == 0) continue;
            int key_status = 0;
            if (nb::isinstance<bool>(item.second)) {
                int val = nb::cast<bool>(item.second) ? 1 : 0;
                fits_update_key(fptr_, TLOGICAL, key.c_str(), &val, nullptr, &key_status);
            } else if (nb::isinstance<nb::str>(item.second)) {
                    std::string val = detail::sanitize_fits_string(nb::cast<std::string>(item.second));
                    fits_update_key(fptr_, TSTRING, key.c_str(), (void*)val.c_str(), nullptr, &key_status);
            } else if (nb::isinstance<int>(item.second)) {
                    long long val = nb::cast<long long>(item.second);
                    fits_update_key(fptr_, TLONGLONG, key.c_str(), &val, nullptr, &key_status);
            } else if (nb::isinstance<double>(item.second) || nb::isinstance<float>(item.second)) {
                    double val = nb::cast<double>(item.second);
                    fits_update_key(fptr_, TDOUBLE, key.c_str(), &val, nullptr, &key_status);
            }
            if (key_status != 0) {
                throw std::runtime_error("Failed to write FITS header keyword: " + key);
            }
        }
    };
    for (auto handle : hdus) {
        nb::object hdu_obj = nb::cast<nb::object>(handle);
        if (nb::hasattr(hdu_obj, "_raw_data")) {
            nb::dict data_dict = nb::cast<nb::dict>(hdu_obj.attr("_raw_data"));
            nb::dict header_dict;
            if (nb::hasattr(hdu_obj, "header"))
                header_dict = nb::cast<nb::dict>(hdu_obj.attr("header"));
            write_table_hdu(fptr_, data_dict, header_dict, nb::none(), false);
            continue;
        }
        nb::object data_obj;
        bool has_data = false;
        if (nb::hasattr(hdu_obj, "to_tensor")) {
            data_obj = hdu_obj.attr("to_tensor")();
            has_data = true;
        }
        if (!has_data && nb::hasattr(hdu_obj, "data")) {
            data_obj = hdu_obj.attr("data"); has_data = true;
        } else if (!has_data && nb::isinstance<nb::dict>(hdu_obj)) {
            nb::dict d = nb::cast<nb::dict>(hdu_obj);
            if (d.contains("data")) { data_obj = d["data"]; has_data = true; }
        }
        if (!has_data) throw std::runtime_error("Compressed writing requires image data");
        nb::ndarray<> tensor = nb::cast<nb::ndarray<>>(data_obj);
        int naxis = tensor.ndim();
        if (naxis <= 0) throw std::runtime_error("Unsupported image ndim for compressed write");
        std::vector<long> naxes(naxis);
        for (int i = 0; i < naxis; ++i) naxes[i] = tensor.shape(i);
        std::reverse(naxes.begin(), naxes.end());
        status = 0;
        fits_set_compression_type(fptr_, compression_type, &status);
        if (status != 0) throw std::runtime_error("Failed to set compression type");
        std::vector<long> tilesize(naxis, 1);
        tilesize[0] = naxes[0];
        fits_set_tile_dim(fptr_, naxis, tilesize.data(), &status);
        if (status != 0) throw std::runtime_error("Failed to set tile dimensions");
        int bitpix = FLOAT_IMG, datatype = TFLOAT;
        nb::dlpack::dtype dt = tensor.dtype();
        if (dt.code == (uint8_t)nb::dlpack::dtype_code::UInt && dt.bits == 8) { bitpix = BYTE_IMG; datatype = TBYTE; }
        else if (dt.code == (uint8_t)nb::dlpack::dtype_code::Int && dt.bits == 8) { bitpix = SBYTE_IMG; datatype = TSBYTE; }
        else if (dt.code == (uint8_t)nb::dlpack::dtype_code::Int && dt.bits == 16) { bitpix = SHORT_IMG; datatype = TSHORT; }
        else if (dt.code == (uint8_t)nb::dlpack::dtype_code::Int && dt.bits == 32) { bitpix = LONG_IMG; datatype = TINT; }
        else if (dt.code == (uint8_t)nb::dlpack::dtype_code::Int && dt.bits == 64) { bitpix = LONGLONG_IMG; datatype = TLONGLONG; }
        else if (dt.code == (uint8_t)nb::dlpack::dtype_code::Float && dt.bits == 32) { bitpix = FLOAT_IMG; datatype = TFLOAT; }
        else if (dt.code == (uint8_t)nb::dlpack::dtype_code::Float && dt.bits == 64) { bitpix = DOUBLE_IMG; datatype = TDOUBLE; }
        else throw std::runtime_error("Unsupported tensor dtype for compressed write");
        fits_create_img(fptr_, bitpix, naxis, naxes.data(), &status);
        if (status != 0) {
            char err_text[31];
            fits_get_errstatus(status, err_text);
            throw std::runtime_error("Error creating compressed image: status=" + std::to_string(status) +
                                     " msg=" + std::string(err_text));
        }
        long nelements = 1;
        for (long dim : naxes) nelements *= dim;
        fits_write_img(fptr_, datatype, 1, nelements, tensor.data(), &status);
        if (status != 0) {
            char err_text[31];
            fits_get_errstatus(status, err_text);
            throw std::runtime_error("Error writing compressed image: status=" + std::to_string(status) +
                                     " msg=" + std::string(err_text));
        }
        write_header_dict(hdu_obj);
    }
    return true;
}

std::string FITSFile::read_header_to_string(int hdu_num) {
    int status = 0;
    ensure_hdu(hdu_num, &status);
    char* header_str = nullptr;
    int nkeys = 0;
    if (fits_hdr2str(fptr_, 0, nullptr, 0, &header_str, &nkeys, &status)) return "";
    std::string result(header_str);
    if (header_str) fits_free_memory(header_str, &status);
    return result;
}

bool FITSFile::ensure_raw_fd(size_t required_end) {
    if (filename_.find('[') != std::string::npos) return false;
    if (!raw_fd_ready_) {
        raw_fd_ready_ = true;
        raw_fd_ = detail::open_readonly_fd(filename_);
        if (raw_fd_ == -1) return false;
        struct stat sb {};
        if (fstat(raw_fd_, &sb) != 0) {
            ::close(raw_fd_); raw_fd_ = -1; raw_file_size_ = 0; return false;
        }
        raw_file_size_ = sb.st_size;
    }
    return raw_fd_ != -1 && required_end <= static_cast<size_t>(raw_file_size_);
}

void FITSFile::close_raw_fd() {
    if (raw_fd_ != -1) { ::close(raw_fd_); raw_fd_ = -1; }
    raw_file_size_ = 0;
    raw_fd_ready_ = false;
}

// ===========================================================================
// SubsetReader implementation
// ===========================================================================
SubsetReader::SubsetReader(const std::string& filename, int hdu_num)
    : file_(filename.c_str(), 0), hdu_num_(hdu_num) {
    if (hdu_num_ < 0) throw std::runtime_error("HDU index must be non-negative");
    init_from_hdu();
}

void SubsetReader::init_from_hdu() {
    int status = 0;
    file_.ensure_hdu(hdu_num_, &status);
    if (status != 0) throw std::runtime_error("Could not move to HDU");
    const auto& info = file_.get_image_info(hdu_num_);
    const int bitpix = std::get<0>(info);
    const int naxis = std::get<1>(info);
    const auto& naxes = std::get<2>(info);
    if (naxis < 2) throw std::runtime_error("SubsetReader requires at least 2D image HDU");
    max_x_ = static_cast<long>(naxes[0]);
    max_y_ = static_cast<long>(naxes[1]);
    const auto scale = file_.get_scale_info_for_hdu(hdu_num_);
    // Match full-image logical dtypes — do not float-promote signed-byte /
    // unsigned integer conventions (that lost to fitsio int8 cutouts).
    if (scale.scaled && bitpix == BYTE_IMG && scale.bscale == 1.0 && scale.bzero == -128.0) {
        dtype_ = torch::kInt8; datatype_ = TSBYTE; return;
    }
    if (scale.scaled && bitpix == SHORT_IMG && scale.bscale == 1.0 && scale.bzero == 32768.0) {
        dtype_ = torch::kUInt16; datatype_ = TUSHORT; return;
    }
    if (scale.scaled && bitpix == LONG_IMG && scale.bscale == 1.0 && scale.bzero == 2147483648.0) {
        dtype_ = torch::kUInt32; datatype_ = TUINT; return;
    }
    if (scale.scaled) {
        dtype_ = torch::kFloat32; datatype_ = TFLOAT; return;
    }
    switch (bitpix) {
        case BYTE_IMG:     dtype_ = torch::kUInt8;  datatype_ = TBYTE;      break;
        case SHORT_IMG:    dtype_ = torch::kInt16;  datatype_ = TSHORT;     break;
        case LONG_IMG:     dtype_ = torch::kInt32;  datatype_ = TINT;       break;
        case LONGLONG_IMG: dtype_ = torch::kInt64;  datatype_ = TLONGLONG;  break;
        case FLOAT_IMG:    dtype_ = torch::kFloat32; datatype_ = TFLOAT;     break;
        case DOUBLE_IMG:   dtype_ = torch::kFloat64; datatype_ = TDOUBLE;    break;
        default: throw std::runtime_error("Unsupported BITPIX");
    }
}

torch::Tensor SubsetReader::read(long x1, long y1, long x2, long y2) {
    if (closed_) throw std::runtime_error("SubsetReader is closed");
    if (x1 < 0) x1 = 0; if (y1 < 0) y1 = 0;
    if (x2 > max_x_) x2 = max_x_; if (y2 > max_y_) y2 = max_y_;
    if (x2 <= x1 || y2 <= y1)
        return torch::empty({0, 0}, torch::TensorOptions().dtype(dtype_));
    long width = x2 - x1, height = y2 - y1;
    auto tensor = torch::empty({height, width}, torch::TensorOptions().dtype(dtype_));
    long fpixel[2] = {x1 + 1, y1 + 1};
    long lpixel[2] = {x2, y2};
    long inc[2] = {1, 1};
    int status = 0, anynul = 0;
    fits_read_subset(file_.get_fptr(), datatype_, fpixel, lpixel, inc,
                     nullptr, tensor.data_ptr(), &anynul, &status);
    if (status != 0) {
        char err_text[31];
        fits_get_errstatus(status, err_text);
        throw std::runtime_error("Error reading subset: status=" + std::to_string(status) +
                                 " msg=" + std::string(err_text));
    }
    return tensor;
}

void SubsetReader::close() {
    if (!closed_) { file_.close(); closed_ = true; }
}

} // namespace torchfits
