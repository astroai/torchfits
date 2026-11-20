#pragma once

// Torch-free tensor for the _C extension.
//
// The extension used to decode into at::Tensor and hand results to Python
// through libtorch_python, which tied every wheel to one PyTorch minor: there
// is no stable C++ ABI across minors. The decoders only ever used tensors as
// typed, contiguous, row-major buffers, so `native::Tensor` is exactly that --
// a shared buffer, a dtype and a shape -- plus the handful of element-wise
// post-processing steps the table reader applies (FITS scaling, null masks,
// unsigned-convention sign flips). Results reach Python as DLPack capsules
// (native_py.h), so torch is only ever touched through its Python API.
//
// Every tensor is C-contiguous. Views (slice along dim 0, dtype
// reinterpretation) share storage with their source, like torch views do.

#include <algorithm>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "core/parallel.h"

namespace torchfits {
namespace native {

enum class ScalarType : int8_t {
    Bool,
    UInt8,
    Int8,
    Int16,
    UInt16,
    Int32,
    UInt32,
    Int64,
    UInt64,
    Float16,
    BFloat16,
    Float32,
    Float64,
    ComplexFloat,
    ComplexDouble,
};

constexpr ScalarType kBool = ScalarType::Bool;
constexpr ScalarType kUInt8 = ScalarType::UInt8;
constexpr ScalarType kByte = ScalarType::UInt8;
constexpr ScalarType kInt8 = ScalarType::Int8;
constexpr ScalarType kChar = ScalarType::Int8;
constexpr ScalarType kInt16 = ScalarType::Int16;
constexpr ScalarType kShort = ScalarType::Int16;
constexpr ScalarType kUInt16 = ScalarType::UInt16;
constexpr ScalarType kInt32 = ScalarType::Int32;
constexpr ScalarType kInt = ScalarType::Int32;
constexpr ScalarType kUInt32 = ScalarType::UInt32;
constexpr ScalarType kInt64 = ScalarType::Int64;
constexpr ScalarType kLong = ScalarType::Int64;
constexpr ScalarType kUInt64 = ScalarType::UInt64;
constexpr ScalarType kFloat16 = ScalarType::Float16;
constexpr ScalarType kHalf = ScalarType::Float16;
constexpr ScalarType kBFloat16 = ScalarType::BFloat16;
constexpr ScalarType kFloat32 = ScalarType::Float32;
constexpr ScalarType kFloat = ScalarType::Float32;
constexpr ScalarType kFloat64 = ScalarType::Float64;
constexpr ScalarType kDouble = ScalarType::Float64;
constexpr ScalarType kComplexFloat = ScalarType::ComplexFloat;
constexpr ScalarType kComplexDouble = ScalarType::ComplexDouble;

inline size_t element_size(ScalarType t) {
    switch (t) {
        case ScalarType::Bool:
        case ScalarType::UInt8:
        case ScalarType::Int8:
            return 1;
        case ScalarType::Int16:
        case ScalarType::UInt16:
        case ScalarType::Float16:
        case ScalarType::BFloat16:
            return 2;
        case ScalarType::Int32:
        case ScalarType::UInt32:
        case ScalarType::Float32:
            return 4;
        case ScalarType::Int64:
        case ScalarType::UInt64:
        case ScalarType::Float64:
        case ScalarType::ComplexFloat:
            return 8;
        case ScalarType::ComplexDouble:
            return 16;
    }
    return 0;
}

// The torch/NumPy spelling of each dtype (also the raw Arrow transport name).
inline const char* dtype_name(ScalarType t) {
    switch (t) {
        case ScalarType::Bool: return "bool";
        case ScalarType::UInt8: return "uint8";
        case ScalarType::Int8: return "int8";
        case ScalarType::Int16: return "int16";
        case ScalarType::UInt16: return "uint16";
        case ScalarType::Int32: return "int32";
        case ScalarType::UInt32: return "uint32";
        case ScalarType::Int64: return "int64";
        case ScalarType::UInt64: return "uint64";
        case ScalarType::Float16: return "float16";
        case ScalarType::BFloat16: return "bfloat16";
        case ScalarType::Float32: return "float32";
        case ScalarType::Float64: return "float64";
        case ScalarType::ComplexFloat: return "complex64";
        case ScalarType::ComplexDouble: return "complex128";
    }
    return "unknown";
}

inline bool is_integral(ScalarType t, bool include_bool) {
    switch (t) {
        case ScalarType::Bool:
            return include_bool;
        case ScalarType::UInt8:
        case ScalarType::Int8:
        case ScalarType::Int16:
        case ScalarType::UInt16:
        case ScalarType::Int32:
        case ScalarType::UInt32:
        case ScalarType::Int64:
        case ScalarType::UInt64:
            return true;
        default:
            return false;
    }
}

template <typename T>
constexpr ScalarType scalar_type_of() {
    if constexpr (std::is_same_v<T, bool>) return ScalarType::Bool;
    else if constexpr (std::is_same_v<T, uint8_t>) return ScalarType::UInt8;
    else if constexpr (std::is_same_v<T, int8_t>) return ScalarType::Int8;
    else if constexpr (std::is_same_v<T, int16_t>) return ScalarType::Int16;
    else if constexpr (std::is_same_v<T, uint16_t>) return ScalarType::UInt16;
    else if constexpr (std::is_same_v<T, int32_t>) return ScalarType::Int32;
    else if constexpr (std::is_same_v<T, uint32_t>) return ScalarType::UInt32;
    else if constexpr (std::is_same_v<T, int64_t>) return ScalarType::Int64;
    else if constexpr (std::is_same_v<T, uint64_t>) return ScalarType::UInt64;
    else if constexpr (std::is_same_v<T, float>) return ScalarType::Float32;
    else if constexpr (std::is_same_v<T, double>) return ScalarType::Float64;
    else if constexpr (std::is_same_v<T, std::complex<float>>) return ScalarType::ComplexFloat;
    else if constexpr (std::is_same_v<T, std::complex<double>>) return ScalarType::ComplexDouble;
    else static_assert(sizeof(T) == 0, "unsupported element type");
}

struct Device {
    bool is_cpu() const { return true; }
};

struct TensorOptions {
    ScalarType dtype_ = ScalarType::Float32;
    TensorOptions dtype(ScalarType t) const {
        TensorOptions o = *this;
        o.dtype_ = t;
        return o;
    }
};

namespace detail {

// 64-byte alignment, the same as c10's CPU allocator, so SIMD byte-swap loops
// and downstream torch kernels see the alignment they always had. Contents are
// left uninitialized, like torch::empty.
inline std::shared_ptr<void> allocate(size_t nbytes) {
    void* p = nullptr;
    if (posix_memalign(&p, 64, nbytes == 0 ? 1 : nbytes) != 0 || p == nullptr) {
        throw std::bad_alloc();
    }
    return std::shared_ptr<void>(p, [](void* q) { std::free(q); });
}

// Applies f(T*) with the tensor's real/integer element pointer type.
template <typename F>
inline void visit_real(ScalarType t, void* data, F&& f) {
    switch (t) {
        case ScalarType::Bool: f(static_cast<bool*>(data)); return;
        case ScalarType::UInt8: f(static_cast<uint8_t*>(data)); return;
        case ScalarType::Int8: f(static_cast<int8_t*>(data)); return;
        case ScalarType::Int16: f(static_cast<int16_t*>(data)); return;
        case ScalarType::UInt16: f(static_cast<uint16_t*>(data)); return;
        case ScalarType::Int32: f(static_cast<int32_t*>(data)); return;
        case ScalarType::UInt32: f(static_cast<uint32_t*>(data)); return;
        case ScalarType::Int64: f(static_cast<int64_t*>(data)); return;
        case ScalarType::UInt64: f(static_cast<uint64_t*>(data)); return;
        case ScalarType::Float32: f(static_cast<float*>(data)); return;
        case ScalarType::Float64: f(static_cast<double*>(data)); return;
        default:
            throw std::runtime_error(
                std::string("unsupported dtype for this operation: ") + dtype_name(t));
    }
}

}  // namespace detail

class Tensor {
public:
    Tensor() = default;

    static Tensor empty(std::vector<int64_t> shape, ScalarType dtype) {
        Tensor t;
        t.shape_ = std::move(shape);
        t.dtype_ = dtype;
        int64_t n = 1;
        for (int64_t d : t.shape_) {
            if (d < 0) throw std::runtime_error("negative tensor dimension");
            if (d != 0 && n > std::numeric_limits<int64_t>::max() / d) {
                throw std::runtime_error("tensor size overflows int64");
            }
            n *= d;
        }
        const size_t esize = native::element_size(dtype);
        if (static_cast<uint64_t>(n) > std::numeric_limits<size_t>::max() / esize) {
            throw std::runtime_error("tensor byte size overflows size_t");
        }
        t.storage_ = detail::allocate(static_cast<size_t>(n) * esize);
        t.data_ = static_cast<std::byte*>(t.storage_.get());
        return t;
    }

    bool defined() const { return storage_ != nullptr; }
    ScalarType scalar_type() const { return dtype_; }
    ScalarType dtype() const { return dtype_; }
    Device device() const { return {}; }
    bool is_contiguous() const { return true; }
    Tensor contiguous() const { return *this; }

    int64_t dim() const { return static_cast<int64_t>(shape_.size()); }
    const std::vector<int64_t>& sizes() const { return shape_; }
    int64_t size(int64_t d) const { return shape_.at(static_cast<size_t>(wrap_dim(d))); }
    int64_t stride(int64_t d) const {
        d = wrap_dim(d);
        int64_t s = 1;
        for (size_t k = static_cast<size_t>(d) + 1; k < shape_.size(); ++k) s *= shape_[k];
        return s;
    }
    int64_t numel() const {
        int64_t n = 1;
        for (int64_t d : shape_) n *= d;
        return n;
    }
    size_t element_size() const { return native::element_size(dtype_); }
    size_t nbytes() const { return static_cast<size_t>(numel()) * element_size(); }

    void* data_ptr() const { return data_; }
    const void* const_data_ptr() const { return data_; }
    template <typename T>
    T* data_ptr() const {
        if (scalar_type_of<std::remove_cv_t<T>>() != dtype_) {
            throw std::runtime_error(
                std::string("expected ") + dtype_name(scalar_type_of<std::remove_cv_t<T>>()) +
                " data, tensor is " + dtype_name(dtype_));
        }
        return reinterpret_cast<T*>(data_);
    }

    // Owner of the underlying buffer (shared by every view of it).
    const std::shared_ptr<void>& storage() const { return storage_; }

    Tensor clone() const {
        Tensor t = empty(shape_, dtype_);
        if (nbytes() != 0) std::memcpy(t.data_, data_, nbytes());
        return t;
    }

    // Rows [start, end) along dimension 0, sharing storage.
    Tensor slice(int64_t d, int64_t start, int64_t end) const {
        if (wrap_dim(d) != 0) throw std::runtime_error("slice: only dimension 0 is supported");
        const int64_t n = shape_.at(0);
        start = std::clamp<int64_t>(start, 0, n);
        end = std::clamp<int64_t>(end, start, n);
        Tensor t = *this;
        t.shape_[0] = end - start;
        t.data_ = data_ + static_cast<size_t>(start) * static_cast<size_t>(stride(0)) * element_size();
        return t;
    }

    // Reinterpret the same bytes as another dtype of equal width.
    Tensor view(ScalarType t) const {
        if (native::element_size(t) != element_size()) {
            throw std::runtime_error("view: dtype widths differ");
        }
        Tensor out = *this;
        out.dtype_ = t;
        return out;
    }

    // Element-wise dtype conversion (C cast semantics; bool is "!= 0").
    // Returns *this when the dtype already matches, like torch's .to().
    Tensor to(ScalarType t) const {
        if (t == dtype_) return *this;
        Tensor out = empty(shape_, t);
        const int64_t n = numel();
        detail::visit_real(dtype_, data_, [&](auto* src) {
            detail::visit_real(t, out.data_, [&](auto* dst) {
                using D = std::remove_pointer_t<decltype(dst)>;
                for (int64_t i = 0; i < n; ++i) {
                    if constexpr (std::is_same_v<D, bool>) {
                        dst[i] = src[i] != 0;
                    } else {
                        dst[i] = static_cast<D>(src[i]);
                    }
                }
            });
        });
        return out;
    }

    // Boolean mask of elements equal to `value` (converted to the element type).
    template <typename S>
    Tensor eq(S value) const {
        Tensor mask = empty(shape_, ScalarType::Bool);
        bool* m = reinterpret_cast<bool*>(mask.data_);
        const int64_t n = numel();
        detail::visit_real(dtype_, data_, [&](auto* p) {
            using T = std::remove_pointer_t<decltype(p)>;
            const T v = static_cast<T>(value);
            for (int64_t i = 0; i < n; ++i) m[i] = p[i] == v;
        });
        return mask;
    }

    Tensor& masked_fill_(const Tensor& mask, double value) {
        if (mask.dtype_ != ScalarType::Bool || mask.numel() != numel()) {
            throw std::runtime_error("masked_fill_: mask must be a bool tensor of the same size");
        }
        const bool* m = static_cast<const bool*>(mask.const_data_ptr());
        const int64_t n = numel();
        detail::visit_real(dtype_, data_, [&](auto* p) {
            using T = std::remove_pointer_t<decltype(p)>;
            const T v = static_cast<T>(value);
            for (int64_t i = 0; i < n; ++i) {
                if (m[i]) p[i] = v;
            }
        });
        return *this;
    }

    // In-place scale by a real factor. Floating and complex tensors only:
    // the scalar is applied in the tensor's own precision, as torch does.
    Tensor& mul_(double factor) {
        const int64_t n = numel();
        switch (dtype_) {
            case ScalarType::Float32: scale_inplace<float>(static_cast<float>(factor), n); break;
            case ScalarType::Float64: scale_inplace<double>(factor, n); break;
            case ScalarType::ComplexFloat:
                scale_inplace<std::complex<float>>(static_cast<float>(factor), n);
                break;
            case ScalarType::ComplexDouble:
                scale_inplace<std::complex<double>>(factor, n);
                break;
            default:
                throw std::runtime_error(
                    std::string("mul_: unsupported dtype ") + dtype_name(dtype_));
        }
        return *this;
    }

    Tensor& add_(double offset) {
        const int64_t n = numel();
        switch (dtype_) {
            case ScalarType::Float32: shift_inplace<float>(static_cast<float>(offset), n); break;
            case ScalarType::Float64: shift_inplace<double>(offset, n); break;
            default:
                throw std::runtime_error(
                    std::string("add_: unsupported dtype ") + dtype_name(dtype_));
        }
        return *this;
    }

    template <typename C>
    Tensor& add_(std::complex<C> offset) {
        const int64_t n = numel();
        if (dtype_ == ScalarType::ComplexFloat) {
            shift_inplace<std::complex<float>>(std::complex<float>(offset), n);
        } else if (dtype_ == ScalarType::ComplexDouble) {
            shift_inplace<std::complex<double>>(std::complex<double>(offset), n);
        } else {
            throw std::runtime_error("add_: complex offset needs a complex tensor");
        }
        return *this;
    }

    // In-place XOR with `bits` converted to the (integer) element type.
    Tensor& bitwise_xor_(int64_t bits) {
        if (!is_integral(dtype_, /*include_bool=*/false)) {
            throw std::runtime_error(
                std::string("bitwise_xor_: unsupported dtype ") + dtype_name(dtype_));
        }
        const int64_t n = numel();
        detail::visit_real(dtype_, data_, [&](auto* p) {
            using T = std::remove_pointer_t<decltype(p)>;
            if constexpr (std::is_integral_v<T> && !std::is_same_v<T, bool>) {
                const T mask = static_cast<T>(bits);
                for (int64_t i = 0; i < n; ++i) p[i] = static_cast<T>(p[i] ^ mask);
            }
        });
        return *this;
    }

private:
    int64_t wrap_dim(int64_t d) const {
        const int64_t n = dim();
        if (d < 0) d += n;
        if (d < 0 || d >= std::max<int64_t>(n, 1)) {
            throw std::runtime_error("dimension out of range");
        }
        return d;
    }

    template <typename T, typename S>
    void scale_inplace(S factor, int64_t n) {
        T* p = reinterpret_cast<T*>(data_);
        for (int64_t i = 0; i < n; ++i) p[i] *= factor;
    }

    template <typename T>
    void shift_inplace(T offset, int64_t n) {
        T* p = reinterpret_cast<T*>(data_);
        for (int64_t i = 0; i < n; ++i) p[i] += offset;
    }

    std::shared_ptr<void> storage_;
    std::byte* data_ = nullptr;
    std::vector<int64_t> shape_;
    ScalarType dtype_ = ScalarType::Float32;
};

// ---------------------------------------------------------------- factories

template <typename Int, typename = std::enable_if_t<std::is_integral_v<Int>>>
inline Tensor empty(const std::vector<Int>& shape, TensorOptions o = {}) {
    return Tensor::empty(std::vector<int64_t>(shape.begin(), shape.end()), o.dtype_);
}
inline Tensor empty(std::initializer_list<int64_t> shape, TensorOptions o = {}) {
    return Tensor::empty(std::vector<int64_t>(shape), o.dtype_);
}
inline Tensor empty(const int64_t* dims, size_t ndim, TensorOptions o = {}) {
    return Tensor::empty(std::vector<int64_t>(dims, dims + ndim), o.dtype_);
}
inline Tensor zeros(std::initializer_list<int64_t> shape, TensorOptions o = {}) {
    Tensor t = Tensor::empty(std::vector<int64_t>(shape), o.dtype_);
    if (t.nbytes() != 0) std::memset(t.data_ptr(), 0, t.nbytes());
    return t;
}
// A new tensor holding a copy of `count` elements at `src`.
inline Tensor copy_of(const void* src, int64_t count, ScalarType dtype) {
    Tensor t = Tensor::empty({count}, dtype);
    if (t.nbytes() != 0) std::memcpy(t.data_ptr(), src, t.nbytes());
    return t;
}

// ---------------------------------------------------------------- threads

// at::parallel_for's partitioning on torchfits' own pool: at most one chunk
// per pool thread, each at least `grain` items, so a 1M-row decode is a
// handful of tasks rather than one task per 2048 rows. f(begin, end) must be
// safe to run concurrently on disjoint ranges.
template <typename F>
inline void parallel_for(int64_t begin, int64_t end, int64_t grain, F&& f) {
    if (end <= begin) return;
    grain = std::max<int64_t>(grain, 1);
    const int64_t range = end - begin;
    const int64_t threads = std::max(1, core::thread_count());
    const int64_t per_thread = (range + threads - 1) / threads;
    core::parallel_for(begin, end, std::max(grain, per_thread), std::forward<F>(f));
}

}  // namespace native
}  // namespace torchfits
