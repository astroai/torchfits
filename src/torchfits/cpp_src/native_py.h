#pragma once

// Python boundary for native::Tensor.
//
// A decoded buffer leaves C++ as an unversioned DLPack capsule whose deleter
// owns the native storage; torch.utils.dlpack.from_dlpack adopts it without a
// copy. Nothing here includes or links libtorch: torch is reached only through
// its public Python API, so one extension serves every torch release that can
// import a DLPack capsule.
//
// Must be called with the GIL held.

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <atomic>
#include <memory>
#include <stdexcept>
#include <vector>

#include "native_tensor.h"

namespace nb = nanobind;

namespace torchfits {
namespace native {

inline nb::dlpack::dtype dlpack_dtype(ScalarType t) {
    using code = nb::dlpack::dtype_code;
    auto make = [](code c, int bits) {
        nb::dlpack::dtype d{};
        d.code = static_cast<uint8_t>(c);
        d.bits = static_cast<uint8_t>(bits);
        d.lanes = 1;
        return d;
    };
    switch (t) {
        case ScalarType::Bool: return make(code::Bool, 8);
        case ScalarType::UInt8: return make(code::UInt, 8);
        case ScalarType::Int8: return make(code::Int, 8);
        case ScalarType::Int16: return make(code::Int, 16);
        case ScalarType::UInt16: return make(code::UInt, 16);
        case ScalarType::Int32: return make(code::Int, 32);
        case ScalarType::UInt32: return make(code::UInt, 32);
        case ScalarType::Int64: return make(code::Int, 64);
        case ScalarType::UInt64: return make(code::UInt, 64);
        case ScalarType::Float16: return make(code::Float, 16);
        case ScalarType::BFloat16: return make(code::Bfloat, 16);
        case ScalarType::Float32: return make(code::Float, 32);
        case ScalarType::Float64: return make(code::Float, 64);
        case ScalarType::ComplexFloat: return make(code::Complex, 64);
        case ScalarType::ComplexDouble: return make(code::Complex, 128);
    }
    throw std::runtime_error("unsupported dtype at the DLPack boundary");
}

namespace detail {

// A capsule that keeps the tensor's storage alive for as long as Python (or a
// DLPack consumer) references it.
inline nb::capsule storage_owner(const Tensor& t) {
    auto* keep = new std::shared_ptr<void>(t.storage());
    return nb::capsule(keep, [](void* p) noexcept {
        delete static_cast<std::shared_ptr<void>*>(p);
    });
}

template <typename Framework>
inline nb::object export_tensor(const Tensor& t) {
    if (!t.defined()) {
        throw std::runtime_error("cannot export an undefined tensor");
    }
    std::vector<size_t> shape(t.sizes().begin(), t.sizes().end());
    nb::ndarray<Framework, nb::device::cpu> array(
        t.data_ptr(), shape.size(), shape.data(), storage_owner(t),
        /*strides=*/nullptr, dlpack_dtype(t.scalar_type()));
    return nb::cast(array, nb::rv_policy::reference);
}

// torch.utils.dlpack.from_dlpack, resolved once. Importing it is what loads
// torch, so the first tensor result -- not `import torchfits._C` -- pays that.
//
// Not a function-local static initializer: the import can release the GIL,
// and a second thread blocked on the static's guard while holding the GIL
// would deadlock the first. Two threads may both resolve it; the loser drops
// its reference.
inline PyObject* torch_from_dlpack() {
    static std::atomic<PyObject*> cached{nullptr};
    if (PyObject* fn = cached.load(std::memory_order_acquire)) {
        return fn;
    }
    nb::object mod = nb::module_::import_("torch.utils.dlpack");
    PyObject* fn = nb::object(mod.attr("from_dlpack")).release().ptr();
    PyObject* expected = nullptr;
    if (!cached.compare_exchange_strong(expected, fn, std::memory_order_acq_rel)) {
        Py_DECREF(fn);
        return expected;
    }
    return fn;  // process lifetime
}

// DLPack's unversioned DLManagedTensor: the layout every DLPack consumer has
// accepted since DLPack 0.x (DLTensor is mirrored by nb::dlpack::dltensor).
struct DLManagedTensor {
    nb::dlpack::dltensor dl_tensor;
    void* manager_ctx;
    void (*deleter)(DLManagedTensor* self);
};

// One allocation per exported tensor: the managed tensor, the shape it points
// at, and the storage reference it keeps alive. The consumer calls `deleter`
// when its tensor is freed -- possibly on a thread without the GIL, which is
// fine: releasing a shared_ptr touches no Python state.
struct ExportedTensor {
    static constexpr size_t kMaxDims = 16;
    DLManagedTensor managed{};
    std::shared_ptr<void> storage;
    int64_t shape[kMaxDims] = {};

    static void destroy(DLManagedTensor* self) noexcept {
        delete static_cast<ExportedTensor*>(self->manager_ctx);
    }
};

constexpr const char* kDLTensorCapsule = "dltensor";

inline void dlpack_capsule_destructor(PyObject* cap) noexcept {
    // Still named "dltensor": nobody adopted it, so it still owns the data.
    // A consumer renames it ("used_dltensor") and calls the deleter itself.
    if (PyCapsule_IsValid(cap, kDLTensorCapsule)) {
        auto* managed = static_cast<DLManagedTensor*>(
            PyCapsule_GetPointer(cap, kDLTensorCapsule));
        managed->deleter(managed);
    }
}

// A "dltensor" capsule over `t` (zero copy).
inline nb::object dlpack_capsule(const Tensor& t) {
    if (!t.defined()) {
        throw std::runtime_error("cannot export an undefined tensor");
    }
    const auto& dims = t.sizes();
    if (dims.size() > ExportedTensor::kMaxDims) {
        throw std::runtime_error("cannot export a tensor with more than 16 dimensions");
    }
    auto holder = std::make_unique<ExportedTensor>();
    holder->storage = t.storage();
    for (size_t i = 0; i < dims.size(); ++i) holder->shape[i] = dims[i];
    nb::dlpack::dltensor& dl = holder->managed.dl_tensor;
    dl.data = t.data_ptr();
    dl.device.device_type = 1;  // kDLCPU
    dl.device.device_id = 0;
    dl.ndim = static_cast<int32_t>(dims.size());
    dl.dtype = dlpack_dtype(t.scalar_type());
    dl.shape = holder->shape;
    dl.strides = nullptr;  // C-contiguous
    dl.byte_offset = 0;
    holder->managed.manager_ctx = holder.get();
    holder->managed.deleter = &ExportedTensor::destroy;

    PyObject* capsule =
        PyCapsule_New(&holder->managed, kDLTensorCapsule, &dlpack_capsule_destructor);
    if (capsule == nullptr) {
        throw nb::python_error();
    }
    holder.release();  // owned by the capsule (and then by its consumer)
    return nb::steal(capsule);
}

}  // namespace detail

// A torch.Tensor that owns `t`'s storage (zero copy).
inline nb::object to_torch(const Tensor& t) {
    // A raw capsule skips the __dlpack__ protocol round trip; from_dlpack
    // consumes it directly (and renames it, so it is adopted exactly once).
    nb::object capsule = detail::dlpack_capsule(t);
    PyObject* args[] = {nullptr, capsule.ptr()};
    PyObject* out = PyObject_Vectorcall(
        detail::torch_from_dlpack(), args + 1, 1 | PY_VECTORCALL_ARGUMENTS_OFFSET, nullptr);
    if (out == nullptr) {
        throw nb::python_error();
    }
    return nb::steal(out);
}

// A numpy.ndarray that owns `t`'s storage (zero copy).
inline nb::object to_numpy(const Tensor& t) {
    return detail::export_tensor<nb::numpy>(t);
}

}  // namespace native
}  // namespace torchfits

// Historical spellings used throughout the bindings.
inline nb::object tensor_to_python(const torchfits::native::Tensor& tensor) {
    return torchfits::native::to_torch(tensor);
}

inline nb::object tensor_to_numpy_object(const torchfits::native::Tensor& tensor) {
    return torchfits::native::to_numpy(tensor);
}
