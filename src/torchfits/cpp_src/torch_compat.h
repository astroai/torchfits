#pragma once

#include <nanobind/nanobind.h>
#include "torchfits_torch.h"
#include <ATen/DLConvertor.h>
#include <Python.h>
#include <vector>
#include <stdexcept>
#include <nanobind/ndarray.h>

namespace nb = nanobind;

// Forward declare THPVariableClass
extern PyObject* THPVariableClass;

// Check if object is a Tensor
inline bool THPVariable_Check(PyObject* obj) {
    return PyObject_IsInstance(obj, THPVariableClass);
}

// Extern declaration for THPVariable_Wrap (exported by libtorch_python)
extern PyObject* THPVariable_Wrap(const at::TensorBase& var);

inline nb::object tensor_to_numpy_object(const torch::Tensor& tensor) {
    PyObject* tensor_obj = THPVariable_Wrap(tensor);
    if (!tensor_obj) {
        throw std::runtime_error("Failed to wrap tensor for NumPy conversion");
    }

    PyObject* numpy_obj = PyObject_CallMethod(tensor_obj, "numpy", nullptr);
    Py_DECREF(tensor_obj);
    if (!numpy_obj) {
        throw nb::python_error();
    }
    return nb::steal(numpy_obj);
}

template <typename T>
inline nb::ndarray<nb::numpy, T, nb::c_contig> alloc_numpy_array(
    const std::vector<size_t>& shape
) {
    size_t nelem = 1;
    for (size_t d : shape) {
        nelem *= d;
    }
    const size_t nbytes = nelem * sizeof(T);

    PyObject* ba = PyByteArray_FromStringAndSize(nullptr, (Py_ssize_t) nbytes);
    if (!ba) {
        throw std::runtime_error("Failed to allocate bytearray for numpy result");
    }
    nb::object owner = nb::steal(ba);
    void* data = (void*) PyByteArray_AsString(owner.ptr());
    if (!data) {
        throw std::runtime_error("Failed to get bytearray buffer for numpy result");
    }
    return nb::ndarray<nb::numpy, T, nb::c_contig>(
        data, shape.size(), shape.data(), owner
    );
}


// Deleter for DLPack capsule created by tensor_to_python_dlpack.
// Only called if torch.from_dlpack() did NOT consume the capsule.
static void _dlpack_capsule_deleter(PyObject* capsule) {
    auto* dl = static_cast<DLManagedTensor*>(
        PyCapsule_GetPointer(capsule, "dltensor")
    );
    if (dl && dl->deleter) {
        dl->deleter(dl);
    }
}

// Helper function to convert torch::Tensor to Python object.
// Uses DLPack to avoid THPVariable_Wrap which can segfault on some
// Python/torch build combinations (e.g. CPython 3.12 + macOS).
inline nb::object tensor_to_python(const torch::Tensor& tensor) {
    DLManagedTensor* dl_tensor = at::toDLPack(tensor);

    PyObject* capsule = PyCapsule_New(dl_tensor, "dltensor", _dlpack_capsule_deleter);
    if (!capsule) {
        dl_tensor->deleter(dl_tensor);
        throw std::runtime_error("Failed to create PyCapsule for DLPack");
    }

    PyObject* torch_mod = PyImport_ImportModule("torch");
    if (!torch_mod) {
        Py_DECREF(capsule);
        dl_tensor->deleter(dl_tensor);
        throw nb::python_error();
    }

    PyObject* result = PyObject_CallMethod(torch_mod, "from_dlpack", "O", capsule);
    Py_DECREF(torch_mod);
    Py_DECREF(capsule);
    if (!result) {
        throw nb::python_error();
    }
    return nb::steal(result);
}

// Helper function to convert Python object to torch::Tensor - FAST PATH
inline torch::Tensor python_to_tensor(nb::object obj) {
    if (!PyObject_HasAttrString(obj.ptr(), "__dlpack__")) {
        throw std::runtime_error("Object does not implement __dlpack__");
    }
    PyObject* capsule_obj = PyObject_CallMethod(obj.ptr(), "__dlpack__", nullptr);
    if (!capsule_obj) {
        throw nb::python_error();
    }
    nb::object capsule = nb::steal(capsule_obj);

    auto* dl_managed = static_cast<DLManagedTensor*>(
        PyCapsule_GetPointer(capsule.ptr(), "dltensor")
    );
    if (!dl_managed) {
        throw nb::python_error();
    }
    auto t = at::fromDLPack(dl_managed);

    // Mark the capsule as consumed to avoid the capsule destructor calling deleter
    // a second time after ATen takes ownership.
    if (PyCapsule_SetName(capsule.ptr(), "used_dltensor") != 0) {
        PyErr_Clear();
    }
    return t;
}
