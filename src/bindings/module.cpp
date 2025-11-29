#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#ifdef CUPROX_HAS_CUDA
#include "cuprox/core/dense_vector.cuh"
#include "cuprox/core/sparse_matrix.cuh"
#include "cuprox/core/cuda_context.cuh"
#endif

namespace py = pybind11;

PYBIND11_MODULE(_core, m) {
    m.doc() = "cuProx C++ core module";

#ifdef CUPROX_HAS_CUDA
    m.attr("cuda_available") = true;

    m.def("get_device_name", []() {
        cudaDeviceProp prop;
        int device;
        cudaGetDevice(&device);
        cudaGetDeviceProperties(&prop, device);
        return std::string(prop.name);
    }, "Get the name of the current CUDA device");

    m.def("get_device_memory", []() {
        cudaDeviceProp prop;
        int device;
        cudaGetDevice(&device);
        cudaGetDeviceProperties(&prop, device);
        return prop.totalGlobalMem / (1024 * 1024);  // MB
    }, "Get total device memory in MB");

    m.def("synchronize", []() {
        cuprox::CudaContext::instance().synchronize();
    }, "Synchronize CUDA device");

#else
    m.attr("cuda_available") = false;
#endif

    m.attr("__version__") = "0.1.0";
}

