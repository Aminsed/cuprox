#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#ifdef CUPROX_HAS_CUDA
#include "cuprox/core/dense_vector.cuh"
#include "cuprox/core/sparse_matrix.cuh"
#include "cuprox/core/cuda_context.cuh"
#include "cuprox/solvers/pdhg.cuh"
#endif

namespace py = pybind11;

#ifdef CUPROX_HAS_CUDA

// Solve LP using PDHG
py::dict solve_lp_pdhg(
    py::array_t<cuprox::Index, py::array::c_style | py::array::forcecast> row_offsets,
    py::array_t<cuprox::Index, py::array::c_style | py::array::forcecast> col_indices,
    py::array_t<double, py::array::c_style | py::array::forcecast> values,
    py::array_t<double, py::array::c_style | py::array::forcecast> c,
    py::array_t<double, py::array::c_style | py::array::forcecast> b,
    py::array_t<double, py::array::c_style | py::array::forcecast> lb,
    py::array_t<double, py::array::c_style | py::array::forcecast> ub,
    cuprox::Index num_rows,
    cuprox::Index num_cols,
    int max_iters,
    double eps_abs,
    double eps_rel,
    bool verbose
) {
    // Get buffer info
    auto ro_buf = row_offsets.request();
    auto ci_buf = col_indices.request();
    auto v_buf = values.request();
    auto c_buf = c.request();
    auto b_buf = b.request();
    auto lb_buf = lb.request();
    auto ub_buf = ub.request();
    
    cuprox::Index nnz = static_cast<cuprox::Index>(v_buf.size);
    
    // Create LP problem
    cuprox::LPProblem<double> lp;
    lp.A = cuprox::CsrMatrix<double>::from_csr(
        num_rows, num_cols, nnz,
        static_cast<cuprox::Index*>(ro_buf.ptr),
        static_cast<cuprox::Index*>(ci_buf.ptr),
        static_cast<double*>(v_buf.ptr)
    );
    
    // Resize and copy vectors to device
    lp.c.resize(num_cols);
    lp.c.copy_from_host(static_cast<double*>(c_buf.ptr), num_cols);
    
    lp.b.resize(num_rows);
    lp.b.copy_from_host(static_cast<double*>(b_buf.ptr), num_rows);
    
    lp.lb.resize(num_cols);
    lp.lb.copy_from_host(static_cast<double*>(lb_buf.ptr), num_cols);
    
    lp.ub.resize(num_cols);
    lp.ub.copy_from_host(static_cast<double*>(ub_buf.ptr), num_cols);
    
    // Set l = b, u = b for equality constraints
    lp.l.resize(num_rows);
    lp.l.copy_from_host(static_cast<double*>(b_buf.ptr), num_rows);
    
    lp.u.resize(num_rows);
    lp.u.copy_from_host(static_cast<double*>(b_buf.ptr), num_rows);
    
    // Configure solver
    cuprox::PdhgSettings<double> settings;
    settings.max_iters = max_iters;
    settings.eps_abs = eps_abs;
    settings.eps_rel = eps_rel;
    settings.verbose = verbose;
    settings.scaling = false;  // Disabled for now
    
    // Solve
    cuprox::PdhgSolver<double> solver(settings);
    auto result = solver.solve(lp);
    
    // Convert result to Python
    auto x_host = result.x.to_host();
    auto y_host = result.y.to_host();
    
    py::array_t<double> x_out(x_host.size());
    py::array_t<double> y_out(y_host.size());
    
    std::copy(x_host.begin(), x_host.end(), x_out.mutable_data());
    std::copy(y_host.begin(), y_host.end(), y_out.mutable_data());
    
    // Status string
    std::string status_str = cuprox::status_to_string(result.status);
    
    py::dict out;
    out["x"] = x_out;
    out["y"] = y_out;
    out["status"] = status_str;
    out["objective"] = result.primal_obj;
    out["primal_residual"] = result.primal_res;
    out["dual_residual"] = result.dual_res;
    out["iterations"] = result.iterations;
    out["solve_time"] = result.solve_time;
    
    return out;
}

#endif  // CUPROX_HAS_CUDA

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

    m.def("solve_lp_pdhg", &solve_lp_pdhg,
        py::arg("row_offsets"),
        py::arg("col_indices"),
        py::arg("values"),
        py::arg("c"),
        py::arg("b"),
        py::arg("lb"),
        py::arg("ub"),
        py::arg("num_rows"),
        py::arg("num_cols"),
        py::arg("max_iters") = 10000,
        py::arg("eps_abs") = 1e-6,
        py::arg("eps_rel") = 1e-6,
        py::arg("verbose") = false,
        "Solve LP using PDHG on GPU");

#else
    m.attr("cuda_available") = false;
#endif

    m.attr("__version__") = "0.1.0";
}
