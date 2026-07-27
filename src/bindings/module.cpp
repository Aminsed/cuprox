#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#ifdef CUPROX_HAS_CUDA
#include "cuprox/core/dense_vector.cuh"
#include "cuprox/core/sparse_matrix.cuh"
#include "cuprox/core/cuda_context.cuh"
#include "cuprox/solvers/pdhg.cuh"
#include "cuprox/solvers/admm.cuh"
#include "cuprox/solvers/batch_pdhg.cuh"
#endif

namespace py = pybind11;

#ifdef CUPROX_HAS_CUDA

// Solve LP using PDHG
py::dict solve_lp_pdhg(
    py::array_t<cuprox::Index, py::array::c_style | py::array::forcecast> row_offsets,
    py::array_t<cuprox::Index, py::array::c_style | py::array::forcecast> col_indices,
    py::array_t<double, py::array::c_style | py::array::forcecast> values,
    py::array_t<double, py::array::c_style | py::array::forcecast> c,
    py::array_t<double, py::array::c_style | py::array::forcecast> l,
    py::array_t<double, py::array::c_style | py::array::forcecast> u,
    py::array_t<double, py::array::c_style | py::array::forcecast> lb,
    py::array_t<double, py::array::c_style | py::array::forcecast> ub,
    cuprox::Index num_rows,
    cuprox::Index num_cols,
    int max_iters,
    double eps_abs,
    double eps_rel,
    bool verbose,
    bool scaling
) {
    // Get buffer info
    auto ro_buf = row_offsets.request();
    auto ci_buf = col_indices.request();
    auto v_buf = values.request();
    auto c_buf = c.request();
    auto l_buf = l.request();
    auto u_buf = u.request();
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
    
    
    lp.lb.resize(num_cols);
    lp.lb.copy_from_host(static_cast<double*>(lb_buf.ptr), num_cols);
    
    lp.ub.resize(num_cols);
    lp.ub.copy_from_host(static_cast<double*>(ub_buf.ptr), num_cols);
    
    // Row bounds as given. This previously copied b into both l and u, which
    // silently turned every inequality the caller asked for into an equality.
    lp.l.resize(num_rows);
    lp.l.copy_from_host(static_cast<double*>(l_buf.ptr), num_rows);

    lp.u.resize(num_rows);
    lp.u.copy_from_host(static_cast<double*>(u_buf.ptr), num_rows);

    // b is retained only so the Ruiz scaling has a right-hand side to work on;
    // the solver itself reads l and u.
    lp.b.resize(num_rows);
    lp.b.copy_from_host(static_cast<double*>(l_buf.ptr), num_rows);
    
    // Configure solver
    cuprox::PdhgSettings<double> settings;
    settings.max_iters = max_iters;
    settings.eps_abs = eps_abs;
    settings.eps_rel = eps_rel;
    settings.verbose = verbose;
    settings.scaling = scaling;
    
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
        py::arg("l"),
        py::arg("u"),
        py::arg("lb"),
        py::arg("ub"),
        py::arg("num_rows"),
        py::arg("num_cols"),
        py::arg("max_iters") = 10000,
        py::arg("eps_abs") = 1e-6,
        py::arg("eps_rel") = 1e-6,
        py::arg("verbose") = false,
        py::arg("scaling") = false,
        "Solve an LP with l <= Ax <= u, lb <= x <= ub, using PDHG on the GPU.");

    // Batched LP: one shared A and bounds, many objectives / right-hand sides.
    // This is what makes MPC horizons and Monte-Carlo scenario sets cheap, and
    // the kernel behind it was compiled but never exposed, so the Python
    // solve_batch was a sequential list comprehension.
    m.def("solve_batch_lp_pdhg", [](
        py::array_t<cuprox::Index, py::array::c_style | py::array::forcecast> row_offsets,
        py::array_t<cuprox::Index, py::array::c_style | py::array::forcecast> col_indices,
        py::array_t<double, py::array::c_style | py::array::forcecast> values,
        py::array_t<double, py::array::c_style | py::array::forcecast> c_batch,
        py::array_t<double, py::array::c_style | py::array::forcecast> b_batch,
        py::array_t<double, py::array::c_style | py::array::forcecast> lb,
        py::array_t<double, py::array::c_style | py::array::forcecast> ub,
        cuprox::Index batch_size,
        cuprox::Index num_rows,
        cuprox::Index num_cols,
        int max_iters,
        double eps_abs,
        double eps_rel,
        bool verbose
    ) {
        auto ro = row_offsets.request();
        auto ci = col_indices.request();
        auto va = values.request();
        auto cb = c_batch.request();
        auto bb = b_batch.request();
        auto lbb = lb.request();
        auto ubb = ub.request();

        auto A = cuprox::CsrMatrix<double>::from_csr(
            num_rows, num_cols, static_cast<cuprox::Index>(va.size),
            static_cast<cuprox::Index*>(ro.ptr),
            static_cast<cuprox::Index*>(ci.ptr),
            static_cast<double*>(va.ptr));

        cuprox::DeviceVector<double> d_lb, d_ub;
        d_lb.resize(num_cols);
        d_lb.copy_from_host(static_cast<double*>(lbb.ptr), num_cols);
        d_ub.resize(num_cols);
        d_ub.copy_from_host(static_cast<double*>(ubb.ptr), num_cols);

        auto problem = cuprox::make_batch_lp<double>(
            A, static_cast<double*>(cb.ptr), static_cast<double*>(bb.ptr),
            d_lb, d_ub, batch_size, num_cols, num_rows);

        cuprox::BatchPdhgSettings<double> settings;
        settings.max_iters = max_iters;
        settings.eps_abs = eps_abs;
        settings.eps_rel = eps_rel;
        settings.verbose = verbose;

        auto result = cuprox::BatchPdhgSolver<double>(settings).solve(problem);

        const size_t xn = static_cast<size_t>(batch_size) * num_cols;
        const size_t yn = static_cast<size_t>(batch_size) * num_rows;
        std::vector<double> h_x(xn), h_obj(batch_size);
        std::vector<int> h_status(batch_size), h_iters(batch_size);
        cuprox::copy_device_to_host(h_x.data(), result.x.get(), xn);
        cuprox::copy_device_to_host(h_obj.data(), result.objectives.get(), batch_size);
        cuprox::copy_device_to_host(h_status.data(), result.statuses.get(), batch_size);
        cuprox::copy_device_to_host(h_iters.data(), result.iterations.get(), batch_size);

        py::array_t<double> x_out({static_cast<py::ssize_t>(batch_size),
                                   static_cast<py::ssize_t>(num_cols)});
        std::copy(h_x.begin(), h_x.end(), x_out.mutable_data());
        py::array_t<double> obj_out(batch_size);
        std::copy(h_obj.begin(), h_obj.end(), obj_out.mutable_data());
        py::array_t<int> st_out(batch_size);
        std::copy(h_status.begin(), h_status.end(), st_out.mutable_data());
        py::array_t<int> it_out(batch_size);
        std::copy(h_iters.begin(), h_iters.end(), it_out.mutable_data());

        py::dict out;
        out["x"] = x_out;
        out["objectives"] = obj_out;
        out["statuses"] = st_out;
        out["iterations"] = it_out;
        out["solve_time"] = result.total_solve_time;
        return out;
    },
        py::arg("row_offsets"), py::arg("col_indices"), py::arg("values"),
        py::arg("c_batch"), py::arg("b_batch"), py::arg("lb"), py::arg("ub"),
        py::arg("batch_size"), py::arg("num_rows"), py::arg("num_cols"),
        py::arg("max_iters") = 5000, py::arg("eps_abs") = 1e-5,
        py::arg("eps_rel") = 1e-5, py::arg("verbose") = false,
        "Solve a batch of LPs that share A, lb and ub, on the GPU.");

    m.def("solve_qp_admm", [](
        py::array_t<cuprox::Index, py::array::c_style | py::array::forcecast> P_row_offsets,
        py::array_t<cuprox::Index, py::array::c_style | py::array::forcecast> P_col_indices,
        py::array_t<double, py::array::c_style | py::array::forcecast> P_values,
        py::array_t<cuprox::Index, py::array::c_style | py::array::forcecast> A_row_offsets,
        py::array_t<cuprox::Index, py::array::c_style | py::array::forcecast> A_col_indices,
        py::array_t<double, py::array::c_style | py::array::forcecast> A_values,
        py::array_t<double, py::array::c_style | py::array::forcecast> q,
        py::array_t<double, py::array::c_style | py::array::forcecast> l,
        py::array_t<double, py::array::c_style | py::array::forcecast> u,
        py::array_t<double, py::array::c_style | py::array::forcecast> var_lb,
        py::array_t<double, py::array::c_style | py::array::forcecast> var_ub,
        cuprox::Index P_n,
        cuprox::Index A_m,
        cuprox::Index A_n,
        int max_iters,
        double eps_abs,
        double eps_rel,
        double rho,
        bool verbose
    ) {
        // Get buffer info
        auto P_ro = P_row_offsets.request();
        auto P_ci = P_col_indices.request();
        auto P_v = P_values.request();
        auto A_ro = A_row_offsets.request();
        auto A_ci = A_col_indices.request();
        auto A_v = A_values.request();
        auto q_buf = q.request();
        auto l_buf = l.request();
        auto u_buf = u.request();
        auto var_lb_buf = var_lb.request();
        auto var_ub_buf = var_ub.request();
        
        // Create QP problem
        cuprox::QPProblem<double> qp;
        qp.P = cuprox::CsrMatrix<double>::from_csr(
            P_n, P_n, static_cast<cuprox::Index>(P_v.size),
            static_cast<cuprox::Index*>(P_ro.ptr),
            static_cast<cuprox::Index*>(P_ci.ptr),
            static_cast<double*>(P_v.ptr)
        );
        qp.A = cuprox::CsrMatrix<double>::from_csr(
            A_m, A_n, static_cast<cuprox::Index>(A_v.size),
            static_cast<cuprox::Index*>(A_ro.ptr),
            static_cast<cuprox::Index*>(A_ci.ptr),
            static_cast<double*>(A_v.ptr)
        );
        
        qp.q.resize(A_n);
        qp.q.copy_from_host(static_cast<double*>(q_buf.ptr), A_n);
        qp.l.resize(A_m);
        qp.l.copy_from_host(static_cast<double*>(l_buf.ptr), A_m);
        qp.u.resize(A_m);
        qp.u.copy_from_host(static_cast<double*>(u_buf.ptr), A_m);
        
        // Variable bounds
        qp.lb.resize(A_n);
        qp.lb.copy_from_host(static_cast<double*>(var_lb_buf.ptr), A_n);
        qp.ub.resize(A_n);
        qp.ub.copy_from_host(static_cast<double*>(var_ub_buf.ptr), A_n);
        
        // Configure solver
        cuprox::AdmmSettings<double> settings;
        settings.max_iters = max_iters;
        settings.eps_abs = eps_abs;
        settings.eps_rel = eps_rel;
        settings.rho = rho;
        settings.verbose = verbose;
        
        // Solve
        cuprox::AdmmSolver<double> solver(settings);
        auto result = solver.solve(qp);
        
        // Convert result
        auto x_host = result.x.to_host();
        auto y_host = result.y.to_host();
        
        py::array_t<double> x_out(x_host.size());
        py::array_t<double> y_out(y_host.size());
        
        std::copy(x_host.begin(), x_host.end(), x_out.mutable_data());
        std::copy(y_host.begin(), y_host.end(), y_out.mutable_data());
        
        py::dict out;
        out["x"] = x_out;
        out["y"] = y_out;
        out["status"] = cuprox::status_to_string(result.status);
        out["objective"] = result.primal_obj;
        out["primal_residual"] = result.primal_res;
        out["dual_residual"] = result.dual_res;
        out["iterations"] = result.iterations;
        out["solve_time"] = result.solve_time;
        
        return out;
    },
        py::arg("P_row_offsets"),
        py::arg("P_col_indices"),
        py::arg("P_values"),
        py::arg("A_row_offsets"),
        py::arg("A_col_indices"),
        py::arg("A_values"),
        py::arg("q"),
        py::arg("l"),
        py::arg("u"),
        py::arg("var_lb"),
        py::arg("var_ub"),
        py::arg("P_n"),
        py::arg("A_m"),
        py::arg("A_n"),
        py::arg("max_iters") = 4000,
        py::arg("eps_abs") = 1e-6,
        py::arg("eps_rel") = 1e-6,
        py::arg("rho") = 1.0,
        py::arg("verbose") = false,
        "Solve QP using ADMM on GPU");

#else
    m.attr("cuda_available") = false;
#endif

    m.attr("__version__") = "0.1.0";
}
