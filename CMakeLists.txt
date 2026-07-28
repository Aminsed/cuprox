cmake_minimum_required(VERSION 3.24 FATAL_ERROR)

# ============================================================================
# cuProx - GPU-Accelerated First-Order LP/QP Solver
# ============================================================================

project(cuprox
    VERSION 0.2.0
    DESCRIPTION "GPU-accelerated LP/QP optimization solver using first-order methods"
    LANGUAGES CXX
)

# ============================================================================
# Options
# ============================================================================

option(CUPROX_BUILD_TESTS "Build unit tests" ON)
option(CUPROX_BUILD_PYTHON "Build Python bindings" ON)
option(CUPROX_BUILD_BENCHMARKS "Build benchmarks" OFF)
option(CUPROX_CPU_ONLY "Build without CUDA (CPU fallback only)" OFF)
option(CUPROX_USE_DOUBLE "Use double precision (FP64) by default" ON)

# ============================================================================
# CUDA Detection and Setup
# ============================================================================

if(NOT CUPROX_CPU_ONLY)
    # ---------------------------------------------------------------------------
    # Find nvcc before the CUDA language is enabled.
    #
    # CMake searches only PATH for nvcc. The usual Linux install puts the toolkit
    # under /usr/local/cuda and only the driver on PATH, which would otherwise be
    # indistinguishable from having no CUDA at all -- this is why `pip install`
    # failed on machines that had a perfectly good toolkit.
    # ---------------------------------------------------------------------------
    if(NOT DEFINED CMAKE_CUDA_COMPILER AND NOT DEFINED ENV{CUDACXX})
        file(GLOB _cuda_roots /usr/local/cuda-*)
        find_program(_nvcc nvcc
            HINTS ENV CUDA_HOME ENV CUDA_PATH ENV CUDA_ROOT
                  /usr/local/cuda ${_cuda_roots}
            PATH_SUFFIXES bin)
        if(_nvcc)
            set(CMAKE_CUDA_COMPILER "${_nvcc}")
        endif()
    endif()

    include(CheckLanguage)
    check_language(CUDA)

    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        message(STATUS "CUDA compiler found: ${CMAKE_CUDA_COMPILER}")
        message(STATUS "CUDA version: ${CMAKE_CUDA_COMPILER_VERSION}")

        # CUDA standards
        set(CMAKE_CUDA_STANDARD 17)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)

        # Target architectures: Volta (7.0) through Blackwell (10.0)
        # Users can override with -DCMAKE_CUDA_ARCHITECTURES="80;86;90"
        if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
            set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86;89;90")
        endif()
        message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

        # CUDA flags
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
        set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -O3")
        set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -G -g")

        # Find CUDA Toolkit
        find_package(CUDAToolkit REQUIRED)
        set(CUPROX_HAS_CUDA TRUE)

    else()
        message(FATAL_ERROR
            "cuProx is a GPU solver and no CUDA compiler was found.\n"
            "Install the CUDA toolkit (11.4 or newer), then either put nvcc on\n"
            "PATH or point CUDACXX at it:\n"
            "    export CUDACXX=/usr/local/cuda/bin/nvcc\n"
            "To build the CPU fallback on purpose, pass -DCUPROX_CPU_ONLY=ON.\n"
            "Refusing to build silently without CUDA: the resulting package\n"
            "would import cleanly and then fail on every solve.")
    endif()
else()
    set(CUPROX_HAS_CUDA FALSE)
    message(STATUS "Building CPU-only version (CUPROX_CPU_ONLY=ON)")
endif()

# ============================================================================
# C++ Setup
# ============================================================================

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Compiler warnings
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ============================================================================
# Dependencies
# ============================================================================

# Eigen (for CPU fallback and testing)
find_package(Eigen3 3.3 QUIET)
if(Eigen3_FOUND)
    message(STATUS "Eigen3 found: ${EIGEN3_INCLUDE_DIR}")
else()
    message(STATUS "Eigen3 not found. Will be fetched.")
    include(FetchContent)
    FetchContent_Declare(
        Eigen
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG 3.4.0
    )
    FetchContent_MakeAvailable(Eigen)
endif()

# ============================================================================
# Core Library
# ============================================================================

add_subdirectory(src/cuprox)

# ============================================================================
# Python Bindings
# ============================================================================

if(CUPROX_BUILD_PYTHON)
    find_package(Python 3.9 COMPONENTS Interpreter Development.Module REQUIRED)
    # pybind11 >= 2.12 is a hard requirement, not a preference.
    #
    # Earlier releases read NumPy's internal array-descriptor layout, which
    # changed in NumPy 2.0. Built against NumPy >= 2 they hand back arrays with
    # a stride of 0, so every element aliases element 0. cuProx shipped exactly
    # that: solutions came back as x[0] repeated while the objective (computed
    # in C++) stayed correct, which is about the hardest version of the bug to
    # notice. The floor below is what stops it recurring.
    set(CUPROX_MIN_PYBIND11 2.12)
    find_package(pybind11 ${CUPROX_MIN_PYBIND11} CONFIG QUIET)

    if(NOT pybind11_FOUND AND Python_EXECUTABLE)
        # Prefer the copy in the active interpreter before downloading one.
        execute_process(
            COMMAND "${Python_EXECUTABLE}" -c
                    "import pybind11; print(pybind11.get_cmake_dir())"
            OUTPUT_VARIABLE _pybind_dir RESULT_VARIABLE _pybind_rc
            ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
        if(_pybind_rc EQUAL 0 AND _pybind_dir)
            find_package(pybind11 ${CUPROX_MIN_PYBIND11} CONFIG QUIET
                         PATHS "${_pybind_dir}" NO_DEFAULT_PATH)
        endif()
    endif()

    if(NOT pybind11_FOUND)
        message(STATUS "pybind11 not found. Will be fetched.")
        include(FetchContent)
        FetchContent_Declare(
            pybind11
            GIT_REPOSITORY https://github.com/pybind/pybind11.git
            GIT_TAG v2.13.6
        )
        FetchContent_MakeAvailable(pybind11)
    endif()

    add_subdirectory(src/bindings)
endif()

# ============================================================================
# Tests
# ============================================================================

if(CUPROX_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests/cpp)
endif()

# ============================================================================
# Installation
# ============================================================================

include(GNUInstallDirs)

install(TARGETS cuprox_core
    EXPORT cuproxTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# ============================================================================
# Summary
# ============================================================================

message(STATUS "")
message(STATUS "============================================================")
message(STATUS "cuProx Configuration Summary")
message(STATUS "============================================================")
message(STATUS "  Version:         ${PROJECT_VERSION}")
message(STATUS "  CUDA:            ${CUPROX_HAS_CUDA}")
if(CUPROX_HAS_CUDA)
    message(STATUS "  CUDA Version:    ${CMAKE_CUDA_COMPILER_VERSION}")
    message(STATUS "  Architectures:   ${CMAKE_CUDA_ARCHITECTURES}")
endif()
message(STATUS "  Build Tests:     ${CUPROX_BUILD_TESTS}")
message(STATUS "  Build Python:    ${CUPROX_BUILD_PYTHON}")
message(STATUS "  Precision:       ${CUPROX_USE_DOUBLE}")
message(STATUS "============================================================")
message(STATUS "")

