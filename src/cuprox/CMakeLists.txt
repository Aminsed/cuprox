# ============================================================================
# cuProx Core Library
# ============================================================================

# Collect source files
set(CUPROX_CORE_SOURCES
    # Core (CPU)
    # core/error.cpp  # Will be added later
)

set(CUPROX_CORE_HEADERS
    core/types.hpp
    core/error.hpp
)

# ============================================================================
# Create Library Target
# ============================================================================

if(CUPROX_HAS_CUDA)
    # GPU-enabled build
    set(CUPROX_CUDA_SOURCES
        # Will be populated as we implement:
        # core/memory.cu
        # core/dense_vector.cu
        # core/sparse_matrix.cu
        # linalg/blas.cu
        # linalg/spmv.cu
        # linalg/projections.cu
        # preprocess/scaling.cu
        # solvers/pdhg.cu
        # solvers/admm.cu
    )
    
    add_library(cuprox_core STATIC
        ${CUPROX_CORE_SOURCES}
        ${CUPROX_CUDA_SOURCES}
        ${CUPROX_CORE_HEADERS}
    )
    
    target_compile_definitions(cuprox_core PUBLIC CUPROX_HAS_CUDA)
    
    target_link_libraries(cuprox_core PUBLIC
        CUDA::cudart
        CUDA::cublas
        CUDA::cusparse
    )
    
else()
    # CPU-only build
    set(CUPROX_CPU_SOURCES
        # cpu_fallback/cpu_pdhg.cpp
        # cpu_fallback/cpu_admm.cpp
    )
    
    add_library(cuprox_core STATIC
        ${CUPROX_CORE_SOURCES}
        ${CUPROX_CPU_SOURCES}
        ${CUPROX_CORE_HEADERS}
    )
endif()

# ============================================================================
# Target Properties
# ============================================================================

target_include_directories(cuprox_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
    $<INSTALL_INTERFACE:include>
)

# Precision configuration
if(CUPROX_USE_DOUBLE)
    target_compile_definitions(cuprox_core PUBLIC CUPROX_USE_DOUBLE)
else()
    target_compile_definitions(cuprox_core PUBLIC CUPROX_USE_FLOAT)
endif()

# Eigen dependency (for CPU fallback and utilities)
if(TARGET Eigen3::Eigen)
    target_link_libraries(cuprox_core PUBLIC Eigen3::Eigen)
endif()

# ============================================================================
# Alias for cleaner usage
# ============================================================================

add_library(cuprox::core ALIAS cuprox_core)

