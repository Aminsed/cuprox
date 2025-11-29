# ============================================================================
# cuProx Core Library
# ============================================================================

set(CUPROX_CORE_HEADERS
    core/types.hpp
    core/error.hpp
)

# ============================================================================
# Create Library Target
# ============================================================================

if(CUPROX_HAS_CUDA)
    set(CUPROX_CUDA_HEADERS
        core/memory.cuh
        core/cuda_context.cuh
        core/dense_vector.cuh
        core/sparse_matrix.cuh
        linalg/projections.cuh
    )
    
    set(CUPROX_CUDA_SOURCES
        core/dense_vector.cu
        linalg/projections.cu
    )
    
    add_library(cuprox_core STATIC
        ${CUPROX_CUDA_SOURCES}
        ${CUPROX_CORE_HEADERS}
        ${CUPROX_CUDA_HEADERS}
    )
    
    target_compile_definitions(cuprox_core PUBLIC CUPROX_HAS_CUDA)
    
    set_target_properties(cuprox_core PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
    
    target_link_libraries(cuprox_core PUBLIC
        CUDA::cudart
        CUDA::cublas
        CUDA::cusparse
    )
    
else()
    # CPU-only build - minimal header-only
    add_library(cuprox_core INTERFACE)
    target_compile_definitions(cuprox_core INTERFACE CUPROX_CPU_ONLY)
endif()

# ============================================================================
# Target Properties
# ============================================================================

if(CUPROX_HAS_CUDA)
    target_include_directories(cuprox_core PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
        $<INSTALL_INTERFACE:include>
    )
else()
    target_include_directories(cuprox_core INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
        $<INSTALL_INTERFACE:include>
    )
endif()

# ============================================================================
# Alias for cleaner usage
# ============================================================================

add_library(cuprox::core ALIAS cuprox_core)
