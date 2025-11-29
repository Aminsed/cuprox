# ============================================================================
# cuProx C++ Tests
# ============================================================================

include(FetchContent)

# Google Test
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

include(GoogleTest)

# ============================================================================
# Test Executables
# ============================================================================

if(CUPROX_HAS_CUDA)
    # Dense vector tests
    add_executable(test_dense_vector test_dense_vector.cpp)
    target_link_libraries(test_dense_vector PRIVATE
        cuprox::core
        GTest::gtest_main
    )
    gtest_discover_tests(test_dense_vector)
    
    # Sparse matrix tests
    add_executable(test_sparse_matrix test_sparse_matrix.cpp)
    target_link_libraries(test_sparse_matrix PRIVATE
        cuprox::core
        GTest::gtest_main
    )
    gtest_discover_tests(test_sparse_matrix)
    
    # PDHG solver tests
    add_executable(test_pdhg test_pdhg.cpp)
    target_link_libraries(test_pdhg PRIVATE
        cuprox::core
        GTest::gtest_main
    )
    gtest_discover_tests(test_pdhg)
    
    # ADMM solver tests
    add_executable(test_admm test_admm.cpp)
    target_link_libraries(test_admm PRIVATE
        cuprox::core
        GTest::gtest_main
    )
    gtest_discover_tests(test_admm)
    
    # Scaling tests
    add_executable(test_scaling test_scaling.cpp)
    target_link_libraries(test_scaling PRIVATE
        cuprox::core
        GTest::gtest_main
    )
    gtest_discover_tests(test_scaling)
endif()
