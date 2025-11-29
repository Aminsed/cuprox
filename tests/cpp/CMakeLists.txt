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
    add_executable(test_dense_vector test_dense_vector.cpp)
    target_link_libraries(test_dense_vector PRIVATE
        cuprox::core
        GTest::gtest_main
    )
    gtest_discover_tests(test_dense_vector)
    
    add_executable(test_sparse_matrix test_sparse_matrix.cpp)
    target_link_libraries(test_sparse_matrix PRIVATE
        cuprox::core
        GTest::gtest_main
    )
    gtest_discover_tests(test_sparse_matrix)
endif()

