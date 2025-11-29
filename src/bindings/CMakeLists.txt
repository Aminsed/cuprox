# ============================================================================
# cuProx Python Bindings
# ============================================================================

pybind11_add_module(_core MODULE
    module.cpp
)

target_link_libraries(_core PRIVATE cuprox::core)

# Install to correct location for scikit-build
install(TARGETS _core DESTINATION cuprox)

