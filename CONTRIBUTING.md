# Contributing to cuProx

Thank you for your interest in contributing to cuProx! This document provides guidelines and instructions for contributing.

## Development Setup

### Prerequisites

- Python 3.9+
- CUDA Toolkit 11.4+ (optional, for GPU development)
- CMake 3.24+
- A C++17 compatible compiler

### Local Development Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/cuprox.git
cd cuprox

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or: venv\Scripts\activate  # Windows

# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pip install pre-commit
pre-commit install

# Run tests to verify setup
pytest tests/python -v
```

### Building with CUDA

If you have CUDA installed:

```bash
# Full build with CUDA
pip install -e .

# Verify CUDA is detected
python -c "import cuprox; print(cuprox.__cuda_available__)"
```

### Building CPU-only

```bash
# CPU-only build (no CUDA required)
pip install -e . --config-settings=cmake.define.CUPROX_CPU_ONLY=ON

# Or set environment variable
export CUPROX_CPU_ONLY=1
pip install -e .
```

## Development Workflow

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

### 2. Make Changes

Follow the coding standards:
- **Python**: Black formatting, Ruff linting
- **C++**: Follow existing style, use clang-format
- **Tests**: Write tests for new features (TDD encouraged)

### 3. Run Checks Locally

```bash
# Run all pre-commit hooks
pre-commit run --all-files

# Run tests
pytest tests/python -v

# Run specific test file
pytest tests/python/test_model.py -v

# Run with coverage
pytest tests/python --cov=python/cuprox --cov-report=html
```

### 4. Commit Changes

```bash
# Commits will be checked by pre-commit hooks
git add .
git commit -m "feat: add new feature description"
```

Commit message format:
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation only
- `test:` Adding tests
- `refactor:` Code refactoring
- `perf:` Performance improvement
- `ci:` CI/CD changes

### 5. Push and Create PR

```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub.

## Testing

### Running Tests

```bash
# All CPU tests
pytest tests/python -v -m "not gpu"

# Only GPU tests (requires GPU)
pytest tests/python -v -m "gpu"

# Specific test
pytest tests/python/test_solver_lp.py::TestSolveLPSimple -v

# With coverage report
pytest tests/python --cov=python/cuprox --cov-report=term-missing
```

### Writing Tests

- Place tests in `tests/python/`
- Use pytest fixtures from `conftest.py`
- Mark GPU tests with `@pytest.mark.gpu`
- Mark slow tests with `@pytest.mark.slow`

Example:

```python
import pytest
import numpy as np
from cuprox import solve

def test_simple_lp(simple_lp):
    """Test solving a simple LP."""
    result = solve(**simple_lp)
    assert result.status == "optimal"
    assert abs(result.objective - simple_lp["expected_obj"]) < 0.1

@pytest.mark.gpu
def test_gpu_performance(large_lp):
    """Test GPU solver performance."""
    result = solve(**large_lp, params={"device": "gpu"})
    assert result.status == "optimal"
```

## CI/CD

### Automated Checks (on every PR)

1. **Lint & Format** - Black, Ruff
2. **Type Check** - mypy
3. **CPU Tests** - pytest on Python 3.9-3.12
4. **Build Check** - Verify package builds

### GPU Testing

GPU tests require a self-hosted runner. To set up:

1. Install GitHub Actions runner on your GPU machine
2. Add labels: `self-hosted`, `gpu`
3. GPU tests will run automatically on `main` branch

### Releasing

Releases are triggered by version tags:

```bash
# Create and push a tag
git tag v0.1.0
git push origin v0.1.0
```

This triggers:
1. Build sdist and wheels
2. Publish to TestPyPI
3. Publish to PyPI
4. Create GitHub Release

## Code Style

### Python

- **Formatter**: Black (line length 100)
- **Linter**: Ruff
- **Type hints**: Required for public APIs

```python
def solve(
    c: np.ndarray,
    A: sparse.csr_matrix,
    b: np.ndarray,
    *,
    params: Optional[Dict[str, Any]] = None,
) -> SolveResult:
    """
    Solve an LP problem.
    
    Args:
        c: Objective coefficients
        A: Constraint matrix
        b: Constraint RHS
        params: Solver parameters
        
    Returns:
        SolveResult with solution
    """
    ...
```

### C++

- **Standard**: C++17
- **Formatter**: clang-format
- Follow existing code style

```cpp
namespace cuprox {

/**
 * @brief GPU vector class
 */
template <typename Scalar>
class DeviceVector {
public:
    explicit DeviceVector(int n);
    
    // Rule of five
    ~DeviceVector();
    DeviceVector(const DeviceVector&) = delete;
    DeviceVector& operator=(const DeviceVector&) = delete;
    DeviceVector(DeviceVector&&) noexcept;
    DeviceVector& operator=(DeviceVector&&) noexcept;
    
private:
    Scalar* data_ = nullptr;
    int size_ = 0;
};

}  // namespace cuprox
```

## Getting Help

- **Issues**: Report bugs or request features
- **Discussions**: Ask questions or share ideas
- **Email**: [your-email@example.com]

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

