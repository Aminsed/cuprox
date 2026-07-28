"""Single source of truth for the package version.

pyproject.toml reads it back through the scikit-build-core regex metadata
provider and ``cuprox.__version__`` re-exports it, so there is exactly one
place to bump.
"""

__version__ = "0.2.0"
