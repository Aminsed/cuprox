"""
Utility functions for PyTorch integration.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert PyTorch tensor to NumPy array.

    Args:
        tensor: PyTorch tensor (any device)

    Returns:
        NumPy array (float64)
    """
    return tensor.detach().cpu().numpy().astype(np.float64)


def to_torch(
    array: np.ndarray,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Convert NumPy array to PyTorch tensor.

    Args:
        array: NumPy array
        device: Target device
        dtype: Target dtype

    Returns:
        PyTorch tensor
    """
    import torch

    return torch.from_numpy(array.astype(np.float64)).to(device=device, dtype=dtype)


def check_torch_available() -> None:
    """
    Raise ImportError if PyTorch is not available.

    Raises:
        ImportError: If torch is not installed
    """
    try:
        import torch  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "PyTorch is required for cuprox.torch. " "Install with: pip install torch"
        ) from e


def get_device_and_dtype(tensor: torch.Tensor) -> tuple:
    """
    Extract device and dtype from tensor.

    Args:
        tensor: PyTorch tensor

    Returns:
        Tuple of (device, dtype)
    """
    return tensor.device, tensor.dtype
