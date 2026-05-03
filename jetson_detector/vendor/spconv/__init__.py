"""Small PointPillars-only spconv compatibility shim.

This is not a sparse convolution implementation. It exists so OpenPCDet's
PointPillars path can run on Jetson systems where no aarch64 spconv wheel is
available. The KITTI PointPillars config uses dense 2D convolutions and only
needs spconv's voxel generator at runtime.
"""

from __future__ import annotations

from . import constants, conv, utils
from .conv import (
    SparseConv2d,
    SparseConv3d,
    SparseConvolution,
    SparseConvTensor,
    SparseInverseConv2d,
    SparseInverseConv3d,
    SparseModule,
    SparseSequential,
    SubMConv2d,
    SubMConv3d,
)

__version__ = "2.3.0"

__all__ = [
    "__version__",
    "constants",
    "conv",
    "utils",
    "SparseModule",
    "SparseSequential",
    "SparseConvolution",
    "SparseConvTensor",
    "SubMConv2d",
    "SubMConv3d",
    "SparseConv2d",
    "SparseConv3d",
    "SparseInverseConv2d",
    "SparseInverseConv3d",
]
