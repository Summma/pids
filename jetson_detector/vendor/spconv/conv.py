from __future__ import annotations

import torch.nn as nn


class SparseConvTensor:
    def __init__(self, features, indices=None, spatial_shape=None, batch_size=None):
        self.features = features
        self.indices = indices
        self.spatial_shape = spatial_shape
        self.batch_size = batch_size

    def replace_feature(self, features):
        self.features = features
        return self

    def dense(self):
        raise RuntimeError("The bundled spconv shim does not implement sparse tensor densification")


class SparseModule(nn.Module):
    pass


class SparseConvolution(SparseModule):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, *args, **kwargs):
        raise RuntimeError("The bundled spconv shim does not implement sparse convolutions")


class SparseSequential(nn.Sequential):
    pass


class SubMConv2d(SparseConvolution):
    pass


class SubMConv3d(SparseConvolution):
    pass


class SparseConv2d(SparseConvolution):
    pass


class SparseConv3d(SparseConvolution):
    pass


class SparseInverseConv2d(SparseConvolution):
    pass


class SparseInverseConv3d(SparseConvolution):
    pass
