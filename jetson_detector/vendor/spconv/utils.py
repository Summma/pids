from __future__ import annotations

import numpy as np


class VoxelGenerator:
    def __init__(
        self,
        voxel_size,
        point_cloud_range,
        max_num_points,
        max_voxels,
        **kwargs,
    ):
        self.voxel_size = np.asarray(voxel_size, dtype=np.float32)
        self.point_cloud_range = np.asarray(point_cloud_range, dtype=np.float32)
        self.max_num_points = int(max_num_points)
        self.max_voxels = int(max_voxels)
        xyz_range = self.point_cloud_range[3:6] - self.point_cloud_range[:3]
        self.grid_size = np.round(xyz_range / self.voxel_size).astype(np.int32)

    def generate(self, points):
        points = np.asarray(points, dtype=np.float32)
        if points.ndim != 2:
            raise ValueError(f"expected points shape (N,C), got {points.shape}")

        lower = self.point_cloud_range[:3]
        upper = self.point_cloud_range[3:6]
        keep = np.all((points[:, :3] >= lower) & (points[:, :3] < upper), axis=1)
        points = points[keep]

        voxels = np.zeros(
            (self.max_voxels, self.max_num_points, points.shape[1]),
            dtype=np.float32,
        )
        coords = np.zeros((self.max_voxels, 3), dtype=np.int32)
        num_points = np.zeros((self.max_voxels,), dtype=np.int32)

        if points.size == 0:
            return voxels[:0], coords[:0], num_points[:0]

        voxel_xyz = np.floor((points[:, :3] - lower) / self.voxel_size).astype(np.int32)
        valid = np.all((voxel_xyz >= 0) & (voxel_xyz < self.grid_size), axis=1)
        points = points[valid]
        voxel_xyz = voxel_xyz[valid]
        if points.size == 0:
            return voxels[:0], coords[:0], num_points[:0]

        keys = (
            voxel_xyz[:, 2].astype(np.int64) * int(self.grid_size[1]) * int(self.grid_size[0])
            + voxel_xyz[:, 1].astype(np.int64) * int(self.grid_size[0])
            + voxel_xyz[:, 0].astype(np.int64)
        )
        unique_keys, first_idx, inverse = np.unique(keys, return_index=True, return_inverse=True)
        selected_unique = np.argsort(first_idx, kind="stable")[: self.max_voxels]
        voxel_count = int(selected_unique.size)
        if voxel_count == 0:
            return voxels[:0], coords[:0], num_points[:0]

        coords_xyz = voxel_xyz[first_idx[selected_unique]]
        coords[:voxel_count, 0] = coords_xyz[:, 2]
        coords[:voxel_count, 1] = coords_xyz[:, 1]
        coords[:voxel_count, 2] = coords_xyz[:, 0]

        unique_to_voxel = np.full(unique_keys.shape[0], -1, dtype=np.int32)
        unique_to_voxel[selected_unique] = np.arange(voxel_count, dtype=np.int32)
        point_voxels = unique_to_voxel[inverse]
        selected_points = point_voxels >= 0
        point_voxels = point_voxels[selected_points]
        points = points[selected_points]

        order = np.argsort(point_voxels, kind="stable")
        sorted_voxels = point_voxels[order]
        sorted_points = points[order]
        counts_full = np.bincount(sorted_voxels, minlength=voxel_count).astype(np.int32)
        starts = np.r_[0, np.cumsum(counts_full[:-1])]
        ranks = np.arange(sorted_voxels.size, dtype=np.int32) - np.repeat(starts, counts_full)
        within_cap = ranks < self.max_num_points
        voxels[sorted_voxels[within_cap], ranks[within_cap]] = sorted_points[within_cap]
        num_points[:voxel_count] = np.minimum(counts_full, self.max_num_points)
        return voxels[:voxel_count], coords[:voxel_count], num_points[:voxel_count]


VoxelGeneratorV2 = VoxelGenerator
