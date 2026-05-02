"""Thermal-camera ↔ lidar projection.

Given a lidar point cloud and a thermal frame plus calibration (intrinsics
+ 6-DoF extrinsics), project each 3D point into the thermal image and sample
the corresponding pixel — used to color the lidar cloud by thermal value.

Coordinate conventions:
  - Ouster lidar:  +X forward, +Y left, +Z up (right-handed).
  - OpenCV camera: +X right, +Y down, +Z forward.
A constant base rotation R_LIDAR_TO_CAM_BASE handles the axis remap; the
user-tuned roll/pitch/yaw in Extrinsics is applied on top of that.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ThermalIntrinsics:
    """Pinhole intrinsics. Defaults: Boson 640 with a 14 mm lens (~50° HFOV)."""
    width: int = 640
    height: int = 512
    fx: float = 686.0
    fy: float = 686.0
    cx: float = 320.0
    cy: float = 256.0


@dataclass
class Extrinsics:
    tx: float = 0.0       # m (in cam frame, after base rotation)
    ty: float = 0.0
    tz: float = 0.0
    roll_deg: float = 0.0
    pitch_deg: float = 0.0
    yaw_deg: float = 0.0


# Lidar (X fwd, Y left, Z up) → camera (X right, Y down, Z fwd):
#   cam_x = -lidar_y;  cam_y = -lidar_z;  cam_z = lidar_x
R_LIDAR_TO_CAM_BASE = np.array(
    [[0.0, -1.0, 0.0],
     [0.0, 0.0, -1.0],
     [1.0, 0.0, 0.0]],
    dtype=np.float64,
)


def euler_to_R(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """ZYX intrinsic Tait-Bryan rotation, angles in radians."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def project_points(
    xyz_lidar: np.ndarray,
    intr: ThermalIntrinsics,
    extr: Extrinsics,
) -> tuple[np.ndarray, np.ndarray]:
    """Project (N,3) lidar points to (u,v) thermal pixels.

    Returns:
        uv: (N, 2) int32, valid entries hold pixel coords; invalid are -1.
        valid: (N,) bool, True where the point lands inside the image.
    """
    R_user = euler_to_R(
        np.deg2rad(extr.roll_deg),
        np.deg2rad(extr.pitch_deg),
        np.deg2rad(extr.yaw_deg),
    )
    R = R_user @ R_LIDAR_TO_CAM_BASE
    t = np.array([extr.tx, extr.ty, extr.tz], dtype=np.float64)

    P_cam = (R @ xyz_lidar.T).T + t  # (N, 3)
    Z = P_cam[:, 2]
    in_front = Z > 0.05

    safe_z = np.where(in_front, Z, 1.0)
    u = (intr.fx * P_cam[:, 0] / safe_z + intr.cx).astype(np.int32)
    v = (intr.fy * P_cam[:, 1] / safe_z + intr.cy).astype(np.int32)
    in_image = (u >= 0) & (u < intr.width) & (v >= 0) & (v < intr.height)

    final = in_front & in_image
    uv = np.full((len(xyz_lidar), 2), -1, dtype=np.int32)
    uv[final, 0] = u[final]
    uv[final, 1] = v[final]
    return uv, final


def colorize_with_thermal(
    xyz_lidar: np.ndarray,
    thermal_bgr: np.ndarray,
    intr: ThermalIntrinsics,
    extr: Extrinsics,
    fallback_rgb: tuple[float, float, float] = (0.25, 0.25, 0.28),
) -> tuple[np.ndarray, np.ndarray]:
    """Sample thermal pixels for each lidar point. Returns (rgba, valid)."""
    uv, valid = project_points(xyz_lidar, intr, extr)
    colors = np.zeros((len(xyz_lidar), 4), dtype=np.float32)
    colors[:, :3] = fallback_rgb
    colors[:, 3] = 1.0
    if valid.any():
        h, w = thermal_bgr.shape[:2]
        # Defensive bounds (in case the thermal frame size changed since calibration)
        u = np.clip(uv[valid, 0], 0, w - 1)
        v = np.clip(uv[valid, 1], 0, h - 1)
        sample = thermal_bgr[v, u]  # BGR
        colors[valid, 0] = sample[:, 2] / 255.0
        colors[valid, 1] = sample[:, 1] / 255.0
        colors[valid, 2] = sample[:, 0] / 255.0
    return colors, valid


# Common Boson lens presets (HFOV in degrees → fx/fy assuming square pixels)
LENS_PRESETS: dict[str, float] = {
    "Boson 8.7mm (~95°)": 95.0,
    "Boson 14mm (~50°)": 50.0,
    "Boson 18mm (~40°)": 40.0,
    "Boson 24mm (~24°)": 24.0,
    "Boson 36mm (~16°)": 16.0,
    "Boson 60mm (~10°)": 10.0,
}


def intrinsics_from_hfov(width: int, height: int, hfov_deg: float) -> ThermalIntrinsics:
    f = width / (2.0 * np.tan(np.deg2rad(hfov_deg) / 2.0))
    return ThermalIntrinsics(
        width=width, height=height, fx=f, fy=f, cx=width / 2.0, cy=height / 2.0
    )
