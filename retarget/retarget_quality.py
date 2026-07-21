"""Shared kinematic quality checks for the X2 retargeting pipelines."""

from __future__ import annotations

import numpy as np


ROOT_LINEAR_VELOCITY_LIMIT = 10.0  # m/s; catches mocap discontinuities
ROOT_ANGULAR_VELOCITY_LIMIT = 15.0  # rad/s
JOINT_LIMIT_FRACTION_LIMIT = 0.25

# Deep crouches/crawls legitimately use leg hard stops, straight arms use the
# elbow stop, and several X2 arm joints have narrow asymmetric ranges.  The
# pathological dataset-wide saturation found in this pipeline is specifically
# the torso task pinning the waist.  Record every joint, but only make sustained
# saturation fatal for the three waist axes.
ENFORCED_SATURATION_JOINTS = {
    "waist_yaw_joint",
    "waist_pitch_joint",
    "waist_roll_joint",
}


def root_motion_metrics(qpos: np.ndarray, fps: float) -> tuple[float, float]:
    """Return maximum root linear and angular speeds."""
    if len(qpos) < 2:
        return 0.0, 0.0
    linear = np.linalg.norm(np.diff(qpos[:, :3], axis=0), axis=1) * fps
    quat = np.asarray(qpos[:, 3:7], dtype=np.float64)
    norm = np.linalg.norm(quat, axis=1, keepdims=True)
    quat = quat / np.maximum(norm, 1e-12)
    # q and -q encode the same orientation, hence abs(dot).
    dots = np.clip(np.abs(np.sum(quat[:-1] * quat[1:], axis=1)), 0.0, 1.0)
    angular = 2.0 * np.arccos(dots) * fps
    return float(linear.max()), float(angular.max())


def joint_limit_report(model, mujoco, dof: np.ndarray, margin: float = 0.02):
    """Return per-joint saturation fractions and unexpected saturated joints."""
    hinge_ids = [
        joint_id
        for joint_id in range(model.njnt)
        if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_HINGE
    ]
    if dof.shape[1] != len(hinge_ids):
        raise ValueError(f"dof width {dof.shape[1]} != {len(hinge_ids)} hinge joints")
    lo = model.jnt_range[hinge_ids, 0]
    hi = model.jnt_range[hinge_ids, 1]
    fractions = ((dof > hi - margin) | (dof < lo + margin)).mean(axis=0)
    names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        for joint_id in hinge_ids
    ]
    report = {
        name: round(float(fraction), 3)
        for name, fraction in zip(names, fractions)
        if fraction > 0.05
    }
    unexpected = {
        name: round(float(fraction), 3)
        for name, fraction in zip(names, fractions)
        if fraction > JOINT_LIMIT_FRACTION_LIMIT
        and name in ENFORCED_SATURATION_JOINTS
    }
    return report, unexpected


def foot_collision_geoms(model, mujoco) -> list[int]:
    """Resolve the X2 sole collision spheres from the model."""
    body_ids = {
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        for name in ("left_ankle_roll_link", "right_ankle_roll_link")
    }
    geom_ids = [
        geom_id
        for geom_id in range(model.ngeom)
        if model.geom_bodyid[geom_id] in body_ids
        and model.geom_contype[geom_id] != 0
        and model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_SPHERE
    ]
    if not geom_ids:
        raise RuntimeError("X2 sole collision spheres were not found")
    return geom_ids


def sole_height(model, data, geom_ids: list[int]) -> float:
    """Return the actual lowest point of all X2 sole collision spheres."""
    return min(
        float(data.geom_xpos[geom_id, 2] - model.geom_size[geom_id, 0])
        for geom_id in geom_ids
    )
