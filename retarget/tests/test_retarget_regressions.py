from __future__ import annotations

import pathlib
import sys

import numpy as np


RETARGET_ROOT = pathlib.Path(__file__).resolve().parents[1]
ANY2ANY_ROOT = RETARGET_ROOT.parent
GMR_ROOT = ANY2ANY_ROOT.parent / "GMR"
sys.path.insert(0, str(ANY2ANY_ROOT))
sys.path.insert(0, str(GMR_ROOT))


def test_target_sample_indices_are_exact_rate():
    from general_motion_retargeting.utils.smpl import target_sample_indices

    # 100 Hz no longer degrades to 33.33 Hz when 30 Hz was requested.
    indices = target_sample_indices(1001, 100.0, 30.0)
    assert len(indices) == 301
    assert np.allclose(np.diff(indices) / 100.0, 1.0 / 30.0)

    # Floating metadata just below 60 must not make int(src / target) == 1.
    indices = target_sample_indices(601, 59.9999885559082, 30.0)
    assert 300 <= len(indices) <= 301
    assert np.allclose(np.diff(indices) / 59.9999885559082, 1.0 / 30.0)

    # Low-rate inputs are upsampled and truthfully labelled at the target rate.
    indices = target_sample_indices(251, 25.0, 30.0)
    assert len(indices) == 301


def test_long_clips_are_balanced_into_bounded_non_overlapping_segments():
    from retarget.batch_retarget_x2 import segment_frame_ranges

    # A tiny tail must be redistributed, not rejected as a sub-2-second clip.
    ranges = segment_frame_ranges(2430, 30.0, 40.0)  # 81 seconds
    assert ranges[0][0] == 0
    assert ranges[-1][1] == 2430
    assert all(left[1] == right[0] for left, right in zip(ranges, ranges[1:]))
    durations = [(end - start) / 30.0 for start, end in ranges]
    assert len(ranges) == 3
    assert max(durations) <= 40.0
    assert min(durations) >= 2.0

    # Motions already within the bound retain their original single output.
    assert segment_frame_ranges(1200, 30.0, 40.0) == [(0, 1200)]


def test_root_motion_metrics_detect_discontinuity_and_quaternion_sign_flip():
    from retarget.retarget_quality import root_motion_metrics

    qpos = np.zeros((4, 8), dtype=np.float64)
    qpos[:, 3] = 1.0
    qpos[2, 0] = 2.0
    linear, angular = root_motion_metrics(qpos, 30.0)
    assert linear == 60.0
    assert angular == 0.0

    # Quaternion sign changes do not represent physical angular motion.
    qpos[1, 3] = -1.0
    _, angular = root_motion_metrics(qpos, 30.0)
    assert angular == 0.0


def test_real_sole_and_unexpected_waist_limit_are_resolved_from_model():
    import mujoco

    from retarget.retarget_quality import (
        foot_collision_geoms,
        joint_limit_report,
        sole_height,
    )

    model = mujoco.MjModel.from_xml_path(
        str(GMR_ROOT / "assets/agibot_x2/x2_mocap.xml")
    )
    data = mujoco.MjData(model)
    data.qpos[3] = 1.0
    mujoco.mj_forward(model, data)
    geoms = foot_collision_geoms(model, mujoco)
    expected = min(
        data.geom_xpos[geom, 2] - model.geom_size[geom, 0] for geom in geoms
    )
    assert sole_height(model, data, geoms) == expected

    hinge_ids = [
        joint_id for joint_id in range(model.njnt)
        if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_HINGE
    ]
    names = [model.joint(joint_id).name for joint_id in hinge_ids]
    dof = np.zeros((20, len(hinge_ids)))
    waist = names.index("waist_pitch_joint")
    dof[:, waist] = model.jnt_range[hinge_ids[waist], 1]
    report, unexpected = joint_limit_report(model, mujoco, dof)
    assert report["waist_pitch_joint"] == 1.0
    assert unexpected["waist_pitch_joint"] == 1.0


def test_x2_waist_posture_regularizer_is_active():
    import mujoco
    import mink

    from general_motion_retargeting import GeneralMotionRetargeting

    retargeter = GeneralMotionRetargeting(
        actual_human_height=1.8,
        src_human="smplx",
        tgt_robot="agibot_x2",
        verbose=False,
    )
    posture = next(task for task in retargeter.tasks1 if isinstance(task, mink.PostureTask))
    joint_id = mujoco.mj_name2id(
        retargeter.model, mujoco.mjtObj.mjOBJ_JOINT, "waist_pitch_joint"
    )
    assert posture.cost[retargeter.model.jnt_dofadr[joint_id]] == 20.0
