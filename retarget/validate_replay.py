"""Kinematic replay validation for the G1 -> X2 scattering matrix.

Pipeline:
  1. Load a SONIC reference motion (joint_pos + tracked body poses).
  2. Resolve the CSV joint order by forward-kinematics cross-check against the
     stored body positions on the G1 model (IsaacLab vs MuJoCo candidates).
  3. Map joints through the scattering matrix to the X2 and replay with FK.
  4. Report sanity metrics (joint-limit violations, foot clearance) and render
     a side-by-side video (G1 left, X2 right).

Run from the repo root (Desktop/Byte) inside .venv_sim.
"""

import csv
import os
import sys

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kinematic_alignment import (
    G1_ISAACLAB_TO_MUJOCO_DOF,
    G1_MUJOCO_JOINTS,
    X2_MUJOCO_JOINTS,
    X2_UNMATCHED_JOINTS,
    g1_to_x2,
)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
G1_XML = os.path.join(ROOT, "Sonic_Retarget/GR00T-WholeBodyControl/gear_sonic/data/assets/robot_description/mjcf/g1_29dof_rev_1_0.xml")
X2_XML = os.path.join(ROOT, "aimdk/X2_URDF-v1.3.0/x2_ultra.xml")
MOTION = os.path.join(ROOT, "Sonic_Retarget/GR00T-WholeBodyControl/gear_sonic_deploy/reference/example/dance_in_da_party_001__A464")
OUT_VIDEO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "replay_g1_vs_x2.mp4")

# 14 tracked bodies: IsaacLab body indices from the motion metadata mapped to
# link names via G1_ISAACLAB_JOINTS (gear_sonic robots/g1.py body list).
G1_ISAACLAB_BODIES = [
    "pelvis", "left_hip_pitch_link", "right_hip_pitch_link", "waist_yaw_link",
    "left_hip_roll_link", "right_hip_roll_link", "waist_roll_link",
    "left_hip_yaw_link", "right_hip_yaw_link", "torso_link",
    "left_knee_link", "right_knee_link",
    "left_shoulder_pitch_link", "right_shoulder_pitch_link",
    "left_ankle_pitch_link", "right_ankle_pitch_link",
    "left_shoulder_roll_link", "right_shoulder_roll_link",
    "left_ankle_roll_link", "right_ankle_roll_link",
    "left_shoulder_yaw_link", "right_shoulder_yaw_link",
    "left_elbow_link", "right_elbow_link",
    "left_wrist_roll_link", "right_wrist_roll_link",
    "left_wrist_pitch_link", "right_wrist_pitch_link",
    "left_wrist_yaw_link", "right_wrist_yaw_link",
]
TRACKED_IDX = [0, 4, 10, 18, 5, 11, 19, 9, 16, 22, 28, 17, 23, 29]


def load_csv(path):
    with open(path) as f:
        rows = list(csv.reader(f))
    return np.array(rows[1:], dtype=np.float64)


def load_motion(d):
    joint_pos = load_csv(os.path.join(d, "joint_pos.csv"))          # (T, 29)
    body_pos = load_csv(os.path.join(d, "body_pos.csv")).reshape(-1, 14, 3)
    body_quat = load_csv(os.path.join(d, "body_quat.csv")).reshape(-1, 14, 4)
    return joint_pos, body_pos, body_quat


def fk_body_positions(model, data, root_pos, root_quat, qj, body_names):
    data.qpos[:3] = root_pos
    data.qpos[3:7] = root_quat  # wxyz
    data.qpos[7:] = qj
    mujoco.mj_kinematics(model, data)
    ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) for n in body_names]
    return np.array([data.xpos[i] for i in ids])


def resolve_joint_order(g1_model, g1_data, joint_pos, body_pos, body_quat):
    """FK cross-check: which interpretation of the CSV reproduces body_pos?"""
    tracked_names = [G1_ISAACLAB_BODIES[i] for i in TRACKED_IDX]
    frames = np.linspace(0, len(joint_pos) - 1, 25, dtype=int)
    errs = {}
    for label, to_mujoco in [
        ("isaaclab", lambda q: q[G1_ISAACLAB_TO_MUJOCO_DOF]),
        ("mujoco", lambda q: q),
    ]:
        e = []
        for t in frames:
            pos = fk_body_positions(
                g1_model, g1_data, body_pos[t, 0], body_quat[t, 0],
                to_mujoco(joint_pos[t]), tracked_names)
            e.append(np.linalg.norm(pos - body_pos[t], axis=-1).mean())
        errs[label] = float(np.mean(e))
    best = min(errs, key=errs.get)
    print(f"joint-order resolution: {errs} -> CSV is in '{best}' order")
    if errs[best] > 0.05:
        raise RuntimeError(f"FK mismatch too large ({errs[best]:.3f} m) — check body list / quat convention")
    return best


def main():
    g1 = mujoco.MjModel.from_xml_path(G1_XML)
    g1d = mujoco.MjData(g1)
    x2 = mujoco.MjModel.from_xml_path(X2_XML)
    x2d = mujoco.MjData(x2)

    # sanity: joint orders in the loaded models match the alignment module
    for model, names in [(g1, G1_MUJOCO_JOINTS), (x2, X2_MUJOCO_JOINTS)]:
        hinges = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
                  for i in range(model.njnt) if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE]
        assert hinges == names, "model joint order drifted from kinematic_alignment"

    joint_pos, body_pos, body_quat = load_motion(MOTION)
    T = len(joint_pos)
    print(f"motion: {os.path.basename(MOTION)}  frames={T}")

    order = resolve_joint_order(g1, g1d, joint_pos, body_pos, body_quat)
    q_g1_mj = joint_pos[:, G1_ISAACLAB_TO_MUJOCO_DOF] if order == "isaaclab" else joint_pos

    # --- scattering matrix: G1 (29, mujoco order) -> X2 (31, mujoco order) ---
    q_x2 = g1_to_x2(q_g1_mj)

    # --- sanity metrics on the X2 ---
    lo, hi = x2.jnt_range[1:, 0], x2.jnt_range[1:, 1]  # skip free joint
    viol = np.maximum(q_x2 - hi, 0) + np.maximum(lo - q_x2, 0)
    per_joint = viol.max(axis=0)
    print("\njoint-limit violations on X2 (max rad over motion):")
    bad = [(X2_MUJOCO_JOINTS[j], per_joint[j]) for j in np.argsort(-per_joint)[:8] if per_joint[j] > 1e-6]
    if bad:
        for name, v in bad:
            print(f"  {name:32s} {v:+.3f} rad")
    else:
        print("  none")
    clipped = np.clip(q_x2, lo, hi)
    clip_ratio = (np.abs(clipped - q_x2) > 1e-6).mean()
    print(f"fraction of joint samples clipped: {clip_ratio:.2%}")

    # foot clearance: min foot height over the motion on both robots
    for model, data, qj, feet in [
        (g1, g1d, q_g1_mj, ["left_ankle_roll_link", "right_ankle_roll_link"]),
        (x2, x2d, clipped, ["left_ankle_roll_link", "right_ankle_roll_link"]),
    ]:
        zmin = []
        for t in range(0, T, 5):
            pos = fk_body_positions(model, data, body_pos[t, 0], body_quat[t, 0], qj[t], feet)
            zmin.append(pos[:, 2].min())
        name = "G1" if model is g1 else "X2"
        print(f"{name}: foot-link z range over motion: [{min(zmin):+.3f}, {max(zmin):+.3f}] m")

    # --- side-by-side render ---
    try:
        os.environ.setdefault("MUJOCO_GL", "egl")
        import cv2
        h, w = 480, 480
        rg1 = mujoco.Renderer(g1, h, w)
        rx2 = mujoco.Renderer(x2, h, w)
        cam = mujoco.MjvCamera()
        cam.distance, cam.elevation, cam.azimuth = 3.0, -15, 135
        vw = cv2.VideoWriter(OUT_VIDEO, cv2.VideoWriter_fourcc(*"mp4v"), 30, (2 * w, h))
        for t in range(0, T, 2):  # ~15 fps playback of 30 fps data
            frames = []
            for model, data, r, qj in [(g1, g1d, rg1, q_g1_mj), (x2, x2d, rx2, clipped)]:
                data.qpos[:3] = body_pos[t, 0]
                data.qpos[3:7] = body_quat[t, 0]
                data.qpos[7:] = qj[t]
                mujoco.mj_forward(model, data)
                cam.lookat[:] = body_pos[t, 0]
                r.update_scene(data, cam)
                frames.append(r.render())
            side = np.hstack(frames)
            cv2.putText(side, f"G1 (source)  frame {t}/{T}", (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(side, "X2 (via S_r)", (w + 10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            vw.write(cv2.cvtColor(side, cv2.COLOR_RGB2BGR))
        vw.release()
        print(f"\nvideo written: {OUT_VIDEO}")
    except Exception as e:  # rendering is best-effort; metrics above are the gate
        print(f"\nrender skipped ({type(e).__name__}: {e})")


if __name__ == "__main__":
    main()
