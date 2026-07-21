"""Build a self-contained deployment package for the X2 robot.

The robot has rclpy + onnxruntime + numpy but NO torch / isaaclab / joblib-with-
tensors, so everything the 50 Hz loop needs is precomputed here into one npz:
Phi matrices, zero-pose offsets, default pose, per-joint PD gains, HARD joint
limits (deployment restores the limit protection that training zeroed out), the
policy's obs block order (hash-random per export -> detected, never hardcoded),
and the reference clip already mapped to G1 space.

Usage (workstation, isaaclab env):
  python pack_deploy.py --onnx <run>/exported/..._g1.onnx \
      --capture <run-capture-dir> --clip <name> \
      --motion-file <exact-motion-lib-file.pkl> --out deploy_pkg.npz
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_A2A = os.path.dirname(_HERE)
sys.path.insert(0, _A2A)
sys.path.insert(0, os.path.join(_A2A, "any2any_train"))
sys.path.insert(0, os.path.join(_A2A, "sim2sim"))

import aligned_env  # noqa: E402  (installs the isaaclab stubs)
from x2 import X2_ISAACLAB_TO_MUJOCO_MAPPING  # noqa: E402

import mujoco_sim2sim_x2 as S  # noqa: E402  (Align/Clip/ObsBuilder: MuJoCo-validated)

HIST = 10
FUTURE_SKIP = 5


def onnx_policy(onnx_path):
    import onnxruntime as ort

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name

    def pol(obs):
        x = np.asarray(obs, dtype=np.float32).reshape(1, -1)
        return sess.run(None, {name: x})[0].ravel()

    return pol


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--capture", required=True, help="dir with constants.npz + gt_step0.npz")
    ap.add_argument("--clip", required=True, help="reference clip name (motion_lib)")
    ap.add_argument(
        "--motion-file",
        help="exact motion-lib .pkl for --clip; avoids resolving a same-named clip from the wrong dataset",
    )
    ap.add_argument(
        "--block-order",
        default="",
        help="force ONNX observation block order, e.g. maob|cmf|pro; "
        "required when the capture belongs to a different checkpoint",
    )
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    C = np.load(os.path.join(args.capture, "constants.npz"), allow_pickle=True)
    names_il = [str(n) for n in C["joint_names_il"]]

    al = S.Align()
    clip_path = args.motion_file or S.find_clip(args.clip)
    if args.motion_file and not os.path.isfile(clip_path):
        raise FileNotFoundError(f"motion file not found: {clip_path}")
    clip = S.Clip(clip_path)
    print(f"reference motion: {os.path.realpath(clip_path)}")

    # reference, pre-mapped to G1 space exactly as ObsBuilder does
    MJ2IL = np.asarray(X2_ISAACLAB_TO_MUJOCO_MAPPING["mujoco_to_isaaclab_dof"])
    ref_il = clip.dof[:, MJ2IL]
    ref_pos_g1 = al.pos(ref_il)
    dvel = np.diff(clip.dof, axis=0, append=clip.dof[-1:]) * S.TARGET_FPS
    if len(dvel) > 1:
        dvel[-1] = dvel[-2]
    ref_vel_g1 = al.lin(dvel[:, MJ2IL])

    # HARD limits straight from the URDF-derived sim model: training zeroed the
    # soft margin (soft == hard) to unpin the waist; deployment clamps position
    # targets back inside the mechanical range so the drives cannot stall
    # against the end stops.
    import mujoco
    m = mujoco.MjModel.from_xml_path(S.X2_XML)
    lim_mj = m.jnt_range[1:].copy()          # skip the free root joint
    mj2il = np.asarray(
        X2_ISAACLAB_TO_MUJOCO_MAPPING["mujoco_to_isaaclab_dof"], dtype=np.int64
    )
    mj_names = [
        mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        for joint_id in range(1, m.njnt)
    ]
    mapped_names = [mj_names[index] for index in mj2il]
    if mapped_names != names_il:
        mismatch = next(
            (
                (index, expected, actual)
                for index, (expected, actual) in enumerate(zip(names_il, mapped_names))
                if expected != actual
            ),
            None,
        )
        raise RuntimeError(
            "MuJoCo-to-IsaacLab limit mapping does not match captured joint order: "
            f"first mismatch={mismatch}, captured={names_il}, mapped={mapped_names}"
        )
    lim_il = lim_mj[mj2il]

    rel_bias_path = os.path.join(args.capture, "rel_bias.npy")
    rel_bias = np.load(rel_bias_path) if os.path.exists(rel_bias_path) else np.zeros(29)

    # Reuse the harness's validated detector whenever a matching capture is
    # available. Fine-tuned exports without a capture may provide the order
    # established by closed-loop permutation testing instead.
    if args.block_order:
        order = tuple(args.block_order.split("|"))
        if sorted(order) != sorted(S.BLOCKS):
            raise ValueError(f"invalid --block-order: {args.block_order}")
        order_err = None
    else:
        order, order_err = S.detect_block_order(onnx_policy(args.onnx), args.capture)

    np.savez(
        args.out,
        M=al.M, MT=al.MT, OFF=al.OFF,
        joint_names_il=np.array(names_il),
        default_joint_pos=C["default_joint_pos"],
        stiffness=C["stiffness"], damping=C["damping"],
        effort_limits=C["effort_limits"],
        act_scale=C["actterm_joint_pos_scale"], act_offset=C["actterm_joint_pos_offset"],
        joint_limits_il=lim_il,
        rel_bias=rel_bias,
        block_order=np.array(order),
        ref_pos_g1=ref_pos_g1, ref_vel_g1=ref_vel_g1, ref_rot=clip.rot,
        clip_name=np.array(args.clip), hist=HIST, future_skip=FUTURE_SKIP,
    )
    if order_err is None:
        print(f"block order: {'|'.join(order)}  (forced)")
    else:
        print(f"block order: {'|'.join(order)}  (action err {order_err:.4f})")
    print(f"clip {args.clip}: {clip.T} frames")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
