"""SONIC-style interactive demo: hang the robot, lower it, then pick a trajectory and play.

Reuses the validated sim2sim harness (obs contract, Phi alignment, ONNX policy,
position-servo physics) from mujoco_sim2sim_x2.py — only the state machine is new.

States
  HANG   root frozen on an invisible gantry, joints PD-held at the default pose
  LOWER  gantry descends until the feet are on the ground, then lets go
  STAND  policy closed-loop against a "stand still" reference (holds balance)
  PLAY   policy closed-loop against the selected trajectory

Keys (focus the MuJoCo window)
  SPACE  lower & release   (HANG -> STAND)
  N / P  next / prev trajectory in the menu
  ENTER  play selected trajectory from where the robot stands
         (the clip's root is re-anchored to the robot's current xy + heading —
          this is the real deployment flow; works for clips that start upright)
  T      teleport (RSI) onto the clip's first frame, then play
         (use for clips that do NOT start standing, e.g. crawl)
  S      stop -> back to STAND
  R      re-hang
  ESC/close window to quit

Usage
  cd Any2Any/sim2sim
  env -u MUJOCO_GL python demo_viewer.py \
      --onnx <run>/exported/model_step_012000_g1.onnx --capture_dir v17/capture
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np

import mujoco
import mujoco.viewer

from mujoco_sim2sim_x2 import (
    CTRL_DT, PHYS_DT, IL2MJ, MJ2IL,
    Align, Clip, ObsBuilder, Policy,
    build_model, detect_block_order, find_clip, quat_rotate_inverse,
)

# trajectories offered in the menu (name shown, clip id resolved via find_clip)
DEFAULT_MENU = [
    ("salute (bow + hand to head)", "salute_head"),
    ("bow", "EyesJapanDataset__Eyes_Japan_Dataset__hamada__greeting-07-bow-hamada_stageii"),
    ("walk to bow", "ACCAD__Male2MartialArtsStances_c3d__D7_-_walk_to_bow_stageii"),
    ("walk", "walk3_subject1"),
    ("run", "run1_subject2"),
    ("sprint", "sprint1_subject4"),
    ("dance", "dance2_subject5"),
    ("run -> jump -> walk", "ACCAD__Female1Running_c3d__C20_-__run_to_jump_to_walk_stageii"),
    ("crawl  (use T: does not start standing)", "KIT__3__crawl03_stageii"),
]

HANG_HEIGHT = 1.15      # m, root height while suspended
STAND_HEIGHT = 0.74     # m, X2 default standing root height
LOWER_SPEED = 0.25      # m/s of gantry descent

HANG, LOWER, STAND, PLAY = "HANG", "LOWER", "STAND", "PLAY"


def yaw_of(q):
    """Heading (rotation about z) of a wxyz quaternion."""
    w, x, y, z = q
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


class StandClip:
    """A 'stand still' reference: the default pose held at a fixed root.

    Same duck-type as Clip for what ObsBuilder consumes (dof / rot / T); every
    frame is identical, so ObsBuilder's future-frame clamping keeps it valid
    forever at t=0.
    """

    def __init__(self, dof_mj, pos, quat, T=20):
        self.name = "stand"
        self.T = T
        self.dof = np.tile(np.asarray(dof_mj, dtype=np.float64), (T, 1))
        self.trans = np.tile(np.asarray(pos, dtype=np.float64), (T, 1))
        self.rot = np.tile(np.asarray(quat, dtype=np.float64), (T, 1))


def anchor_clip(cl, t0, pos, quat):
    """Re-anchor a clip so frame t0's root xy+heading matches the robot's.

    Rotates the whole trajectory about z by the heading difference and translates
    it onto the robot's current xy. Absolute height is preserved (floor is z=0 in
    both the clip and the sim), so the motion's ground clearance stays correct.
    """
    dpsi = yaw_of(quat) - yaw_of(cl.rot[t0])
    c, s = np.cos(dpsi), np.sin(dpsi)
    Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    qz = np.array([np.cos(dpsi / 2), 0.0, 0.0, np.sin(dpsi / 2)])

    out = Clip.__new__(Clip)                 # shallow copy with rewritten root
    out.name = cl.name
    out.T = cl.T
    out.dof = cl.dof                         # joints are root-invariant
    delta = cl.trans - cl.trans[t0]
    out.trans = delta @ Rz.T + np.array([pos[0], pos[1], cl.trans[t0][2]])
    out.rot = np.stack([quat_mul(qz, q) for q in cl.rot])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--capture_dir", default="v17/capture")
    ap.add_argument("--cam_dist", type=float, default=3.0)
    args = ap.parse_args()

    al = Align()
    C = np.load(os.path.join(args.capture_dir, "constants.npz"), allow_pickle=True)
    def_il = C["default_joint_pos"].astype(np.float64)
    scale_il = C["actterm_joint_pos_scale"].astype(np.float64)
    off_il = C["actterm_joint_pos_offset"].astype(np.float64)
    rb = os.path.join(args.capture_dir, "rel_bias.npy")
    rel_bias = np.load(rb) if os.path.exists(rb) else None
    def_mj = def_il[IL2MJ]

    model, _, (lo, hi) = build_model(C)
    data = mujoco.MjData(model)
    pol = Policy(args.onnx)
    order, oerr = detect_block_order(pol, args.capture_dir)
    print(f"onnx block order: [{' | '.join(order)}] (err {oerr:.4f})")

    menu = [(n, c) for n, c in DEFAULT_MENU if _exists(c)]
    if not menu:
        raise SystemExit("no menu clips found")

    st = {"phase": HANG, "sel": 0, "hang_z": HANG_HEIGHT, "t": 0,
          "prev_a": np.zeros(29), "ob": None, "clip": None, "follow": True}

    def print_menu():
        print("\n" + "=" * 62)
        print(f"  state: {st['phase']}")
        for i, (name, _) in enumerate(menu):
            print(f"  {'>' if i == st['sel'] else ' '} [{i}] {name}")
        print("  SPACE=lower&release  N/P=select  ENTER=play  T=teleport+play  "
              "S=stand  R=re-hang  F=cam follow")
        print("=" * 62, flush=True)

    def hang():
        data.qpos[:3] = [0, 0, st["hang_z"]]
        data.qpos[3:7] = [1, 0, 0, 0]
        data.qpos[7:] = def_mj
        data.qvel[:] = 0
        mujoco.mj_forward(model, data)

    def make_ob(clip):
        st["clip"] = clip
        st["ob"] = ObsBuilder(al, clip, def_il, rel_bias=rel_bias, order=order)
        st["t"] = 0

    def to_stand():
        st["phase"] = STAND
        make_ob(StandClip(def_mj, data.qpos[:3].copy(), data.qpos[3:7].copy()))
        print(f"[STAND] balancing in place", flush=True)

    def to_play(teleport):
        name, cid = menu[st["sel"]]
        cl = Clip(find_clip(cid))
        if teleport:                                   # RSI onto the clip's start
            data.qpos[:3] = cl.trans[0]
            data.qpos[3:7] = cl.rot[0]
            data.qpos[7:] = cl.dof[0]
            lin_w, ang_w = cl.root_vel(0)
            data.qvel[:3] = lin_w
            data.qvel[3:6] = quat_rotate_inverse(cl.rot[0], ang_w)
            data.qvel[6:] = cl.dof_vel(0)
            mujoco.mj_forward(model, data)
            play_clip = cl
        else:                                          # start from where it stands
            play_clip = anchor_clip(cl, 0, data.qpos[:3].copy(), data.qpos[3:7].copy())
        st["phase"] = PLAY
        make_ob(play_clip)
        print(f"[PLAY] {name}  ({cl.T / 50:.1f}s, {'teleport' if teleport else 'from stand'})",
              flush=True)

    def key_cb(k):
        if k == 32:                                    # SPACE
            if st["phase"] == HANG:
                st["phase"] = LOWER
                print("[LOWER] gantry descending...", flush=True)
        elif k in (78, 110):                           # N
            st["sel"] = (st["sel"] + 1) % len(menu)
            print_menu()
        elif k in (80, 112):                           # P
            st["sel"] = (st["sel"] - 1) % len(menu)
            print_menu()
        elif k in (257, 335):                          # ENTER
            if st["phase"] in (STAND, PLAY):
                to_play(teleport=False)
            else:
                print("  (lower it first: SPACE)", flush=True)
        elif k in (84, 116):                           # T
            to_play(teleport=True)
        elif k in (83, 115):                           # S
            to_stand()
        elif k in (82, 114):                           # R
            st["phase"] = HANG
            st["hang_z"] = HANG_HEIGHT
            st["prev_a"] = np.zeros(29)
            hang()
            print("[HANG] suspended", flush=True)
        elif k in (70, 102):                           # F: camera follow on/off
            st["follow"] = not st["follow"]
            print(f"[camera] follow {'ON' if st['follow'] else 'OFF (free pan)'}",
                  flush=True)

    hang()
    print_menu()
    print("[HANG] suspended — press SPACE to lower & release", flush=True)

    with mujoco.viewer.launch_passive(model, data, key_callback=key_cb) as viewer:
        viewer.cam.distance = args.cam_dist
        viewer.cam.elevation, viewer.cam.azimuth = -12, 140
        viewer.cam.lookat[:] = data.qpos[:3]
        nsub = int(round(CTRL_DT / PHYS_DT))
        prev_root = data.qpos[:3].copy()

        while viewer.is_running():
            wall = time.time()
            ph = st["phase"]

            if ph in (HANG, LOWER):
                if ph == LOWER:
                    st["hang_z"] -= LOWER_SPEED * CTRL_DT
                    if st["hang_z"] <= STAND_HEIGHT:   # feet down -> let go
                        st["hang_z"] = STAND_HEIGHT
                        to_stand()
                data.ctrl[:] = np.clip(def_mj, lo, hi)  # PD-hold the default pose
                for _ in range(nsub):
                    mujoco.mj_step(model, data)
                if st["phase"] in (HANG, LOWER):        # keep the root on the gantry
                    data.qpos[:3] = [0, 0, st["hang_z"]]
                    data.qpos[3:7] = [1, 0, 0, 0]
                    data.qvel[:6] = 0
                    mujoco.mj_forward(model, data)
            else:                                       # STAND / PLAY: policy closed loop
                cl = st["clip"]
                f = min(st["t"], cl.T - 1) if ph == PLAY else 0
                obs = st["ob"].build(f, data.qpos[7:][MJ2IL], data.qvel[6:][MJ2IL],
                                     data.qpos[3:7].copy(), data.qvel[3:6].copy(),
                                     st["prev_a"])
                a = pol(obs)
                st["prev_a"] = a
                q_des = off_il + al.act_back(a) * scale_il
                data.ctrl[:] = np.clip(q_des[IL2MJ], lo, hi)
                for _ in range(nsub):
                    mujoco.mj_step(model, data)
                if ph == PLAY:
                    st["t"] += 1
                    if st["t"] >= cl.T - 2:             # trajectory finished
                        print("[PLAY] done -> STAND", flush=True)
                        to_stand()

            # follow by DELTA, never by assignment: writing lookat outright each
            # frame would stomp the user's right-drag pan (and their zoom target)
            if st["follow"]:
                viewer.cam.lookat[:] += data.qpos[:3] - prev_root
            prev_root[:] = data.qpos[:3]

            viewer.sync()
            lag = CTRL_DT - (time.time() - wall)
            if lag > 0:
                time.sleep(lag)                         # real-time 50 Hz


def _exists(cid):
    try:
        find_clip(cid)
        return True
    except FileNotFoundError:
        return False


if __name__ == "__main__":
    main()
