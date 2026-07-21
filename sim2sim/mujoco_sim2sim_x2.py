"""MuJoCo sim2sim deployment harness for Sonic->X2 (Any2Any), v13+.

Closes the loop: MuJoCo physics -> X2 state -> Phi alignment -> G1-space obs
(1570) -> g1.onnx -> 29 G1 actions -> Phi+ -> 31 X2 position targets -> PD.

Obs contract (validated against IsaacSim clean ground truth, see --validate):
  onnx input [1,1570] = the 3 blocks {cmf 580, maob 60, proprio 930} concatenated
    in an order that is HASH-RANDOM PER EXPORT (the exporter builds the key list
    via list(set(...))): v10 and v17 came out [cmf|maob|pro], v13/v15 [maob|cmf|pro].
    `detect_block_order()` recovers it automatically by permutation-testing against
    a captured (obs, action) pair — the true order sits at the FSQ noise floor
    (<0.3) while every wrong one is >2. Never hardcode it.
  - cmf  = [pos_block 10x29 | vel_block 10x29], G1-IsaacLab order.
           slot i = ref frame (t + 5*i) @50fps; pos = absolute ref dof mapped
           via Phi (with zero-pose offsets); vel = forward diff (f+1 - f)*50,
           mapped linearly.
  - maob = 10 frames x 6D relative orientation; slot i = ref frame
           t + 5*max(0, i-1) (slot0 and slot1 both current frame; validated
           exactly 0.0 vs capture, while t+5*i gives ~0.1 errs on slots 1-9).
           R_rel = R(robot_root)^T @ R(ref_root); 6D = the (3,2) submatrix
           mat[..., :2] flattened row-major: [r00,r01,r10,r11,r20,r21]
           (matches matrix_from_quat(...)[..., :2].reshape(E,-1)).
  - proprio (block-major, 10-frame history each, oldest->newest, reset fills
           with current frame): base_ang_vel 10x3 (base frame) | joint_pos_rel
           10x29 (q - q_default, G1-IL) | joint_vel 10x29 | prev_actions 10x29
           (raw policy outputs, step0 zeros) | gravity_dir 10x3.

Physics (mirrors IsaacLab ImplicitActuator as closely as MuJoCo allows):
  implicitfast integrator, dt=0.005, 4 substeps per control step (50 Hz);
  position servos gain=[kp,0,0] bias=[0,-kp,-kd]; per-joint kp/kd/armature/
  effort limits captured live from the IsaacSim env (constants.npz);
  frictionloss zeroed (IsaacLab default has none).

Usage (from Any2Any/sim2sim/; --capture_dir defaults to ./capture):
  # validation gate (obs blocks vs IsaacSim clean capture):
  python mujoco_sim2sim_x2.py --validate --onnx <run>/exported/..._g1.onnx
  # closed-loop physics run + video:
  python mujoco_sim2sim_x2.py --clip walk1_subject1 --steps 500 \
      --onnx <run>/exported/..._g1.onnx --render out/walk1.mp4 --ref_ghost
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import tempfile
import xml.etree.ElementTree as ET

import joblib
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))     # Any2Any/sim2sim
_A2A = os.path.dirname(_HERE)                          # Any2Any
sys.path.insert(0, _A2A)                               # kinematic_alignment.py
sys.path.insert(0, os.path.join(_A2A, "any2any_train"))  # aligned_env.py (+x2 maps)

import aligned_env  # noqa: E402  (installs isaaclab stubs, imports x2/g1 maps)
from x2 import X2_ISAACLAB_TO_MUJOCO_MAPPING  # noqa: E402

X2_XML = "/home/taie/Desktop/Byte/Sonic_Retarget/GMR/assets/agibot_x2/x2_mocap.xml"
_DATA = "/home/taie/Desktop/Byte/Sonic_Retarget/GR00T-WholeBodyControl/data"
MOTION_DIRS = [
    f"{_DATA}/motion_lib_lafan_x2/robot",       # keep [0] = lafan (validate() matches here)
    f"{_DATA}/motion_lib_traindist_test/robot",
    f"{_DATA}/motion_lib_omomo_x2/robot",
    f"{_DATA}/motion_lib_jump_demo/robot",       # V15 jump/leap demo distribution
    f"{_DATA}/motion_lib_crawl_demo/robot",      # crawl / kneel-to-crawl / crawl-to-stand
    f"{_DATA}/motion_lib_x2_v16/robot",          # full 15,269-clip training lib (v17)
    f"{_DATA}/motion_lib_salute/robot",          # curated salute clip
    f"{_DATA}/motion_lib_video/robot",           # motion captured from a phone video
]
TARGET_FPS = 50.0
FUTURE_SKIP = 5          # cmf future-frame stride @50fps (dt_future_ref_frames=0.1)
HIST = 10                # obs history length
CTRL_DT = 0.02           # 50 Hz control
PHYS_DT = 0.005          # 4 substeps

IL2MJ = np.asarray(X2_ISAACLAB_TO_MUJOCO_MAPPING["isaaclab_to_mujoco_dof"])
MJ2IL = np.asarray(X2_ISAACLAB_TO_MUJOCO_MAPPING["mujoco_to_isaaclab_dof"])

# Arm joints (X2 MJ indices 15..28) for the optional stiffness multiplier that
# emulates IsaacLab's rigid implicit tracking on the low-inertia arm chain.
ARM_MJ = np.arange(15, 29)
# Waist joints (yaw/pitch/roll). The waist is the weakest torso joint (kp 28.5 ->
# ~18 Nm at full error) and references routinely pin waist_pitch at its +18deg limit
# (100% of frames in the AMASS standing clip). IsaacLab's implicit actuator holds
# that; MuJoCo's explicit servo loses to gravity once the CoM shifts and the torso
# falls to the OPPOSITE limit (measured: q_des +18deg, actual -17deg -> permanent
# backward lean). waist_kp_mult=4 restores tracking (err 30deg+ -> ~3deg median).
WAIST_MJ = np.arange(12, 15)


# --------------------------------------------------------------------------
# alignment (single source of truth: the training-side FrameMapper)
# --------------------------------------------------------------------------
class Align:
    def __init__(self):
        fm = aligned_env.FrameMapper("cpu")
        self.M = fm.m.cpu().numpy().astype(np.float64)      # (29,31) act: g1->x2
        self.MT = fm.mt.cpu().numpy().astype(np.float64)    # (31,29) fwd: x2->g1
        self.OFF = fm.off_x2.cpu().numpy().astype(np.float64)  # (31,) X2-IL zero offsets

    def pos(self, q_x2_il):   # absolute positions
        return (q_x2_il - self.OFF) @ self.MT

    def lin(self, q_x2_il):   # velocities / relative quantities
        return q_x2_il @ self.MT

    def act_back(self, a_g1_il):
        return a_g1_il @ self.M


# --------------------------------------------------------------------------
# small rotation helpers (quats are wxyz everywhere in this file)
# --------------------------------------------------------------------------
def quat_to_mat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def rel_ori_6d(q_robot, q_ref):
    """R_rel = R(robot)^T R(ref); 6D = mat[..., :2] flattened row-major."""
    r = quat_to_mat(q_robot).T @ quat_to_mat(q_ref)
    return r[:, :2].reshape(-1)  # [r00,r01,r10,r11,r20,r21]


def quat_rotate_inverse(q, v):
    return quat_to_mat(q).T @ np.asarray(v, dtype=np.float64)


# --------------------------------------------------------------------------
# reference clip
# --------------------------------------------------------------------------
class Clip:
    """Loads a motion_lib pkl and resamples to 50 fps.

    pkl schema: {name: {root_trans_offset (T,3), dof (T,31 X2-MJ order),
    root_rot (T,4 **xyzw**), fps 30.0}}
    """

    def __init__(self, path):
        raw = joblib.load(path)
        self.name, m = next(iter(raw.items()))
        fps = float(m["fps"])
        dof = np.asarray(m["dof"], dtype=np.float64)          # (T,31) X2-MJ
        trans = np.asarray(m["root_trans_offset"], dtype=np.float64)
        rot_xyzw = np.asarray(m["root_rot"], dtype=np.float64)
        rot = rot_xyzw[:, [3, 0, 1, 2]]                       # -> wxyz
        # enforce quaternion sign continuity before interpolation
        for t in range(1, len(rot)):
            if np.dot(rot[t], rot[t - 1]) < 0:
                rot[t] = -rot[t]
        T = len(dof)
        src_t = np.arange(T) / fps
        dst_t = np.arange(int(np.floor(src_t[-1] * TARGET_FPS)) + 1) / TARGET_FPS
        self.dof = self._interp(dst_t, src_t, dof)
        self.trans = self._interp(dst_t, src_t, trans)
        r = self._interp(dst_t, src_t, rot)
        self.rot = r / np.linalg.norm(r, axis=1, keepdims=True)
        self.T = len(self.dof)

    @staticmethod
    def _interp(dst_t, src_t, arr):
        return np.stack([np.interp(dst_t, src_t, arr[:, i]) for i in range(arr.shape[1])], axis=1)

    def dof_vel(self, f):
        f = min(f, self.T - 2)
        return (self.dof[f + 1] - self.dof[f]) * TARGET_FPS

    def root_vel(self, f):
        f = min(f, self.T - 2)
        lin_w = (self.trans[f + 1] - self.trans[f]) * TARGET_FPS
        # angular velocity from quat forward diff: w_world = 2*(dq * q^-1)_vec
        q0, q1 = self.rot[f], self.rot[f + 1]
        dq = (q1 - q0) * TARGET_FPS
        w, x, y, z = q0
        # quaternion multiply dq * conj(q0), vector part
        cw, cx, cy, cz = w, -x, -y, -z
        vx = dq[0] * cx + dq[1] * cw + dq[2] * cz - dq[3] * cy
        vy = dq[0] * cy - dq[1] * cz + dq[2] * cw + dq[3] * cx
        vz = dq[0] * cz + dq[1] * cy - dq[2] * cx + dq[3] * cw
        ang_w = 2.0 * np.array([vx, vy, vz])
        return lin_w, ang_w


def find_clip(name):
    for d in MOTION_DIRS:
        p = os.path.join(d, name if name.endswith(".pkl") else name + ".pkl")
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"clip {name} not found in {MOTION_DIRS}")


# --------------------------------------------------------------------------
# observation builder
# --------------------------------------------------------------------------
BLOCKS = ("cmf", "maob", "pro")


def detect_block_order(pol, capture_dir):
    """Recover the ONNX input block order for THIS export.

    The exporter concatenates obs groups via list(set(...)), so the order is
    hash-random per export (v13/v15 came out [maob|cmf|pro], v17 [cmf|maob|pro]).
    Permutation-test the 3 blocks against a captured (obs, action) pair: the true
    order lands at the FSQ noise floor (<0.3), every wrong one is >2.
    """
    import itertools
    gt = np.load(os.path.join(capture_dir, "gt_step0.npz"), allow_pickle=True)
    tok, pro, act = gt["obs.tokenizer"], gt["obs.actor_obs"], gt["action_g1"]
    blk = {"cmf": tok[3:583], "maob": tok[594:654], "pro": pro}
    scored = sorted(
        (float(np.abs(pol(np.concatenate([blk[k] for k in p])) - act).max()), p)
        for p in itertools.permutations(BLOCKS))
    err, order = scored[0]
    if err > 1.0:
        raise RuntimeError(
            f"no ONNX block order reproduces the captured action "
            f"(best {'|'.join(order)} err={err:.3f}); capture and ONNX may be from "
            f"different checkpoints")
    return order, err


class ObsBuilder:
    def __init__(self, align: Align, clip: Clip, default_il, hist_newest_first=False,
                 rel_bias=None, order=BLOCKS):
        self.al = align
        self.clip = clip
        self.def_il = default_il
        self.rel_bias = np.zeros(29) if rel_bias is None else np.asarray(rel_bias)
        self.newest_first = hist_newest_first
        self.order = tuple(order)
        # pre-map reference to G1-IL space (positions + forward-diff velocities)
        ref_il = clip.dof[:, MJ2IL]                 # (T,31) X2-IL
        self.ref_pos_g1 = align.pos(ref_il)         # (T,29) absolute
        dvel = np.diff(clip.dof, axis=0, append=clip.dof[-1:]) * TARGET_FPS
        dvel[-1] = dvel[-2] if len(dvel) > 1 else dvel[-1]
        self.ref_vel_g1 = align.lin(dvel[:, MJ2IL])  # (T,29)
        self.reset()

    def reset(self):
        self.h_ang, self.h_jp, self.h_jv, self.h_act, self.h_grav = [], [], [], [], []

    def _push(self, buf, v):
        buf.append(np.asarray(v, dtype=np.float64))
        if len(buf) == 1:
            buf.extend([buf[0]] * (HIST - 1))
        if len(buf) > HIST:
            del buf[0]

    def _flat(self, buf):
        seq = buf[::-1] if self.newest_first else buf
        return np.concatenate(seq)

    def proprio(self, q_x2_il, v_x2_il, root_quat, ang_vel_body, prev_action_g1):
        jp_rel = self.al.lin(q_x2_il - self.def_il) + self.rel_bias
        jv = self.al.lin(v_x2_il)
        grav = quat_rotate_inverse(root_quat, [0.0, 0.0, -1.0])
        for buf, v in ((self.h_ang, ang_vel_body), (self.h_jp, jp_rel),
                       (self.h_jv, jv), (self.h_act, prev_action_g1),
                       (self.h_grav, grav)):
            self._push(buf, v)
        return np.concatenate([self._flat(self.h_ang), self._flat(self.h_jp),
                               self._flat(self.h_jv), self._flat(self.h_act),
                               self._flat(self.h_grav)])

    def cmf(self, t, ref_rate=1.0):
        # When the adaptive clock slows the current reference, its preview must
        # slow by the same amount.  Otherwise slot 9 would still advertise a pose
        # 0.9 source-seconds ahead even though the clock is deliberately holding.
        idx = np.rint(t + FUTURE_SKIP * ref_rate * np.arange(HIST)).astype(int)
        idx = np.minimum(idx, self.clip.T - 1)
        return np.concatenate([self.ref_pos_g1[idx].ravel(),
                               self.ref_vel_g1[idx].ravel()])

    def maob(self, t, root_quat, ref_rate=1.0):
        # slot schedule t + 5*max(0, i-1): empirically exact vs IsaacSim capture
        # (slots 0 and 1 both use the current frame; cmf uses t + 5*i)
        out = []
        for i in range(HIST):
            f = int(round(t + FUTURE_SKIP * ref_rate * max(0, i - 1)))
            f = min(f, self.clip.T - 1)
            out.append(rel_ori_6d(root_quat, self.clip.rot[f]))
        return np.concatenate(out)

    def build(self, t, q_x2_il, v_x2_il, root_quat, ang_vel_body, prev_action_g1,
              ref_rate=1.0):
        b = {  # proprio() mutates the history buffers -> call exactly once
            "pro": self.proprio(q_x2_il, v_x2_il, root_quat, ang_vel_body, prev_action_g1),
            "cmf": self.cmf(t, ref_rate),
            "maob": self.maob(t, root_quat, ref_rate),
        }
        return np.concatenate([b[k] for k in self.order])  # order detected per export


# --------------------------------------------------------------------------
# physics model construction
# --------------------------------------------------------------------------
def build_model(constants, arm_kp_mult=1.0, waist_kp_mult=1.0, zero_frictionloss=True):
    import mujoco

    kp_il = constants["stiffness"].astype(np.float64)
    kd_il = constants["damping"].astype(np.float64)
    arm_il = constants["armature"].astype(np.float64)
    eff_il = constants["effort_limits"].astype(np.float64)
    kp = kp_il[IL2MJ]
    kd = kd_il[IL2MJ]
    armature = arm_il[IL2MJ]
    effort = eff_il[IL2MJ]
    kp_mult = np.ones(31)
    kp_mult[ARM_MJ] = arm_kp_mult
    kp_mult[WAIST_MJ] = waist_kp_mult

    tree = ET.parse(X2_XML)
    root = tree.getroot()
    # absolute meshdir so the patched xml can live anywhere
    comp = root.find("compiler")
    comp.set("meshdir", os.path.join(os.path.dirname(X2_XML), "meshes"))
    opt = root.find("option")
    if opt is None:
        opt = ET.SubElement(root, "option")
    opt.set("timestep", str(PHYS_DT))
    opt.set("integrator", "implicitfast")
    # offscreen framebuffer for rendering
    vis = root.find("visual") or ET.SubElement(root, "visual")
    glb = vis.find("global") if vis.find("global") is not None else ET.SubElement(vis, "global")
    glb.set("offwidth", "1440")
    glb.set("offheight", "1080")
    # the X2 MJCF ships no ground plane -> inject one, with MuJoCo's stock
    # checkerboard texture/material so it looks like the default scene
    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")
    ET.SubElement(asset, "texture", {
        "name": "groundplane", "type": "2d", "builtin": "checker", "mark": "edge",
        "rgb1": "0.2 0.3 0.4", "rgb2": "0.1 0.2 0.3", "markrgb": "0.8 0.8 0.8",
        "width": "300", "height": "300"})
    ET.SubElement(asset, "material", {
        "name": "groundplane", "texture": "groundplane", "texuniform": "true",
        "texrepeat": "5 5", "reflectance": "0.2"})
    ET.SubElement(asset, "texture", {
        "name": "skybox", "type": "skybox", "builtin": "gradient",
        "rgb1": "0.3 0.5 0.7", "rgb2": "0 0 0", "width": "512", "height": "3072"})
    wb = root.find("worldbody")
    ET.SubElement(wb, "geom", {"name": "ground", "type": "plane",
                               "size": "0 0 0.05", "pos": "0 0 0",
                               "material": "groundplane"})
    ET.SubElement(wb, "light", {"pos": "0 0 3.5", "dir": "0 0 -1",
                                "directional": "true"})
    # joint order from document
    jorder, jrange = [], {}
    for j in root.iter("joint"):
        n = j.get("name")
        if n and n != "floating_base_joint":
            jorder.append(n)
            jrange[n] = j.get("range")
            j.set("armature", f"{armature[len(jorder)-1]:.9f}")
            if zero_frictionloss:
                j.set("frictionloss", "0")
    assert len(jorder) == 31, jorder
    # replace all actuators with position servos in joint (document) order
    for act in root.findall("actuator"):
        root.remove(act)
    act = ET.SubElement(root, "actuator")
    for i, n in enumerate(jorder):
        ET.SubElement(act, "position", {
            "name": f"pos_{n}", "joint": n,
            "kp": f"{kp[i]*kp_mult[i]:.6f}", "kv": f"{kd[i]:.6f}",
            "forcerange": f"{-effort[i]:.3f} {effort[i]:.3f}",
            "ctrlrange": jrange[n],
        })
    # unique per process: parallel renders sharing one path race each other and a
    # reader can hit the file mid-write (XML_ERROR_EMPTY_DOCUMENT)
    fd, xml_path = tempfile.mkstemp(prefix="x2_sim2sim_", suffix=".xml")
    os.close(fd)
    tree.write(xml_path)
    model = mujoco.MjModel.from_xml_path(xml_path)   # meshes are absolute -> fully
    os.unlink(xml_path)                              # compiled here, safe to remove
    # sanity: mujoco joint order must match kinematic_alignment's X2_MUJOCO_JOINTS
    from kinematic_alignment import X2_MUJOCO_JOINTS
    mj_names = [model.joint(i).name for i in range(1, model.njnt)]
    assert mj_names == X2_MUJOCO_JOINTS, f"joint order mismatch:\n{mj_names}"
    lo = model.jnt_range[1:, 0].copy()
    hi = model.jnt_range[1:, 1].copy()
    return model, xml_path, (lo, hi)


# --------------------------------------------------------------------------
# ONNX policy
# --------------------------------------------------------------------------
class Policy:
    def __init__(self, onnx_path):
        import onnxruntime as ort
        self.sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self.in_name = self.sess.get_inputs()[0].name
        self.out_name = self.sess.get_outputs()[0].name

    def __call__(self, obs1570):
        a = self.sess.run([self.out_name],
                          {self.in_name: obs1570[None].astype(np.float32)})[0]
        return a.reshape(-1).astype(np.float64)


# --------------------------------------------------------------------------
# validation gate: obs blocks vs IsaacSim clean capture
# --------------------------------------------------------------------------
def validate(capture_dir, onnx_path):
    al = Align()
    C = np.load(os.path.join(capture_dir, "constants.npz"), allow_pickle=True)
    steps = [np.load(os.path.join(capture_dir, f"gt_step{t}.npz"), allow_pickle=True)
             for t in range(3)]
    def_il = C["default_joint_pos"].astype(np.float64)

    print("=" * 70)
    print("A. ONNX block order + network parity (captured obs -> onnx vs captured action)")
    pol = Policy(onnx_path)
    order, oerr = detect_block_order(pol, capture_dir)
    print(f"  detected block order: [{' | '.join(order)}]   (best-perm err {oerr:.4f})")

    def blk(s):
        return {"cmf": s["obs.tokenizer"][3:583], "maob": s["obs.tokenizer"][594:654],
                "pro": s["obs.actor_obs"]}
    for t, s in enumerate(steps):
        a = pol(np.concatenate([blk(s)[k] for k in order]))
        print(f"  step{t}: max|action err| = {np.abs(a - s['action_g1']).max():.6f}")

    print("=" * 70)
    print("B. proprio rebuild from raw state (blocks: ang|jp|jv|act|grav)")
    gt_slices = {"ang": (0, 30), "jp": (30, 320), "jv": (320, 610),
                 "act": (610, 900), "grav": (900, 930)}
    # constant rel_bias for the joint_pos_rel block, measured from step0
    s0 = steps[0]
    jp_mine0 = al.lin(s0["joint_pos_il"].astype(np.float64) - def_il)
    rel_bias = s0["obs.actor_obs"][30:320].reshape(HIST, 29)[-1] - jp_mine0
    np.save(os.path.join(capture_dir, "rel_bias.npy"), rel_bias)
    print(f"  rel_bias (measured, |max|={np.abs(rel_bias).max():.5f}) "
          f"-> saved rel_bias.npy")
    errs = {}
    clip_dummy = type("D", (), {"dof": np.zeros((2, 31)), "T": 2,
                                "rot": np.array([[1., 0, 0, 0]] * 2)})()
    ob = ObsBuilder(al, clip_dummy, def_il, rel_bias=rel_bias)
    for t, s in enumerate(steps):
        prev_a = steps[t - 1]["action_g1"] if t > 0 else np.zeros(29)
        ang_b = quat_rotate_inverse(s["root_quat_w"], s["root_ang_vel_w"])
        pro = ob.proprio(s["joint_pos_il"].astype(np.float64),
                         s["joint_vel_il"].astype(np.float64),
                         s["root_quat_w"].astype(np.float64), ang_b, prev_a)
        for k, (a0, b0) in gt_slices.items():
            e = np.abs(pro[a0:b0] - s["obs.actor_obs"][a0:b0]).max()
            errs[k] = max(errs.get(k, 0), e)
    print("  [oldest-first+rel_bias] "
          + "  ".join(f"{k}={v:.5f}" for k, v in errs.items()))

    print("=" * 70)
    print("C. cmf/maob rebuild (identify clip+frame by cmf pos slot0 over all frames)")
    cmf_gt0 = steps[0]["obs.tokenizer"][3:583]
    tgt = cmf_gt0[:29]
    best = (1e9, None, -1)
    for d in MOTION_DIRS[:1]:
        for f in sorted(os.listdir(d)):
            cl = Clip(os.path.join(d, f))
            pos_all = al.pos(cl.dof[:, MJ2IL])            # (T,29)
            e = np.abs(pos_all - tgt).max(axis=1)
            i = int(np.argmin(e))
            if e[i] < best[0]:
                best = (e[i], os.path.join(d, f), i)
    err0, path, t_abs = best
    print(f"  matched: {os.path.basename(path)} frame {t_abs} (err {err0:.5f})")
    cl = Clip(path)
    ob = ObsBuilder(al, cl, def_il, rel_bias=rel_bias)
    for t, s in enumerate(steps):
        tok = s["obs.tokenizer"]
        e_cmf = np.abs(ob.cmf(t_abs + t) - tok[3:583]).max()
        e_maob = np.abs(ob.maob(t_abs + t, s["root_quat_w"].astype(np.float64))
                        - tok[594:654]).max()
        print(f"  step{t}: cmf err {e_cmf:.4f}   maob err {e_maob:.6f}")

    print("=" * 70)
    print("D. end-to-end: rebuilt obs -> onnx vs captured action")
    ob = ObsBuilder(al, cl, def_il, rel_bias=rel_bias, order=order)
    for t, s in enumerate(steps):
        prev_a = steps[t - 1]["action_g1"] if t > 0 else np.zeros(29)
        ang_b = quat_rotate_inverse(s["root_quat_w"], s["root_ang_vel_w"])
        obs = ob.build(t_abs + t, s["joint_pos_il"].astype(np.float64),
                       s["joint_vel_il"].astype(np.float64),
                       s["root_quat_w"].astype(np.float64), ang_b, prev_a)
        a = pol(obs)
        print(f"  step{t}: max|action err| = {np.abs(a - s['action_g1']).max():.6f}")


# --------------------------------------------------------------------------
# closed-loop run
# --------------------------------------------------------------------------
def run(args):
    import mujoco

    al = Align()
    C = np.load(os.path.join(args.capture_dir, "constants.npz"), allow_pickle=True)
    def_il = C["default_joint_pos"].astype(np.float64)
    scale_il = C["actterm_joint_pos_scale"].astype(np.float64)
    off_il = C["actterm_joint_pos_offset"].astype(np.float64)

    rb_path = os.path.join(args.capture_dir, "rel_bias.npy")
    rel_bias = np.load(rb_path) if os.path.exists(rb_path) else None

    model, xml_path, (lo, hi) = build_model(C, arm_kp_mult=args.arm_kp_mult,
                                            waist_kp_mult=args.waist_kp_mult)
    data = mujoco.MjData(model)
    pol = Policy(args.onnx)
    if args.block_order:
        # explicit override for exports with no matching capture (e.g. v19:
        # order established by closed-loop permutation test instead)
        order = tuple(args.block_order.split("|"))
        print(f"onnx block order: [{' | '.join(order)}] (forced via --block_order)")
    else:
        order, oerr = detect_block_order(pol, args.capture_dir)   # hash-random per export
        print(f"onnx block order: [{' | '.join(order)}] (err {oerr:.4f})")
    # A direct path keeps one-off/live references out of the curated motion
    # libraries.  This is especially useful for camera mimic tests, where a new
    # clip is produced every few seconds and must not pollute training data.
    cl = Clip(args.motion_path if args.motion_path else find_clip(args.clip))
    ob = ObsBuilder(al, cl, def_il, rel_bias=rel_bias, order=order)
    print(f"clip {cl.name}: {cl.T} frames @50fps ({cl.T/50:.1f}s); model {xml_path}")

    t0 = args.start
    nsub = int(round(CTRL_DT / PHYS_DT))
    steps = args.steps if args.adaptive_clock else min(args.steps, cl.T - t0 - 2)

    def rsi_reset(f):
        """Put the robot on reference frame f (pose + velocities), RSI-style."""
        data.qpos[:3] = cl.trans[f]
        data.qpos[3:7] = cl.rot[f]
        data.qpos[7:] = cl.dof[f]
        lin_w, ang_w = cl.root_vel(f)
        data.qvel[:3] = lin_w
        data.qvel[3:6] = quat_rotate_inverse(cl.rot[f], ang_w)  # free-joint ang vel is body-local
        data.qvel[6:] = cl.dof_vel(f)
        mujoco.mj_forward(model, data)

    def policy_step(f, prev_a, ref_rate=1.0):
        """One 50 Hz control step: state -> obs -> onnx -> Phi+ -> PD -> physics."""
        obs = ob.build(f, data.qpos[7:][MJ2IL], data.qvel[6:][MJ2IL],
                       data.qpos[3:7].copy(), data.qvel[3:6].copy(), prev_a,
                       ref_rate=ref_rate)
        a_g1 = pol(obs)
        q_des_il = off_il + al.act_back(a_g1) * scale_il
        data.ctrl[:] = np.clip(q_des_il[IL2MJ], lo, hi)
        for _ in range(nsub):
            mujoco.mj_step(model, data)
        return a_g1

    rsi_reset(t0)

    # ---- interactive MuJoCo window: closed-loop policy, real time, looped ----
    if args.viewer:
        import time as _time
        import mujoco.viewer
        print(f"opening MuJoCo viewer — looping {cl.name} ({steps / 50:.1f}s). "
              f"Close the window to quit.")
        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer.cam.distance = args.cam_dist
            viewer.cam.elevation, viewer.cam.azimuth = -12, 140
            viewer.cam.lookat[:] = data.qpos[:3]
            prev_root = data.qpos[:3].copy()
            t, prev_a = 0, np.zeros(29)
            while viewer.is_running():
                if t >= steps:                       # replay the clip from the top
                    rsi_reset(t0)
                    ob.reset()
                    prev_a = np.zeros(29)
                    t = 0
                    _time.sleep(0.4)
                wall = _time.time()
                prev_a = policy_step(t0 + t, prev_a)
                # follow by DELTA, not assignment: overwriting lookat every frame
                # would stomp the user's right-drag pan
                viewer.cam.lookat[:] += data.qpos[:3] - prev_root
                prev_root[:] = data.qpos[:3]
                viewer.sync()
                t += 1
                lag = CTRL_DT - (_time.time() - wall)
                if lag > 0:
                    _time.sleep(lag)                 # pace to real-time 50 Hz
        return

    render = args.render
    if render:
        os.environ.setdefault("MUJOCO_GL", "egl")
        import cv2
        w, h = (1440, 540) if args.ref_ghost else (720, 540)
        vw = cv2.VideoWriter(render, cv2.VideoWriter_fourcc(*"mp4v"), 50, (w, h))
        rend = mujoco.Renderer(model, 540, 720)
        cam = mujoco.MjvCamera()
        cam.distance, cam.elevation, cam.azimuth = args.cam_dist, -12, 140
        ref_data = mujoco.MjData(model) if args.ref_ghost else None

    prev_a = np.zeros(29)
    ref_phase = float(t0)
    ref_rate = 1.0
    held_steps = 0
    fall_step = None
    jerr_sum = 0.0
    completed = False
    for t in range(steps):
        f = min(int(round(ref_phase)), cl.T - 2) if args.adaptive_clock else t0 + t
        prev_a = policy_step(f, prev_a, ref_rate=ref_rate)

        # tracking bookkeeping (paper-style root criteria)
        dh = abs(data.qpos[2] - cl.trans[f][2])
        qr = quat_to_mat(data.qpos[3:7]).T @ quat_to_mat(cl.rot[f])
        ori_err = np.arccos(np.clip((np.trace(qr) - 1) / 2, -1, 1))
        jerr = np.abs(data.qpos[7:] - cl.dof[f])
        jerr_sum += jerr.mean()

        if args.adaptive_clock:
            # Two independent safety signals.  Full speed below the soft
            # thresholds, progressively slower above them, and a small crawl
            # rate rather than a permanent deadlock at the hard threshold.
            ori_deg = np.degrees(ori_err)
            joint_rate = np.clip((args.adaptive_joint_hard - jerr.mean()) /
                                 (args.adaptive_joint_hard - args.adaptive_joint_soft),
                                 args.adaptive_min_rate, 1.0)
            ori_rate = np.clip((args.adaptive_ori_hard - ori_deg) /
                               (args.adaptive_ori_hard - args.adaptive_ori_soft),
                               args.adaptive_min_rate, 1.0)
            target_rate = float(min(joint_rate, ori_rate))
            # Slow immediately for safety, recover smoothly to avoid reference
            # acceleration spikes when tracking catches up.
            ref_rate = target_rate if target_rate < ref_rate else min(
                target_rate, ref_rate + args.adaptive_recovery
            )
            held_steps += int(ref_rate < 0.999)
            ref_phase += ref_rate
            if ref_phase >= cl.T - 2:
                completed = True
        if fall_step is None and (dh > 0.25 or ori_err > 0.8):
            fall_step = t
            print(f"  [term] step {t} ({t/50:.2f}s): root_h_err={dh:.3f} "
                  f"ori_err={np.degrees(ori_err):.1f}deg")
        if t % 100 == 0:
            print(f"  t={t/50:5.2f}s  h={data.qpos[2]:.3f} (ref {cl.trans[f][2]:.3f})  "
                  f"ori={np.degrees(ori_err):5.1f}deg  jerr={jerr.mean():.3f}rad"
                  + (f"  ref={f/50:.2f}s rate={ref_rate:.2f}" if args.adaptive_clock else ""))

        if render:
            cam.lookat[:] = data.qpos[:3]
            rend.update_scene(data, cam)
            frame = rend.render()
            if args.ref_ghost:
                ref_data.qpos[:3] = cl.trans[f]
                ref_data.qpos[3:7] = cl.rot[f]
                ref_data.qpos[7:] = cl.dof[f]
                mujoco.mj_forward(model, ref_data)
                cam.lookat[:] = cl.trans[f]
                rend.update_scene(ref_data, cam)
                ref_frame = rend.render()
                frame = np.concatenate([frame, ref_frame], axis=1)
            vw.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        if completed:
            print(f"  adaptive clock completed reference at wall t={t/50:.2f}s")
            break

    if render:
        vw.release()
        print(f"video: {render}")
    executed_steps = t + 1
    surv = (fall_step if fall_step is not None else executed_steps) / 50.0
    print(f"RESULT clip={cl.name} survived={surv:.2f}s / {executed_steps/50:.2f}s "
          f"({'fell' if fall_step is not None else 'no root-level fall'}) "
          f"mean_joint_err={jerr_sum/max(1,executed_steps):.3f}rad arm_kp_mult={args.arm_kp_mult}"
          + (f" adaptive_held={held_steps}/{executed_steps} final_ref={ref_phase/50:.2f}s"
             if args.adaptive_clock else ""))


def _default_capture():
    """Newest per-version capture (sim2sim/<ver>/capture), else legacy ./capture."""
    cands = [c for c in glob.glob(os.path.join(_HERE, "*", "capture"))
             if os.path.exists(os.path.join(c, "constants.npz"))]
    return max(cands, key=os.path.getmtime) if cands else os.path.join(_HERE, "capture")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--capture_dir",
                    default=os.environ.get("ANY2ANY_CAPTURE_DIR", _default_capture()))
    ap.add_argument("--onnx", default="")
    ap.add_argument("--block_order", default="", help="force obs block order e.g. maob|cmf|pro (skip capture detection)")
    ap.add_argument("--clip", default="walk1_subject1")
    ap.add_argument("--motion_path", default="",
                    help="direct motion_lib .pkl path; bypasses --clip lookup")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--render", default="")
    ap.add_argument("--viewer", action="store_true",
                    help="open an interactive MuJoCo window (real-time, looped) "
                         "instead of writing an mp4")
    ap.add_argument("--ref_ghost", action="store_true",
                    help="side-by-side kinematic reference in the video")
    ap.add_argument("--arm_kp_mult", type=float, default=1.0)
    ap.add_argument("--waist_kp_mult", type=float, default=1.0,
                    help="waist stiffness multiplier; ~4 emulates IsaacLab's implicit "
                         "actuator so the torso can hold the routinely limit-pinned "
                         "waist_pitch reference against gravity (see WAIST_MJ note)")
    ap.add_argument("--adaptive_clock", action="store_true",
                    help="slow reference playback when pose tracking error grows")
    ap.add_argument("--adaptive_joint_soft", type=float, default=0.20)
    ap.add_argument("--adaptive_joint_hard", type=float, default=0.35)
    ap.add_argument("--adaptive_ori_soft", type=float, default=15.0,
                    help="root orientation soft threshold in degrees")
    ap.add_argument("--adaptive_ori_hard", type=float, default=35.0,
                    help="root orientation hard threshold in degrees")
    ap.add_argument("--adaptive_min_rate", type=float, default=0.10)
    ap.add_argument("--adaptive_recovery", type=float, default=0.04,
                    help="maximum playback-rate recovery per 50 Hz step")
    ap.add_argument("--cam_dist", type=float, default=3.0)
    args = ap.parse_args()
    if args.validate:
        validate(args.capture_dir, args.onnx)
    else:
        run(args)


if __name__ == "__main__":
    main()
