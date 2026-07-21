"""Convert retargeted X2 pickles -> gear_sonic motion_lib format.

Mirrors gear_sonic/data_process/convert_soma_csv_to_motion_lib.py exactly
(same output keys / dtypes / conventions), but for the AgiBot X2:

  input  (batch_retarget_x2.py output):
      {robot, fps, root_pos (T,3), root_rot (T,4 wxyz), dof_pos (T,31), source}
  output (motion_lib directory mode, one joblib pkl per motion):
      {root_trans_offset (T,3) f32,
       pose_aa (T, 1+31, 3) f32,   # [0]=root rotvec, [1+j]=axis_j * dof_j
       dof (T,31) f32,
       root_rot (T,4) f32 xyzw,    # scipy convention, as in the official converter
       fps}

Joint axes are parsed from the GMR X2 MJCF (x2_mocap.xml) so they match the
retargeting joint order (X2_MUJOCO_JOINTS) by construction.

Usage (any env with numpy/scipy/joblib, e.g. isaaclab):
  python convert_x2_to_motion_lib.py \
      --src ../retargeted_dataset --out <gear_sonic>/data/motion_lib_x2/robot \
      [--subsets ACCAD CMU ...]
"""

from __future__ import annotations

import argparse
import pathlib
import pickle
import sys
import xml.etree.ElementTree as ET

import joblib
import numpy as np
from scipy.spatial import transform

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from kinematic_alignment import X2_MUJOCO_JOINTS  # noqa: E402

X2_XML = HERE.parent.parent / "GMR" / "assets" / "agibot_x2" / "x2_mocap.xml"
N_DOF = len(X2_MUJOCO_JOINTS)  # 31

# IsaacLab soft limits (mid +/- 0.9*half of the URDF range) for the 4 chain-end
# wrist joints; see convert_one() for why references are clamped to these.
WRIST_SOFT_LIMITS = {
    "left_wrist_pitch_joint": (-0.5022, 0.5022),   # hard [-0.558, 0.558]
    "left_wrist_roll_joint": (-1.45625, 0.60925),  # hard [-1.571, 0.724]
    "right_wrist_pitch_joint": (-0.5022, 0.5022),  # hard [-0.558, 0.558]
    "right_wrist_roll_joint": (-0.60925, 1.45625),  # hard [-0.724, 1.571]
}


def parse_x2_axes(xml_path=X2_XML) -> np.ndarray:
    """(31, 3) hinge axes in X2_MUJOCO_JOINTS order, from the MJCF."""
    root = ET.parse(xml_path).getroot()
    axes = {}
    for j in root.iter("joint"):
        name = j.attrib.get("name")
        if name in X2_MUJOCO_JOINTS:
            axes[name] = np.array(
                [float(v) for v in j.attrib.get("axis", "0 0 1").split()],
                dtype=np.float32)
    missing = [n for n in X2_MUJOCO_JOINTS if n not in axes]
    assert not missing, f"joints missing from {xml_path}: {missing}"
    return np.stack([axes[n] for n in X2_MUJOCO_JOINTS])


def parse_hb_node_table(xml_path=X2_XML):
    """Node table exactly as Humanoid_Batch parses the MJCF: one node per
    <body> in document order (root first). Each node maps to the hinge joint
    it carries (None for joint-less nodes like the GMR toe bodies).

    Returns (node_joint_names, node_axes): lists of length num_bodies-?; row k
    corresponds to pose_aa[:, 1+k]. Verified against the probe: HB body_names
    for x2_mocap.xml has 34 nodes incl. left/right_toe_link with no joint.
    """
    root = ET.parse(xml_path).getroot()
    worldbody = root.find("worldbody")
    nodes = []

    def walk(body):
        joints = body.findall("joint")
        hinge = None
        axis = np.zeros(3, dtype=np.float32)
        for j in joints:
            if j.attrib.get("type", "hinge") in ("hinge", "slide") and "name" in j.attrib:
                hinge = j.attrib["name"]
                axis = np.array([float(v) for v in j.attrib.get("axis", "0 0 1").split()],
                                dtype=np.float32)
                break
        nodes.append((body.attrib.get("name", "?"), hinge, axis))
        for child in body.findall("body"):
            walk(child)

    root_body = worldbody.find("body")
    walk(root_body)
    return nodes  # [(body_name, joint_name|None, axis)] — nodes[0] is the root


NODE_TABLE = None  # filled by main()/self_test via parse_hb_node_table


def convert_one(pkl_path: pathlib.Path, node_table) -> dict:
    with open(pkl_path, "rb") as f:
        d = pickle.load(f)
    root_pos = np.asarray(d["root_pos"], dtype=np.float32)          # (T,3)
    root_rot_wxyz = np.asarray(d["root_rot"], dtype=np.float32)     # (T,4) wxyz
    dof = np.asarray(d["dof_pos"], dtype=np.float32)                # (T,31)
    T = len(root_pos)
    assert dof.shape == (T, N_DOF), f"{pkl_path}: dof {dof.shape}"

    # Clamp the 4 chain-end wrist joints to IsaacLab SOFT limits (mid +/- 0.9*half).
    # GMR saturates them at HARD limits; refs in the soft-hard band fight the
    # joint_limit penalty (w=-10) ~80x harder than the ori-tracking gain, pinning
    # the wrists. These joints are co-located chain ends: clamping provably moves
    # NO tracked body position (only hand orientation, <= 6.6 deg).
    for _jn, (_lo, _hi) in WRIST_SOFT_LIMITS.items():
        _c = X2_MUJOCO_JOINTS.index(_jn)
        dof[:, _c] = np.clip(dof[:, _c], _lo, _hi)

    # scipy wants xyzw
    root_rot_xyzw = root_rot_wxyz[:, [1, 2, 3, 0]]
    # pose_aa: ONE ROW PER HB NODE (x2_mocap.xml has 34 bodies incl. joint-less
    # GMR toe bodies -> 1 root row + 33 node rows; zero rows for fixed nodes).
    # Row k+1 corresponds to node_table[k+1]'s hinge joint.
    n_nodes = len(node_table)                       # 34 for x2_mocap.xml
    j_index = {n: X2_MUJOCO_JOINTS.index(n)
               for _, n, _ in node_table if n is not None}
    pose_aa = np.zeros((T, n_nodes, 3), dtype=np.float32)
    pose_aa[:, 0, :] = transform.Rotation.from_quat(root_rot_xyzw).as_rotvec()
    for k, (_body, joint, axis) in enumerate(node_table):
        if k == 0 or joint is None:
            continue
        pose_aa[:, k, :] = axis[None, :] * dof[:, j_index[joint], None]

    return {
        "root_trans_offset": root_pos,
        "pose_aa": pose_aa,
        "dof": dof,
        "root_rot": root_rot_xyzw.astype(np.float32),
        "fps": float(d.get("fps", 30.0)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(HERE.parent / "retargeted_dataset"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--subsets", nargs="*", default=None,
                    help="limit to these top-level subset dirs")
    args = ap.parse_args()

    src = pathlib.Path(args.src)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    node_table = parse_hb_node_table()
    n_joints = sum(1 for _, j, _ in node_table if j is not None)
    print(f"HB node table: {len(node_table)} nodes, {n_joints} hinge joints "
          f"from {X2_XML.name}")

    pkls = sorted(src.rglob("*.pkl"))
    if args.subsets:
        allow = set(args.subsets)
        pkls = [p for p in pkls if p.relative_to(src).parts[0] in allow]
    n_ok = n_err = 0
    for p in pkls:
        rel = p.relative_to(src)
        # motion key mirrors the source path, flattened (motion_lib dir mode)
        key = str(rel.with_suffix("")).replace("/", "__")
        try:
            entry = convert_one(p, node_table)
            joblib.dump({key: entry}, out / f"{key}.pkl")
            n_ok += 1
        except Exception as e:  # noqa: BLE001
            n_err += 1
            print(f"ERROR {rel}: {type(e).__name__}: {e}")
    print(f"converted {n_ok} motions -> {out}  ({n_err} errors)")


def self_test():
    node_table = parse_hb_node_table()
    joints = [j for _, j, _ in node_table if j is not None]
    assert len(joints) == N_DOF, f"expected {N_DOF} hinge joints, got {len(joints)}"
    assert set(joints) == set(X2_MUJOCO_JOINTS)
    assert node_table[0][1] is None or node_table[0][0] == "pelvis"
    # probe-verified: x2_mocap.xml has 34 nodes incl. two joint-less toe bodies
    assert len(node_table) == 34, f"node count {len(node_table)} (expected 34)"
    toe_rows = [k for k, (b, j, _) in enumerate(node_table)
                if j is None and "toe" in b]
    assert len(toe_rows) == 2, f"toe nodes: {toe_rows}"
    # synthetic round trip: identity root, single-joint bend lands on the
    # correct NODE row (not the naive 1+j row)
    T = 5
    fake = {
        "root_pos": np.zeros((T, 3)),
        "root_rot": np.tile([1, 0, 0, 0], (T, 1)).astype(np.float32),  # wxyz id
        "dof_pos": np.zeros((T, N_DOF), dtype=np.float32),
        "fps": 30.0,
    }
    fake["dof_pos"][:, X2_MUJOCO_JOINTS.index("left_knee_joint")] = 0.7
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".pkl") as f:
        pickle.dump(fake, open(f.name, "wb"))
        entry = convert_one(pathlib.Path(f.name), node_table)
    assert entry["pose_aa"].shape == (T, len(node_table), 3)
    assert np.allclose(entry["pose_aa"][:, 0, :], 0), "identity root -> zero rotvec"
    knee_node = next(k for k, (_b, j, _a) in enumerate(node_table)
                     if j == "left_knee_joint")
    knee_row = entry["pose_aa"][:, knee_node, :]
    assert np.allclose(np.linalg.norm(knee_row, axis=1), 0.7), "knee angle magnitude"
    mask = np.ones(len(node_table), bool); mask[[0, knee_node]] = False
    assert np.allclose(entry["pose_aa"][:, mask, :], 0), "other rows must stay zero"
    assert np.allclose(entry["root_rot"], [0, 0, 0, 1]), "wxyz->xyzw identity"
    print(f"convert_x2_to_motion_lib self_test: OK ({len(node_table)} HB nodes, "
          f"{N_DOF} joints, knee at node {knee_node})")


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        self_test()
    else:
        main()
