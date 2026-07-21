"""Quality validation for a single retargeted X2 clip.

Checks the goal's three visual criteria numerically, plus a numeric diff
against a reference pkl (the old machine's calibrated v9 output) when given:
  - knees not locked straight  -> knee joint range & percentile flexion
  - hands not flipped backward -> wrist/shoulder at-limit fractions
  - stance foot on the ground  -> sole height stats via MuJoCo FK

Run inside the gmr env from the GMR repo root:
  python ../Any2Any/validate_single.py --pkl <new.pkl> [--ref <v9.pkl>]
"""

import argparse
import pathlib
import pickle

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
X2_XML = HERE.parent / "GMR" / "assets" / "agibot_x2" / "x2_mocap.xml"


def load(p):
    with open(p, "rb") as f:
        return pickle.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", required=True)
    ap.add_argument("--ref", default=None, help="reference pkl (old-machine v9) for numeric diff")
    args = ap.parse_args()

    import mujoco

    d = load(args.pkl)
    qpos = np.concatenate([d["root_pos"], d["root_rot"], d["dof_pos"]], axis=1)
    fps = d["fps"]
    dof = d["dof_pos"]
    print(f"clip: {d.get('source','?')}")
    print(f"frames {len(qpos)} @ {fps} fps  | dof dim {dof.shape[1]}")

    model = mujoco.MjModel.from_xml_path(str(X2_XML))
    data = mujoco.MjData(model)
    hinge_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
                   for j in range(model.njnt)
                   if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
    name2col = {n: i for i, n in enumerate(hinge_names)}
    lo, hi = model.jnt_range[1:, 0], model.jnt_range[1:, 1]

    print("\n--- 1) knees (locked-straight check) ---")
    ok = True
    for n in hinge_names:
        if "knee" in n.lower():
            col = name2col[n]
            v = dof[:, col]
            rng = v.max() - v.min()
            print(f"  {n}: min {v.min():+.3f} max {v.max():+.3f} range {rng:.3f} rad "
                  f"(p10 {np.percentile(v,10):+.3f} / p90 {np.percentile(v,90):+.3f})")
            if rng < 0.25:
                ok = False
                print(f"    ⚠️ range < 0.25 rad — knee looks locked")
    print(f"  knees verdict: {'PASS' if ok else 'FAIL'}")
    knee_ok = ok

    print("\n--- 2) wrists/shoulders (hand flip-back check) ---")
    at_limit = ((dof > hi - 0.02) | (dof < lo + 0.02)).mean(axis=0)
    ok = True
    worst = []
    for n in hinge_names:
        l = n.lower()
        if any(k in l for k in ("wrist", "shoulder", "elbow")):
            frac = at_limit[name2col[n]]
            if frac > 0.02:
                worst.append((frac, n))
    worst.sort(reverse=True)
    for frac, n in worst[:8]:
        # hardware shoulder_roll adduction clamp (±0.06) is expected per memory
        expected = "shoulder_roll" in n
        tag = " (expected hw clamp)" if expected else ""
        print(f"  {n}: at-limit {frac*100:.1f}%{tag}")
        if not expected and frac > 0.30:
            ok = False
    if not worst:
        print("  (no arm joint pinned at limits)")
    print(f"  arms verdict: {'PASS' if ok else 'FAIL — wrist/elbow pinned at limit (flip-back symptom)'}")
    arm_ok = ok

    print("\n--- 3) stance foot on ground (MuJoCo FK) ---")
    foot_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f)
                for f in ["left_ankle_roll_link", "right_ankle_roll_link"]]
    soles = []
    for t in range(0, len(qpos), 3):
        data.qpos[:] = qpos[t]
        mujoco.mj_forward(model, data)
        soles.append(min(data.xpos[i][2] for i in foot_ids) - 0.068)
    soles = np.array(soles)
    stance_med = float(np.percentile(soles, 20))
    print(f"  sole height: min {soles.min():+.4f} m | p20(stance) {stance_med:+.4f} m | max {soles.max():+.4f} m")
    foot_ok = (-0.06 < soles.min()) and (stance_med < 0.06)
    print(f"  foot verdict: {'PASS' if foot_ok else 'FAIL'} "
          f"(penetration>{-0.06}, stance float<{0.06})")

    if args.ref:
        print("\n--- 4) numeric diff vs reference (old-machine v9) ---")
        r = load(args.ref)
        if r["dof_pos"].shape != d["dof_pos"].shape:
            print(f"  shape differs: new {d['dof_pos'].shape} vs ref {r['dof_pos'].shape} "
                  f"(different frame count -> configs may differ)")
        else:
            dd = np.abs(d["dof_pos"] - r["dof_pos"])
            dr = np.abs(d["root_pos"] - r["root_pos"])
            print(f"  dof  |Δ| max {dd.max():.2e} rad, mean {dd.mean():.2e}")
            print(f"  root |Δ| max {dr.max():.2e} m")
            if dd.max() < 1e-3:
                print("  ✅ matches v9 baseline (pipeline exactly reproduced)")
            elif dd.max() < 0.05:
                print("  ~ near-identical to v9 (minor solver/lib version noise)")
            else:
                print("  ⚠️ diverges from v9 — inspect video before batch")

    print(f"\n=== OVERALL: {'PASS ✅' if (knee_ok and arm_ok and foot_ok) else 'FAIL ❌'} ===")


if __name__ == "__main__":
    main()
