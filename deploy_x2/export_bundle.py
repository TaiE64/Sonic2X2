"""deploy_pkg.npz -> x2_bundle/ (raw float32 + meta.txt) for the C++ pipeline.

The robot-side controller is C++ with no numpy; this flattens everything the
50 Hz loop needs into raw little-endian float32 files plus a text manifest.
Array semantics (shapes, mapping conventions) are documented in pack_deploy.py;
this file only changes the container format.

Usage:
  python export_bundle.py --pkg /tmp/deploy_pkg.npz --out <dir>/x2_bundle
"""

from __future__ import annotations

import argparse
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkg", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    P = np.load(args.pkg, allow_pickle=True)
    os.makedirs(args.out, exist_ok=True)

    arrays = {
        "M": P["M"],                    # (29,31) action g1->x2
        "MT": P["MT"],                  # (31,29) fwd x2->g1
        "OFF": P["OFF"],                # (31,) zero-pose offsets
        "default": P["default_joint_pos"],
        "stiffness": P["stiffness"],
        "damping": P["damping"],
        "act_scale": P["act_scale"],
        "act_offset": P["act_offset"],
        "lim_lo": P["joint_limits_il"][:, 0],
        "lim_hi": P["joint_limits_il"][:, 1],
        "rel_bias": P["rel_bias"],      # (29,)
        "ref_pos_g1": P["ref_pos_g1"],  # (T,29)
        "ref_vel_g1": P["ref_vel_g1"],  # (T,29)
        "ref_rot": P["ref_rot"],        # (T,4) wxyz
    }
    for name, a in arrays.items():
        a32 = np.ascontiguousarray(np.asarray(a), dtype=np.float32)
        a32.tofile(os.path.join(args.out, f"{name}.f32"))

    names = [str(n) for n in P["joint_names_il"]]
    order = [str(x) for x in P["block_order"]]
    with open(os.path.join(args.out, "meta.txt"), "w") as f:
        f.write(f"T {len(P['ref_pos_g1'])}\n")
        f.write(f"hist {int(P['hist'])}\n")
        f.write(f"future_skip {int(P['future_skip'])}\n")
        f.write(f"block_order {'|'.join(order)}\n")
        f.write(f"clip {str(P['clip_name'])}\n")
        f.write("names " + ",".join(names) + "\n")

    print(f"bundle -> {args.out}")
    print(f"  T={len(P['ref_pos_g1'])} blocks={'|'.join(order)}")
    print(f"  files: {sorted(os.listdir(args.out))}")


if __name__ == "__main__":
    main()
