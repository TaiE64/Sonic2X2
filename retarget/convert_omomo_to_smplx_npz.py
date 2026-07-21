"""OMOMO (joblib .p) -> per-sequence SMPL-X .npz for the Any2Any eval set.

Reads train/test_diffusion_manip_seq_joints24.p and writes one .npz per
sequence with the exact keys GMR's load_smplx_file expects (same as AMASS
stageii): pose_body (N,63), root_orient (N,3), trans (N,3), betas (16,),
gender, mocap_frame_rate (30 for OMOMO).

Usage (isaaclab env):
  python convert_omomo_to_smplx_npz.py --src <dir with the two .p> --out <dir>
"""

import argparse
import pathlib

import joblib
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    src = pathlib.Path(args.src)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    n = 0
    for part in ("train", "test"):
        p = src / f"{part}_diffusion_manip_seq_joints24.p"
        if not p.exists():
            print(f"missing {p}, skipping")
            continue
        data = joblib.load(p)
        for key in data:
            s = data[key]
            seq = s["seq_name"]
            betas = np.asarray(s["betas"], dtype=np.float64).reshape(-1)
            if betas.shape[0] < 16:
                betas = np.pad(betas, (0, 16 - betas.shape[0]))
            np.savez(
                out / f"{part}__{seq}.npz",
                pose_body=np.asarray(s["pose_body"], dtype=np.float64),
                root_orient=np.asarray(s["root_orient"], dtype=np.float64),
                trans=np.asarray(s["trans"], dtype=np.float64),
                betas=betas[:16],
                gender=str(s.get("gender", "neutral")),
                mocap_frame_rate=np.array(30.0),
            )
            n += 1
    print(f"wrote {n} sequences -> {out}")


if __name__ == "__main__":
    main()
