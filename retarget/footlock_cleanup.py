"""Foot-skate cleanup for X2 motion_lib pkls (conservative root-XY servo).

Root cause: GMR root-local scaling makes stance-foot targets drift at
(1-scale)*root_velocity (~16 cm/s on walks). Physically untrackable ->
structural world-frame drift during training/eval.

Fix (minimal intervention): adjust ONLY root_trans_offset XY per frame so
detected stance feet stay anchored. Joints/pose_aa/root_rot/Z untouched.

Conservative stance detection (won't touch intentional slides/pivots):
  foot z < (10th pct + 3cm)  AND  |v_z| < 0.15 m/s  AND  |v_xy| < 0.40 m/s

Causal servo with per-frame correction cap (default 1 cm/frame) keeps the
root velocity profile smooth. Originals are never overwritten -- output goes
to a sibling directory; a per-clip QC line (before/after skate, total root
correction) is appended to <outdir>/footlock_qc.jsonl.

Usage:
  python footlock_cleanup.py --in data/motion_lib_x2/robot --out data/motion_lib_x2_footlock/robot [--only NAME] [--workers 6]
"""
import argparse, glob, json, os, sys

import joblib
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "GR00T-WholeBodyControl"))
from omegaconf import OmegaConf
from gear_sonic.utils.motion_lib.torch_humanoid_batch import Humanoid_Batch

Z_BAND = 0.03          # stance: foot within 3cm of its low percentile
VZ_MAX = 0.15          # stance: vertical speed below (m/s)
VXY_MAX = 0.40         # stance: horizontal speed below (m/s) -- pivots/slides above this are left alone
GAIN = 0.5             # servo gain per frame toward anchor
MAX_STEP = 0.01        # max root correction per frame (m)

_HB = None

def get_hb():
    global _HB
    if _HB is None:
        cfg = OmegaConf.create({
            "asset": {"assetRoot": "gear_sonic/data/assets/robot_description/mjcf/",
                      "assetFileName": "x2.xml"},
            "extend_config": [],
        })
        _HB = Humanoid_Batch(cfg)
    return _HB


def fk_feet(v):
    hb = get_hb()
    names = list(hb.body_names)
    gt = hb.fk_batch(
        torch.tensor(v["pose_aa"][None]).float(),
        torch.tensor(v["root_trans_offset"][None]).float(),
        return_full=True,
    ).global_translation[0].numpy()
    return gt[:, names.index("left_ankle_roll_link")], gt[:, names.index("right_ankle_roll_link")]


def stance_mask(foot, fps):
    z = foot[:, 2]
    zmin = np.percentile(z, 10)
    vel = np.diff(foot, axis=0, append=foot[-1:]) * fps
    return (z < zmin + Z_BAND) & (np.abs(vel[:, 2]) < VZ_MAX) & \
           (np.linalg.norm(vel[:, :2], axis=-1) < VXY_MAX)


def skate_metric(foot, fps):
    z = foot[:, 2]
    zmin = np.percentile(z, 10)
    vel = np.diff(foot, axis=0) * fps
    st = (z[:-1] < zmin + Z_BAND) & (np.abs(vel[:, 2]) < VZ_MAX)
    if st.sum() <= 5:
        return None
    return float(np.linalg.norm(vel[st][:, :2], axis=-1).mean())


def _clean_mask(m, close_gap=2, min_run=4):
    """Close short gaps, then drop runs shorter than min_run."""
    m = m.copy()
    # close gaps
    t = 0
    T = len(m)
    while t < T:
        if not m[t]:
            j = t
            while j < T and not m[j]:
                j += 1
            if t > 0 and j < T and (j - t) <= close_gap:
                m[t:j] = True
            t = j
        else:
            t += 1
    # drop short runs
    t = 0
    while t < T:
        if m[t]:
            j = t
            while j < T and m[j]:
                j += 1
            if (j - t) < min_run:
                m[t:j] = False
            t = j
        else:
            t += 1
    return m


def footlock(v):
    """Exact-solve foot lock: within each stance segment the foot is pinned to
    its (delta-continuous) touchdown point; delta holds through swing; light
    box smoothing removes segment-boundary steps. Returns (T,2) XY delta for
    root_trans_offset."""
    fps = v.get("fps", 30)
    lf, rf = fk_feet(v)
    T = len(lf)
    feet = [lf, rf]
    masks = [_clean_mask(stance_mask(lf, fps)), _clean_mask(stance_mask(rf, fps))]
    delta = np.zeros((T, 2))
    anchors = [None, None]
    d = np.zeros(2)
    for t in range(T):
        reqs = []
        for i in (0, 1):
            if masks[i][t]:
                if anchors[i] is None:
                    anchors[i] = feet[i][t, :2] + d   # touchdown point, delta-continuous
                reqs.append(anchors[i] - feet[i][t, :2])
            else:
                anchors[i] = None
        if reqs:
            d = np.mean(reqs, axis=0)                 # exact (average in double support)
        delta[t] = d
    # light box smoothing (kills steps at segment boundaries, keeps lock)
    k = 3
    pad = np.pad(delta, ((k, k), (0, 0)), mode="edge")
    kernel = np.ones(2 * k + 1) / (2 * k + 1)
    for c in (0, 1):
        delta[:, c] = np.convolve(pad[:, c], kernel, mode="same")[k:-k]
    return delta


def process(fpath, outdir):
    m = joblib.load(fpath)
    key = list(m.keys())[0]
    v = m[key]
    fps = v.get("fps", 30)
    lf0, rf0 = fk_feet(v)
    before = [skate_metric(f, fps) for f in (lf0, rf0)]
    delta = footlock(v)
    v = dict(v)
    rt = np.array(v["root_trans_offset"], dtype=np.float64).copy()
    rt[:, :2] += delta
    v["root_trans_offset"] = rt.astype(np.float32)
    lf1, rf1 = fk_feet(v)
    after = [skate_metric(f, fps) for f in (lf1, rf1)]
    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, os.path.basename(fpath))
    joblib.dump({key: v}, out)
    rec = {
        "clip": os.path.basename(fpath),
        "skate_before_cm_s": [None if s is None else round(s * 100, 1) for s in before],
        "skate_after_cm_s": [None if s is None else round(s * 100, 1) for s in after],
        "total_correction_cm": round(float(np.linalg.norm(delta[-1]) * 100), 1),
        "max_step_cm": round(float(np.max(np.linalg.norm(np.diff(delta, axis=0), axis=-1)) * 100), 2) if len(delta) > 1 else 0.0,
    }
    with open(os.path.join(outdir, "footlock_qc.jsonl"), "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="indir", required=True)
    ap.add_argument("--out", dest="outdir", required=True)
    ap.add_argument("--only", default=None, help="substring filter for single-clip validation")
    ap.add_argument("--skip_done", action="store_true")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.indir, "*.pkl")))
    if args.only:
        files = [f for f in files if args.only in os.path.basename(f)]
    print(f"{len(files)} clips -> {args.outdir}")
    for i, f in enumerate(files):
        if args.skip_done and os.path.exists(os.path.join(args.outdir, os.path.basename(f))):
            continue
        rec = process(f, args.outdir)
        if i < 20 or (i + 1) % 200 == 0:
            print(f"[{i+1}/{len(files)}] {rec['clip'][:50]:50s} "
                  f"skate {rec['skate_before_cm_s']} -> {rec['skate_after_cm_s']} cm/s "
                  f"corr {rec['total_correction_cm']}cm")


if __name__ == "__main__":
    main()
