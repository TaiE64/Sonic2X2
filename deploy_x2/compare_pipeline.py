"""Assert the C++ pipeline output matches the numpy reference.

  python compare_pipeline.py --expect expect.f32 --got cpp_out.f32 --n 40

Each step is obs[1570] + pos[31] = 1601 float32. Reports the worst element in
the obs blocks (cmf/maob/pro) and in the targets separately, so a mismatch
points at which stage diverged. Threshold 2e-3 (float32 round-trip noise).
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

STEP = 1601
BLOCKS = [("cmf", 0, 580), ("maob", 580, 640), ("pro", 640, 1570), ("targets", 1570, 1601)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--expect", required=True)
    ap.add_argument("--got", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--tol", type=float, default=2e-3)
    args = ap.parse_args()

    exp = np.fromfile(args.expect, dtype=np.float32).reshape(args.n, STEP)
    got = np.fromfile(args.got, dtype=np.float32).reshape(args.n, STEP)

    print(f"{'block':>9s} {'max|diff|':>12s} {'@step':>6s} {'@idx':>6s}")
    ok = True
    for name, lo, hi in BLOCKS:
        d = np.abs(exp[:, lo:hi] - got[:, lo:hi])
        m = d.max()
        s, i = np.unravel_index(d.argmax(), d.shape)
        flag = "OK" if m < args.tol else "FAIL"
        if m >= args.tol:
            ok = False
        print(f"{name:>9s} {m:12.6f} {s:6d} {i:6d}  {flag}")

    print()
    if ok:
        print(f"PASS — C++ pipeline matches numpy reference within {args.tol}")
        sys.exit(0)
    else:
        print("FAIL — C++ pipeline diverges from reference")
        sys.exit(1)


if __name__ == "__main__":
    main()
