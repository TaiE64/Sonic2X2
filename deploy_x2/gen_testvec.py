"""Generate cross-validation vectors for the C++ X2PolicyPipeline.

Emits, for N random-but-deterministic steps:
  testvec.f32  : int32 N, then per step  q[31] v[31] quat_wxyz[4] w[3] act[29]
  expect.f32   : per step  obs[1570] pos_targets[31]  (numpy reference)

The numpy reference here IS the authority the C++ must match — it reuses the
exact obs math validated bit-for-bit against IsaacSim in mujoco_sim2sim_x2.py.
Blend-in is off (C++ test sets blend_steps=0) so only the pipeline math is
compared. compare_pipeline.py then asserts max|C++ - numpy| < 2e-3.
"""

from __future__ import annotations

import argparse
import struct

import numpy as np

HIST = 10


def quat_to_mat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


class Ref:
    """Reference pipeline: same math as mujoco_sim2sim_x2.ObsBuilder + act_back."""

    def __init__(self, pkg):
        P = np.load(pkg, allow_pickle=True)
        self.M = P["M"].astype(np.float64)
        self.MT = P["MT"].astype(np.float64)
        self.OFF = P["OFF"].astype(np.float64)
        self.default = P["default_joint_pos"].astype(np.float64)
        self.act_scale = P["act_scale"].astype(np.float64)
        self.act_offset = P["act_offset"].astype(np.float64)
        self.lim = P["joint_limits_il"].astype(np.float64)
        self.rel_bias = P["rel_bias"].astype(np.float64)
        self.order = [str(x) for x in P["block_order"]]
        self.ref_pos = P["ref_pos_g1"].astype(np.float64)
        self.ref_vel = P["ref_vel_g1"].astype(np.float64)
        self.ref_rot = P["ref_rot"].astype(np.float64)
        self.T = len(self.ref_pos)
        self.skip = int(P["future_skip"])
        self.reset()

    def reset(self):
        self.t = 0
        self.prev = np.zeros(29)
        self.h = {k: [] for k in ("ang", "jp", "jv", "act", "grav")}

    def _push(self, k, v):
        b = self.h[k]
        b.append(np.asarray(v, float))
        if len(b) == 1:
            b.extend([b[0]] * (HIST - 1))
        if len(b) > HIST:
            del b[0]

    def step(self, q, v, quat, w):
        jp = (q - self.default) @ self.MT + self.rel_bias
        jv = v @ self.MT
        grav = quat_to_mat(quat).T @ np.array([0., 0., -1.])
        for k, val in (("ang", w), ("jp", jp), ("jv", jv), ("act", self.prev), ("grav", grav)):
            self._push(k, val)
        pro = np.concatenate([np.concatenate(self.h[k])
                              for k in ("ang", "jp", "jv", "act", "grav")])
        idx = np.minimum(self.t + self.skip * np.arange(HIST), self.T - 1)
        cmf = np.concatenate([self.ref_pos[idx].ravel(), self.ref_vel[idx].ravel()])
        maob = np.concatenate([
            (quat_to_mat(quat).T @ quat_to_mat(
                self.ref_rot[min(self.t + self.skip * max(0, i - 1), self.T - 1)]))[:, :2].reshape(-1)
            for i in range(HIST)])
        blk = {"cmf": cmf, "maob": maob, "pro": pro}
        obs = np.concatenate([blk[k] for k in self.order])
        return obs

    def targets(self, action, q):
        tgt = self.act_offset + (action @ self.M) * self.act_scale
        tgt = np.clip(tgt, self.lim[:, 0], self.lim[:, 1])
        self.prev = action.copy()
        self.t = min(self.t + 1, self.T - 1)
        return tgt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkg", required=True)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--testvec", required=True)
    ap.add_argument("--expect", required=True)
    args = ap.parse_args()

    rng = np.random.default_rng(1234)          # deterministic
    ref = Ref(args.pkg)

    tv = open(args.testvec, "wb")
    tv.write(struct.pack("i", args.n))
    ex = open(args.expect, "wb")

    for _ in range(args.n):
        q = (ref.default + rng.normal(0, 0.15, 31)).astype(np.float32)
        v = rng.normal(0, 0.3, 31).astype(np.float32)
        quat = rng.normal(0, 1, 4)
        quat = (quat / np.linalg.norm(quat)).astype(np.float32)
        w = rng.normal(0, 0.4, 3).astype(np.float32)
        act = rng.normal(0, 0.7, 29).astype(np.float32)
        for a in (q, v, quat, w, act):
            tv.write(a.astype(np.float32).tobytes())

        obs = ref.step(q.astype(np.float64), v.astype(np.float64),
                       quat.astype(np.float64), w.astype(np.float64))
        tgt = ref.targets(act.astype(np.float64), q.astype(np.float64))
        ex.write(obs.astype(np.float32).tobytes())
        ex.write(tgt.astype(np.float32).tobytes())

    tv.close()
    ex.close()
    print(f"testvec: {args.n} steps -> {args.testvec}")
    print(f"expect:  obs1570+pos31 per step -> {args.expect}")
    print(f"block order: {'|'.join(ref.order)}")


if __name__ == "__main__":
    main()
