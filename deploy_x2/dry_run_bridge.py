"""READ-ONLY dry-run bridge: X2 telemetry -> obs -> policy -> printed targets.

SAFETY: this node NEVER publishes. It creates no publisher at all, so it cannot
move the robot even if it wanted to. Its only job is to prove the deployment
chain is numerically correct against live hardware telemetry:

    /aima/hal/joint/{leg,waist,arm,head}/state  (31 joints, by name)
    /aima/hal/imu/torso/state                   (orientation + angular velocity)
        -> X2-IsaacLab vector -> Phi -> G1 obs (1570) -> ONNX -> 29 actions
        -> Phi+ -> 31 X2 position targets -> limit clamp -> PRINT ONLY

Everything the loop needs comes from deploy_pkg.npz (built by pack_deploy.py on
the workstation): Phi matrices, defaults, PD gains, hard limits, block order and
the reference clip. Robot side needs only numpy + onnxruntime + rclpy.

Run on the robot:
  python3 dry_run_bridge.py --pkg deploy_pkg.npz --onnx x2_policy.onnx [--steps 250]
"""

from __future__ import annotations

import argparse

import numpy as np
import onnxruntime as ort
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from aimdk_msgs.msg import JointStateArray
from sensor_msgs.msg import Imu

GROUPS = ("leg", "waist", "arm", "head")


def quat_to_mat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def rel_ori_6d(q_robot, q_ref):
    r = quat_to_mat(q_robot).T @ quat_to_mat(q_ref)
    return r[:, :2].reshape(-1)


class DryRun(Node):
    def __init__(self, pkg_path, onnx_path, steps, static_ref=False, dump=False,
                 block_order=None):
        super().__init__("x2_policy_dry_run")
        P = np.load(pkg_path, allow_pickle=True)
        self.P = P
        self.M, self.MT, self.OFF = P["M"], P["MT"], P["OFF"]
        self.names_il = [str(n) for n in P["joint_names_il"]]
        self.default_il = P["default_joint_pos"].astype(np.float64)
        self.lim_il = P["joint_limits_il"].astype(np.float64)
        self.act_scale = P["act_scale"].astype(np.float64)
        self.act_offset = P["act_offset"].astype(np.float64)
        self.rel_bias = P["rel_bias"].astype(np.float64)
        self.order = (tuple(block_order.split("|")) if block_order
                      else tuple(str(x) for x in P["block_order"]))
        if sorted(self.order) != ["cmf", "maob", "pro"]:
            raise ValueError(f"invalid block order: {self.order}")
        self.ref_pos_g1, self.ref_vel_g1 = P["ref_pos_g1"], P["ref_vel_g1"]
        self.ref_rot = P["ref_rot"]
        self.hist, self.skip = int(P["hist"]), int(P["future_skip"])
        self.T = len(self.ref_pos_g1)

        self.sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self.in_name = self.sess.get_inputs()[0].name

        # live telemetry, indexed by joint NAME (never by array order: the HAL
        # groups are separate topics and their concatenation is not the IL order)
        self.q = {}
        self.dq = {}
        self.root_quat = None
        self.ang_vel = None
        self.h = {k: [] for k in ("ang", "jp", "jv", "act", "grav")}
        self.prev_action = np.zeros(29)
        self.t = 0
        self.steps = steps
        self.n_infer = 0
        self.static_ref = static_ref
        self.static_snap = None
        self.dump = dump

        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST, depth=1)
        for g in GROUPS:
            self.create_subscription(
                JointStateArray, f"/aima/hal/joint/{g}/state",
                lambda m, _g=g: self._on_joints(m), qos)
        self.create_subscription(Imu, "/aima/hal/imu/torso/state", self._on_imu, qos)
        self.create_timer(0.02, self._tick)          # 50 Hz
        self.get_logger().info(
            f"DRY RUN (no publisher exists). blocks={'|'.join(self.order)} "
            f"clip_frames={self.T}")

    def _on_joints(self, msg):
        for j in msg.joints:
            self.q[j.name] = j.position
            self.dq[j.name] = j.velocity

    def _on_imu(self, msg):
        o = msg.orientation
        self.root_quat = np.array([o.w, o.x, o.y, o.z])   # wxyz, as the policy expects
        a = msg.angular_velocity
        self.ang_vel = np.array([a.x, a.y, a.z])

    def _push(self, key, v):
        buf = self.h[key]
        buf.append(np.asarray(v, dtype=np.float64))
        if len(buf) == 1:
            buf.extend([buf[0]] * (self.hist - 1))
        if len(buf) > self.hist:
            del buf[0]

    def _tick(self):
        missing = [n for n in self.names_il if n not in self.q]
        if missing or self.root_quat is None:
            if self.t == 0:
                self.get_logger().warn(
                    f"waiting: {len(missing)} joints / imu={self.root_quat is not None}")
            return

        q_il = np.array([self.q[n] for n in self.names_il])
        v_il = np.array([self.dq[n] for n in self.names_il])

        # Static-self-reference check: pin the reference to the pose the robot is
        # actually holding. Tracking error is then ~0 by construction, so a
        # correct obs chain must make the policy go quiet (|a| ~ the 2.09 seen at
        # IsaacSim reset). A loud policy here means the chain is wrong, not that
        # the robot is behind -- which the clip-driven open-loop run cannot tell
        # apart, since there the reference runs away from a robot that never moves.
        if self.static_ref:
            if self.static_snap is None:
                pos_g1 = (q_il - self.OFF) @ self.MT
                self.ref_pos_g1 = np.repeat(pos_g1[None], self.hist * self.skip + 2, 0)
                self.ref_vel_g1 = np.zeros_like(self.ref_pos_g1)
                self.ref_rot = np.repeat(self.root_quat[None], len(self.ref_pos_g1), 0)
                self.T = len(self.ref_pos_g1)
                self.static_snap = True
                self.get_logger().info("static self-reference pinned to current pose")
            self.t = 0

        jp_rel = (q_il - self.default_il) @ self.MT + self.rel_bias
        jv = v_il @ self.MT
        grav = quat_to_mat(self.root_quat).T @ np.array([0.0, 0.0, -1.0])
        for k, v in (("ang", self.ang_vel), ("jp", jp_rel), ("jv", jv),
                     ("act", self.prev_action), ("grav", grav)):
            self._push(k, v)
        pro = np.concatenate([np.concatenate(self.h[k])
                              for k in ("ang", "jp", "jv", "act", "grav")])

        idx = np.minimum(self.t + self.skip * np.arange(self.hist), self.T - 1)
        cmf = np.concatenate([self.ref_pos_g1[idx].ravel(), self.ref_vel_g1[idx].ravel()])
        maob = np.concatenate([
            rel_ori_6d(self.root_quat,
                       self.ref_rot[min(self.t + self.skip * max(0, i - 1), self.T - 1)])
            for i in range(self.hist)])

        blk = {"cmf": cmf, "maob": maob, "pro": pro}
        obs = np.concatenate([blk[k] for k in self.order])
        action = self.sess.run(None, {self.in_name: obs.astype(np.float32)[None]})[0].ravel()

        # action -> X2 position targets (exactly the sim2sim-validated form:
        # map to X2 space FIRST, then apply the X2-space action scale; the
        # offset already carries the default pose), then RESTORE the limit
        # protection training zeroed out (soft == hard there; on hardware a
        # target past the end stop means the drive stalls into it)
        tgt_il = self.act_offset + (action @ self.M) * self.act_scale
        tgt_clamped = np.clip(tgt_il, self.lim_il[:, 0], self.lim_il[:, 1])
        n_clamped = int((np.abs(tgt_clamped - tgt_il) > 1e-9).sum())

        self.prev_action = action
        if not self.static_ref:
            self.t = (self.t + 1) % self.T
        self.n_infer += 1

        if self.n_infer == 5 and self.dump:
            np.savez("/tmp/robot_obs_dump.npz", obs=obs, action=action,
                     q_il=q_il, v_il=v_il, root_quat=self.root_quat,
                     ang_vel=self.ang_vel, grav=grav, jp_rel=jp_rel, jv=jv,
                     cmf=cmf, maob=maob, pro=pro)
            self.get_logger().info("obs dumped to /tmp/robot_obs_dump.npz")
        if self.n_infer % 25 == 0:
            err = float(np.abs(tgt_clamped - q_il).max())
            self.get_logger().info(
                f"[{self.n_infer}] obs={obs.shape[0]} |a|max={np.abs(action).max():.3f} "
                f"tgt-q max={err:.3f}rad clamped={n_clamped}/31 "
                f"grav_z={grav[2]:+.2f}")
        if self.steps and self.n_infer >= self.steps:
            self.get_logger().info("dry run complete (nothing was published)")
            raise SystemExit(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkg", required=True)
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--dump", action="store_true", help="dump one obs for offline diff")
    ap.add_argument("--static_ref", action="store_true",
                    help="pin the reference to the robot's current pose "
                         "(zero-error check: the policy must go quiet)")
    ap.add_argument("--block_order",
                    help="diagnostic override, e.g. maob|cmf|pro")
    args = ap.parse_args()
    rclpy.init()
    node = DryRun(args.pkg, args.onnx, args.steps, args.static_ref, args.dump,
                  args.block_order)
    try:
        rclpy.spin(node)
    except SystemExit:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
