"""AlignedEnv: kinematic-alignment proxy between the X2 Isaac Lab env and the
frozen G1 SONIC policy (paper Sec 3.3.1, Level 1+2).

Verified layout facts (fingerprint probes v1-v3, 2026-07-08):
  - robot obs joint blocks AND motion-lib command blocks arrive in
    **X2 IsaacLab order** (31); each history/future frame is a contiguous
    31-vector (frame-major).
  - The G1 policy expects **G1 IsaacLab order** (29) everywhere.
  - Mapping chain per frame: X2-IL --IL2MJ--> X2-MJ --Phi--> G1-MJ --MJ2IL--> G1-IL
    (positions carry the calibrated zero-pose offsets; velocities/actions are
    linear; actions map back through the inverse chain, head joints held at 0).

Group schemas (from the built env, probe-verified):
  actor_obs (990) : base_ang_vel 10x3 | joint_pos 10x31 | joint_vel 10x31 |
                    actions 10x31 | gravity_dir 10x3          -> 930
  critic_obs(1745): command_multi_future [pos 10x31 | vel 10x31] |
                    anchor_pos 3 | anchor_ori 6 | body_pos 42 | body_ori 84 |
                    base_lin_vel 30 | base_ang_vel 30 |
                    joint_pos 10x31 | joint_vel 10x31 | actions 10x31 -> 1645
  tokenizer       : command_multi_future_nonflat (10,62)->(10,58) [pos|vel per
                    frame after flatten]; all other terms pass through.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import torch

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE))
from kinematic_alignment import (  # noqa: E402
    G1_ISAACLAB_TO_MUJOCO_DOF, G1_MUJOCO_TO_ISAACLAB_DOF, OFFSET_VEC, PHI,
    PHI_PINV, N_TARGET, T_SOURCE, X2_MUJOCO_JOINTS,
)

sys.path.insert(0, "/home/taie/Desktop/Byte/Sonic_Retarget/GR00T-WholeBodyControl/gear_sonic/envs/manager_env/robots")
try:
    from x2 import X2_ISAACLAB_TO_MUJOCO_MAPPING  # noqa: E402
except ModuleNotFoundError:  # outside Isaac Sim: use the selftest's stubs
    import x2_selftest  # noqa: E402
    x2_selftest._install_isaaclab_stubs()
    from x2 import X2_ISAACLAB_TO_MUJOCO_MAPPING  # noqa: E402

X2_IL2MJ = np.asarray(X2_ISAACLAB_TO_MUJOCO_MAPPING["isaaclab_to_mujoco_dof"])
X2_MJ2IL = np.asarray(X2_ISAACLAB_TO_MUJOCO_MAPPING["mujoco_to_isaaclab_dof"])


class FrameMapper:
    """Vectorized per-frame joint mapping X2-IL(31) <-> G1-IL(29)."""

    def __init__(self, device):
        dev = torch.device(device)
        # X2-IL -> G1-IL composite permutation-with-projection, as a single
        # (29,31) matrix (and its transpose for the way back): rows pick the
        # X2-IL slot feeding each G1-IL slot.
        # Convention (matches g1.py usage, probe-verified):
        #   q_mj = q_il[IL2MJ]  =>  IL2MJ[mj_idx] = IL slot of MuJoCo joint mj_idx
        #   q_il = q_mj[MJ2IL]  =>  MJ2IL[il_idx] = MuJoCo joint at IL slot il_idx
        m = np.zeros((T_SOURCE, N_TARGET), dtype=np.float32)
        for g1_il in range(T_SOURCE):
            g1_mj = int(G1_MUJOCO_TO_ISAACLAB_DOF[g1_il])   # MJ joint at this IL slot
            # PHI row g1_mj = D^-1 S: a single +-1 for most joints, but the hip
            # roll/yaw rows carry TWO entries (the D decoupling rotation). Sum
            # over all nonzero columns, not just the argmax, or D is dropped.
            for x2_mj in range(N_TARGET):
                w = float(PHI[g1_mj, x2_mj])
                if abs(w) > 1e-9:
                    m[g1_il, int(X2_IL2MJ[x2_mj])] = w      # IL slot of X2-MJ joint
        self.m = torch.as_tensor(m, device=dev)                 # (29,31)
        self.mt = self.m.T.contiguous()                         # (31,29)
        # X2-IL offsets for the reverse direction
        off_x2 = np.zeros(N_TARGET, dtype=np.float32)
        for x2_mj, v in enumerate(OFFSET_VEC):
            off_x2[int(X2_IL2MJ[x2_mj])] = v   # IL slot of MJ joint x2_mj (conv A)
        self.off_x2 = torch.as_tensor(off_x2, device=dev)       # (31,)

    def pos(self, q_x2il):                    # positions: subtract offsets
        return (q_x2il - self.off_x2) @ self.mt

    def lin(self, q_x2il):                    # velocities / prev-actions
        return q_x2il @ self.mt

    def act_back(self, a_g1il):               # policy action -> X2-IL, head=0
        return a_g1il @ self.m


def _map_frames(x, mapper_fn, width_in, width_out):
    """Apply a per-frame mapper over a flat (..., F*width_in) block."""
    lead = x.shape[:-1]
    f = x.shape[-1] // width_in
    y = mapper_fn(x.reshape(*lead, f, width_in))
    return y.reshape(*lead, f * width_out)


class AlignedEnv:
    """Wraps the gear_sonic env wrapper; presents G1-dim spaces to the trainer."""

    # joint_pos obs terms use joint_pos_rel (q - q_default) -> kind "pos_rel";
    # motion-lib command blocks are ABSOLUTE dof -> kind "pos".
    POLICY_BLOCKS = [("base_ang_vel", 30, "pass"), ("joint_pos", 310, "pos_rel"),
                     ("joint_vel", 310, "lin"), ("actions", 310, "lin"),
                     ("gravity_dir", 30, "pass")]
    CRITIC_BLOCKS = [("cmd_pos", 310, "pos"), ("cmd_vel", 310, "lin"),
                     ("anchor_pos", 3, "pass"), ("anchor_ori", 6, "pass"),
                     ("body_pos", 42, "pass"), ("body_ori", 84, "pass"),
                     ("base_lin_vel", 30, "pass"), ("base_ang_vel", 30, "pass"),
                     ("joint_pos", 310, "pos_rel"), ("joint_vel", 310, "lin"),
                     ("actions", 310, "lin")]

    # G1 default joint pose (g1.py init_state, IL order via the mapping) —
    # needed to convert relative observations between the two conventions.
    G1_DEFAULT_PATTERNS = [(".*_hip_pitch_joint", -0.312), (".*_knee_joint", 0.669),
                           (".*_ankle_pitch_joint", -0.363), (".*_elbow_joint", 0.6),
                           ("left_shoulder_roll_joint", 0.2), ("left_shoulder_pitch_joint", 0.2),
                           ("right_shoulder_roll_joint", -0.2), ("right_shoulder_pitch_joint", 0.2)]

    def __init__(self, inner, device):
        self._inner = inner
        self.map = FrameMapper(device)
        self._head_zeros = None
        # rel-obs bias: obs_g1_rel = P(obs_x2_rel) + b,
        # b = P(q_def_x2 - offset) - q_def_g1 = pos(q_def_x2_il) - q_def_g1_il
        import re as _re
        from kinematic_alignment import G1_MUJOCO_JOINTS
        q_def_g1_mj = np.zeros(T_SOURCE, dtype=np.float32)
        for pat, val in self.G1_DEFAULT_PATTERNS:
            rx = _re.compile("^" + pat + "$")
            for i, n in enumerate(G1_MUJOCO_JOINTS):
                if rx.match(n):
                    q_def_g1_mj[i] = val
        # q_il = q_mj[MJ2IL] (convention A: MJ2IL[il_idx] = mj_idx at that slot)
        q_def_g1_il = torch.as_tensor(q_def_g1_mj, device=device)[
            torch.as_tensor(G1_MUJOCO_TO_ISAACLAB_DOF, device=device)]
        try:
            q_def_x2_il = inner.env.unwrapped.scene["robot"].data.default_joint_pos[0].to(device)
        except Exception:
            q_def_x2_il = torch.zeros(N_TARGET, device=device)
        self.rel_bias = self.map.pos(q_def_x2_il.unsqueeze(0))[0] - q_def_g1_il
        mb = float(self.rel_bias.abs().max())
        print(f"[aligned_env] joint_pos_rel bias |b|max = {mb:.4f} rad "
              f"({'OK, Phi-consistent defaults' if mb < 0.02 else 'NONZERO — check robot default pose!'})")

    # -- generic pass-through -------------------------------------------------
    def __getattr__(self, name):
        return getattr(self._inner, name)

    # -- obs transforms --------------------------------------------------------
    def _map_concat(self, x, blocks):
        out, i = [], 0
        for _n, w, kind in blocks:
            chunk = x[..., i:i + w]
            i += w
            if kind == "pass":
                out.append(chunk)
            elif kind == "pos":
                out.append(_map_frames(chunk, self.map.pos, N_TARGET, T_SOURCE))
            elif kind == "pos_rel":
                out.append(_map_frames(
                    chunk, lambda q: self.map.lin(q) + self.rel_bias,
                    N_TARGET, T_SOURCE))
            elif kind == "lin":
                out.append(_map_frames(chunk, self.map.lin, N_TARGET, T_SOURCE))
        assert i == x.shape[-1], f"schema covered {i}/{x.shape[-1]}"
        return torch.cat(out, dim=-1)

    # flattened tokenizer layout (TokenizerCfg term order, X2 widths; verified
    # by probe: 3+620+10+1+6+60+240+9+12+720+60+60 = 1801)
    TOKENIZER_FLAT = [("encoder_index", 3, "pass"),
                      ("command_multi_future_nonflat", 620, "cmd"),
                      ("command_z_multi_future_nonflat", 10, "pass"),
                      ("command_z", 1, "pass"),
                      ("motion_anchor_ori_b", 6, "pass"),
                      ("motion_anchor_ori_b_mf_nonflat", 60, "pass"),
                      ("command_multi_future_lower_body", 240, "pass"),
                      ("vr_3point_local_target", 9, "pass"),
                      ("vr_3point_local_orn_target", 12, "pass"),
                      ("smpl_joints_multi_future_local_nonflat", 720, "pass"),
                      ("smpl_root_ori_b_multi_future", 60, "pass"),
                      ("joint_pos_multi_future_wrist_for_smpl", 60, "pass")]

    def _map_cmd_flat(self, flat):
        """[pos FxN | vel FxN] -> [pos FxT | vel FxT]."""
        half = flat.shape[-1] // 2
        pos = _map_frames(flat[..., :half], self.map.pos, N_TARGET, T_SOURCE)
        vel = _map_frames(flat[..., half:], self.map.lin, N_TARGET, T_SOURCE)
        return torch.cat([pos, vel], dim=-1)

    def _map_tokenizer(self, tok):
        out = dict(tok)
        cmf = tok["command_multi_future_nonflat"]          # (E,10,62)
        e, f, _ = cmf.shape
        mapped = self._map_cmd_flat(cmf.reshape(e, -1))
        out["command_multi_future_nonflat"] = mapped.reshape(e, f, 2 * T_SOURCE)
        return out

    def _map_tokenizer_flat(self, x):
        out, i = [], 0
        for _n, w, kind in self.TOKENIZER_FLAT:
            chunk = x[..., i:i + w]
            i += w
            out.append(self._map_cmd_flat(chunk) if kind == "cmd" else chunk)
        assert i == x.shape[-1], f"tokenizer schema covered {i}/{x.shape[-1]}"
        return torch.cat(out, dim=-1)

    def transform_obs(self, obs):
        if obs is None:
            return obs
        new = {}
        for k, v in obs.items():
            if k == "actor_obs":
                new[k] = self._map_concat(v, self.POLICY_BLOCKS)
            elif k == "critic_obs":
                new[k] = self._map_concat(v, self.CRITIC_BLOCKS)
            elif isinstance(v, dict):
                new[k] = self._map_tokenizer(v)
            elif k == "tokenizer_obs" or (v.dim() >= 1 and v.shape[-1] == 1801):
                new[k] = self._map_tokenizer_flat(v)
            else:
                new[k] = v
        return new

    def transform_action(self, a_g1):
        return self.map.act_back(a_g1)

    # -- trainer-facing interface (manager_env_wrapper conventions) ------------
    # trainer calls: reset_all(); step({"actions": a, ...}) -> (obs, rew, dones, infos)
    def reset(self, *args, **kwargs):
        return self.transform_obs(self._inner.reset(*args, **kwargs))

    def reset_all(self, *args, **kwargs):
        return self.transform_obs(self._inner.reset_all(*args, **kwargs))

    def step(self, actions, *args, **kwargs):
        # actions may be a plain dict or a TensorDict; NEVER mutate the
        # trainer's copy (PPO stores the G1-space action for its update).
        if hasattr(actions, "keys") and "actions" in set(actions.keys()):
            mapped = actions.copy() if hasattr(actions, "copy") else dict(actions)
            mapped["actions"] = self.transform_action(actions["actions"])
        else:
            mapped = self.transform_action(actions)
        res = self._inner.step(mapped, *args, **kwargs)
        if isinstance(res, tuple):
            return (self.transform_obs(res[0]), *res[1:])
        return self.transform_obs(res)


def self_test():
    torch.manual_seed(0)
    import re as _re
    from kinematic_alignment import (G1_MUJOCO_JOINTS, g1_to_x2)

    def g1_il_slot(name):   # IL slot of a G1 MuJoCo joint (conv A: IL2MJ[mj]=il)
        return int(G1_ISAACLAB_TO_MUJOCO_DOF[G1_MUJOCO_JOINTS.index(name)])

    def x2_il_slot(name):
        return int(X2_IL2MJ[X2_MUJOCO_JOINTS.index(name)])

    m = FrameMapper("cpu")
    # round trips
    a = torch.randn(16, T_SOURCE)
    x2 = m.act_back(a)
    assert torch.allclose(m.lin(x2), a, atol=1e-6), "action round trip"
    head_il = [x2_il_slot(n) for n in ("head_yaw_joint", "head_pitch_joint")]
    assert torch.allclose(x2[:, head_il], torch.zeros(16, 2)), "head hold"
    # zero-pose offsets land on the right IL slot
    q0 = m.pos(torch.zeros(1, N_TARGET))
    assert abs(q0[0, g1_il_slot("left_elbow_joint")].item() - 1.579) < 1e-3
    # semantic slot: X2 knee -> G1 knee through IL frames
    fp = torch.zeros(1, N_TARGET)
    fp[0, x2_il_slot("left_knee_joint")] = 0.7
    g1v = m.lin(fp)
    assert g1v[0, g1_il_slot("left_knee_joint")] == 0.7 and g1v.abs().sum() == 0.7

    # ---- REAL-ENV cross-validation (no self-consistent-fake blind spot) ----
    import numpy as _np, pathlib as _pl
    npz = _pl.Path("/tmp/claude-1000/-home-taie-Desktop-Byte/"
                   "7ab7af16-a868-4841-95ca-1a71a2e529ec/scratchpad/probe_defaults.npz")
    if npz.exists():
        d = _np.load(npz, allow_pickle=True)
        env_names = list(d["_joint_names_il"])
        # mapper's implied X2-IL order must equal the env's joint_names
        implied = [X2_MUJOCO_JOINTS[int(np.argmax(
            np.asarray(X2_IL2MJ) == k))] for k in range(N_TARGET)]
        # X2_IL2MJ[mj] = il  =>  joint at IL slot k = name of mj with IL2MJ[mj]==k
        assert implied == env_names, (
            "mapper IL order != env order:\n" +
            "\n".join(f"{k}: {a} vs {b}" for k, (a, b)
                      in enumerate(zip(implied, env_names)) if a != b))
        # bias with the REAL env default pose must be ~0 (DR jitter only)
        class _F: pass
        fake = _F(); fake.env = _F(); fake.env.unwrapped = _F()
        fake.env.unwrapped.scene = {"robot": type("R", (), {"data": type("D", (), {
            "default_joint_pos": torch.tensor(d["_default_joint_pos_il"][:1])})()})()}
        ae = AlignedEnv(fake, "cpu")
        mb = float(ae.rel_bias.abs().max())
        assert mb < 0.03, f"real-env bias {mb:.4f} rad (expected ~DR jitter)"
        print(f"REAL-ENV cross-validation: order match 31/31, bias {mb:.4f} rad ✓")
    else:
        print("(probe_defaults.npz missing — real-env check skipped)")
    print("aligned_env self_test: OK")


if __name__ == "__main__":
    self_test()
