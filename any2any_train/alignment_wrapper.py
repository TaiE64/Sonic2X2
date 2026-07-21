"""Kinematic-alignment wrapper for the Sonic G1 -> AgiBot X2 transfer.

Implements the paper's two-level alignment (Sec. 3.3.1) around a frozen
source-convention policy:

  Level 2 (joint-level, Eq. 4/5):   Phi = J D^-1 S = S   (X2: D = J = I)
      observations' joint-structured terms:  q_tilde = Phi q_x2
      action out:                            a_x2    = Phi^+ a_tilde

  Level 1 (observation layout): reorder the target robot's observation blocks
  into the source policy's expected layout. The block schema is read from the
  Sonic env config (single source of truth) rather than hard-coded.

Torch-first (works on batched tensors); numpy accepted and returned as numpy.
The matrices come from kinematic_alignment.py (validated by FK replay).
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from kinematic_alignment import (  # noqa: E402
    G1_ISAACLAB_TO_MUJOCO_DOF, G1_MUJOCO_TO_ISAACLAB_DOF,
    OFFSET_VEC, PHI, PHI_PINV, N_TARGET, T_SOURCE,
)


def _t(x, ref: torch.Tensor | None = None):
    if isinstance(x, torch.Tensor):
        return x
    return torch.as_tensor(x, dtype=torch.float32,
                           device=ref.device if ref is not None else "cpu")


class JointAlignment:
    """Level-2 mapping between X2 (31) and G1 (29) joint vectors.

    Sonic's training artifacts store DoF in IsaacLab order; the scattering
    matrix is defined in MuJoCo order. `frame` selects the source-side
    convention so both consumers can use one object:
        frame="mujoco":   q_g1 in G1 MuJoCo order
        frame="isaaclab": q_g1 in G1 IsaacLab order (Sonic policy convention)
    """

    def __init__(self, device: str | torch.device = "cpu", frame: str = "isaaclab"):
        assert frame in ("mujoco", "isaaclab")
        self.frame = frame
        self.phi = torch.as_tensor(PHI, dtype=torch.float32, device=device)
        self.phi_pinv = torch.as_tensor(PHI_PINV, dtype=torch.float32, device=device)
        self.offset = torch.as_tensor(OFFSET_VEC, dtype=torch.float32, device=device)
        self.il2mj = torch.as_tensor(G1_ISAACLAB_TO_MUJOCO_DOF, device=device)
        self.mj2il = torch.as_tensor(G1_MUJOCO_TO_ISAACLAB_DOF, device=device)

    def to(self, device):
        return JointAlignment(device, self.frame)

    # -- positions (affine: offsets apply) ---------------------------------
    def x2_pos_to_src(self, q_x2):
        """(..., 31) X2 MuJoCo-order joint positions -> (..., 29) source frame."""
        q = _t(q_x2, self.phi)
        g1_mj = (q - self.offset) @ self.phi.T
        return g1_mj[..., self.mj2il] if self.frame == "isaaclab" else g1_mj

    def src_pos_to_x2(self, q_g1):
        """(..., 29) source-frame positions -> (..., 31) X2 MuJoCo order."""
        q = _t(q_g1, self.phi)
        g1_mj = q[..., self.il2mj] if self.frame == "isaaclab" else q
        return g1_mj @ self.phi_pinv.T + self.offset

    # -- velocities / deltas (linear: no offsets) ---------------------------
    def x2_vel_to_src(self, dq_x2):
        dq = _t(dq_x2, self.phi)
        g1_mj = dq @ self.phi.T
        return g1_mj[..., self.mj2il] if self.frame == "isaaclab" else g1_mj

    def src_vel_to_x2(self, dq_g1):
        dq = _t(dq_g1, self.phi)
        g1_mj = dq[..., self.il2mj] if self.frame == "isaaclab" else dq
        return g1_mj @ self.phi_pinv.T

    # -- actions (position offsets => linear rule, paper Sec 3.3.1) --------
    def src_action_to_x2(self, a_src, head_hold: float = 0.0):
        """Policy output (29, source frame) -> X2 executed action (31).

        Actions are joint-position *offsets*, so the zero-pose offsets do not
        apply. Unmatched X2 joints (head) get `head_hold` (0 = hold neutral,
        handled by a separate low-level loop per kinematic_alignment.py).
        """
        a = self.src_vel_to_x2(a_src)
        if head_hold != 0.0:
            a[..., -2:] = head_hold
        return a

    def x2_action_to_src(self, a_x2):
        return self.x2_vel_to_src(a_x2)


class ObservationAlignment:
    """Level-1 layout alignment.

    Sonic composes its actor/critic observations from named blocks (base ang
    vel, projected gravity, dof pos, dof vel, prev action, reference stream).
    This class receives the per-block layout (name, width, kind) extracted
    from the Sonic env config at env-build time and rewrites only the
    joint-structured blocks through the Level-2 mapping; scalar blocks pass
    through untouched, in the source policy's expected order.

    kind: "pass"      -> copied as-is
          "dof_pos"   -> width N_TARGET in, T_SOURCE out, x2_pos_to_src
          "dof_vel"   -> width N_TARGET in, T_SOURCE out, x2_vel_to_src
          "action"    -> width N_TARGET in, T_SOURCE out, x2_action_to_src^-1
    """

    KINDS = ("pass", "dof_pos", "dof_vel", "action")

    def __init__(self, blocks: list[tuple[str, int, str]], joint: JointAlignment):
        for _, _, kind in blocks:
            assert kind in self.KINDS, kind
        self.blocks = blocks
        self.joint = joint
        self.in_width = sum(w for _, w, _ in blocks)
        self.out_width = sum(
            w if k == "pass" else T_SOURCE for _, w, k in blocks)

    def __call__(self, obs_x2: torch.Tensor) -> torch.Tensor:
        assert obs_x2.shape[-1] == self.in_width, \
            f"obs width {obs_x2.shape[-1]} != schema {self.in_width}"
        out, i = [], 0
        for _, w, kind in self.blocks:
            chunk = obs_x2[..., i:i + w]
            i += w
            if kind == "pass":
                out.append(chunk)
            elif kind == "dof_pos":
                out.append(self.joint.x2_pos_to_src(chunk))
            elif kind == "dof_vel":
                out.append(self.joint.x2_vel_to_src(chunk))
            elif kind == "action":
                out.append(self.joint.x2_action_to_src(chunk))
        return torch.cat(out, dim=-1)


def self_test():
    torch.manual_seed(0)
    j = JointAlignment(frame="mujoco")
    # round trip positions (matched joints)
    q_g1 = torch.randn(64, T_SOURCE)
    back = j.x2_pos_to_src(j.src_pos_to_x2(q_g1))
    assert torch.allclose(back, q_g1, atol=1e-5), "pos round-trip failed"
    # actions: linear rule, no offset leakage
    a_g1 = torch.randn(64, T_SOURCE)
    a_x2 = j.src_action_to_x2(a_g1)
    assert torch.allclose(j.x2_action_to_src(a_x2), a_g1, atol=1e-5)
    assert torch.allclose(a_x2[..., -2:], torch.zeros(64, 2)), "head must hold 0"
    # isaaclab frame round trip
    ji = JointAlignment(frame="isaaclab")
    assert torch.allclose(ji.x2_pos_to_src(ji.src_pos_to_x2(q_g1)), q_g1, atol=1e-5)
    # a G1 MuJoCo impulse must land on the same-named X2 joint through the
    # isaaclab path exactly once. IL index of MuJoCo joint m is
    # G1_ISAACLAB_TO_MUJOCO_DOF[m] (the array is used as q_mj = q_il[IL2MJ],
    # so IL2MJ[m] is where MuJoCo joint m lives in the IsaacLab vector).
    from kinematic_alignment import (G1_MUJOCO_JOINTS, X2_MUJOCO_JOINTS,
                                     SEMANTIC_JOINT_MAP)
    for probe in ("left_knee_joint", "waist_roll_joint", "right_wrist_yaw_joint"):
        v = torch.zeros(T_SOURCE)
        v[G1_ISAACLAB_TO_MUJOCO_DOF[G1_MUJOCO_JOINTS.index(probe)]] = 1.0
        out = ji.src_vel_to_x2(v)
        tgt, sgn = SEMANTIC_JOINT_MAP.get(probe, (probe, 1.0))  # chain-position + FK sign
        assert out[X2_MUJOCO_JOINTS.index(tgt)] == sgn and abs(out).sum() == 1.0, probe
    # Level-1 block rewiring: widths and pass-through integrity
    blocks = [("base_ang_vel", 3, "pass"), ("proj_gravity", 3, "pass"),
              ("dof_pos", N_TARGET, "dof_pos"), ("dof_vel", N_TARGET, "dof_vel"),
              ("prev_action", N_TARGET, "action"), ("ref", 32, "pass")]
    oa = ObservationAlignment(blocks, ji)
    obs = torch.randn(8, oa.in_width)
    out = oa(obs)
    assert out.shape == (8, oa.out_width)
    assert torch.equal(out[..., :6], obs[..., :6])          # scalars pass
    assert torch.equal(out[..., -32:], obs[..., -32:])      # ref passes
    print(f"alignment_wrapper self_test: OK (obs {oa.in_width}->{oa.out_width}, "
          f"src frame={ji.frame})")


if __name__ == "__main__":
    self_test()
