# AgiBot X2 Ultra robot configuration (31 DoF), structured after h2.py.
#
# URDF source: aimdk/X2_URDF-v1.3.0/x2_ultra_simple_collision.urdf
# NOTE: the simple-collision variant (ball feet + primitive collisions) is used
# on purpose; the mesh-feet URDF (x2_ultra.urdf) ruins locomotion training.

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
import isaaclab.sim as sim_utils

ASSET_DIR = "gear_sonic/data/assets"

# REPRODUCTION APPROXIMATION (repro-plan gap: vendor PD gains for the X2 are
# unknown). We reuse the G1/H2 gain scheme, which derives stiffness/damping
# from the reflected rotor inertia (armature) of a motor class and a target
# closed-loop response: K = armature * w^2, D = 2 * zeta * armature * w with
# w = 2*pi*10Hz and zeta = 2. X2 joints are assigned to the existing motor
# classes by matching each joint's URDF effort limit to the same-role joint on
# G1 (35 kg) / H2 (larger, same 31-DoF layout); X2 (~42 kg, URDF total 41.97) effort
# limits sit between the two, so the class assignment (not the absolute
# gains) is what is matched:
#
#   X2 joint (URDF effort)          class      G1 analog (effort) / H2 analog
#   hip pitch/roll, knee (120)      7520_22    hip/knee (139) / (417)
#   hip yaw (120)                   7520_14    hip yaw (88) / (264)
#   waist yaw (120)                 7520_14    waist yaw (88) / (264)
#   waist pitch/roll (48)           2x 5020    waist roll/pitch (50) / (150)
#   ankle pitch/roll (36/24)        2x 5020    ankles (50) / (150)
#   shoulder p/r (36), y (24),
#   elbow (24), wrist yaw (24)      5020       shoulder/elbow/wrist_roll (25)
#   wrist pitch/roll (4.8)          4010       wrist pitch/yaw (5)
#   head yaw/pitch (2.6/0.6)        4010       small actuators; H2's 2x 5020
#                                              head gains are far too stiff
#                                              for the X2's 2.6/0.6 Nm motors
#
# Note the X2 wrist chain is yaw->pitch->roll (forearm rotation first), so the
# X2 wrist_yaw plays the role of the G1 wrist_roll (5020 class) and the X2
# wrist pitch/roll are the small distal 4010-class joints.
# Effort/velocity limits below are taken directly from the X2 URDF.

ARMATURE_5020 = 0.003609725
ARMATURE_7520_14 = 0.010177520
ARMATURE_7520_22 = 0.025101925
ARMATURE_4010 = 0.00425

NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10Hz
DAMPING_RATIO = 2.0

STIFFNESS_5020 = ARMATURE_5020 * NATURAL_FREQ**2
STIFFNESS_7520_14 = ARMATURE_7520_14 * NATURAL_FREQ**2
STIFFNESS_7520_22 = ARMATURE_7520_22 * NATURAL_FREQ**2
STIFFNESS_4010 = ARMATURE_4010 * NATURAL_FREQ**2

DAMPING_5020 = 2.0 * DAMPING_RATIO * ARMATURE_5020 * NATURAL_FREQ
DAMPING_7520_14 = 2.0 * DAMPING_RATIO * ARMATURE_7520_14 * NATURAL_FREQ
DAMPING_7520_22 = 2.0 * DAMPING_RATIO * ARMATURE_7520_22 * NATURAL_FREQ
DAMPING_4010 = 2.0 * DAMPING_RATIO * ARMATURE_4010 * NATURAL_FREQ

# IsaacLab enumerates joints/bodies breadth-first over the URDF kinematic tree
# (verified against the known-correct g1.py mapping). MuJoCo order is the
# depth-first chain order of x2_ultra.xml: left leg, right leg, waist (yaw,
# pitch, roll), left arm, right arm, head — matching
# Any2Any/kinematic_alignment.py X2_MUJOCO_JOINTS.
# Convention: q_mujoco = q_isaaclab[X2_ISAACLAB_TO_MUJOCO_DOF].

# --- EMPIRICAL index maps (regenerated 2026-07-08 from the live Isaac Lab
# env via the any2any probe: IsaacLab sorts BFS siblings ALPHABETICALLY, which
# the original URDF-document-order derivation got wrong for head/shoulder
# levels). Convention (same as g1.py): q_mj = q_il[IL2MJ]; q_il = q_mj[MJ2IL].
X2_ISAACLAB_TO_MUJOCO_DOF = [
    0, 3, 6, 9, 14, 19, 1, 4, 7, 10, 15, 20,
    2, 5, 8, 12, 17, 21, 23, 25, 27, 29, 13, 18,
    22, 24, 26, 28, 30, 11, 16,
]
X2_MUJOCO_TO_ISAACLAB_DOF = [
    0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 29,
    15, 22, 4, 10, 30, 16, 23, 5, 11, 17, 24, 18,
    25, 19, 26, 20, 27, 21, 28,
]
# --- BODY maps FIXED 2026-07-10 ------------------------------------------
# CRITICAL BUG (present v1..v13): the two BODY maps below were copied from
# g1.py and never corrected for X2. X2's FK skeleton (x2.xml, Humanoid_Batch)
# has 34 bodies because it INCLUDES left_toe_link(idx7) and right_toe_link
# (idx14); the IsaacLab robot has only 32 (no toe bodies). The old maps were
# built as if the FK output were also a 32-body no-toe chain, so every body
# from the toes upward was shifted -> 25/32 isaaclab slots pulled the WRONG
# FK body. Net effect: the reference markers / obs / 14-body tracking reward
# fed the policy a SCRAMBLED upper body (e.g. right_shoulder marker actually
# sat on left_wrist, torso on waist_yaw). DOF maps were unaffected (toes have
# no DoF), which is why the robot still stood & the 5-point reward was only
# mildly off. Regenerated from probe_final.npz ground truth:
#   mujoco_to_isaaclab_body[k] = hb34.index(il_body[k])   (len == #isaaclab)
# Verified: all 14 tracked bodies + reward_point resolve to themselves.
X2_MUJOCO_TO_ISAACLAB_BODY = [
    0, 1, 8, 15, 2, 9, 16, 3, 10, 17, 4, 11,
    32, 18, 25, 5, 12, 33, 19, 26, 6, 13, 20, 27,
    21, 28, 22, 29, 23, 30, 24, 31,
]
# inverse (len == #FK bodies == 34); toe slots (mj 7,14) have no isaaclab body
# so they point at their parent ankle_roll. NOTE: unused by the X2 reference
# pipeline (only mujoco_to_isaaclab_body is consumed), kept for consistency.
X2_ISAACLAB_TO_MUJOCO_BODY = [
    0, 1, 4, 7, 10, 15, 20, 20, 2, 5, 8, 11,
    16, 21, 21, 3, 6, 9, 13, 18, 22, 24, 26, 28,
    30, 14, 19, 23, 25, 27, 29, 31, 12, 17,
]
# NOTE: despite the (g1.py-inherited) name, this list holds BODY/link names
# in IsaacLab order — commands.py indexes cfg.body_names against it.
X2_ISAACLAB_JOINTS = ['pelvis', 'left_hip_pitch_link', 'right_hip_pitch_link', 'waist_yaw_link', 'left_hip_roll_link', 'right_hip_roll_link', 'waist_pitch_link', 'left_hip_yaw_link', 'right_hip_yaw_link', 'torso_link', 'left_knee_link', 'right_knee_link', 'head_yaw_link', 'left_shoulder_pitch_link', 'right_shoulder_pitch_link', 'left_ankle_pitch_link', 'right_ankle_pitch_link', 'head_pitch_link', 'left_shoulder_roll_link', 'right_shoulder_roll_link', 'left_ankle_roll_link', 'right_ankle_roll_link', 'left_shoulder_yaw_link', 'right_shoulder_yaw_link', 'left_elbow_link', 'right_elbow_link', 'left_wrist_yaw_link', 'right_wrist_yaw_link', 'left_wrist_pitch_link', 'right_wrist_pitch_link', 'left_wrist_roll_link', 'right_wrist_roll_link']

X2_ISAACLAB_TO_MUJOCO_MAPPING = {
    "isaaclab_joints": X2_ISAACLAB_JOINTS,
    "isaaclab_to_mujoco_dof": X2_ISAACLAB_TO_MUJOCO_DOF,
    "mujoco_to_isaaclab_dof": X2_MUJOCO_TO_ISAACLAB_DOF,
    "isaaclab_to_mujoco_body": X2_ISAACLAB_TO_MUJOCO_BODY,
    "mujoco_to_isaaclab_body": X2_MUJOCO_TO_ISAACLAB_BODY,
}


# X2 Robot Configuration
X2_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        replace_cylinders_with_capsules=True,
        asset_path=f"{ASSET_DIR}/robot_description/urdf/x2/x2.urdf",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # X2 root-to-sole is ~0.67 m standing; ~0.74 m leaves margin for the
        # slightly knee-bent pose below to settle onto the ground at reset.
        pos=(0.0, 0.0, 0.74),
        joint_pos={
            # Phi-consistent default pose: q_x2 = Phi^+(q_default_G1), i.e. the
            # X2 pose that is SEMANTICALLY IDENTICAL to G1's default (applies
            # the calibrated zero-pose offsets from kinematic_alignment.py:
            # elbows -1.579/-1.585, shoulder_roll -/+0.061). This makes
            # (a) joint_pos_rel observations bias-free under the alignment
            # wrapper and (b) the zero-action target pose match the source
            # policy's zero-action semantics. All values within URDF limits
            # (elbow [-2.3556,0]; shoulder_roll asymmetric, abduction side).
            ".*_hip_pitch_joint": -0.312,
            ".*_knee_joint": 0.669,
            ".*_ankle_pitch_joint": -0.363,
            "left_elbow_joint": -0.979,   # 0.6 - 1.579
            "right_elbow_joint": -0.985,  # 0.6 - 1.585
            "left_shoulder_roll_joint": 0.139,   # 0.2 - 0.061
            "left_shoulder_pitch_joint": 0.2,
            "right_shoulder_roll_joint": -0.139,  # -0.2 + 0.061
            "right_shoulder_pitch_joint": 0.2,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
            ],
            effort_limit_sim={
                ".*_hip_yaw_joint": 120.0,
                ".*_hip_roll_joint": 120.0,
                ".*_hip_pitch_joint": 120.0,
                ".*_knee_joint": 120.0,
            },
            velocity_limit_sim={
                ".*_hip_yaw_joint": 11.936,
                ".*_hip_roll_joint": 11.936,
                ".*_hip_pitch_joint": 11.936,
                ".*_knee_joint": 11.936,
            },
            stiffness={
                ".*_hip_pitch_joint": STIFFNESS_7520_22,
                ".*_hip_roll_joint": STIFFNESS_7520_22,
                ".*_hip_yaw_joint": STIFFNESS_7520_14,
                ".*_knee_joint": STIFFNESS_7520_22,
            },
            damping={
                ".*_hip_pitch_joint": DAMPING_7520_22,
                ".*_hip_roll_joint": DAMPING_7520_22,
                ".*_hip_yaw_joint": DAMPING_7520_14,
                ".*_knee_joint": DAMPING_7520_22,
            },
            armature={
                ".*_hip_pitch_joint": ARMATURE_7520_22,
                ".*_hip_roll_joint": ARMATURE_7520_22,
                ".*_hip_yaw_joint": ARMATURE_7520_14,
                ".*_knee_joint": ARMATURE_7520_22,
            },
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            effort_limit_sim={
                ".*_ankle_pitch_joint": 36.0,
                ".*_ankle_roll_joint": 24.0,
            },
            velocity_limit_sim={
                ".*_ankle_pitch_joint": 13.087,
                ".*_ankle_roll_joint": 15.077,
            },
            stiffness=2.0 * STIFFNESS_5020,
            damping=2.0 * DAMPING_5020,
            armature=2.0 * ARMATURE_5020,
        ),
        "waist": ImplicitActuatorCfg(
            effort_limit_sim=48.0,
            velocity_limit_sim=13.088,
            joint_names_expr=["waist_roll_joint", "waist_pitch_joint"],
            stiffness=2.0 * STIFFNESS_5020,
            damping=2.0 * DAMPING_5020,
            armature=2.0 * ARMATURE_5020,
        ),
        "waist_yaw": ImplicitActuatorCfg(
            effort_limit_sim=120.0,
            velocity_limit_sim=11.936,
            joint_names_expr=["waist_yaw_joint"],
            stiffness=STIFFNESS_7520_14,
            damping=DAMPING_7520_14,
            armature=ARMATURE_7520_14,
        ),
        "head": ImplicitActuatorCfg(
            joint_names_expr=["head_yaw_joint", "head_pitch_joint"],
            effort_limit_sim={
                "head_yaw_joint": 2.6,
                "head_pitch_joint": 0.6,
            },
            velocity_limit_sim={
                "head_yaw_joint": 6.019,
                "head_pitch_joint": 6.28,
            },
            stiffness=STIFFNESS_4010,
            damping=DAMPING_4010,
            armature=ARMATURE_4010,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
                ".*_wrist_yaw_joint",
                ".*_wrist_pitch_joint",
                ".*_wrist_roll_joint",
            ],
            effort_limit_sim={
                ".*_shoulder_pitch_joint": 36.0,
                ".*_shoulder_roll_joint": 36.0,
                ".*_shoulder_yaw_joint": 24.0,
                ".*_elbow_joint": 24.0,
                ".*_wrist_yaw_joint": 24.0,
                ".*_wrist_pitch_joint": 4.8,
                ".*_wrist_roll_joint": 4.8,
            },
            velocity_limit_sim={
                ".*_shoulder_pitch_joint": 13.088,
                ".*_shoulder_roll_joint": 13.088,
                ".*_shoulder_yaw_joint": 15.077,
                ".*_elbow_joint": 15.077,
                ".*_wrist_yaw_joint": 15.077,
                ".*_wrist_pitch_joint": 4.188,
                ".*_wrist_roll_joint": 4.188,
            },
            stiffness={
                ".*_shoulder_pitch_joint": STIFFNESS_5020,
                ".*_shoulder_roll_joint": STIFFNESS_5020,
                ".*_shoulder_yaw_joint": STIFFNESS_5020,
                ".*_elbow_joint": STIFFNESS_5020,
                ".*_wrist_yaw_joint": STIFFNESS_5020,
                ".*_wrist_pitch_joint": STIFFNESS_4010,
                ".*_wrist_roll_joint": STIFFNESS_4010,
            },
            damping={
                ".*_shoulder_pitch_joint": DAMPING_5020,
                ".*_shoulder_roll_joint": DAMPING_5020,
                ".*_shoulder_yaw_joint": DAMPING_5020,
                ".*_elbow_joint": DAMPING_5020,
                ".*_wrist_yaw_joint": DAMPING_5020,
                ".*_wrist_pitch_joint": DAMPING_4010,
                ".*_wrist_roll_joint": DAMPING_4010,
            },
            armature={
                ".*_shoulder_pitch_joint": ARMATURE_5020,
                ".*_shoulder_roll_joint": ARMATURE_5020,
                ".*_shoulder_yaw_joint": ARMATURE_5020,
                ".*_elbow_joint": ARMATURE_5020,
                ".*_wrist_yaw_joint": ARMATURE_5020,
                ".*_wrist_pitch_joint": ARMATURE_4010,
                ".*_wrist_roll_joint": ARMATURE_4010,
            },
        ),
    },
)

# X2 Action Scale
X2_ACTION_SCALE = {}
for a in X2_CFG.actuators.values():
    e = a.effort_limit_sim
    s = a.stiffness
    names = a.joint_names_expr
    if not isinstance(e, dict):
        e = dict.fromkeys(names, e)
    if not isinstance(s, dict):
        s = dict.fromkeys(names, s)
    for n in names:
        if n in e and n in s and s[n]:
            X2_ACTION_SCALE[n] = 0.25 * e[n] / s[n]

# --- Any2Any action-space fidelity override (paper Sec 3.4: "keep the action
# space identical to the source pretraining"): use G1's per-joint action scale
# on the SEMANTICALLY matched joint, so the frozen policy's action units mean
# the same thing on the X2. Wrists pair by chain position (G1 roll<->X2 yaw,
# G1 yaw<->X2 roll — opposite chain order); heads keep the derived value
# (no G1 counterpart; held at zero by the alignment anyway).
from gear_sonic.envs.manager_env.robots.g1 import G1_MODEL_12_ACTION_SCALE as _G1S

_SEM = {".*_wrist_roll_joint": ".*_wrist_yaw_joint",
        ".*_wrist_yaw_joint": ".*_wrist_roll_joint"}
for _xj in list(X2_ACTION_SCALE):
    _gj = _SEM.get(_xj, _xj)          # X2 joint's G1 semantic partner
    if _gj in _G1S:
        X2_ACTION_SCALE[_xj] = _G1S[_gj]
