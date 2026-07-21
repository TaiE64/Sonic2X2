# Sonic2AgiX2: SONIC (G1) → AgiBot X2 Whole-Body Control Transfer

Closed-loop policy in MuJoCo (left) vs. reference (right):

![walk-to-bow](media/x2_walk_to_bow.gif)

Reproduction of the **Any2Any** cross-embodiment transfer (arXiv:2605.23733) that
adapts NVIDIA's GEAR-SONIC whole-body controller (trained for the Unitree G1) onto
the **AgiBot X2 Ultra** humanoid (31 DoF), via LoRA on the actor dynamics decoder +
critic with a fixed kinematic G1↔X2 alignment. This drop covers **v19** (deployed)
and the **v20** data-fix retrain (rebuilt retargeting: neutral canonical shape,
exact 30 Hz grid, ground snap, bad-clip rejection).

> **This repository is an overlay, not a standalone runtime.** It contains only the
> code we wrote (retargeting, kinematic alignment, LoRA wiring, the MuJoCo sim2sim
> gate, and the X2 real-robot deploy tooling). It plugs into NVIDIA's trainer and
> requires assets we cannot redistribute. See **Prerequisites**.

## What this is (and isn't)

The transfer is a thin, verified layer on top of the SONIC trainer. To actually run
it you bring three things yourself — none can legally ship in this repo:

| You provide | Why it's not here |
|---|---|
| [NVlabs/GR00T-WholeBodyControl](https://github.com/NVlabs/GR00T-WholeBodyControl) (Isaac Sim trainer) | Apache-2.0 upstream; clone it directly |
| SONIC base policy weights | NVIDIA Open Model License — not redistributable |
| AMASS motion data (+ LAFAN/OMOMO) | AMASS is registration-gated, research-only |
| Isaac Sim / IsaacLab + an NVIDIA GPU | runtime dependency |

## Layout

```
any2any_train/          LoRA + kinematic-alignment core
  aligned_env.py          G1↔X2 FrameMapper (Phi / Phi+ alignment, zero-pose offsets)
  lora.py, wire_sonic_lora.py, merge_lora_checkpoint.py
  convert_x2_to_motion_lib.py
retarget/               AMASS/LAFAN/OMOMO → X2 retargeting pipeline
  batch_retarget_*.py, retarget_x2.py, kinematic_alignment.py, calibrate_offsets.py, ...
sim2sim/                MuJoCo closed-loop validation harness (no Isaac Sim needed)
  mujoco_sim2sim_x2.py    obs-contract gate + physics rollout + video
gear_sonic_x2_overlay/  X2 robot config to drop into the GR00T tree
  x2.py                   -> gear_sonic/envs/manager_env/robots/x2.py
deploy_x2/              real-robot deployment tooling (numpy/ONNX/ROS2)
  pack_deploy.py, export_bundle.py, gen_testvec.py, compare_pipeline.py,
  dry_run_bridge.py, set_mode.py, js_joy_node.py
data_manifests/         training-data curation lists (clip names only) for
                        v16-v20 motion libraries
docs/                   REPRODUCE_PLAN.md, CHECKLIST.md, BUGS_AND_FIXES.md
```

## Setup

1. Clone the trainer and install per its instructions (Isaac Sim + IsaacLab):
   ```bash
   git clone https://github.com/NVlabs/GR00T-WholeBodyControl.git
   export GROOT_ROOT=$(pwd)/GR00T-WholeBodyControl
   export SONIC2X2_ROOT=$(pwd)/Sonic2X2
   ```
2. Overlay the X2 robot config:
   ```bash
   cp $SONIC2X2_ROOT/gear_sonic_x2_overlay/x2.py \
      $GROOT_ROOT/gear_sonic/envs/manager_env/robots/x2.py
   ```
3. Obtain the SONIC base weights (NVIDIA) and AMASS data separately.

Paths are read from env vars (`GROOT_ROOT`, `X2_MJCF`, `HEFT_DATA_ROOT`); grep for
`/path/to/` to see everything you must point at your own checkout.

## Pipeline

1. **Retarget** AMASS/LAFAN/OMOMO to X2 (`retarget/batch_retarget_*.py`) → motion_lib.
2. **Wire LoRA** onto the SONIC actor decoder + critic (`any2any_train/wire_sonic_lora.py`).
3. **Train** with the GR00T trainer (`gear_sonic/train_agent_trl.py`) resuming from the
   SONIC checkpoint, with `motion_file` pointed at your retargeted X2 lib.
4. **Merge + export** (`merge_lora_checkpoint.py`, then the trainer's ONNX export).
5. **Validate in MuJoCo** (`sim2sim/mujoco_sim2sim_x2.py --validate` gate, then closed-loop).
6. **Deploy** to the robot (`deploy_x2/pack_deploy.py` → `export_bundle.py` → on-robot
   C++ controller / `dry_run_bridge.py`).

The MuJoCo harness is Isaac-Sim-free and the fastest way to sanity-check a checkpoint.

## Reproduction notes

`docs/CHECKLIST.md` and `docs/BUGS_AND_FIXES.md` are the hard-won engineering record —
the index/mapping, semantic-pairing, obs-contract, and soft-limit pitfalls that make or
break an A→B embodiment transfer. Read them before adapting to another robot.

## License

Apache-2.0 (see `LICENSE`). Portions derive from NVlabs/GR00T-WholeBodyControl
(Apache-2.0); model weights and motion data are **not** included and carry their own,
non-redistributable licenses.
