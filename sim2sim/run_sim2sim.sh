#!/bin/bash
# Sim2sim pipeline for Sonic->X2: merge -> export ONNX -> capture GT ->
# validate obs contract -> closed-loop MuJoCo run with video.
#
# Per-version layout: every version gets its own folder sim2sim/<version>/
# holding that version's videos + capture/ (obs ground truth).
#
# Usage:
#   bash run_sim2sim.sh <run_dir> <version> [clip] [steps] [ckpt]
# e.g. latest last.pt of a run:
#   bash run_sim2sim.sh logs_rl/.../sonic2x2_v13_lora64-20260710_223237 v13 walk1_subject1 1500
# e.g. a named snapshot (V15 etc.):
#   bash run_sim2sim.sh logs_rl/.../sonic2x2_v13_lora64-20260711_154525 v15 jumps1_subject1 1500 \
#        logs_rl/.../sonic2x2_v13_lora64-20260711_154525/snap_v15_iter5499.pt
#
# Re-run whenever there is a newer checkpoint; every stage is idempotent.
# NOTE the ONNX input block order is hash-dependent per export (list(set()));
# the --validate stage catches it — if section A errors are >1, re-run the
# permutation test (see mujoco_sim2sim_x2.py docstring) and fix ObsBuilder.build.
set -e
cd /home/taie/Desktop/Byte/Sonic_Retarget/GR00T-WholeBodyControl
PY=/home/taie/miniconda3/envs/isaaclab/bin/python
A2A=/home/taie/Desktop/Byte/Sonic_Retarget/Any2Any
S2S=$A2A/sim2sim
RUN=$1
VER=${2:?usage: run_sim2sim.sh <run_dir> <version> [clip] [steps] [ckpt]}
CLIP=${3:-walk1_subject1}
STEPS=${4:-1500}
CKPT=${5:-$RUN/last.pt}
OUT=$S2S/$VER
CAP=$OUT/capture
mkdir -p $OUT

# 1. snapshot (only if pointing at the live last.pt) + merge (LoRA -> plain SONIC)
if [ "$(basename $CKPT)" = "last.pt" ]; then
    while [ $(( $(date +%s) - $(stat -c %Y $CKPT) )) -lt 5 ]; do sleep 1; done
    SNAP=$RUN/snap_${VER}.pt; cp $CKPT $SNAP
else
    SNAP=$CKPT                       # already a stable named snapshot
fi
$PY $A2A/any2any_train/merge_lora_checkpoint.py \
    --ckpt $SNAP --out $RUN/merged_${VER}.pt

# 2. export ONNX (needs IsaacSim; ~2 min; writes $RUN/exported/model_step_XXXXXX_g1.onnx)
WANDB_MODE=offline $PY -u gear_sonic/eval_agent_trl.py \
    +checkpoint=$RUN/merged_${VER}.pt +headless=True +export_onnx_only=True \
    ++num_envs=1 ++manager_env.commands.motion.motion_lib_cfg.multi_thread=False \
    ++manager_env.commands.motion.motion_lib_cfg.motion_file=data/motion_lib_lafan_x2_mini/robot
ONNX=$(ls -t $RUN/exported/*_g1.onnx | head -1)

# 3. capture clean ground truth into this version's folder (corruption off)
ANY2ANY_PLAY=1 ANY2ANY_CAPTURE=1 ANY2ANY_CAPTURE_DIR=$CAP WANDB_MODE=offline \
$PY -u gear_sonic/train_agent_trl.py \
    +exp=manager/universal_token/all_modes/sonic2x2_v13 num_envs=1 headless=True \
    +checkpoint=$SNAP \
    ++manager_env.commands.motion.motion_lib_cfg.motion_file=data/motion_lib_lafan_x2/robot \
    ++manager_env.commands.motion.motion_lib_cfg.multi_thread=False \
    ++manager_env.observations.policy.enable_corruption=False \
    ++manager_env.observations.tokenizer.enable_corruption=False

# 4. validation gate (expect: proprio 0.0000, maob ~1e-5, cmf ~0.05, action ~0.1)
$PY $S2S/mujoco_sim2sim_x2.py --validate --capture_dir $CAP --onnx $ONNX

# 5. closed-loop MuJoCo physics + video -> sim2sim/<version>/<clip>.mp4
MUJOCO_GL=egl $PY $S2S/mujoco_sim2sim_x2.py \
    --clip $CLIP --steps $STEPS --capture_dir $CAP --onnx $ONNX \
    --render $OUT/${CLIP}.mp4 --ref_ghost
echo "### DONE -> $OUT/${CLIP}.mp4 (capture in $CAP)"
