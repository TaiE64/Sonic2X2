# sim2sim — MuJoCo 部署验证

把 Sonic→X2 (Any2Any) 训好的策略拿到 MuJoCo 物理里闭环跑,验证 sim2sim 迁移。
详细复现记录见 `../BUGS_AND_FIXES.md` 的 "sim2sim MuJoCo 部署重建" 一节。

## 目录约定:每个策略版本一个文件夹

```
sim2sim/
  mujoco_sim2sim_x2.py     harness
  run_sim2sim.sh           一键管线
  README.md
  x2_sim2sim_generated.xml 每次跑自动生成的物理模型(可删)
  v17/                     ← V17,按用途分子目录(v13/v15 及 v17_BEST 权重已于
                             2026-07-20 清理;capture/ 因 v19 依赖保留):
    capture/               该版本 obs 真值
    benchmark/             数据集片段实测(LAFAN/AMASS 具名 clip + 日志)
    demos/                 能力演示(moonwalk/waltz/单腿/恢复跨步/凌空踢…)
    video_tracking/        手机视频→机器人跟踪成品(video2robot.sh 输出)
  v19/                     ← V19 (step9000 导出);无 capture,块序用 --block_order
                             或对 v17 capture 自动检测(已验证可行):
    demos/                 能力演示(13 条 AMASS clip)
    demos_inplace_hard_s9000/    原地高难度 battery(11 条 + results.txt)
    demos_inplace_hard_s12000/   同一 battery @续训 step12000(对比用)
    generalization/        手机视频泛化测试
```

每个版本自带 `capture/`(从 IsaacSim 捕获的干净真值 constants.npz / gt_step0..2.npz /
rel_bias.npy),让 harness 无需 IsaacSim 即可独立跑该版本的 `--validate` 和闭环渲染。
`--capture_dir` 不传时自动选**最新**的 `v*/capture`。

## 文件

- `mujoco_sim2sim_x2.py` — harness。`--validate` 是 obs 契约验证门(纯 CPU,逐块对
  IsaacSim 干净真值);默认闭环跑 MuJoCo 物理并渲染视频。obs 契约细节写在文件头 docstring。
  捕获探针在 `../../GR00T-WholeBodyControl/gear_sonic/train_agent_trl.py` 的 PLAY 块(ANY2ANY_CAPTURE)。
- `run_sim2sim.sh <run_dir> <version> [clip] [steps] [ckpt]` — 一键管线:merge LoRA →
  导 ONNX(需 IsaacSim)→ 捕获真值到 `<version>/capture` → 验证门 → 闭环渲染到 `<version>/<clip>.mp4`。
  `ckpt` 省略=用 run 的 last.pt(会快照);也可传具名快照(如 `snap_v15_iter5499.pt`)。

## 用法

```bash
# 一键全流程(每个版本各自成文件夹):
#   last.pt:
bash run_sim2sim.sh logs_rl/.../sonic2x2_v13_lora64-20260710_223237 v13 walk1_subject1 1500
#   具名快照(V15):
bash run_sim2sim.sh logs_rl/.../sonic2x2_v13_lora64-20260711_154525 v15 jumps1_subject1 1500 \
     logs_rl/.../sonic2x2_v13_lora64-20260711_154525/snap_v15_iter5499.pt

# 手动验证门(纯 CPU;--capture_dir 不传=自动选最新版本):
python mujoco_sim2sim_x2.py --validate --onnx <run_dir>/exported/model_step_XXXXXX_g1.onnx

# 单条 clip 闭环 + 视频(手动指定版本 capture 与输出):
MUJOCO_GL=egl python mujoco_sim2sim_x2.py --clip walk1_subject1 --steps 1500 \
    --onnx <run_dir>/exported/..._g1.onnx --capture_dir v15/capture \
    --render v15/walk1_subject1.mp4 --ref_ghost
```

## 结果快照

- **V13 (iter12350)** — `v13/`:walk/jumps1/dance2/aiming2 全程 30s 零摔倒;fight/fightAndSports/sprint
  在 11–17s 摔(root ori>46° 失衡)。
- **V15 (iter5499, jump 专项)** — `v15/`:平衡更稳,sprint 摔@21.8s(v13 仅 10.9s)、fight 19.4s、
  关节误差普遍更低;跳跃/腾跃(jumps2/leap_demo)摔在腾空/落地相(root_h 高度发散,MuJoCo 落地接触 gap)。

验证门标准:proprio 精确 0.0000、maob ~1e-5、cmf ~0.05(30→50fps 插值底噪)、端到端动作 ~0.1(FSQ 容忍)。

⚠️ ONNX 输入块序**每次导出可能变**(导出代码用 `list(set())`);验证门 A 段误差 >1 即错序,
按 `mujoco_sim2sim_x2.py` docstring 的排列穷举重定 `ObsBuilder.build` 拼接顺序。
