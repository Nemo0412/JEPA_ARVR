# 方法、脚本与代码对应表

更新日期：2026-06-20

这份文档用于帮助合作者快速了解我们做了哪些方法，以及每个方法对应哪些运行脚本、配置文件和核心代码。文档只保留复现和读代码所需的信息。

> **实验历史与指标**请到 VJEPA2-EXP 仓库查看 `logs/DASHBOARD.md` 和各分支卡。
> **中文超参表**见 [docs/human_readable_zh/超参表(zh).md](human_readable_zh/超参表(zh).md)。
> **近期 HPC Run 入口**见 [docs/RECENT_RUNS.md](RECENT_RUNS.md)。

---

## 当前标准（2026-06-15 起）

| 项 | 值 |
|---|---|
| Backbone | ViT-L/256（已从 ViT-G/384 迁回） |
| 数据划分 | `p01_fixed`（train 5964 / val 375 / test 1045） |
| 类别空间 | `phd_reference`（全 HD-EPIC taxonomy + P01 primary verb-noun pairs） |
| 时序采样 | `phd_reference`（action-start anchor，anticipation_point=[1,1]） |
| Label 来源 | `primary_verb_noun`（verb_classes[0] / noun_classes[0]） |
| Probe | singleprobe（已从 20-head sweep 迁移） |
| 精度 | fp32（bf16 路径存在但需 GradScaler 修复） |
| 指标 | Action Top-1/3/5 + class-mean Recall@5；verb/noun Top-3 |

**旧版 B1–B10（ViT-G/384、20-head、legacy split）已归档。** 不要用归档分支的指标作为当前 baseline。

---

## 总览

| 方法 | 主要目的 | 运行入口 | 核心代码 | 状态 |
|---|---|---|---|---|
| B1 Clean / LoRA baseline | HD-EPIC action anticipation 干净基线 | `scripts/run_hdepic_action_anticipation.slurm`, `scripts/run_hdepic_lora_probe.slurm` | `app/hdepic_lora_action_anticipation/eval.py`, `vjepa2/evals/action_anticipation_frozen/*` | **归档** |
| B2 Gaze fusion | gaze 作为附加模态接入 probe | `scripts/run_hdepic_lora_rnn_gaze_train.slurm` 等 | `app/hdepic_lora_action_anticipation/gaze.py`, `gaze_rnn.py` | **归档** |
| B3 Long horizon | 10s / 更长 horizon 的 anticipation | `scripts/run_hdepic_lora_val_horizons.slurm` 等 | `modelcustom/vit_encoder_predictor_rollout.py` | **归档** |
| B5 Binary input adapter | gaze binary/distance map 调制 RGB 输入 | `scripts/run_hdepic_lora_binary_input_adapter_train.slurm` | `binary_input_adapter.py`, `binary_map_utils.py` | **归档** |
| B6 Future latent compare | observed/predicted/oracle latent 差异分析 | `scripts/run_hdepic_future_latent_compare.slurm` | `future_latent_compare.py` | **归档** |
| B7 Long-history gaze | 固定视频窗口，延长 gaze history | `scripts/run_hdepic_lora_rnn_long_gaze_train.slurm` | `gaze_rnn.py`, `gaze.py` | **归档** |
| B8 Encoder-output gaze injection | encoder output / predictor input 间注入 gaze | `scripts/run_hdepic_lora_encoder_gaze_inject_train.slurm` | `encoder_output_gaze_adapter.py` | **归档** |
| B10 SLAM pose / multimodal RNN | SLAM pose 或 pose+gaze token late fusion | `scripts/run_hdepic_lora_rnn_pose_train.slurm` | `pose_slam.py`, `gaze_rnn.py` | **归档** |
| **B11 Encoder-LoRA + Interframe Pose Matrix** | ViT-L/256 singleprobe + encoder-LoRA + gaze/pose adapter | `scripts/run_hdepic_lora_binary_gaze_pose_matrix_train.slurm`, `scripts/run_hdepic_singleprobe_1s_*.slurm` | `binary_input_adapter.py`, `pose_slam.py`, `pose_map_builder.py`, `encoder_lora.py` | **Active** |
| **Encoder-LoRA** | 在 frozen ViT encoder 注入 LoRA（rank=8, attn.qkv+attn.proj） | `scripts/run_hdepic_lora_probe.slurm`（通过 `ENCODER_LORA_ENABLED=1`） | `encoder_lora.py`, `eval.py::LoRALinear` | **Active** |
| **Predictor-LoRA** | 在 frozen predictor 注入 LoRA（与 encoder-LoRA 同参） | `scripts/run_hdepic_lora_probe.slurm`（通过 `PREDICTOR_LORA_ENABLED=1`） | `predictor_lora.py` | **Active** |
| **LTM (Latent Memory)** | latent memory / oracle compare / direct probe 诊断 | `scripts/run_d3_e18a_direct_probe_train.slurm`, `scripts/run_d3_e20_oracle_probe_train.slurm` | `app/hdepic_lora_action_anticipation/` LTM 分支代码 | **Active** |

---

## 公共训练入口

大部分 LoRA / probe 方法最终都会经过：

- `scripts/run_hdepic_lora_probe.slurm`
  - 生成或改写 eval config。
  - 设置 batch size、worker 数、LR、checkpoint、tag、gaze mode、encoder-LoRA、predictor-LoRA 等公共训练参数。
  - 多数专用脚本只是先设置环境变量，再调用这个共享入口。
- `app/hdepic_lora_action_anticipation/eval.py`
  - HD-EPIC action anticipation eval / train 的主要扩展入口。
  - 将不同 `gaze.mode`、past-window、adapter、pose、encoder-LoRA、predictor-LoRA 等配置接入模型和 dataloader。
- `app/hdepic_lora_action_anticipation/gaze.py`
  - gaze 数据读取、对齐、map/token 构造、coverage 诊断和 dataloader 逻辑。
- `app/hdepic_lora_action_anticipation/gaze_rnn.py`
  - RNN / MLP / pose / multimodal token encoder，以及 probe-side fusion 模块。

---

## 归档：B1–B10（Pre-ViT-L，2026-06-15 前）

以下方法在 ViT-G/384、20-head、legacy split 下完成。代码仍在本仓库，但指标不应作为当前 baseline。完整历史见 VJEPA2-EXP `logs/archive/pre-vitl/ARCHIVE_INDEX.md`。

### B1: Clean / LoRA Baseline

用途：建立 HD-EPIC action anticipation 的 clean baseline。

运行脚本：
- `scripts/run_hdepic_action_anticipation.slurm`
- `scripts/run_hdepic_lora_probe.slurm`

核心代码：
- `app/hdepic_lora_action_anticipation/eval.py`
- `vjepa2/evals/action_anticipation_frozen/eval.py`
- `vjepa2/evals/action_anticipation_frozen/dataloader.py`
- `vjepa2/evals/action_anticipation_frozen/models.py`

### B2: Gaze Fusion

用途：在 frozen encoder/predictor 后，将 gaze 信息接入 probe pooler 的 K/V。

变体：RNN Gaze Fuse / MLP Gaze Fuse / Token Gate / Overlay / Video-token RNN

运行脚本：
- `scripts/run_hdepic_lora_rnn_gaze_train.slurm`
- `scripts/run_hdepic_lora_mlp_gaze_train.slurm`
- `scripts/run_hdepic_lora_token_gaze_train.slurm`
- `scripts/run_hdepic_lora_overlay_gaze_train.slurm`

核心代码：
- `app/hdepic_lora_action_anticipation/gaze_rnn.py`（GazeTrajectoryLoader、GazeTrajectoryEncoder、GazeFusedAttentivePooler）
- `app/hdepic_lora_action_anticipation/gaze.py`

### B3: Long-Horizon Prediction

用途：3.5s、10s、60s 等更长 anticipation horizon。

运行脚本：
- `scripts/run_hdepic_lora_val_horizons.slurm`
- `scripts/run_hdepic_lora_ar_val_horizons.slurm`
- `scripts/run_hdepic_lora_past_window_train.slurm`

核心代码：
- `app/hdepic_lora_action_anticipation/modelcustom/vit_encoder_predictor_rollout.py`（AutoregressiveAnticipativeWrapper）

### B5: Binary Input Adapter

用途：gaze disk 或 distance map 作为额外输入通道，在 RGB 进入 frozen encoder 前用小 adapter 做条件化。

运行脚本：
- `scripts/run_hdepic_lora_binary_input_adapter_train.slurm`
- `scripts/run_hdepic_lora_binary_input_adapter_distance_lr.slurm`
- `scripts/run_hdepic_lora_binary_input_adapter_zero_val.slurm`

核心代码：
- `app/hdepic_lora_action_anticipation/binary_input_adapter.py`（BinaryMapInputAdapter、BinaryInputAdaptedModel、tokens_proxy gradient gating）
- `app/hdepic_lora_action_anticipation/binary_map_utils.py`
- `app/hdepic_lora_action_anticipation/binary_map_aug.py`

**已知问题：** disable_train_aug=true 导致 train/val gap ~34pt；zero-channel val 几乎等同正常 val，说明 gain 不能直接归因于 gaze channel。当前视为 dead-end 候选。

### B6: Future Latent Compare / Failure Analysis

用途：拆解 observed latent、predicted future latent、oracle future latent 的差异和失败来源。

运行脚本：
- `scripts/run_hdepic_future_latent_compare.slurm`
- `scripts/run_hdepic_future_latent_failure_modes.slurm`
- `scripts/run_hdepic_lora_valonly_dump.slurm`
- `scripts/run_hdepic_rescore_window_cpu.slurm`

核心代码：
- `app/hdepic_lora_action_anticipation/future_latent_compare.py`
- `scripts/analyze_future_latent_failure_modes.py`
- `scripts/rescore_window.py`

### B7: Long-History Gaze

用途：视频观察窗口不变，只延长 gaze history（默认 20s）。

运行脚本：
- `scripts/run_hdepic_lora_rnn_long_gaze_train.slurm`

### B8: Encoder-Output Gaze Injection

用途：gaze 注入点在 encoder output 和 predictor input 之间，影响 future-token prediction。

运行脚本：
- `scripts/run_hdepic_lora_encoder_gaze_inject_train.slurm`

核心代码：
- `app/hdepic_lora_action_anticipation/encoder_output_gaze_adapter.py`（EncoderOutputGazeAdapter、EncoderOutputGazeAdaptedModel）

### B10: SLAM Pose / Multimodal RNN Fuse

用途：SLAM closed-loop pose / head-motion trajectory 作为与 gaze 平行的模态，接入 late-fusion probe path。

运行脚本：
- `scripts/run_hdepic_lora_rnn_pose_train.slurm`
- `scripts/run_hdepic_lora_rnn_multimodal_train.slurm`

核心代码：
- `app/hdepic_lora_action_anticipation/pose_slam.py`（SlamPoseLoader、PoseTrajectoryLoader、pose_6d/pose_vel/pose_full）
- `app/hdepic_lora_action_anticipation/gaze_rnn.py`（PoseTrajectoryEncoder）

### B5+B10 Hybrid

用途：B5 binary map gaze（pixel adapter）+ B10 pose GRU（probe K/V）同时注入。

运行脚本：
- `scripts/run_hdepic_lora_binary_pose_train.slurm`
- `scripts/submit_binary_pose_train_adaptive.sh`
- `scripts/launch_binary_pose_train_watchdog.sh`

GAZE_MODE: `binary_input_adapter_pose_rnn_fuse`

---

## Active：B11 及 ViT-L 新方法（2026-06-15 起）

### B11: Encoder-LoRA + Interframe Pose Matrix

当前主线。ViT-L/256 singleprobe，encoder-LoRA (rank=8, alpha=16, attn.qkv+attn.proj) + gaze/pose 5ch adapter。

**方法：** 将相邻 video frame 之间的高频 SLAM 采样（~1000 Hz）打包为 `[K_max, D]` 矩阵，rasterize 到每帧左上角 patch，与 B5 binary gaze 合并为 5 通道 adapter 输入。probe 侧不再使用 pose GRU。

**运行脚本：**
- `scripts/run_hdepic_lora_binary_gaze_pose_matrix_train.slurm`：全量训练（w10, mem=768G, bs=2）
- `scripts/run_hdepic_lora_binary_gaze_pose_matrix_smoke.slurm`：smoke test
- `scripts/run_hdepic_lora_binary_gaze_pose_matrix_valonly_filtered.slurm`：formal filtered val
- `scripts/run_hdepic_lora_b11_zero_aux_valonly.slurm`：zero-channel control（force_zero_map/force_zero_pose）
- `scripts/run_hdepic_lora_b11_latent_effect.slurm`：latent effect diagnostic
- `scripts/run_hdepic_singleprobe_1s_*.slurm`：singleprobe 1s 变体
- `scripts/run_hdepic_singleprobe_ar10s_*.slurm`：AR 10s 变体

**核心代码：**
- `app/hdepic_lora_action_anticipation/binary_input_adapter.py`（5ch BinaryMapInputAdapter）
- `app/hdepic_lora_action_anticipation/pose_slam.py`（query_interframe_matrices）
- `app/hdepic_lora_action_anticipation/pose_map_builder.py`（InterframePoseMapBuilder、GazePoseInputMapBuilder）
- `app/hdepic_lora_action_anticipation/encoder_lora.py`（inject_encoder_lora）
- `app/hdepic_lora_action_anticipation/eval.py`（GAZE_MODE=binary_input_adapter_gaze_pose_matrix）

### Encoder-LoRA

在 frozen ViT encoder 中注入 trainable LoRA adapter（rank=8, alpha=16, lr_mult=0.5），target_suffixes=`attn.qkv, attn.proj`，不碰 MLP。与 JEPA_ARVR 参考实现对齐。

**运行入口：** `scripts/run_hdepic_lora_probe.slurm` + `ENCODER_LORA_ENABLED=1`

**核心代码：**
- `app/hdepic_lora_action_anticipation/encoder_lora.py`
- `app/hdepic_lora_action_anticipation/eval.py::LoRALinear`

### Predictor-LoRA

与 encoder-LoRA 相同的 LoRA 策略，但注入到 `model.predictor.predictor_blocks`（全部 12 blocks）。encoder 完全冻结。

**运行入口：** `scripts/run_hdepic_lora_probe.slurm` + `PREDICTOR_LORA_ENABLED=1` + `ENCODER_LORA_ENABLED=0`

**核心代码：**
- `app/hdepic_lora_action_anticipation/predictor_lora.py`

### LTM (Latent Memory) 实验

**D3 诊断系列：**
- D3-E11 latent distribution @10s：`scripts/run_d3_e11_latent_distribution_10s.slurm`
- D3-E12 norm rescale @10s：`scripts/run_d3_e12_norm_rescale_10s.slurm`
- D3-E14 ctx0 prediction dist @10s：`scripts/run_d3_e14_ctx0_prediction_dist_10s.slurm`
- D3-E17 junk future @10s：`scripts/run_d3_e17_junk_future_10s.slurm`
- D3-E18a direct probe train / valonly：`scripts/run_d3_e18a_direct_probe_train.slurm`
- D3-E20 oracle probe train：`scripts/run_d3_e20_oracle_probe_train.slurm`

---

## 对齐参考（PhD-Reference Alignment）

2026-06-15 起，HD-EPIC 路径默认对齐 JEPA_ARVR PhD 参考实现。详见 [docs/HD_EPIC_SYNC_NOTES.md](HD_EPIC_SYNC_NOTES.md) 和 VJEPA2-EXP `docs/HD_EPIC_REFERENCE_ALIGNMENT_DEBUG.md`。

关键默认值变更：
- `class_space=phd_reference`（全 taxonomy，非 train_only）
- `temporal_sampling=phd_reference`（action-start anchor）
- `split=p01_fixed`（固定 train/val/test 视频列表）
- label source `primary_verb_noun`

设置 `LORA_CLASS_SPACE=train_only` + `LORA_TEMPORAL_SAMPLING=legacy` 可退回旧行为（仅用于复现历史 run）。

---

## 数据准备与健康检查脚本

- `scripts/download_hdepic_data_cpu.slurm`：HD-EPIC data download
- `scripts/download_hdepic_gaze_cpu.slurm`：SLAM-and-Gaze data download
- `scripts/download_ek100_vitg384_inference_ckpts.sh`：EK100 / V-JEPA checkpoint download
- `scripts/refresh_hdepic_vjepa_annotations.slurm`：refresh V-JEPA 格式 annotations
- `scripts/convert_hdepic_to_vjepa_csv.py`：annotation conversion（默认 `--split-preset p01_fixed`）
- `scripts/check_hdepic_video_health.py` / `scripts/check_hdepic_video_health_cpu.slurm`：video health check
- `scripts/inspect_hdepic_gaze_data.py` / `scripts/inspect_hdepic_gaze_data_cpu.slurm`：gaze coverage / sync audit
- `scripts/inspect_hdepic_slam_pose_data.py` / `scripts/inspect_hdepic_slam_pose_data_cpu.slurm`：SLAM pose audit
- `scripts/create_debug_subset.py` / `scripts/create_debug_subset_cpu.slurm`：deterministic debug subset
- `configs/debug_subset_p01.json`：debug subset definition

---

## 给合作者的阅读顺序

1. 先看本文档总览和**当前标准**，确定感兴趣的方法对应哪些脚本和代码。
2. 要复现实验，优先看对应 `scripts/run_*.slurm`，再看它引用的 `configs/generated/*.yaml`。
3. 要理解模型改动，优先看 `app/hdepic_lora_action_anticipation/` 下对应模块。
4. 中文超参对照看 [docs/human_readable_zh/超参表(zh).md](human_readable_zh/超参表(zh).md)。
5. 近期 HPC 实验入口看 [docs/RECENT_RUNS.md](RECENT_RUNS.md)。
6. 完整实验历史和指标去 VJEPA2-EXP 仓库看 `logs/DASHBOARD.md`。
