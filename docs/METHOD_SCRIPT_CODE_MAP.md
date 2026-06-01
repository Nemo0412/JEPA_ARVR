# 方法、脚本与代码对应表

更新日期：2026-06-01

这份文档用于帮助合作者快速了解我们做了哪些方法，以及每个方法对应哪些运行脚本、配置文件和核心代码。文档只保留复现和读代码所需的信息。

## 总览

| 方法 | 主要目的 | 运行入口 | 核心代码 |
|---|---|---|---|
| B1 Clean / LoRA baseline | 建立 HD-EPIC action anticipation 干净基线 | `scripts/run_hdepic_action_anticipation.slurm`, `scripts/run_hdepic_lora_probe.slurm` | `app/hdepic_lora_action_anticipation/eval.py`, `vjepa2/evals/action_anticipation_frozen/*` |
| B2 Gaze fusion | 将 gaze 作为附加模态接入 probe | `scripts/run_hdepic_lora_rnn_gaze_train.slurm`, `scripts/run_hdepic_lora_mlp_gaze_train.slurm`, `scripts/run_hdepic_lora_token_gaze_train.slurm`, `scripts/run_hdepic_lora_overlay_gaze_train.slurm` | `app/hdepic_lora_action_anticipation/gaze.py`, `app/hdepic_lora_action_anticipation/gaze_rnn.py`, `app/hdepic_lora_action_anticipation/eval.py` |
| B3 Long horizon | 做 10s / 更长 horizon 的 anticipation 和 past-window 训练 | `scripts/run_hdepic_lora_val_horizons.slurm`, `scripts/run_hdepic_lora_past_window_train.slurm`, `scripts/run_hdepic_lora_past_window_baseline.slurm` | `app/hdepic_lora_action_anticipation/modelcustom/vit_encoder_predictor_rollout.py`, `app/hdepic_lora_action_anticipation/eval.py` |
| B5 Binary input adapter | 用 gaze binary/distance map 在输入侧调制 RGB | `scripts/run_hdepic_lora_binary_input_adapter_train.slurm`, `scripts/run_hdepic_lora_binary_input_adapter_distance_lr.slurm`, `scripts/run_hdepic_lora_binary_input_adapter_zero_val.slurm` | `app/hdepic_lora_action_anticipation/binary_input_adapter.py`, `app/hdepic_lora_action_anticipation/binary_map_utils.py`, `app/hdepic_lora_action_anticipation/binary_map_aug.py` |
| B6 Future latent compare | 分析 observed / predicted / oracle latent 的差异和失败模式 | `scripts/run_hdepic_future_latent_compare.slurm`, `scripts/run_hdepic_future_latent_failure_modes.slurm`, `scripts/run_hdepic_lora_valonly_dump.slurm`, `scripts/run_hdepic_rescore_window_cpu.slurm` | `app/hdepic_lora_action_anticipation/future_latent_compare.py`, `scripts/analyze_future_latent_failure_modes.py`, `scripts/rescore_window.py` |
| B7 Long-history gaze | 固定视频窗口，只延长 gaze history | `scripts/run_hdepic_lora_rnn_long_gaze_train.slurm` | `app/hdepic_lora_action_anticipation/gaze_rnn.py`, `app/hdepic_lora_action_anticipation/gaze.py`, `app/hdepic_lora_action_anticipation/eval.py` |
| B8 Encoder-output gaze injection | 在 encoder output / predictor input 之间注入 gaze | `scripts/run_hdepic_lora_encoder_gaze_inject_train.slurm` | `app/hdepic_lora_action_anticipation/encoder_output_gaze_adapter.py`, `app/hdepic_lora_action_anticipation/gaze_rnn.py` |
| B10 SLAM pose / multimodal RNN | 用 SLAM pose 或 pose+gaze token 做 late fusion | `scripts/run_hdepic_lora_rnn_pose_train.slurm`, `scripts/run_hdepic_lora_rnn_multimodal_train.slurm`, `scripts/run_hdepic_lora_pose_smoke.slurm` | `app/hdepic_lora_action_anticipation/pose_slam.py`, `app/hdepic_lora_action_anticipation/gaze_rnn.py`, `app/hdepic_lora_action_anticipation/eval.py` |

## 公共训练入口

大部分 LoRA / probe 方法最终都会经过：

- `scripts/run_hdepic_lora_probe.slurm`
  - 生成或改写 eval config。
  - 设置 batch size、worker 数、LR cap、checkpoint、tag、gaze mode 等公共训练参数。
  - 多数专用脚本只是先设置环境变量，再调用这个共享入口。
- `app/hdepic_lora_action_anticipation/eval.py`
  - HD-EPIC action anticipation eval / train 的主要扩展入口。
  - 将不同 `gaze.mode`、past-window、adapter、pose 等配置接入模型和 dataloader。
- `app/hdepic_lora_action_anticipation/gaze.py`
  - gaze 数据读取、对齐、map/token 构造、coverage 诊断和 dataloader 逻辑。
- `app/hdepic_lora_action_anticipation/gaze_rnn.py`
  - RNN / MLP / pose / multimodal token encoder，以及 probe-side fusion 模块。

## B1: Clean / LoRA Baseline

用途：建立 HD-EPIC action anticipation 的 clean baseline，后续 gaze、binary adapter、long-horizon 都需要和它对齐 metric scope 后再比较。

运行脚本：

- `scripts/run_hdepic_action_anticipation.slurm`: 原始 action anticipation 准备/训练入口。
- `scripts/run_hdepic_lora_probe.slurm`: 当前更常用的 LoRA probe 统一入口。
- `scripts/run_hdepic_ek100_probe_valonly_horizons.slurm`: EK100 probe / val-only horizon 相关入口。

常用配置：

- `configs/generated/hdepic_action_anticipation_vitl384.yaml`
- `configs/generated/hdepic_lora_probe_vitg384.yaml`
- `configs/generated/hdepic_ek100_probe_valonly_1s.yaml`
- `configs/generated/hdepic_ek100_probe_valonly_10s.yaml`

核心代码：

- `app/hdepic_lora_action_anticipation/eval.py`
- `vjepa2/evals/action_anticipation_frozen/eval.py`
- `vjepa2/evals/action_anticipation_frozen/dataloader.py`
- `vjepa2/evals/action_anticipation_frozen/models.py`
- `vjepa2/evals/action_anticipation_frozen/metrics.py`

## B2: Gaze Fusion

用途：在 frozen V-JEPA2 encoder/predictor 后，将 gaze 信息接入 probe pooler 的 K/V 或输入侧视频表示，测试 gaze 是否改善 action anticipation。

### RNN Gaze Fuse

方法：将 gaze trajectory 编码成 token，接入 probe cross-attention K/V。

运行脚本：

- `scripts/run_hdepic_lora_rnn_gaze_train.slurm`
- `scripts/run_hdepic_lora_gaze_val.slurm`

常用配置：

- `configs/generated/hdepic_lora_rnn_gaze.yaml`

核心代码：

- `app/hdepic_lora_action_anticipation/gaze_rnn.py`
- `app/hdepic_lora_action_anticipation/gaze.py`
- `app/hdepic_lora_action_anticipation/eval.py`

### MLP Gaze Fuse

方法：用较简单的 MLP / pooled gaze 表示替代 RNN trajectory encoder，用于判断 RNN 是否真正带来序列建模收益。

运行脚本：

- `scripts/run_hdepic_lora_mlp_gaze_train.slurm`

常用配置：

- `configs/generated/hdepic_lora_mlp_gaze.yaml`

核心代码：

- `app/hdepic_lora_action_anticipation/gaze_rnn.py`
- `app/hdepic_lora_action_anticipation/gaze.py`

### Token Gate

方法：将 gaze 转成 token/gate 信号，而不是直接用 RNN fusion。

运行脚本：

- `scripts/run_hdepic_lora_token_gaze_train.slurm`

常用配置：

- `configs/generated/hdepic_lora_token_gaze.yaml`

核心代码：

- `app/hdepic_lora_action_anticipation/gaze.py`
- `app/hdepic_lora_action_anticipation/gaze_rnn.py`

### Overlay / Binary Overlay

方法：离线或在线把 gaze map 叠到视频输入，作为像素空间的弱注入方式。

运行脚本：

- `scripts/run_hdepic_lora_overlay_gaze_train.slurm`
- `scripts/run_hdepic_lora_binary_overlay_gaze_train.slurm`
- `scripts/build_hdepic_gaze_overlays_cpu.slurm`
- `scripts/build_hdepic_gaze_heatmap_cpu.slurm`
- `scripts/build_hdepic_gaze_heatmap_video.py`

常用配置：

- `configs/generated/hdepic_lora_overlay_gaze_sigma96.yaml`
- `configs/generated/hdepic_lora_binary_overlay_gaze_radius64.yaml`
- `configs/generated/gaze_sigma96/hdepic_lora_valonly_1s.yaml`

核心代码：

- `app/hdepic_lora_action_anticipation/gaze.py`
- `scripts/build_hdepic_gaze_heatmap_video.py`

### Video-token RNN Variants

方法：RNN gaze token 进一步与 video token 交互，例如 nearest-concat、gated-nearest、local-attention、residual-alpha 等。

运行脚本：

- `scripts/run_hdepic_lora_rnn_gaze_train.slurm`
  - 通过 `GAZE_RNN_USE_VIDEO_TOKEN`、`GAZE_RNN_VIDEO_FUSION`、`GAZE_RNN_LOCAL_TEMPORAL_RADIUS`、`GAZE_RNN_RESIDUAL_ALPHA_INIT` 等环境变量切换变体。

核心代码：

- `app/hdepic_lora_action_anticipation/gaze_rnn.py`

## B3: Long-Horizon Prediction

用途：研究 3.5s、10s、60s 等更长 anticipation horizon。当前包括 val-only horizon、autoregressive rollout、past-window train/curriculum 等路径。

运行脚本：

- `scripts/run_hdepic_lora_val_horizons.slurm`: 多 horizon val-only / AR validation。
- `scripts/run_hdepic_lora_ar_val_horizons.slurm`: autoregressive validation。
- `scripts/run_hdepic_lora_past_window_baseline.slurm`: past-window baseline matrix。
- `scripts/run_hdepic_lora_past_window_train.slurm`: past-window training / curriculum。
- `scripts/run_hdepic_lora_past_window_parallel_debug_subset.slurm`: 2-GPU / debug-subset resource probe。

常用配置：

- `configs/generated/hdepic_lora_valonly_1s.yaml`
- `configs/generated/hdepic_lora_valonly_3p5s.yaml`
- `configs/generated/hdepic_lora_valonly_10s.yaml`
- `configs/generated/hdepic_lora_ar_valonly_10s.yaml`
- `configs/generated/past_window_baseline/*.yaml`
- `configs/generated/past_window_train/*.yaml`

核心代码：

- `app/hdepic_lora_action_anticipation/modelcustom/vit_encoder_predictor_rollout.py`
- `app/hdepic_lora_action_anticipation/eval.py`
- `app/hdepic_lora_action_anticipation/gaze.py`

## B5: Binary Input Adapter

用途：把 gaze disk 或 distance map 作为额外输入通道，在 RGB 进入 frozen encoder 前用小 adapter 做条件化。它和 overlay 的区别是：overlay 直接改变视频像素；binary input adapter 是一个可训练的 RGB+map 残差适配器。

运行脚本：

- `scripts/run_hdepic_lora_binary_input_adapter_train.slurm`: 主训练入口。
- `scripts/run_hdepic_lora_binary_input_adapter_distance_lr.slurm`: distance-map + adapter LR multiplier 变体。
- `scripts/run_hdepic_lora_binary_input_adapter_train_augaware_smoke.slurm`: aug-aware smoke / resource probe。
- `scripts/run_hdepic_lora_binary_input_adapter_zero_val.slurm`: zero-channel control。
- `scripts/run_hdepic_binary_input_adapter_diagnostics.slurm`: adapter-space / coverage diagnostics。
- `scripts/check_binary_adapter_latent_effect.slurm`: adapter 是否实际改变 latent 的检查。
- `scripts/submit_b5_distance_lrm05_light_checks.sh`: distance-lrm05 light-check matrix。

常用配置：

- `configs/generated/hdepic_lora_binary_input_adapter_radius64.yaml`
- `configs/generated/hdepic_lora_binary_input_adapter_radius64_gazefixed.yaml`
- `configs/generated/hdepic_lora_binary_input_adapter_radius64_gazefixed_trainzeromap.yaml`
- `configs/generated/hdepic_lora_binary_input_adapter_distance_lrmult05.yaml`
- `configs/generated/hdepic_lora_binary_input_adapter_augaware_smoke*.yaml`
- `configs/generated/hdepic_lora_binary_input_adapter_zero_val.yaml`

核心代码：

- `app/hdepic_lora_action_anticipation/binary_input_adapter.py`
  - `BinaryMapInputAdapter`
  - `BinaryInputAdaptedModel`
  - `train_one_epoch_with_binary_input_adapter`
  - per-head `tokens_proxy` gradient gating
- `app/hdepic_lora_action_anticipation/binary_map_utils.py`
- `app/hdepic_lora_action_anticipation/binary_map_aug.py`
- `app/hdepic_lora_action_anticipation/binary_input_adapter_diagnostics.py`
- `scripts/check_binary_adapter_latent_effect.py`
- `scripts/analyze_binary_adapter_channels.py`

相关稳定性代码：

- `app/hdepic_lora_action_anticipation/binary_input_adapter.py`
- `app/hdepic_lora_action_anticipation/gaze.py`

## B6: Future Latent Compare / Failure Analysis

用途：不只看 final metrics，而是拆解 observed latent、predicted future latent、oracle future latent、head selection、label-window rescore 等失败来源。

运行脚本：

- `scripts/run_hdepic_future_latent_compare.slurm`: observed / direct / AR / oracle latent 对比。
- `scripts/run_hdepic_future_latent_failure_modes.slurm`: native 1s/10s failure-mode diagnostics。
- `scripts/run_hdepic_lora_valonly_dump.slurm`: 导出 val prediction dump。
- `scripts/run_hdepic_rescore_window_cpu.slurm`: 对 prediction dump 做 future-window label rescore。

分析脚本：

- `scripts/analyze_future_latent_failure_modes.py`
- `scripts/analyze_prediction_dump.py`
- `scripts/rescore_window.py`
- `scripts/analyze_hdepic_label_priors.py`

常用配置：

- `configs/generated/future_latent_compare/hdepic_future_latent_compare.yaml`
- `configs/generated/future_latent_compare/path_y_b1_clean_10s.yaml`
- `configs/generated/future_latent_compare/path_y_b2_rnn_gaze_10s.yaml`
- `configs/generated/future_latent_compare/path_y_b5_binary_adapter_10s.yaml`
- `configs/generated/valonly_dump/*.yaml`

核心代码：

- `app/hdepic_lora_action_anticipation/future_latent_compare.py`
- `app/hdepic_lora_action_anticipation/modelcustom/vit_encoder_predictor_rollout.py`
- `scripts/analyze_future_latent_failure_modes.py`
- `scripts/rescore_window.py`

## B7: Long-History Gaze

用途：视频观察窗口不变，只延长 gaze history，测试更早的 gaze 是否包含 intention / task context。它和 B2 的区别是时间轴：B2 主要同窗口 gaze，B7 延长 gaze-only history。

运行脚本：

- `scripts/run_hdepic_lora_rnn_long_gaze_train.slurm`
  - 设置 `GAZE_RNN_HISTORY_SEC`。
  - 默认禁用 video-token conditioning。
  - 最后调用 `scripts/run_hdepic_lora_rnn_gaze_train.slurm` 共享路径。

常用配置：

- `configs/generated/hdepic_lora_rnn_long_gaze.yaml`

核心代码：

- `app/hdepic_lora_action_anticipation/gaze_rnn.py`
- `app/hdepic_lora_action_anticipation/gaze.py`
- `app/hdepic_lora_action_anticipation/eval.py`

## B8: Encoder-Output Gaze Injection

用途：把 gaze 注入点从 B2 的 probe-side 移到 encoder output 和 predictor input 之间。目标是让 gaze 影响 future-token prediction，而不是只影响最后的 classifier/pooler。

运行脚本：

- `scripts/run_hdepic_lora_encoder_gaze_inject_train.slurm`

常用配置：

- `configs/generated/hdepic_lora_encoder_gaze_inject.yaml`

核心代码：

- `app/hdepic_lora_action_anticipation/encoder_output_gaze_adapter.py`
  - `EncoderOutputGazeAdapter`
  - `EncoderOutputGazeAdaptedModel`
- `app/hdepic_lora_action_anticipation/gaze_rnn.py`
- `app/hdepic_lora_action_anticipation/eval.py`

## B10: SLAM Pose / Multimodal RNN Fuse

用途：把 SLAM closed-loop pose / head-motion trajectory 作为与 gaze 平行的模态，接入和 B2 类似的 late-fusion probe path。包含 pose-only 和 pose+gaze multimodal 两条路径。

运行脚本：

- `scripts/inspect_hdepic_slam_pose_data_cpu.slurm`: CPU audit / coverage scan。
- `scripts/inspect_hdepic_slam_pose_data.py`: 本地/CPU 分析脚本。
- `scripts/run_hdepic_lora_pose_smoke.slurm`: debug subset smoke。
- `scripts/run_hdepic_lora_rnn_pose_train.slurm`: pose-only RNN fuse。
- `scripts/run_hdepic_lora_rnn_multimodal_train.slurm`: gaze + pose multimodal RNN fuse。

常用配置：

- `configs/generated/hdepic_lora_rnn_pose.yaml`
- `configs/generated/hdepic_lora_rnn_pose_smoke.yaml`
- `configs/generated/hdepic_lora_rnn_multimodal.yaml`

核心代码：

- `app/hdepic_lora_action_anticipation/pose_slam.py`
  - session mapping
  - streaming zip read
  - clip-relative pose features: `pose_6d`, `pose_vel`, `pose_full`
- `app/hdepic_lora_action_anticipation/gaze_rnn.py`
  - `PoseTrajectoryLoader`
  - pose/gaze dual encoders
  - multimodal token concatenation
- `app/hdepic_lora_action_anticipation/eval.py`
- `app/hdepic_lora_action_anticipation/gaze.py`

## 数据准备与健康检查脚本

这些不是单独方法，但对复现实验很重要：

- `scripts/download_hdepic_data_cpu.slurm`: HD-EPIC data download。
- `scripts/download_hdepic_gaze_cpu.slurm`: SLAM-and-Gaze data download。
- `scripts/download_ek100_vitg384_inference_ckpts.sh`: EK100 / V-JEPA checkpoint download。
- `scripts/refresh_hdepic_vjepa_annotations.slurm`: refresh V-JEPA 格式 annotations。
- `scripts/convert_hdepic_to_vjepa_csv.py`: annotation conversion。
- `scripts/check_hdepic_video_health.py` / `scripts/check_hdepic_video_health_cpu.slurm`: video health check。
- `scripts/inspect_hdepic_gaze_data.py` / `scripts/inspect_hdepic_gaze_data_cpu.slurm`: gaze coverage / sync audit。
- `scripts/create_debug_subset.py` / `scripts/create_debug_subset_cpu.slurm`: deterministic debug subset。
- `configs/debug_subset_p01.json`: debug subset definition。

## 给合作者的阅读顺序

1. 先看本文档总览，确定感兴趣的方法对应哪些脚本和代码。
2. 要复现实验，优先看对应 `scripts/run_*.slurm`，再看它引用的 `configs/generated/*.yaml`。
3. 要理解模型改动，优先看 `app/hdepic_lora_action_anticipation/` 下对应模块。
4. 数据准备和健康检查只需要看本文档列出的脚本入口即可。
