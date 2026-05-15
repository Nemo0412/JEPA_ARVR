# V-JEPA2 action anticipation LoRA probe ver

数据：

- HD-EPIC P01
- 转换后 7384 条样本
- 随机 video-level split
- train: 5744 rows / 22 videos
- val: 1640 rows / 5 videos

模型/配置：

- 运行环境: NYU HPC L40S
- 运行时间: 5~6 h/epoch
- encoder: V-JEPA2 ViT-g/384, checkpoint = vitg-384.pt
- 输入：32 frames, 8 FPS, 384 resolution
- 任务：1s action anticipation
- backbone/predictor 冻结
- 复用 EK100 ViT-g/384 action anticipation probe 的 pooler 权重
- 丢弃 EK100 原分类头，换成 HD-EPIC 的 verb/noun/action heads
- LoRA 加在 probe/pooler 里的 Linear 层上
- trainable params 大概 2.5M / 95.8M，也就是 2.62%
- batch size = 1
- epochs = 3
- metrics: Top-3 accuracy + class-mean Recall@5，和你的实现对齐

结果，epoch 3：

- val action Top-3: 44.7%
- val action Recall@5: 30.6%
- val verb Top-3: 74.6%
- val verb Recall@5: 45.7%
- val noun Top-3: 64.9%
- val noun Recall@5: 50.7%

训练趋势：

- val action Top-3: 30.4 -> 41.5 -> 44.7
- val action Recall@5: 13.5 -> 25.4 -> 30.6

## 训练过程详细数据

- epoch 1: train action Top-3 23.2%, val action Top-3 30.4%, val action Recall@5 13.5%
- epoch 2: train action Top-3 37.3%, val action Top-3 41.5%, val action Recall@5 25.4%
- epoch 3: train action Top-3 46.3%, val action Top-3 44.7%, val action Recall@5 30.6%

### epoch 3 详细结果

train acc  action/verb/noun: 46.3% / 67.6% / 60.3%
train recall action/verb/noun: 30.6% / 39.7% / 45.7%

val acc    action/verb/noun: 44.7% / 74.6% / 64.9%
val recall action/verb/noun: 30.6% / 45.7% / 50.7%
