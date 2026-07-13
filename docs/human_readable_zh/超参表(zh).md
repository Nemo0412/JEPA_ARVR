# JEPA_ARVR(对方代码)对齐超参表 — 中文快照

> 这是给人看的快照副本，**不会实时维护**。需要最新版时看
> VJEPA2-EXP 仓库的 `docs/HYPERPARAMETER_REFERENCE.md`(agent 维护的英文超参权威版，按含义组织)，
> 对齐叙述/run 分类看 VJEPA2-EXP 仓库的 `docs/PHD_CODE_COMPARISON.md`。
> 生成于 2026-06-18，对照 JEPA_ARVR submodule commit `add6d34`。
> 迁移到共享仓库日期：2026-06-20。

## 一句话结论

对方(Leshu Li / Nemo0412 的 JEPA_ARVR)在 `add6d34` 里给参考实现**也加了
encoder-LoRA**(之前只是冻结编码器 + 从头训 probe 的 baseline)。我们的
encoder-LoRA 路径与这个新协议**核心完全对齐**;剩下的差异都是有意为之。

## encoder-LoRA 超参表

下面的"我方默认"是 2026-06-18 改完默认值后,**裸调用 / 最简配置**就能得到的对齐值。

| 名称 | 我方默认 | 参考 | 对齐? | 解释 |
| --- | --- | --- | --- | --- |
| `rank` | 8 | 8 | ✅ | 低秩适配器的秩。 |
| `alpha` | 16.0 | 16.0 | ✅ | 缩放,有效 `ΔW` 乘 `alpha/rank`(=2.0)。 |
| `dropout` | 0.05 | 0.0 | ⚠️ 有意 | LoRA 输入上的 dropout。参考没有;要严格一致就设 0.0。 |
| `last_n_blocks` | 0(=全部) | 全部 | ✅ | `<=0` 表示注入每一个 transformer block。 |
| `target_suffixes` | `attn.qkv, attn.proj` | `attn.qkv, attn.proj` | ✅ | 只注入注意力的 qkv+proj,**不碰 MLP**。 |
| `lr_mult` | 0.5 | 0.5(5e-5 vs 1e-4) | ✅ | LoRA 学习率 = `lr_mult × probe 学习率`。 |
| `weight_decay` | 1e-4 | 1e-4 | ✅ | AdamW 权重衰减。 |
| `activation_checkpointing` | True | 无 | 无关 | 只影响显存/算力,不影响结果。 |

## 主干 / 训练协议(由 config 决定,不在 LoRA 模块里)

| 名称 | 对齐值 | 参考 | 备注 |
| --- | --- | --- | --- |
| 编码器 | `vit_large` / `vitl.pt` | ViT-L `vit_large_rope` | ✅ 已从 ViT-g/384 迁回 ViT-L。 |
| 分辨率 | 256 | 256 | ✅ |
| 精度 `use_bfloat16` | false(fp32) | fp32 | ✅ 参考无 AMP。bf16 路径存在但属有意分歧(需要禁用 GradScaler 的修复)。 |
| `pretrained_probe` | `''`(从头训) | 从头训 | ✅ |
| 学习率/epoch/bs/warmup | 1e-4 / 10 / 8 / 2 | 1e-4 / 10 / 8 / 2 | ✅ |
| `align_reference_metrics` | true | 无 | 把日志里的 acc→Top-3、recall→类均值 Recall@5。 |
| `temporal_sampling` | `phd_reference`(默认) | `obs_end=动作起点-horizon` | ✅ 运行时补丁强制 `anticipation_point=[1,1]` 且 `训练 horizon=验证 horizon`,**覆盖 config 里写的 [0,0]**。设 `legacy` 可退回。 |
| `class_space` | `phd_reference`(默认) | 完整 HD-EPIC 词表 | ✅ 设 `train_only` 可退回。 |
| 数据划分 | `p01_fixed` | 日期划分(20240203 训练) | ✅ p01_fixed = 项目固定的日期划分 + 额外 test。`legacy` = 随机/参与者划分。 |

predictor-LoRA 是**项目独有**功能(参考的 probe 流程没有 predictor rollout),
无对齐对象;其默认值已与 encoder-LoRA 对齐集保持一致。

## 最近几天的 run 对齐情况(2026-06-15 ~ 06-18)

| run / 配置 | 对齐? | 不对齐时的差异 |
| --- | --- | --- |
| `…_p01fixed_rgbonly_probeonly`(上游) | ✅ | 冻结编码器 probe-only = 参考 `LORA_RANK=0` baseline,p01_fixed。 |
| `ltm-s1-*-probeonly-{h1s,h3p5s}`(LTM) | ✅ 协议对齐 | horizon 扩展:参考只验证过 1s,3.5s 是 LTM 有意扩展。subset/fulltest 是实验数据切片,非全 P01。 |
| `ltm-s1-*-encoderlora-diag-h3p5s`(LTM) | ✅ 协议对齐 | 同上 horizon/数据切片;encoder-LoRA 参数对齐。 |
| `ltm_oracle_compare`(LTM) | ✅ 协议对齐 | oracle/天花板诊断;显式套用 phd_reference 补丁。 |
| `…_legacy_rgbonly`(上游) | ❌ 有意 | submit 显式设 `LORA_TEMPORAL_SAMPLING=legacy` + `LORA_CLASS_SPACE=train_only` + legacy 划分,复现历史 job 10847438。LoRA 参数本身是对齐的。 |
| `…_legacy_gazepose_reg`(上游) | ❌ 有意 | 同样 legacy 覆盖 + gaze/pose 融合 + 输出正则(超出参考的项目功能)。 |
| `…_legacy_rgbonly_predictorlora`(上游) | ❌ 有意 | 同样 legacy 覆盖 + predictor-LoRA(项目独有,无参考对应)。 |

**总结:** 所有**非 legacy** 的近期 run 都协议对齐。`legacy_*` 是有意复现旧 baseline,
不是误漂移。对齐的 LTM run 里唯一的非参考变量是预测时长(1s/3.5s vs 参考 1s)和
subset/fulltest 数据切片。
