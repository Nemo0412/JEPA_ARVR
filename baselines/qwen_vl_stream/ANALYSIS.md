# Why V-JEPA (encoder) often beats decoder Qwen-VL on this task

Setting: HD-EPIC P01 **streaming** anticipation (+2/+4/+6s), closed-set
verb/noun/action accuracy.

Baseline here: **Qwen2-VL-2B** (official smallest Qwen-VL, ~2B ≈ ViT 675M + LLM 1.5B).
Still ~5× V-JEPA vanilla (~400M); no official ~400M Qwen-VL exists.

1. **Task head**: JEPA uses CE classification heads; Qwen is a generative LM
   with an attached probe — objective mismatch.
2. **Latent prediction**: JEPA predictor targets future visual latents with an
   anticipation-time condition; Qwen encodes frames for language decoding.
3. **Temporal budget**: JEPA keeps dense tubelet tokens over 4–10s context;
   Qwen must subsample to ~8 frames or explode token/compute cost.
4. **Where parameters sit**: JEPA ~400M mostly vision+predictor; Qwen-2B mostly
   language decoder — suboptimal allocation for egocentric action forecasting.
5. **Protocol fit**: Communicating MTP multi-horizon is native to the JEPA
   stack; decoder VLMs lack a first-class “+t seconds” pathway.

Qwen remains preferable for open-vocab language explanations; not for this
closed-set streaming metric.
