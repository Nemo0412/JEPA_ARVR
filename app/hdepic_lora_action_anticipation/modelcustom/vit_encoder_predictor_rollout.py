"""Autoregressive predictor wrapper for HD-EPIC long-horizon validation.

This module mirrors the upstream action-anticipation modelcustom entry point,
but keeps the implementation outside ``vjepa2/``. The direct upstream wrapper
asks the predictor for absolute future positions in one call; that fails when
the requested horizon exceeds the predictor positional table. This wrapper
instead predicts one valid next chunk at a time and rolls that chunk into a
local context window.
"""

from __future__ import annotations

import logging

import torch

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def _get_model_modules(pretrain_kwargs):
    use_v2_1 = pretrain_kwargs.get("use_v2_1", False)
    if use_v2_1:
        import app.vjepa_2_1.models.predictor as vit_pred
        import app.vjepa_2_1.models.vision_transformer as vit
    else:
        import src.models.predictor as vit_pred
        import src.models.vision_transformer as vit
    return vit, vit_pred


def init_module(
    frames_per_clip: int,
    frames_per_second: int,
    resolution: int,
    checkpoint: str,
    model_kwargs: dict,
    wrapper_kwargs: dict,
    **kwargs,
):
    logger.info("Loading pretrained model from %s", checkpoint)
    checkpoint_data = torch.load(checkpoint, map_location="cpu")
    vit, vit_pred = _get_model_modules(model_kwargs)

    enc_kwargs = model_kwargs["encoder"]
    enc_ckp_key = enc_kwargs.get("checkpoint_key")
    enc_model_name = enc_kwargs.get("model_name")
    encoder = vit.__dict__[enc_model_name](
        img_size=resolution,
        num_frames=frames_per_clip,
        **enc_kwargs,
    )
    pretrained_dict = checkpoint_data[enc_ckp_key]
    pretrained_dict = {k.replace("module.", ""): v for k, v in pretrained_dict.items()}
    pretrained_dict = {k.replace("backbone.", ""): v for k, v in pretrained_dict.items()}
    for k, v in encoder.state_dict().items():
        if k not in pretrained_dict:
            logger.info('encoder key "%s" could not be found in loaded state dict', k)
        elif pretrained_dict[k].shape != v.shape:
            logger.info('encoder key "%s" shape mismatch; keeping initialized value', k)
            pretrained_dict[k] = v
    msg = encoder.load_state_dict(pretrained_dict, strict=False)
    logger.info("loaded pretrained encoder with msg: %s", msg)

    prd_kwargs = model_kwargs["predictor"]
    prd_ckp_key = prd_kwargs.get("checkpoint_key")
    prd_model_name = prd_kwargs.get("model_name")
    teacher_embed_dim = prd_kwargs.get("teacher_embed_dim")
    n_output_distillation = prd_kwargs.get("n_output_distillation", 4)
    prd_out_embed_dim = teacher_embed_dim // n_output_distillation if teacher_embed_dim is not None else None
    predictor = vit_pred.__dict__[prd_model_name](
        img_size=resolution,
        embed_dim=encoder.embed_dim,
        patch_size=encoder.patch_size,
        tubelet_size=encoder.tubelet_size,
        out_embed_dim=prd_out_embed_dim,
        **prd_kwargs,
    )
    pretrained_dict = checkpoint_data[prd_ckp_key]
    pretrained_dict = {k.replace("module.", ""): v for k, v in pretrained_dict.items()}
    pretrained_dict = {k.replace("backbone.", ""): v for k, v in pretrained_dict.items()}
    for k, v in predictor.state_dict().items():
        if k not in pretrained_dict:
            logger.info('predictor key "%s" could not be found in loaded state dict', k)
        elif pretrained_dict[k].shape != v.shape:
            logger.info('predictor key "%s" shape mismatch; keeping initialized value', k)
            pretrained_dict[k] = v
    msg = predictor.load_state_dict(pretrained_dict, strict=False)
    logger.info("loaded pretrained predictor with msg: %s", msg)

    model = AutoregressiveAnticipativeWrapper(
        encoder=encoder,
        predictor=predictor,
        frames_per_second=frames_per_second,
        crop_size=resolution,
        patch_size=encoder.patch_size,
        tubelet_size=encoder.tubelet_size,
        **wrapper_kwargs,
    )
    model.embed_dim = encoder.embed_dim
    return_mode = wrapper_kwargs.get("return_mode", "observed_plus_target")
    num_steps = int(wrapper_kwargs.get("num_steps", 1))
    if num_steps > 1:
        max_adv_sec = (
            max(1, (getattr(predictor, "num_frames", 64) // encoder.tubelet_size) - max(1, (frames_per_clip // encoder.tubelet_size)))
            * encoder.tubelet_size
            / float(frames_per_second)
        )
        logger.info(
            "[ar-rollout] sliding-window mode num_steps=%s (~%.2fs max advance/step); "
            "NOT fine-grained 0.25s/chunk rollout (which needs ~41 steps @10s)",
            num_steps,
            max_adv_sec,
        )
    logger.info(
        "[ar-rollout] AutoregressiveAnticipativeWrapper return_mode=%s | "
        "final_window=probe gets last-N sliding window after AR (JEPA_ARVR anticipate_features); "
        "observed_plus_target=encoder full + final target chunk (legacy default); "
        "observed_plus_rollout=encoder full + all rollout chunks",
        return_mode,
    )
    if hasattr(predictor, "hierarchical_layers") and len(predictor.hierarchical_layers) > 1:
        encoder.return_hierarchical = True
    return model


class AutoregressiveAnticipativeWrapper(torch.nn.Module):
    """Roll the predictor forward in local chunks before classifier pooling."""

    def __init__(
        self,
        encoder,
        predictor,
        frames_per_second=4,
        crop_size=224,
        patch_size=16,
        tubelet_size=2,
        no_predictor=False,
        num_output_frames=2,
        num_steps=1,
        no_encoder=False,
        rollout_stride_chunks=1,
        return_mode="observed_plus_target",
        max_rollout_steps=512,
    ):
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor
        self.grid_size = crop_size // patch_size
        self.tubelet_size = tubelet_size
        self.no_predictor = no_predictor
        self.num_output_frames = max(num_output_frames, tubelet_size)
        self.frames_per_second = frames_per_second
        self.num_steps = num_steps
        self.no_encoder = no_encoder
        self.rollout_stride_chunks = int(rollout_stride_chunks)
        self.return_mode = return_mode
        self.max_rollout_steps = int(max_rollout_steps)

        assert not (self.no_predictor and self.no_encoder), "Anticipative wrapper must use predictor or encoder"
        if self.rollout_stride_chunks != 1:
            raise ValueError("Only rollout_stride_chunks=1 is supported in the first AR prototype")
        if self.return_mode not in {"observed_plus_target", "target_only", "final_window", "observed_plus_rollout"}:
            raise ValueError(f"Unsupported return_mode={self.return_mode}")

    def _max_advance_slots(self, context_tokens: int) -> int:
        spatial_tokens = int(self.grid_size**2)
        window_slots = max(1, context_tokens // spatial_tokens)
        pred_num_frames = int(getattr(self.predictor, "num_frames", 64))
        cap_slots = pred_num_frames // self.tubelet_size
        return max(1, cap_slots - window_slots)

    def _forward_sliding_window(self, x_full, anticipation_times, observed_for_classifier):
        """JEPA_ARVR-style coarse AR: K predictor forwards, ~4s max advance per step."""
        B, N, _ = x_full.size()
        device = x_full.device
        embed_dim = self.encoder.embed_dim
        spatial_tokens = int(self.grid_size**2)
        max_adv = self._max_advance_slots(N)

        total_slabs = (
            anticipation_times.float() * self.frames_per_second / self.tubelet_size
        ).round().to(torch.int64).clamp(min=1)
        k_steps = max(int(self.num_steps), 1)
        k_steps = max(k_steps, int(((total_slabs.max() + max_adv - 1) // max_adv).item()))

        ctx = x_full
        ctx_pos = torch.arange(N, device=device).unsqueeze(0).expand(B, -1)
        advanced = torch.zeros(B, dtype=torch.int64, device=device)
        rollout_for_classifier = []
        target_by_sample = [None for _ in range(B)]

        for step_idx in range(1, k_steps + 1):
            target_slabs = (total_slabs.float() * step_idx / k_steps).round().to(torch.int64)
            adv = (target_slabs - advanced).clamp(min=0)
            adv = torch.minimum(adv, torch.full_like(adv, max_adv))
            adv_max = int(adv.max().item())
            if adv_max <= 0:
                continue

            n_pred = spatial_tokens * adv_max
            tgt_pos = torch.arange(n_pred, device=device).unsqueeze(0).expand(B, -1) + N
            pred_out = self.predictor(ctx, masks_x=ctx_pos, masks_y=tgt_pos)
            pred_full = pred_out[0] if isinstance(pred_out, tuple) else pred_out
            pred_for_classifier = (
                pred_full[:, :, -embed_dim:] if pred_full.size(-1) != embed_dim else pred_full
            )
            pred_for_input = pred_full if pred_full.size(-1) == ctx.size(-1) else pred_for_classifier

            for b in range(B):
                adv_b = int(adv[b].item())
                if adv_b <= 0:
                    continue
                n_pred_b = spatial_tokens * adv_b
                pred_b = pred_for_input[b : b + 1, :n_pred_b, :]
                pred_cls_b = pred_for_classifier[b : b + 1, :n_pred_b, :]
                rollout_for_classifier.append(pred_cls_b)
                full = torch.cat([ctx[b : b + 1], pred_b], dim=1)
                ctx[b : b + 1] = full[:, -N:, :]
                advanced[b] += adv_b
                if advanced[b] >= total_slabs[b]:
                    target_by_sample[b] = pred_cls_b

        if self.return_mode == "final_window":
            return ctx[:, :, -embed_dim:] if ctx.size(-1) != embed_dim else ctx
        if self.return_mode == "observed_plus_rollout":
            return torch.cat([observed_for_classifier, *rollout_for_classifier], dim=1)

        final_window = ctx[:, :, -embed_dim:] if ctx.size(-1) != embed_dim else ctx
        target_candidates = [t for t in target_by_sample if t is not None]
        target_tokens = final_window
        if len(target_candidates) == B and len({t.size(1) for t in target_candidates}) == 1:
            target_tokens = torch.cat(target_candidates, dim=0)

        if self.return_mode == "target_only":
            return target_tokens
        return torch.cat([observed_for_classifier, target_tokens], dim=1)

    def forward(self, x, anticipation_times):
        x_full = self.encoder(x)
        if self.no_predictor:
            return x_full

        B, N, D_full = x_full.size()
        embed_dim = self.encoder.embed_dim
        use_hierarchical = D_full > embed_dim
        x_last_layer = x_full[:, :, -embed_dim:] if use_hierarchical else x_full

        if self.no_encoder:
            observed_for_classifier = torch.rand(B, 0, embed_dim, device=x.device)
        else:
            observed_for_classifier = x_last_layer.clone()

        if int(self.num_steps) > 1:
            return self._forward_sliding_window(x_full, anticipation_times, observed_for_classifier)

        spatial_tokens = int(self.grid_size**2)
        chunk_tokens = int(spatial_tokens * (self.num_output_frames // self.tubelet_size))
        if chunk_tokens <= 0 or chunk_tokens > N:
            raise ValueError(f"Invalid rollout chunk_tokens={chunk_tokens}; context tokens={N}")

        local_ctxt_positions = torch.arange(N, device=x.device).unsqueeze(0).repeat(B, 1)
        local_tgt_positions = torch.arange(chunk_tokens, device=x.device).unsqueeze(0).repeat(B, 1)
        local_tgt_positions += N

        horizon_chunks = (anticipation_times * self.frames_per_second / self.tubelet_size).to(torch.int64)
        rollout_steps = (horizon_chunks + (self.num_output_frames // self.tubelet_size)).clamp(min=1)
        max_steps = int(rollout_steps.max().item())
        if max_steps > self.max_rollout_steps:
            raise ValueError(f"Requested {max_steps} rollout steps, above max_rollout_steps={self.max_rollout_steps}")

        x_window = x_full
        target_by_sample = [None for _ in range(B)]
        rollout_for_classifier = []

        for step in range(max_steps):
            pred_out = self.predictor(
                x_window,
                masks_x=local_ctxt_positions,
                masks_y=local_tgt_positions,
            )
            pred_full = pred_out[0] if isinstance(pred_out, tuple) else pred_out
            pred_for_classifier = pred_full[:, :, -embed_dim:] if pred_full.size(-1) != embed_dim else pred_full
            rollout_for_classifier.append(pred_for_classifier)

            for b in range(B):
                if step == int(rollout_steps[b].item()) - 1:
                    target_by_sample[b] = pred_for_classifier[b : b + 1]

            pred_for_input = pred_full if pred_full.size(-1) == x_window.size(-1) else pred_for_classifier
            x_window = torch.cat([x_window[:, chunk_tokens:, :], pred_for_input], dim=1)

        target_tokens = torch.cat(target_by_sample, dim=0)
        final_window = x_window[:, :, -embed_dim:] if x_window.size(-1) != embed_dim else x_window

        if self.return_mode == "target_only":
            return target_tokens
        if self.return_mode == "final_window":
            return final_window
        if self.return_mode == "observed_plus_rollout":
            return torch.cat([observed_for_classifier, *rollout_for_classifier], dim=1)
        return torch.cat([observed_for_classifier, target_tokens], dim=1)
