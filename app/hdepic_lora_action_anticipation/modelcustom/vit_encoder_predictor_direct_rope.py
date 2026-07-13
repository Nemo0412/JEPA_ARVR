"""Single-step predictor @ long horizon with NTK temporal RoPE scaling (direct_rope).

One encoder forward + one predictor forward per sample (no AR rollout loop).
"""

from __future__ import annotations

import logging

import torch

from app.hdepic_lora_action_anticipation.rope_position_scaling import remap_mask_pair_ntk_temporal

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


def _predictor_trained_grid_depth(predictor, encoder) -> int:
    grid_depth = getattr(predictor, "grid_depth", None)
    if grid_depth is not None:
        return int(grid_depth)
    num_patches = int(getattr(predictor, "num_patches", 0))
    spatial = (encoder.grid_height * encoder.grid_width) if hasattr(encoder, "grid_height") else None
    if num_patches > 0 and spatial:
        return max(int(num_patches // spatial), 1)
    pretrain_frames = int(getattr(predictor, "num_frames", encoder.num_frames))
    tubelet = int(encoder.tubelet_size)
    return max(pretrain_frames // tubelet, 1)


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

    rope_mode = wrapper_kwargs.get("rope_scale_mode", "ntk_temporal")
    model = DirectRopeAnticipativeWrapper(
        encoder=encoder,
        predictor=predictor,
        frames_per_second=frames_per_second,
        crop_size=resolution,
        patch_size=encoder.patch_size,
        tubelet_size=encoder.tubelet_size,
        rope_scale_mode=rope_mode,
        **{k: v for k, v in wrapper_kwargs.items() if k != "rope_scale_mode"},
    )
    model.embed_dim = encoder.embed_dim
    logger.info(
        "[direct-rope] DirectRopeAnticipativeWrapper rope_scale_mode=%s | "
        "single predictor step (NOT AR rollout)",
        rope_mode,
    )
    if hasattr(predictor, "hierarchical_layers") and len(predictor.hierarchical_layers) > 1:
        encoder.return_hierarchical = True
    return model


class DirectRopeAnticipativeWrapper(torch.nn.Module):
    """concat_ar-style wrapper with optional NTK temporal RoPE remap for 10s horizons."""

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
        rope_scale_mode="ntk_temporal",
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
        self.rope_scale_mode = str(rope_scale_mode or "").strip().lower()
        self.spatial_tokens = int(self.grid_size**2)

        assert not (self.no_predictor and self.no_encoder), "Anticipative wrapper must use predictor or encoder"

    def forward(self, x, anticipation_times):
        x_full = self.encoder(x)
        if self.no_predictor:
            return x_full

        B, N, D_full = x_full.size()
        embed_dim = self.encoder.embed_dim
        use_hierarchical = D_full > embed_dim
        x = x_full[:, :, -embed_dim:] if use_hierarchical else x_full

        if self.no_encoder:
            x_accumulate = torch.rand(B, 0, embed_dim, device=x.device)
        else:
            x_accumulate = x.clone()

        ctxt_positions = torch.arange(N, device=x.device).unsqueeze(0).repeat(B, 1)
        anticipation_steps = (anticipation_times * self.frames_per_second / self.tubelet_size).to(torch.int64)
        skip_positions = N + self.spatial_tokens * anticipation_steps
        n_pred = int(self.spatial_tokens * (self.num_output_frames // self.tubelet_size))
        tgt_positions = torch.arange(n_pred, device=x.device).unsqueeze(0).repeat(B, 1)
        tgt_positions = tgt_positions + skip_positions.unsqueeze(1)

        x_pred_input = x_full
        for _ in range(self.num_steps):
            masks_x = ctxt_positions
            masks_y = tgt_positions
            rope_scale = 1.0
            if self.rope_scale_mode == "ntk_temporal":
                trained_grid_depth = _predictor_trained_grid_depth(self.predictor, self.encoder)
                masks_x, masks_y, rope_scale = remap_mask_pair_ntk_temporal(
                    masks_x, masks_y, self.spatial_tokens, trained_grid_depth
                )
            pred_out = self.predictor(x_pred_input, masks_x=masks_x, masks_y=masks_y)
            x_pred_full = pred_out[0] if isinstance(pred_out, tuple) else pred_out
            x_pred = x_pred_full[:, :, -embed_dim:] if x_pred_full.size(-1) != embed_dim else x_pred_full
            x_accumulate = torch.cat([x_accumulate, x_pred], dim=1)
            x_pred_for_input = x_pred_full if x_pred_full.size(-1) == x_pred_input.size(-1) else x_pred
            x_pred_input = torch.cat([x_pred_input[:, n_pred:, :], x_pred_for_input], dim=1)

        return x_accumulate
