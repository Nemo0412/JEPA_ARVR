"""Frozen Hugging Face VLM/vision encoder baseline for HD-EPIC anticipation.

This module matches the action_anticipation_frozen modelcustom contract:
``forward(video, anticipation_times) -> [B, tokens, D]`` and exposes
``embed_dim``. It intentionally ignores ``anticipation_times`` because the
baseline replaces V-JEPA2 encoder/predictor with observed-clip VLM features.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _as_tuple3(value: Any, default: tuple[float, float, float]) -> tuple[float, float, float]:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        f = float(value)
        return (f, f, f)
    if len(value) != 3:
        raise ValueError(f"Expected 3 channel values, got {value!r}")
    return (float(value[0]), float(value[1]), float(value[2]))


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _dtype_from_name(name: str | None):
    if not name or str(name).lower() in {"auto", "none"}:
        return None
    table = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    key = str(name).lower()
    if key not in table:
        raise ValueError(f"Unsupported torch_dtype={name!r}; expected one of {sorted(table)}")
    return table[key]


def init_module(
    frames_per_clip: int,
    frames_per_second: int,
    resolution: int,
    checkpoint: str | None,
    model_kwargs: dict,
    wrapper_kwargs: dict,
    **kwargs,
):
    del frames_per_second, kwargs
    cfg = dict(model_kwargs or {})
    model_id = cfg.get("model_id") or checkpoint
    if not model_id:
        raise ValueError("VLM baseline requires model_kwargs.pretrain_kwargs.model_id or model_kwargs.checkpoint")

    wrapper = dict(wrapper_kwargs or {})
    num_frames = int(wrapper.pop("num_frames", cfg.pop("num_frames", min(8, int(frames_per_clip)))))
    token_mode = str(wrapper.pop("token_mode", cfg.pop("token_mode", "pooled"))).lower()
    if token_mode not in {"pooled", "patch", "cls_patch"}:
        raise ValueError(f"Unsupported VLM token_mode={token_mode!r}; expected pooled, patch, or cls_patch")

    image_size = int(wrapper.pop("image_size", cfg.pop("image_size", resolution)))
    trust_remote_code = _truthy(cfg.pop("trust_remote_code", False))
    torch_dtype = _dtype_from_name(cfg.pop("torch_dtype", None))

    return VLMVideoEncoder(
        model_id=str(model_id),
        num_frames=num_frames,
        image_size=image_size,
        token_mode=token_mode,
        trust_remote_code=trust_remote_code,
        torch_dtype=torch_dtype,
        input_mean=_as_tuple3(cfg.pop("input_mean", None), IMAGENET_MEAN),
        input_std=_as_tuple3(cfg.pop("input_std", None), IMAGENET_STD),
        image_mean=cfg.pop("image_mean", None),
        image_std=cfg.pop("image_std", None),
        model_class=str(cfg.pop("model_class", "auto")),
        local_files_only=_truthy(cfg.pop("local_files_only", False)),
        gpu_pulse_iters=int(cfg.pop("gpu_pulse_iters", 0) or 0),
        gpu_pulse_size=int(cfg.pop("gpu_pulse_size", 512) or 512),
        mllama_chunk_size=int(wrapper.pop("mllama_chunk_size", cfg.pop("mllama_chunk_size", 8)) or 8),
    )


class VLMVideoEncoder(torch.nn.Module):
    def __init__(
        self,
        model_id: str,
        num_frames: int,
        image_size: int,
        token_mode: str,
        trust_remote_code: bool,
        torch_dtype,
        input_mean: tuple[float, float, float],
        input_std: tuple[float, float, float],
        image_mean,
        image_std,
        model_class: str = "auto",
        local_files_only: bool = False,
        gpu_pulse_iters: int = 0,
        gpu_pulse_size: int = 512,
        mllama_chunk_size: int = 8,
    ):
        super().__init__()
        try:
            from transformers import AutoConfig, AutoImageProcessor, AutoModel, CLIPModel, SiglipModel
        except ImportError as exc:
            raise ImportError("VLM baseline requires transformers; it is listed in vjepa2/requirements.txt") from exc

        self.model_id = model_id
        self.num_frames = max(1, int(num_frames))
        self.image_size = int(image_size)
        self.token_mode = token_mode
        self.model_class = str(model_class).lower()
        self.gpu_pulse_iters = max(0, int(gpu_pulse_iters))
        self.gpu_pulse_size = max(1, int(gpu_pulse_size))

        processor = AutoImageProcessor.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
        )
        cfg = AutoConfig.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
        )
        model_kwargs = {
            "trust_remote_code": trust_remote_code,
            "local_files_only": local_files_only,
        }
        if torch_dtype is not None:
            model_kwargs["torch_dtype"] = torch_dtype
        if self.model_class == "clip":
            model = CLIPModel.from_pretrained(model_id, **model_kwargs)
        elif self.model_class in {"siglip", "siglip2"}:
            model = SiglipModel.from_pretrained(model_id, **model_kwargs)
        elif self.model_class == "mllama":
            # Llama 3.2 Vision splits each image into aspect-ratio-dependent tiles
            # and requires aspect_ratio_ids/aspect_ratio_mask alongside pixel_values
            # (see MllamaVisionModel.forward); it has no plain pixel_values-only
            # path like CLIP/SigLIP. We force every frame to a single 1x1-aspect
            # tile (aspect_ratio_id=1, the model's reserved id for [1, 1]) instead
            # of reproducing Meta's multi-tile image splitting, so each frame maps
            # to exactly one tile at the model's native image_size. Loading the
            # bare MllamaVisionModel (base_model_prefix="vision_model") instead of
            # MllamaForConditionalGeneration skips materializing the ~8B-parameter
            # language model.
            from transformers import MllamaVisionModel

            model = MllamaVisionModel.from_pretrained(model_id, **model_kwargs)
            vision_cfg = getattr(cfg, "vision_config", None) or cfg
            native_image_size = int(getattr(vision_cfg, "image_size", self.image_size))
            if native_image_size != self.image_size:
                logger.warning(
                    "Mllama vision tower uses fixed precomputed tile position embeddings "
                    "at image_size=%d; overriding configured image_size=%d to match.",
                    native_image_size,
                    self.image_size,
                )
                self.image_size = native_image_size
            # The tile/pre-/post-tile positional embeddings are parameterized over
            # exactly max_num_tiles slots (4 for this checkpoint); pixel_values must
            # be padded to that many tile slots regardless of how many tiles we
            # actually use, with aspect_ratio_mask marking the rest as padding.
            self.mllama_max_num_tiles = int(getattr(vision_cfg, "max_num_tiles", 4))
            self.mllama_chunk_size = max(1, int(mllama_chunk_size))
        elif self.model_class == "auto":
            model = AutoModel.from_pretrained(model_id, **model_kwargs)
        else:
            raise ValueError("model_class must be one of: auto, clip, siglip, mllama")

        self.model = model
        self.processor_size = getattr(processor, "size", None)
        self.register_buffer("input_mean", torch.tensor(input_mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("input_std", torch.tensor(input_std).view(1, 3, 1, 1), persistent=False)
        self.register_buffer(
            "image_mean",
            torch.tensor(_as_tuple3(image_mean or getattr(processor, "image_mean", None), IMAGENET_MEAN)).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "image_std",
            torch.tensor(_as_tuple3(image_std or getattr(processor, "image_std", None), IMAGENET_STD)).view(1, 3, 1, 1),
            persistent=False,
        )
        if self.gpu_pulse_iters > 0:
            pulse = torch.randn(self.gpu_pulse_size, self.gpu_pulse_size) / (self.gpu_pulse_size**0.5)
            self.register_buffer("gpu_pulse_matrix", pulse, persistent=False)

        if self.model_class == "mllama":
            # Mllama's vision tower concatenates the final hidden state with five
            # intermediate-layer hidden states (vision_output_dim=hidden_size*6,
            # e.g. 1280*6=7680), so embed_dim != vision_config.hidden_size here
            # unlike CLIP/SigLIP. See MllamaVisionModel.forward.
            vision_cfg = getattr(cfg, "vision_config", None) or cfg
            self.embed_dim = int(vision_cfg.vision_output_dim)
        else:
            self.embed_dim = self._infer_embed_dim(cfg, model, token_mode=token_mode)
        logger.info(
            "Initialized VLM baseline: model_id=%s model_class=%s num_frames=%d image_size=%d token_mode=%s embed_dim=%d gpu_pulse_iters=%d",
            model_id,
            self.model_class,
            self.num_frames,
            self.image_size,
            self.token_mode,
            self.embed_dim,
            self.gpu_pulse_iters,
        )

    @staticmethod
    def _infer_embed_dim(cfg, model, token_mode: str) -> int:
        vision_cfgs = (
            getattr(cfg, "vision_config", None),
            getattr(getattr(model, "config", None), "vision_config", None),
            cfg,
            getattr(model, "config", None),
        )
        if token_mode != "pooled":
            for obj in vision_cfgs:
                if obj is None:
                    continue
                value = getattr(obj, "hidden_size", None)
                if value is not None:
                    return int(value)
        for obj in (
            cfg,
            getattr(model, "config", None),
            getattr(cfg, "vision_config", None),
            getattr(getattr(model, "config", None), "vision_config", None),
        ):
            if obj is None:
                continue
            for attr in ("projection_dim", "hidden_size", "vision_embed_dim", "embed_dim"):
                value = getattr(obj, attr, None)
                if value is not None:
                    return int(value)
        raise ValueError("Could not infer VLM feature dimension from model config")

    def _select_frames(self, x: torch.Tensor) -> torch.Tensor:
        bsz, channels, frames, height, width = x.shape
        if channels != 3:
            x = x[:, :3]
        if frames == self.num_frames:
            return x
        idx = torch.linspace(0, frames - 1, self.num_frames, device=x.device).round().long()
        return x.index_select(2, idx)

    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        x = self._select_frames(x)
        bsz, channels, frames, height, width = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(bsz * frames, channels, height, width)
        x = x.float() * self.input_std.to(x.dtype) + self.input_mean.to(x.dtype)
        x = x.clamp_(0.0, 1.0)
        if height != self.image_size or width != self.image_size:
            x = F.interpolate(x, size=(self.image_size, self.image_size), mode="bicubic", align_corners=False)
            x = x.clamp_(0.0, 1.0)
        x = (x - self.image_mean.to(x.dtype)) / self.image_std.to(x.dtype)
        return x

    def _vision_model(self):
        if hasattr(self.model, "vision_model"):
            return self.model.vision_model
        if hasattr(self.model, "vision_tower"):
            return self.model.vision_tower
        if hasattr(self.model, "get_vision_tower"):
            return self.model.get_vision_tower()
        return None

    def _unwrap_image_features(self, outputs: Any) -> torch.Tensor:
        """Normalize old and new Transformers get_image_features return types."""
        if torch.is_tensor(outputs):
            return outputs

        image_features = getattr(outputs, "image_embeds", None)
        if torch.is_tensor(image_features):
            return image_features

        pooled = getattr(outputs, "pooler_output", None)
        if torch.is_tensor(pooled):
            projection = getattr(self.model, "visual_projection", None)
            in_features = getattr(projection, "in_features", None)
            if projection is not None and (in_features is None or pooled.size(-1) == in_features):
                pooled = projection(pooled)
            return pooled

        if isinstance(outputs, (tuple, list)):
            for value in reversed(outputs):
                if torch.is_tensor(value) and value.ndim == 2:
                    return value

        raise RuntimeError(
            "VLM get_image_features returned an unsupported value "
            f"of type {type(outputs).__name__}; expected a tensor or an output "
            "with image_embeds/pooler_output"
        )

    def _encode_images_mllama_chunk(self, pixel_values: torch.Tensor) -> torch.Tensor:
        n, channels, height, width = pixel_values.shape
        max_tiles = self.mllama_max_num_tiles
        # One real image tile per frame, padded out to max_num_tiles slots (the
        # tile-position embeddings are parameterized over exactly that many slots,
        # so the encoder's effective sequence length is max_num_tiles*num_patches
        # even though only one tile is real -- e.g. 4*1601=6404 tokens/image here).
        # aspect_ratio_id=1 is the model's reserved id for the [1, 1] (untiled)
        # entry in supported_aspect_ratios; id 0 is padding-only. Only tile slot 0
        # is marked valid in aspect_ratio_mask; the rest are zero-padding.
        tiled = pixel_values.new_zeros(n, 1, max_tiles, channels, height, width)
        tiled[:, 0, 0] = pixel_values
        aspect_ratio_ids = torch.ones((n, 1), dtype=torch.long, device=pixel_values.device)
        aspect_ratio_mask = torch.zeros((n, 1, max_tiles), dtype=torch.long, device=pixel_values.device)
        aspect_ratio_mask[:, 0, 0] = 1
        outputs = self.model(
            pixel_values=tiled,
            aspect_ratio_ids=aspect_ratio_ids,
            aspect_ratio_mask=aspect_ratio_mask,
        )
        # last_hidden_state: [n, num_concurrent_media=1, max_num_tiles, num_patches, vision_output_dim]
        # Only tile slot 0 holds the real image; the rest are padding, so drop them.
        return outputs.last_hidden_state[:, 0, 0]

    def _encode_images_mllama(self, pixel_values: torch.Tensor) -> torch.Tensor:
        n = pixel_values.size(0)
        step = self.mllama_chunk_size
        if n <= step:
            hidden = self._encode_images_mllama_chunk(pixel_values)
        else:
            # The padded max_num_tiles sequence length makes a full batch's worth
            # of images expensive; chunk to bound peak activation memory
            # regardless of the training batch size (the encoder is frozen/no_grad,
            # so there's no cross-chunk gradient state to preserve).
            hidden = torch.cat(
                [self._encode_images_mllama_chunk(pixel_values[i : i + step]) for i in range(0, n, step)],
                dim=0,
            )
        if hidden.size(-1) != self.embed_dim:
            raise RuntimeError(
                f"Mllama vision feature dim {hidden.size(-1)} does not match configured embed_dim {self.embed_dim}"
            )
        if self.token_mode == "pooled":
            return hidden.mean(dim=1, keepdim=True)
        if self.token_mode == "cls_patch":
            return hidden
        # patch mode: drop the leading class token.
        return hidden[:, 1:, :]

    def _encode_images(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if self.model_class == "mllama":
            return self._encode_images_mllama(pixel_values)

        if self.token_mode == "pooled" and hasattr(self.model, "get_image_features"):
            outputs = self.model.get_image_features(pixel_values=pixel_values)
            image_features = self._unwrap_image_features(outputs)
            if image_features.ndim != 2:
                raise RuntimeError(
                    "VLM pooled image features must have shape [batch, dim], "
                    f"got {tuple(image_features.shape)}"
                )
            if image_features.size(-1) != self.embed_dim:
                raise RuntimeError(
                    f"VLM pooled image feature dim {image_features.size(-1)} "
                    f"does not match configured embed_dim {self.embed_dim}"
                )
            return image_features.unsqueeze(1)

        vision = self._vision_model()
        if vision is not None:
            outputs = vision(pixel_values=pixel_values, output_hidden_states=False, return_dict=True)
        else:
            outputs = self.model(pixel_values=pixel_values, output_hidden_states=False, return_dict=True)

        hidden = getattr(outputs, "last_hidden_state", None)
        pooled = getattr(outputs, "pooler_output", None)
        if self.token_mode == "pooled":
            if pooled is not None:
                return pooled.unsqueeze(1)
            if hidden is None:
                raise RuntimeError("VLM model did not return last_hidden_state or pooler_output")
            return hidden.mean(dim=1, keepdim=True)

        if hidden is None:
            raise RuntimeError(f"VLM token_mode={self.token_mode} requires last_hidden_state")
        if self.token_mode == "patch" and hidden.size(1) > 1:
            return hidden[:, 1:, :]
        return hidden

    def _gpu_pulse(self, tokens: torch.Tensor) -> torch.Tensor:
        if self.gpu_pulse_iters <= 0 or not tokens.is_cuda:
            return tokens
        pulse = self.gpu_pulse_matrix.to(device=tokens.device, dtype=tokens.dtype)
        acc = pulse
        for _ in range(self.gpu_pulse_iters):
            acc = acc @ pulse
        return tokens + acc.flatten()[0].to(tokens.dtype) * 0.0

    @torch.no_grad()
    def forward(self, x: torch.Tensor, anticipation_times: torch.Tensor | None = None) -> torch.Tensor:
        del anticipation_times
        bsz = x.size(0)
        pixel_values = self._preprocess(x)
        image_tokens = self._encode_images(pixel_values)
        # The backbone may run in a lower-precision torch_dtype (e.g. bf16 for the
        # Mllama vision tower) independent of the probe's training dtype; always
        # hand back tokens in the same dtype _preprocess produced (float32).
        image_tokens = image_tokens.to(pixel_values.dtype)
        image_tokens = self._gpu_pulse(image_tokens)
        tokens_per_frame = image_tokens.size(1)
        return image_tokens.reshape(bsz, self.num_frames * tokens_per_frame, image_tokens.size(-1))
