"""Attention-importance token pruning for Qwen2.5-VL (B12 equal-compute study).

Analogous to JEPA_ARVR's ``TokenPruner`` (refer_repo/JEPA_ARVR/vjepa/train_hdepic_probe.py,
branch ``pruning``), but placed where it actually saves backbone compute for a VLM.

Where the V-JEPA2 probe prunes *after* the encoder's last block (so only the cheap probe
head sees fewer tokens -- see B12 open-item #1: that placement does NOT reduce backbone
FLOPs), here we prune the merged **video** vision tokens *between the vision tower and the
LLM*. The 36-layer LLM is Qwen's dominant cost, and it runs on every surviving vision
token, so dropping vision tokens before the LLM genuinely cuts training compute -- which is
what the equal-compute comparison against V-JEPA2 needs.

Importance metric (faithful to the PhD): each merged vision token's importance is the total
attention it *receives* in the vision tower's last full-attention block::

    importance[j] = sum_{head, query_i}  softmax(q_i . k_j / sqrt(d))

i.e. the Qwen ViT analogue of V-JEPA2's encoder-last-layer attention-received score. The
top-K (K = round(N * keep_ratio)) merged tokens are kept, in their original order, and the
remaining columns are dropped from ``inputs_embeds`` / ``position_ids`` / ``attention_mask``
before the LLM runs. Qwen's M-RoPE positions are absolute (temporal, height, width) triplets
computed by ``get_rope_index``; dropping vision tokens leaves the survivors' absolute
positions valid (M-RoPE needs no contiguity), so kept text tokens keep their positions too.

Only the ``qwen25vl`` backend is supported. Install with ``install_qwen_video_token_pruner``
*after* the model is on-device and LoRA is injected; it monkey-patches the inner
``Qwen2_5_VLModel.forward`` and the last vision block's attention, and returns a handle whose
``.remove()`` restores the originals and whose ``.last_stats`` reports realized token counts.

Pinned to the transformers build in the project overlay (5.6.2). The patched inner-forward
mirrors that version's ``Qwen2_5_VLModel.forward``; if transformers is upgraded, re-verify
against the new source before trusting the pruned path.
"""

from __future__ import annotations

import torch


class _PruneState:
    """Mutable per-model handle: holds config, captured importance, and removal closures."""

    def __init__(self, keep_ratio: float, chunk_size: int):
        self.keep_ratio = keep_ratio
        self.chunk_size = chunk_size          # 0 -> no query chunking (segments are tiny here)
        self.patch_importance: torch.Tensor | None = None  # window-order, patch granularity
        self._restore = []                    # list of callables to undo monkey-patches
        self.last_stats: dict | None = None   # {"n_tokens", "n_kept", "seq_len", "pruned_seq_len"}

    def remove(self):
        for fn in reversed(self._restore):
            fn()
        self._restore.clear()


def _patch_last_vision_attention(visual, state: _PruneState):
    """Wrap the last vision block's attention so it also records per-patch attention-received.

    The wrapper calls the original attention (unchanged output / gradients) and, under
    ``no_grad``, recomputes q/k for the same block to accumulate the column-sum of the
    softmax attention within each ``cu_seqlens`` segment (per-frame for a full-attention
    block). Result is stored in window order at patch granularity; the inner-forward patch
    maps it to canonical merged-token order.
    """
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import apply_rotary_pos_emb_vision

    attn = visual.blocks[-1].attn
    orig_forward = attn.forward

    def _forward(hidden_states, cu_seqlens, rotary_pos_emb=None, position_embeddings=None, **kwargs):
        out = orig_forward(
            hidden_states,
            cu_seqlens=cu_seqlens,
            rotary_pos_emb=rotary_pos_emb,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        with torch.no_grad():
            seq = hidden_states.shape[0]
            qkv = attn.qkv(hidden_states).reshape(seq, 3, attn.num_heads, -1).permute(1, 0, 2, 3)
            q, k, _ = qkv.unbind(0)                       # each (seq, heads, head_dim)
            if position_embeddings is not None:
                cos, sin = position_embeddings
                q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)
            q = q.float()
            k = k.float()
            scale = attn.scaling
            imp = torch.zeros(seq, device=hidden_states.device, dtype=torch.float32)
            bounds = cu_seqlens.tolist()
            for a, b in zip(bounds[:-1], bounds[1:]):
                if b <= a:
                    continue
                qs = q[a:b].transpose(0, 1)               # (heads, L, head_dim)
                ks = k[a:b].transpose(0, 1)
                step = state.chunk_size if state.chunk_size and state.chunk_size > 0 else (b - a)
                acc = torch.zeros(b - a, device=hidden_states.device, dtype=torch.float32)
                for ci in range(0, b - a, step):
                    qc = qs[:, ci : ci + step, :]         # (heads, c, head_dim)
                    logits = torch.matmul(qc, ks.transpose(-2, -1)) * scale  # (heads, c, L)
                    acc += logits.softmax(dim=-1).sum(dim=1).sum(dim=0)      # received per key -> (L,)
                imp[a:b] = acc
            state.patch_importance = imp
        return out

    attn.forward = _forward
    state._restore.append(lambda: setattr(attn, "forward", orig_forward))


def _merged_importance_canonical(visual, grid_thw, state: _PruneState, device) -> torch.Tensor:
    """Map captured window-order patch importance to canonical merged-token order.

    The vision tower permutes 4-patch merge units by ``window_index`` before the blocks, then
    ``merger`` collapses each consecutive group of ``spatial_merge_unit`` patches into one
    token and ``argsort(window_index)`` restores canonical order. We replay exactly that to
    turn per-patch importance into per-merged-token importance aligned with ``pooler_output``.
    """
    merge_unit = visual.spatial_merge_unit
    window_index, _ = visual.get_window_index(grid_thw)
    window_index = window_index.to(device)
    merged_w = state.patch_importance.view(-1, merge_unit).sum(dim=-1)   # window order, per merged token
    reverse = torch.argsort(window_index)
    return merged_w[reverse]                                             # canonical merged order


def _make_pruned_forward(inner, state: _PruneState):
    """Build a replacement for Qwen2_5_VLModel.forward that prunes video tokens pre-LLM."""
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLModelOutputWithPast

    visual = inner.visual

    def forward(
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        rope_deltas=None,
        mm_token_type_ids=None,
        second_per_grid_ts=None,
        **kwargs,
    ):
        # Fall back to the stock path for anything outside the train/eval video-probe case
        # this pruner targets (no images, exactly one video, no KV-cache, given input_ids).
        if (
            state.keep_ratio >= 1.0
            or pixel_values_videos is None
            or pixel_values is not None
            or input_ids is None
            or past_key_values is not None
            or position_ids is not None
        ):
            return state.orig_inner_forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                pixel_values=pixel_values,
                pixel_values_videos=pixel_values_videos,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                rope_deltas=rope_deltas,
                mm_token_type_ids=mm_token_type_ids,
                second_per_grid_ts=second_per_grid_ts,
                **kwargs,
            )

        inputs_embeds = inner.get_input_embeddings()(input_ids)

        # Vision tower forward (records patch importance via the attention patch).
        video_embeds = inner.get_video_features(pixel_values_videos, video_grid_thw).pooler_output
        video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
        _, video_mask = inner.get_placeholder_mask(
            input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
        )
        inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        position_ids = inner.compute_3d_position_ids(
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            mm_token_type_ids=mm_token_type_ids,
        )

        device = inputs_embeds.device
        B, L, D = inputs_embeds.shape
        merge_unit = visual.spatial_merge_unit
        imp_canonical = _merged_importance_canonical(visual, video_grid_thw, state, device)
        sizes = (video_grid_thw.prod(dim=-1) // merge_unit).tolist()   # tokens per video

        vmask2d = video_mask[..., 0] if video_mask.dim() == 3 else video_mask  # (B, L) bool
        keep_rows = []
        off = 0
        n_tokens = sizes[0]
        n_kept = max(1, int(round(n_tokens * state.keep_ratio)))
        for b in range(B):
            n_b = sizes[b]
            imp_b = imp_canonical[off : off + n_b]
            off += n_b
            k_b = max(1, int(round(n_b * state.keep_ratio)))
            vpos = vmask2d[b].nonzero(as_tuple=False).squeeze(1)        # (n_b,) absolute cols
            top = torch.topk(imp_b, k_b).indices
            keep_v = vpos[torch.sort(top).values]
            nonvideo = (~vmask2d[b]).nonzero(as_tuple=False).squeeze(1)
            keep_rows.append(torch.sort(torch.cat([nonvideo, keep_v])).values)

        keep_idx = torch.stack(keep_rows, dim=0)                        # (B, L')
        g = keep_idx.unsqueeze(-1).expand(-1, -1, D)
        inputs_embeds = inputs_embeds.gather(1, g)
        position_ids = position_ids.gather(2, keep_idx.unsqueeze(0).expand(3, -1, -1))
        if attention_mask is not None:
            attention_mask = attention_mask.gather(1, keep_idx)

        state.last_stats = {
            "n_tokens": int(n_tokens),
            "n_kept": int(n_kept),
            "seq_len": int(L),
            "pruned_seq_len": int(keep_idx.shape[1]),
        }

        outputs = inner.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )
        return Qwen2_5_VLModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=None,
        )

    return forward


def install_qwen_video_token_pruner(backbone, keep_ratio: float, chunk_size: int = 0) -> _PruneState:
    """Install attention-importance video-token pruning on a Qwen2.5-VL model.

    Args:
        backbone: ``Qwen2_5_VLForConditionalGeneration`` (already on-device, LoRA injected).
        keep_ratio: fraction of merged video tokens to keep, in (0, 1]. >= 1.0 disables pruning
            (the install still returns a handle, but the stock forward is used).
        chunk_size: optional query-axis chunk for the importance softmax (0 = whole segment;
            segments are per-frame and small at our resolutions, so 0 is fine).

    Returns:
        ``_PruneState`` handle: ``.remove()`` restores the original forwards, ``.last_stats``
        holds the most recent realized token counts.
    """
    if not (0.0 < keep_ratio <= 1.0):
        raise ValueError(f"keep_ratio must be in (0, 1], got {keep_ratio}")

    modules = dict(backbone.named_modules())
    inner = modules.get("model") or getattr(backbone, "model", None)
    if inner is None or not hasattr(inner, "visual"):
        raise RuntimeError("Could not locate the inner Qwen2_5_VLModel (expected backbone.model with .visual).")

    state = _PruneState(keep_ratio=keep_ratio, chunk_size=chunk_size)
    state.orig_inner_forward = inner.forward

    if keep_ratio < 1.0:
        _patch_last_vision_attention(inner.visual, state)

    pruned_forward = _make_pruned_forward(inner, state)
    inner.forward = pruned_forward
    state._restore.append(lambda: setattr(inner, "forward", state.orig_inner_forward))

    return state
