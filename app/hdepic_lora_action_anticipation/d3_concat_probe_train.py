"""D3 probe FT on [ctx16 ; future_target] @ long horizon (E20 oracle, E18a direct_rope)."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import torch
import webdataset as wds
from torch.utils.data import DataLoader

from app.hdepic_lora_action_anticipation.future_latent_compare import (
    FutureOracleDataset,
    _build_samples,
    _collate,
    _last_layer,
    _predict_direct,
    decord_worker_init,
)
from app.hdepic_lora_action_anticipation.gaze import ContiguousSplitByWorker, ResampledItems
from app.hdepic_lora_action_anticipation.gaze_rnn import call_classifier
from evals.action_anticipation_frozen.epickitchens import SharedEpoch, split_by_node
from src.utils.checkpoint_loader import robust_checkpoint_loader

logger = logging.getLogger("d3_concat_probe_train")


class D3DataInfo:
    """DataInfo with set_epoch for clip-balanced ResampledItems."""

    def __init__(self, dataloader, sampler=None, shared_epoch=None):
        self.dataloader = dataloader
        self.sampler = sampler
        self.shared_epoch = shared_epoch

    def set_epoch(self, epoch):
        if self.shared_epoch is not None:
            self.shared_epoch.set_value(epoch)
        if self.sampler is not None and hasattr(self.sampler, "set_epoch"):
            self.sampler.set_epoch(int(epoch))


def _clip_balanced_items(samples) -> list[dict]:
    video_to_idx: dict[str, int] = {}
    items = []
    for i, sample in enumerate(samples):
        vid = str(sample.video_id)
        if vid not in video_to_idx:
            video_to_idx[vid] = len(video_to_idx)
        items.append({"sample_idx": i, "video_index": video_to_idx[vid]})
    return items


class _FutureOracleDecodeStage(wds.PipelineStage):
    """Decode obs+oracle clips; per-worker single VideoReader cache."""

    def __init__(self, oracle_ds: FutureOracleDataset):
        self.oracle_ds = oracle_ds

    def run(self, src):
        for item in src:
            idx = int(item["sample_idx"])
            yield self.oracle_ds._getitem_one(self.oracle_ds.samples[idx])


def _spatial_tokens(cfg, encoder) -> int:
    data_cfg = cfg["experiment"]["data"]
    wrapper_cfg = cfg.get("model_kwargs", {}).get("wrapper_kwargs", {})
    grid = int(data_cfg["resolution"] // encoder.patch_size)
    tubelet = int(encoder.tubelet_size)
    n_out = max(int(wrapper_cfg.get("num_output_frames", 2)), tubelet)
    return int(grid * grid * (n_out // tubelet))


def _concat_tokens(obs_last: torch.Tensor, future: torch.Tensor, context_chunks: int, spatial: int) -> torch.Tensor:
    if context_chunks <= 0:
        return future
    prefix = obs_last[:, -context_chunks * spatial :, :]
    return torch.cat([prefix, future], dim=1)


def _labels_from_batch(batch, verb_classes, noun_classes, action_classes, device):
    verbs = batch["verb_raw"].tolist()
    nouns = batch["noun_raw"].tolist()
    verb_labels = torch.tensor([verb_classes[int(v)] for v in verbs], device=device, dtype=torch.long)
    noun_labels = torch.tensor([noun_classes[int(n)] for n in nouns], device=device, dtype=torch.long)
    action_labels = torch.tensor(
        [action_classes[(int(v), int(n))] for v, n in zip(verbs, nouns)],
        device=device,
        dtype=torch.long,
    )
    return verb_labels, noun_labels, action_labels


def _future_method_prefix(d3_cfg: dict) -> str:
    mode = str(d3_cfg.get("future_mode", "oracle")).lower()
    if mode == "oracle":
        return "oracle_target"
    if mode in {"direct_rope", "direct_rope_10s"}:
        return "direct_rope_10s"
    raise ValueError(f"Unknown future_mode={mode!r}")


@torch.no_grad()
def _build_future_tokens(
    encoder,
    predictor,
    observed,
    oracle_clip,
    device,
    use_bfloat16: bool,
    n_spatial: int,
    label_horizon: float,
    full_cfg: dict,
    d3_cfg: dict,
):
    mode = str(d3_cfg.get("future_mode", "oracle")).lower()
    obs_dev = observed if observed.device.type == device.type else observed.to(device, non_blocking=True)
    with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=use_bfloat16):
        if mode == "oracle":
            ora_dev = oracle_clip if oracle_clip.device.type == device.type else oracle_clip.to(
                device, non_blocking=True
            )
            stacked = torch.cat([obs_dev, ora_dev], dim=0)
            tok = encoder(stacked)
            b = obs_dev.shape[0]
            obs_last = _last_layer(tok[:b], encoder.embed_dim)
            ora_last = _last_layer(tok[b:], encoder.embed_dim)
            future = ora_last[:, -n_spatial:, :]
            return obs_last, future
        obs_tok = encoder(obs_dev)
    obs_last = _last_layer(obs_tok, encoder.embed_dim)

    if mode in {"direct_rope", "direct_rope_10s"}:
        rope_mode = str(d3_cfg.get("rope_scale_mode", "ntk_temporal"))
        future, info = _predict_direct(
            encoder,
            predictor,
            obs_tok,
            float(label_horizon),
            full_cfg,
            device,
            dense=False,
            rope_scale_mode=rope_mode if rope_mode else None,
        )
        if future is None:
            raise RuntimeError(f"direct_rope predict failed: {info}")
        return obs_last, future

    raise ValueError(f"Unknown future_mode={mode!r}")


def make_d3_concat_webvid(
    base_path,
    annotations_path,
    batch_size,
    transform,
    frames_per_clip=32,
    fps=8,
    num_workers=8,
    world_size=1,
    rank=0,
    anticipation_time_sec=(0.0, 0.0),
    training=True,
    anticipation_point=(0.0, 0.0),
    d3_concat_cfg=None,
    data_cfg=None,
    **kwargs,
):
    del base_path, transform, kwargs, anticipation_time_sec
    d3_concat_cfg = dict(d3_concat_cfg or {})
    data_cfg = dict(data_cfg or {})
    label_h = float(d3_concat_cfg.get("label_horizon_sec", 10.0))
    fast_decode = bool(d3_concat_cfg.get("train_fast_decode", True)) and bool(training)
    paths, annotations = annotations_path
    samples = _build_samples((paths, annotations))
    ds = FutureOracleDataset(
        samples=samples,
        horizon_sec=label_h,
        frames_per_clip=int(frames_per_clip),
        fps=float(fps),
        anticipation_point=tuple(anticipation_point),
        resolution=int(data_cfg.get("resolution", 384)),
        drop_incomplete_history=bool(d3_concat_cfg.get("drop_incomplete_history", True)),
        training=bool(training) and not fast_decode,
        auto_augment=bool(data_cfg.get("auto_augment", True)) and not fast_decode,
        reprob=0.0 if fast_decode else float(data_cfg.get("reprob", 0.25)),
        random_resize_scale=tuple(data_cfg.get("random_resize_scale", (0.08, 1.0))),
        decord_blocklist_path=d3_concat_cfg.get("decord_blocklist_path"),
        probe_decodable=bool(d3_concat_cfg.get("probe_decodable", False)),
    )
    epoch = SharedEpoch(epoch=0)
    if world_size > 1:
        sampler = torch.utils.data.distributed.DistributedSampler(ds, shuffle=training)
        loader = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=_collate,
            drop_last=training,
            worker_init_fn=decord_worker_init if num_workers > 0 else None,
            persistent_workers=(num_workers > 0),
            prefetch_factor=2 if num_workers > 0 else None,
        )
    elif training:
        items = _clip_balanced_items(ds.samples)
        pipeline = [
            ResampledItems(items, epoch=epoch, training=True),
            split_by_node(rank=rank, world_size=world_size),
            ContiguousSplitByWorker(),
            _FutureOracleDecodeStage(ds),
            wds.batched(batch_size, partial=True, collation_fn=_collate),
        ]
        wds_dataset = wds.DataPipeline(*pipeline)
        loader = DataLoader(
            wds_dataset,
            batch_size=None,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=(num_workers > 0),
            worker_init_fn=decord_worker_init if num_workers > 0 else None,
        )
        sampler = None
    else:
        sampler = None
        loader = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=_collate,
            worker_init_fn=decord_worker_init if num_workers > 0 else None,
            persistent_workers=True,
            prefetch_factor=2,
        )
    loader.num_batches = max(1, len(ds) // max(1, world_size * batch_size))
    loader.num_samples = len(ds)
    logger.info(
        "D3 concat dataloader: split=%s samples=%d horizon=%.1fs mode=%s batch=%d workers=%d fast_decode=%s clip_balanced=%s",
        "train" if training else "val",
        len(ds),
        label_h,
        d3_concat_cfg.get("future_mode", "oracle"),
        batch_size,
        num_workers,
        fast_decode,
        bool(training and world_size == 1),
    )
    return ds, loader, D3DataInfo(dataloader=loader, sampler=sampler, shared_epoch=epoch)


@torch.no_grad()
def _eval_ctx_scan(
    encoder,
    predictor,
    classifiers,
    device,
    cfg: dict,
    d3_cfg: dict,
    annotations: dict,
    use_bfloat16: bool,
):
    fm_path = Path(__file__).resolve().parents[2] / "scripts" / "analyze_future_latent_failure_modes.py"
    import importlib.util

    spec = importlib.util.spec_from_file_location("fm", fm_path)
    fm = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(fm)

    data_cfg = cfg["experiment"]["data"]
    label_h = float(d3_cfg["label_horizon_sec"])
    method_prefix = _future_method_prefix(d3_cfg)
    val_paths, val_ann = annotations["val"]
    samples = _build_samples((val_paths, val_ann))

    ds = FutureOracleDataset(
        samples=samples,
        horizon_sec=label_h,
        frames_per_clip=int(data_cfg["frames_per_clip"]),
        fps=float(data_cfg["frames_per_second"]),
        anticipation_point=tuple(data_cfg.get("val_anticipation_point", [0.0, 0.0])),
        resolution=int(data_cfg["resolution"]),
        drop_incomplete_history=bool(d3_cfg.get("drop_incomplete_history", True)),
        training=False,
    )
    loader = DataLoader(
        ds,
        batch_size=int(d3_cfg.get("eval_batch_size", 4)),
        shuffle=False,
        num_workers=int(d3_cfg.get("eval_num_workers", 4)),
        collate_fn=_collate,
        pin_memory=True,
        worker_init_fn=decord_worker_init,
        persistent_workers=True,
        prefetch_factor=2,
    )
    spatial = _spatial_tokens(cfg, encoder)
    ctx_list = [int(x) for x in d3_cfg.get("eval_context_chunks", [0, 16])]
    use_valid = str(os.environ.get("LORA_VAL_METRIC_SCOPE", "native")).lower() == "filtered"
    # recall@5 always needs the full val-label set; use_valid only gates _topk candidate filtering.
    valid_labels = fm._valid_label_sets(annotations)

    trackers = {
        ctx: {h: {t: fm.MetricTracker() for t in ["verb", "noun", "action"]} for h in range(len(classifiers))}
        for ctx in ctx_list
    }

    max_samples = d3_cfg.get("validate_max_samples")
    seen = 0
    for batch in loader:
        observed = batch["observed"]
        oracle = batch["oracle"]
        obs_last, future = _build_future_tokens(
            encoder,
            predictor,
            observed,
            oracle,
            device,
            use_bfloat16,
            spatial,
            label_h,
            cfg,
            d3_cfg,
        )
        verb_labels, noun_labels, action_labels = _labels_from_batch(
            batch,
            annotations["verbs"],
            annotations["nouns"],
            annotations["actions"],
            device,
        )
        label_cpu = {
            "verb": verb_labels.detach().cpu().tolist(),
            "noun": noun_labels.detach().cpu().tolist(),
            "action": action_labels.detach().cpu().tolist(),
        }
        for ctx in ctx_list:
            tokens = _concat_tokens(obs_last, future, ctx, spatial)
            for head, clf in enumerate(classifiers):
                clf_eval = clf.module if hasattr(clf, "module") else clf
                with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=use_bfloat16):
                    out = call_classifier(clf_eval, tokens, None)
                for task in ["verb", "noun", "action"]:
                    valid = valid_labels[task] if use_valid and valid_labels else None
                    for i in range(out[task].shape[0]):
                        preds, _ = fm._topk(out[task][i], 5, valid)
                        trackers[ctx][head][task].update(label_cpu[task][i], preds)
        if max_samples is not None:
            seen += int(observed.shape[0])
            if seen >= int(max_samples):
                break

    rows = []
    for ctx in ctx_list:
        head_rows = []
        for head in range(len(classifiers)):
            row = {"method": f"{method_prefix}_ctx{ctx}", "context_chunks": ctx, "head": head}
            for task in ["verb", "noun", "action"]:
                vals = trackers[ctx][head][task].values(valid_labels[task])
                for metric, value in vals.items():
                    row[f"{task}_{metric}"] = value
            head_rows.append(row)
        native = fm._vjepa2_native_summary(head_rows)
        if native:
            rows.append(native[0])
    return rows


def train_one_epoch_d3_concat(
    d3_cfg: dict,
    full_cfg: dict,
    action_is_verb_noun,
    ipe,
    device,
    model,
    classifiers,
    scaler,
    optimizer,
    scheduler,
    wd_scheduler,
    data_loader,
    use_bfloat16,
    noun_classes,
    verb_classes,
    action_classes,
    criterion,
    **_,
):
    del action_is_verb_noun, _
    exp_id = str(d3_cfg.get("experiment_id", "D3"))
    context_chunks = int(d3_cfg.get("context_chunks", 16))
    label_h = float(d3_cfg["label_horizon_sec"])
    encoder = model.encoder if hasattr(model, "encoder") else model
    predictor = model.predictor if hasattr(model, "predictor") else None
    if predictor is None:
        raise RuntimeError("direct_rope training requires AnticipativeWrapper with .predictor")
    encoder.eval()
    model.eval()

    spatial = _spatial_tokens(full_cfg, encoder)
    for c in classifiers:
        c.train(mode=True)

    _data_loader = iter(data_loader)
    logged_shapes = False
    for itr in range(ipe):
        try:
            batch = next(_data_loader)
        except StopIteration:
            _data_loader = iter(data_loader)
            batch = next(_data_loader)

        [s.step() for s in scheduler]
        [wds.step() for wds in wd_scheduler]

        observed = batch["observed"].to(device, non_blocking=True)
        oracle = batch["oracle"].to(device, non_blocking=True)
        verb_labels, noun_labels, action_labels = _labels_from_batch(
            batch, verb_classes, noun_classes, action_classes, device
        )

        with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=use_bfloat16):
            obs_last, future = _build_future_tokens(
                encoder,
                predictor,
                observed,
                oracle,
                device,
                use_bfloat16,
                spatial,
                label_h,
                full_cfg,
                d3_cfg,
            )
            tokens = _concat_tokens(obs_last, future, context_chunks, spatial)
            if not logged_shapes:
                logger.info(
                    "%s token shapes: obs=%s future=%s concat=%s (n_spatial=%d ctx=%d mode=%s)",
                    exp_id,
                    tuple(obs_last.shape),
                    tuple(future.shape),
                    tuple(tokens.shape),
                    spatial,
                    context_chunks,
                    d3_cfg.get("future_mode"),
                )
                logged_shapes = True
            outputs = []
            for c in classifiers:
                clf = c.module if hasattr(c, "module") else c
                outputs.append(call_classifier(clf, tokens, None))

            verb_loss = [criterion(o["verb"], verb_labels) for o in outputs]
            noun_loss = [criterion(o["noun"], noun_labels) for o in outputs]
            action_loss = [criterion(o["action"], action_labels) for o in outputs]
            loss = [v + n + a for v, n, a in zip(verb_loss, noun_loss, action_loss)]

        if use_bfloat16:
            [s.scale(l).backward() for s, l in zip(scaler, loss)]
            [s.step(o) for s, o in zip(scaler, optimizer)]
            [s.update() for s in scaler]
        else:
            [L.backward() for L in loss]
            [o.step() for o in optimizer]
        [o.zero_grad() for o in optimizer]

        if itr % 50 == 0:
            logger.info("%s train itr=%d/%d loss=%.4f", exp_id, itr, ipe, float(loss[0].detach()))

        ckpt_every = int(d3_cfg.get("checkpoint_every_iters", 400))
        if ckpt_every > 0 and itr > 0 and itr % ckpt_every == 0:
            ckpt_dir = (
                Path(full_cfg.get("folder", "."))
                / "action_anticipation_frozen"
                / str(full_cfg.get("tag", "d3-concat"))
            )
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "classifiers": [(c.module if hasattr(c, "module") else c).state_dict() for c in classifiers],
                    "epoch": 0,
                    "itr": itr,
                    "ipe": ipe,
                },
                ckpt_dir / "latest.pt",
            )
            logger.info("%s saved mid-epoch checkpoint itr=%d/%d -> %s", exp_id, itr, ipe, ckpt_dir / "latest.pt")

    z = 0.0
    block = {"accuracy": z, "recall": z}
    return {"action": dict(block), "verb": dict(block), "noun": dict(block)}


def validate_d3_concat(**kwargs):
    d3_cfg = kwargs.pop("d3_cfg")
    full_cfg = kwargs.pop("full_cfg")
    annotations_holder = kwargs.pop("annotations_holder")
    device = kwargs["device"]
    model = kwargs["model"]
    classifiers = kwargs["classifiers"]
    use_bfloat16 = kwargs["use_bfloat16"]

    exp_id = str(d3_cfg.get("experiment_id", "D3"))
    encoder = model.encoder if hasattr(model, "encoder") else model
    predictor = model.predictor if hasattr(model, "predictor") else None
    encoder.eval()
    for c in classifiers:
        c.eval()

    if not annotations_holder:
        raise RuntimeError(f"{exp_id} annotations_holder empty; main() patch did not run")
    rows = _eval_ctx_scan(
        encoder,
        predictor,
        classifiers,
        device,
        full_cfg,
        d3_cfg,
        annotations_holder,
        use_bfloat16,
    )
    tag = full_cfg.get("tag", "d3-concat")
    folder = Path(full_cfg.get("folder", ".")) / "action_anticipation_frozen" / tag
    folder.mkdir(parents=True, exist_ok=True)
    out_name = str(d3_cfg.get("eval_json_name", "d3_concat_ctx_eval.json"))
    out_path = folder / out_name
    with out_path.open("w", encoding="utf-8") as f:
        json.dump({"rows": rows, "future_mode": d3_cfg.get("future_mode")}, f, indent=2)
    for row in rows:
        logger.info(
            "%s eval %s action_r5=%.2f noun_top3=%.2f",
            exp_id,
            row.get("method"),
            float(row.get("action_recall5", 0)),
            float(row.get("noun_top3", 0)),
        )
    best = max((float(r.get("action_recall5", 0)) for r in rows), default=0.0)
    block = {"accuracy": best, "recall": best}
    return {"action": dict(block), "verb": dict(block), "noun": dict(block)}


def enable_d3_concat_probe_training(
    base_eval,
    args_eval,
    d3_cfg: dict,
    data_cfg: dict,
    lora_init_classifier_fn,
):
    full_cfg = {
        "tag": args_eval.get("tag"),
        "folder": args_eval.get("folder"),
        "experiment": {"data": data_cfg},
        "model_kwargs": args_eval.get("model_kwargs", {}),
    }
    annotations_holder: dict = {}
    exp_id = str(d3_cfg.get("experiment_id", "D3"))

    def _make(*args, **kwargs):
        kwargs["d3_concat_cfg"] = d3_cfg
        kwargs["data_cfg"] = data_cfg
        kwargs["anticipation_point"] = kwargs.get(
            "anticipation_point",
            data_cfg.get("train_anticipation_point", [0.0, 0.0])
            if kwargs.get("training", True)
            else data_cfg.get("val_anticipation_point", [0.0, 0.0]),
        )
        return make_d3_concat_webvid(*args, **kwargs)

    import evals.action_anticipation_frozen.dataloader as dl
    import evals.action_anticipation_frozen.epickitchens as ek

    ek.make_webvid = _make
    dl.ek100_make_webvid = _make

    init_ckpt = d3_cfg.get("init_checkpoint")

    def init_classifier_with_warmstart(*args, **kwargs):
        classifiers = lora_init_classifier_fn(*args, **kwargs)
        if init_ckpt and os.path.isfile(init_ckpt):
            logger.info("%s warm-start classifiers from %s", exp_id, init_ckpt)
            checkpoint = robust_checkpoint_loader(init_ckpt, map_location="cpu")
            for clf, state in zip(classifiers, checkpoint["classifiers"]):
                clean = {k.removeprefix("module."): v for k, v in state.items()}
                msg = clf.load_state_dict(clean, strict=False)
                logger.info("Loaded warm-start: %s", msg)
        else:
            logger.warning("%s warm-start skipped: init_checkpoint missing (%s)", exp_id, init_ckpt)
        return classifiers

    original_init_module = base_eval.init_module

    def init_module_frozen_encoder(*args, **kwargs):
        model = original_init_module(*args, **kwargs)
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        logger.info("%s: froze encoder/predictor wrapper parameters", exp_id)
        return model

    base_eval.init_module = init_module_frozen_encoder
    base_eval.init_classifier = init_classifier_with_warmstart
    base_eval.train_one_epoch = lambda **kw: train_one_epoch_d3_concat(d3_cfg=d3_cfg, full_cfg=full_cfg, **kw)

    num_epochs = int(args_eval.get("optimization", {}).get("num_epochs", 5))
    validate_call = {"i": 0}

    def validate_with_epoch_policy(**kw):
        epoch_i = validate_call["i"]
        cfg = dict(d3_cfg)
        is_last = epoch_i >= num_epochs - 1
        if not is_last:
            interim = d3_cfg.get("validate_interim_max_samples")
            if interim is None and d3_cfg.get("skip_interim_validate"):
                logger.info("%s skip ctx validate epoch %d/%d", exp_id, epoch_i + 1, num_epochs)
                validate_call["i"] += 1
                z = 0.0
                block = {"accuracy": z, "recall": z}
                return {"action": dict(block), "verb": dict(block), "noun": dict(block)}
            if interim is not None:
                cfg["validate_max_samples"] = int(interim)
                logger.info("%s interim ctx validate epoch %d/%d max_samples=%s", exp_id, epoch_i + 1, num_epochs, interim)
        else:
            cfg.pop("validate_max_samples", None)
            logger.info("%s full ctx validate epoch %d/%d", exp_id, epoch_i + 1, num_epochs)
        result = validate_d3_concat(d3_cfg=cfg, full_cfg=full_cfg, annotations_holder=annotations_holder, **kw)
        validate_call["i"] += 1
        return result

    base_eval.validate = validate_with_epoch_policy

    original_main = base_eval.main

    def main_with_annotations(args_eval, resume_preempt=False):
        from evals.action_anticipation_frozen.dataloader import filter_annotations

        annotations_holder.update(
            filter_annotations(
                data_cfg["dataset"],
                data_cfg["base_path"],
                data_cfg["dataset_train"],
                data_cfg["dataset_val"],
                file_format=data_cfg.get("file_format", 1),
            )
        )
        return original_main(args_eval=args_eval, resume_preempt=resume_preempt)

    base_eval.main = main_with_annotations
    logger.info(
        "Enabled %s concat train: mode=%s label_h=%.1fs ctx=%d init=%s",
        exp_id,
        d3_cfg.get("future_mode"),
        float(d3_cfg["label_horizon_sec"]),
        int(d3_cfg.get("context_chunks", 16)),
        init_ckpt,
    )
