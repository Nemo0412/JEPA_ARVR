"""
REF baseline: frozen V-JEPA2 encoder + attentive motion head (from scratch).

Task: Ego-Exo4D 3D Human Motion Prediction (EgoAgent protocol).
  - 5 egocentric video frames
  - 5 past 3D body poses (optional ablation: video-only)
  - predict future 15 frames x 17 COCO joints
Metrics: MPJPE (cm), MPJVE (cm/s)
"""

from __future__ import annotations

import logging
import os
import pprint
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from app.ref_egoexo4d_motion_prediction.dataset import make_motion_dataloader
from app.ref_egoexo4d_motion_prediction.metrics import compute_mpjpe_mpjve, masked_mse
from app.ref_egoexo4d_motion_prediction.models import AttentiveMotionHead
from evals.action_anticipation_frozen.models import init_module
from evals.action_anticipation_frozen.utils import init_opt
from src.utils.checkpoint_loader import robust_checkpoint_loader
from src.utils.distributed import init_distributed
from src.utils.logging import AverageMeter, CSVLogger

logger = logging.getLogger(__name__)
logging.basicConfig()
logger.setLevel(logging.INFO)
pp = pprint.PrettyPrinter(indent=4)


def _unwrap(module: torch.nn.Module) -> torch.nn.Module:
    return module.module if isinstance(module, DistributedDataParallel) else module


def _encode_video(model: torch.nn.Module, clips: torch.Tensor, device: torch.device, use_bfloat16: bool):
    """Frozen encoder tokens; ``no_predictor`` wrapper ignores anticipation time."""
    clips = clips.to(device, non_blocking=True)
    anticipation = torch.zeros(clips.size(0), device=device)
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=use_bfloat16):
            tokens = model(clips, anticipation)
            if isinstance(tokens, (list, tuple)):
                tokens = tokens[-1]
            embed_dim = model.embed_dim
            if tokens.size(-1) > embed_dim:
                tokens = tokens[:, :, -embed_dim:]
    return tokens


def train_one_epoch(
    *,
    model,
    head,
    loader,
    optimizer,
    scaler,
    scheduler,
    wd_scheduler,
    device,
    use_bfloat16: bool,
    use_past_pose: bool,
    motion_fps: float,
) -> dict[str, float]:
    head.train()
    loss_meter = AverageMeter()
    mpjpe_meter = AverageMeter()
    mpjve_meter = AverageMeter()

    for batch in loader:
        tokens = _encode_video(model, batch["video"], device, use_bfloat16)
        past = batch["past_motion"].to(device, non_blocking=True) if use_past_pose else None
        target = batch["future_motion"].to(device, non_blocking=True)
        mask = batch["future_mask"].to(device, non_blocking=True)

        with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=use_bfloat16):
            pred = head(tokens, past_motion=past)
            loss = masked_mse(pred, target, mask)

        opt = optimizer[0]
        opt.zero_grad(set_to_none=True)
        if scaler[0] is not None:
            scaler[0].scale(loss).backward()
            scaler[0].unscale_(opt)
            scaler[0].step(opt)
            scaler[0].update()
        else:
            loss.backward()
            opt.step()

        metrics = compute_mpjpe_mpjve(pred.detach().float(), target.float(), mask, fps=motion_fps)
        loss_meter.update(loss.item())
        if metrics["mpjpe_cm"] == metrics["mpjpe_cm"]:
            mpjpe_meter.update(metrics["mpjpe_cm"])
        if metrics["mpjve_cm_s"] == metrics["mpjve_cm_s"]:
            mpjve_meter.update(metrics["mpjve_cm_s"])

        for s in scheduler:
            s.step()
        for wds in wd_scheduler:
            wds.step()

    return {
        "loss": loss_meter.avg,
        "mpjpe_cm": mpjpe_meter.avg,
        "mpjve_cm_s": mpjve_meter.avg,
    }


@torch.no_grad()
def validate(
    *,
    model,
    head,
    loader,
    device,
    use_bfloat16: bool,
    use_past_pose: bool,
    motion_fps: float,
) -> dict[str, float]:
    head.eval()
    loss_meter = AverageMeter()
    mpjpe_meter = AverageMeter()
    mpjve_meter = AverageMeter()

    for batch in loader:
        tokens = _encode_video(model, batch["video"], device, use_bfloat16)
        past = batch["past_motion"].to(device, non_blocking=True) if use_past_pose else None
        target = batch["future_motion"].to(device, non_blocking=True)
        mask = batch["future_mask"].to(device, non_blocking=True)

        with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=use_bfloat16):
            pred = head(tokens, past_motion=past)
            loss = masked_mse(pred, target, mask)

        metrics = compute_mpjpe_mpjve(pred.float(), target.float(), mask, fps=motion_fps)
        loss_meter.update(loss.item())
        if metrics["mpjpe_cm"] == metrics["mpjpe_cm"]:
            mpjpe_meter.update(metrics["mpjpe_cm"])
        if metrics["mpjve_cm_s"] == metrics["mpjve_cm_s"]:
            mpjve_meter.update(metrics["mpjve_cm_s"])

    if dist.is_available() and dist.is_initialized():
        for meter in (loss_meter, mpjpe_meter, mpjve_meter):
            t = torch.tensor([meter.sum, meter.count], device=device, dtype=torch.float64)
            dist.all_reduce(t)
            meter.sum = float(t[0].item())
            meter.count = int(t[1].item())

    return {
        "loss": loss_meter.avg,
        "mpjpe_cm": mpjpe_meter.avg,
        "mpjve_cm_s": mpjve_meter.avg,
    }


def load_checkpoint(path: str, head, optimizer, scaler, device):
    ckpt = robust_checkpoint_loader(path, map_location=device)
    state = ckpt.get("motion_head", ckpt)
    _unwrap(head).load_state_dict(state, strict=False)
    if optimizer and ckpt.get("opt"):
        for o, sd in zip(optimizer, ckpt["opt"]):
            o.load_state_dict(sd)
    if scaler and ckpt.get("scaler"):
        for s, sd in zip(scaler, ckpt["scaler"]):
            if s is not None and sd is not None:
                s.load_state_dict(sd)
    return int(ckpt.get("epoch", 0))


def save_checkpoint(path: str, epoch: int, head, optimizer, scaler, batch_size: int, world_size: int):
    ckpt = {
        "epoch": epoch,
        "motion_head": _unwrap(head).state_dict(),
        "opt": [o.state_dict() for o in optimizer],
        "scaler": None if scaler[0] is None else [s.state_dict() for s in scaler],
        "batch_size": batch_size,
        "world_size": world_size,
    }
    torch.save(ckpt, path)


def main(args_eval, resume_preempt=False):
    del resume_preempt
    val_only = bool(args_eval.get("val_only", False))
    tag = args_eval.get("tag", "ref-egoexo-motion")
    folder = Path(args_eval.get("folder", "."))
    run_dir = folder / "egoexo_motion_prediction" / tag
    run_dir.mkdir(parents=True, exist_ok=True)

    args_exp = args_eval["experiment"]
    args_data = args_exp["data"]
    args_cls = args_exp["classifier"]
    args_opt = args_exp["optimization"]
    args_model = args_eval["model_kwargs"]["pretrain_kwargs"]
    args_wrapper = args_eval["model_kwargs"]["wrapper_kwargs"]

    motion_cfg = dict(args_exp.get("motion_prediction", {}))
    use_past_pose = bool(motion_cfg.get("use_past_pose", True))
    context_video_frames = int(motion_cfg.get("context_video_frames", 5))
    context_motion_frames = int(motion_cfg.get("context_motion_frames", 5))
    future_motion_frames = int(motion_cfg.get("future_motion_frames", 15))
    motion_fps = float(motion_cfg.get("motion_fps", 30))

    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)

    world_size, rank = init_distributed()
    logger.info("REF egoexo motion pred rank=%d world_size=%d", rank, world_size)

    model = init_module(
        module_name=args_eval["model_kwargs"]["module_name"],
        frames_per_clip=args_data["frames_per_clip"],
        frames_per_second=args_data["frames_per_second"],
        resolution=args_data["resolution"],
        checkpoint=args_eval["model_kwargs"]["checkpoint"],
        model_kwargs=args_model,
        wrapper_kwargs=args_wrapper,
        device=device,
    )
    for p in model.parameters():
        p.requires_grad = False

    head = AttentiveMotionHead(
        embed_dim=model.embed_dim,
        num_heads=int(args_cls.get("num_heads", 16)),
        num_probe_blocks=int(args_cls.get("num_probe_blocks", 4)),
        context_motion_frames=context_motion_frames,
        future_motion_frames=future_motion_frames,
        use_past_pose=use_past_pose,
    ).to(device)

    pretrained_probe = args_exp.get("probe", {}).get("pretrained_probe")
    if pretrained_probe:
        logger.warning(
            "REF baseline ignores pretrained_probe=%s; motion head is trained from scratch",
            pretrained_probe,
        )

    opt_kwargs = args_opt["optimizer_kwargs"]
    batch_size = int(args_opt["batch_size"])
    num_epochs = int(args_opt["num_epochs"])
    use_bfloat16 = bool(args_opt.get("use_bfloat16", True))

    ds_kwargs = dict(
        context_video_frames=context_video_frames,
        context_motion_frames=context_motion_frames,
        future_motion_frames=future_motion_frames,
        video_target_fps=float(args_data["frames_per_second"]),
        resolution=int(args_data["resolution"]),
        auto_augment=bool(args_data.get("auto_augment", True)),
        reprob=float(args_data.get("reprob", 0.25)),
        random_resize_scale=tuple(args_data.get("random_resize_scale", (0.08, 1.0))),
        max_samples=args_data.get("max_samples"),
    )

    _, train_loader = make_motion_dataloader(
        index_csv=args_data["dataset_train"],
        batch_size=batch_size,
        num_workers=int(args_data.get("num_workers", 4)),
        pin_memory=bool(args_data.get("pin_memory", True)),
        training=True,
        **{**ds_kwargs, "auto_augment": args_data.get("auto_augment", True)},
    )
    _, val_loader = make_motion_dataloader(
        index_csv=args_data["dataset_val"],
        batch_size=batch_size,
        num_workers=int(args_data.get("val_num_workers", args_data.get("num_workers", 4))),
        pin_memory=bool(args_data.get("pin_memory", True)),
        training=False,
        **{**ds_kwargs, "auto_augment": False, "reprob": 0.0},
    )
    ipe = train_loader.num_batches

    optimizer, scaler, scheduler, wd_scheduler = init_opt(
        classifiers=[head],
        opt_kwargs=[opt_kwargs],
        iterations_per_epoch=ipe,
        num_epochs=num_epochs,
        use_bfloat16=use_bfloat16,
    )

    if dist.is_available() and dist.is_initialized() and world_size > 1:
        head = DistributedDataParallel(head, device_ids=[device.index], output_device=device.index)

    latest_path = run_dir / "latest.pt"
    start_epoch = 0
    if args_eval.get("resume_checkpoint") and latest_path.exists():
        start_epoch = load_checkpoint(str(latest_path), head, optimizer, scaler, device)
        for _ in range(start_epoch * ipe):
            scheduler[0].step()
            wd_scheduler[0].step()

    log_file = run_dir / f"log_r{rank}.csv"
    if rank == 0:
        csv_logger = CSVLogger(
            str(log_file),
            ("%d", "epoch"),
            ("%.5f", "train-loss"),
            ("%.5f", "train-mpjpe-cm"),
            ("%.5f", "train-mpjve-cm-s"),
            ("%.5f", "val-loss"),
            ("%.5f", "val-mpjpe-cm"),
            ("%.5f", "val-mpjve-cm-s"),
        )

    for epoch in range(start_epoch, num_epochs):
        train_metrics = {"loss": 0.0, "mpjpe_cm": 0.0, "mpjve_cm_s": 0.0}
        if not val_only:
            train_metrics = train_one_epoch(
                model=model,
                head=head,
                loader=train_loader,
                optimizer=optimizer,
                scaler=scaler,
                scheduler=scheduler,
                wd_scheduler=wd_scheduler,
                device=device,
                use_bfloat16=use_bfloat16,
                use_past_pose=use_past_pose,
                motion_fps=motion_fps,
            )

        val_metrics = validate(
            model=model,
            head=head,
            loader=val_loader,
            device=device,
            use_bfloat16=use_bfloat16,
            use_past_pose=use_past_pose,
            motion_fps=motion_fps,
        )

        if rank == 0:
            csv_logger.log(
                epoch + 1,
                train_metrics["loss"],
                train_metrics["mpjpe_cm"],
                train_metrics["mpjve_cm_s"],
                val_metrics["loss"],
                val_metrics["mpjpe_cm"],
                val_metrics["mpjve_cm_s"],
            )
            save_checkpoint(
                str(latest_path),
                epoch + 1,
                head,
                optimizer,
                scaler,
                batch_size,
                world_size,
            )
            logger.info(
                "epoch=%d train loss=%.4f mpjpe=%.2f mpjve=%.3f | val loss=%.4f mpjpe=%.2f mpjve=%.3f",
                epoch + 1,
                train_metrics["loss"],
                train_metrics["mpjpe_cm"],
                train_metrics["mpjve_cm_s"],
                val_metrics["loss"],
                val_metrics["mpjpe_cm"],
                val_metrics["mpjve_cm_s"],
            )

        if val_only:
            break

    pp.pprint({"val": val_metrics, "run_dir": str(run_dir)})
