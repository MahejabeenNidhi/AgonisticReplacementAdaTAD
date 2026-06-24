import os
import sys
import copy

sys.dont_write_bytecode = True
path = os.path.join(os.path.dirname(__file__), "..")
if path not in sys.path:
    sys.path.insert(0, path)

# ═══ MUST be set before ANY CUDA/torch.distributed initialization ═══
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["CUDNN_DETERMINISTIC"] = "1"
os.environ["PYTHONHASHSEED"] = "42"

import argparse
import torch
import torch.distributed as dist
from torch.distributed.algorithms.ddp_comm_hooks import default as comm_hooks
from torch.nn.parallel import DistributedDataParallel
from torch.cuda.amp import GradScaler
from mmengine.config import Config, DictAction

from opentad.models import build_detector
from opentad.datasets import build_dataset, build_dataloader
from opentad.cores import train_one_epoch, val_one_epoch, eval_one_epoch, build_optimizer, build_scheduler
from opentad.utils import (
    set_seed,
    update_workdir,
    create_folder,
    save_config,
    setup_logger,
    ModelEma,
    save_checkpoint,
    save_best_checkpoint,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a Temporal Action Detector")
    parser.add_argument("config", metavar="FILE", type=str, help="path to config file")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--id", type=int, default=0, help="repeat experiment id")
    parser.add_argument("--resume", type=str, default=None, help="resume from a checkpoint")
    parser.add_argument("--not_eval", action="store_true", help="whether not to eval, only do inference")
    parser.add_argument("--disable_deterministic", action="store_true", help="disable deterministic for faster speed")
    parser.add_argument("--cfg-options", nargs="+", action=DictAction, help="override settings")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # load config
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    os.environ["PYTHONHASHSEED"] = str(args.seed)

    # ═══ Set seed BEFORE DDP init to seed NCCL and CUDA context ═══
    set_seed(args.seed, args.disable_deterministic)

    # DDP init
    args.local_rank = int(os.environ["LOCAL_RANK"])
    args.world_size = int(os.environ["WORLD_SIZE"])
    args.rank = int(os.environ["RANK"])
    print(f"Distributed init (rank {args.rank}/{args.world_size}, local rank {args.local_rank})")
    dist.init_process_group("nccl", rank=args.rank, world_size=args.world_size)
    torch.cuda.set_device(args.local_rank)

    # Re-seed after CUDA init to ensure GPU RNG is properly set
    set_seed(args.seed, args.disable_deterministic)

    cfg = update_workdir(cfg, args.id, args.world_size)
    if args.rank == 0:
        create_folder(cfg.work_dir)
        save_config(args.config, cfg.work_dir)

    # Override evaluation subset so training-time mAP is computed on
    # the validation split, NOT the test split.
    cfg.evaluation.subset = "validation"

    # setup logger
    logger = setup_logger("Train", save_dir=cfg.work_dir, distributed_rank=args.rank)
    logger.info(f"Using torch version: {torch.__version__}, CUDA version: {torch.version.cuda}")
    logger.info(f"Config: \n{cfg.pretty_text}")

    # ═══ Build datasets ═══
    # Train loader (training subset, filter_gt=True)
    train_dataset = build_dataset(cfg.dataset.train, default_args=dict(logger=logger))
    train_loader = build_dataloader(
        train_dataset,
        rank=args.rank,
        world_size=args.world_size,
        shuffle=True,
        drop_last=True,
        seed=args.seed,
        **cfg.solver.train,
    )

    # Val loader for loss monitoring (validation subset, filter_gt=True)
    val_dataset = build_dataset(cfg.dataset.val, default_args=dict(logger=logger))
    val_loader = build_dataloader(
        val_dataset,
        rank=args.rank,
        world_size=args.world_size,
        shuffle=False,
        drop_last=False,
        seed=args.seed,
        **cfg.solver.val,
    )

    # Eval-val loader for mAP / count metrics (validation subset, filter_gt=False)
    eval_val_cfg = copy.deepcopy(cfg.dataset.test)
    eval_val_cfg.subset_name = "validation"
    eval_val_dataset = build_dataset(eval_val_cfg, default_args=dict(logger=logger))
    eval_val_loader = build_dataloader(
        eval_val_dataset,
        rank=args.rank,
        world_size=args.world_size,
        shuffle=False,
        drop_last=False,
        seed=args.seed,
        **cfg.solver.val,
    )

    # Test dataset is intentionally NOT built here.
    # Test data must only be used at final evaluation via tools/test.py.

    # build model
    model = build_detector(cfg.model)

    # DDP
    use_static_graph = getattr(cfg.solver, "static_graph", False)
    model = model.to(args.local_rank)
    model = DistributedDataParallel(
        model,
        device_ids=[args.local_rank],
        output_device=args.local_rank,
        find_unused_parameters=False,
        static_graph=use_static_graph,
    )
    logger.info(f"Using DDP with total {args.world_size} GPUS...")

    # FP16 compression
    use_fp16_compress = getattr(cfg.solver, "fp16_compress", False)
    if use_fp16_compress:
        logger.info("Using FP16 compression ...")
        model.register_comm_hook(state=None, hook=comm_hooks.fp16_compress_hook)

    # Model EMA
    use_ema = getattr(cfg.solver, "ema", False)
    if use_ema:
        logger.info("Using Model EMA...")
        model_ema = ModelEma(model)
    else:
        model_ema = None

    # AMP: automatic mixed precision
    use_amp = getattr(cfg.solver, "amp", False)
    if use_amp:
        logger.info("Using Automatic Mixed Precision...")
        scaler = GradScaler()
    else:
        scaler = None

    # build optimizer and scheduler
    optimizer = build_optimizer(cfg.optimizer, model, logger)
    scheduler, max_epoch = build_scheduler(cfg.scheduler, optimizer, len(train_loader))

    # override the max_epoch
    max_epoch = cfg.workflow.get("end_epoch", max_epoch)

    # resume
    if args.resume is not None:
        logger.info("Resume training from: {}".format(args.resume))
        device = f"cuda:{args.local_rank}"
        checkpoint = torch.load(args.resume, map_location=device)
        resume_epoch = checkpoint["epoch"]
        logger.info("Resume epoch is {}".format(resume_epoch))
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        if model_ema is not None:
            model_ema.module.load_state_dict(checkpoint["state_dict_ema"])
        del checkpoint
        torch.cuda.empty_cache()
    else:
        resume_epoch = -1

    # train the detector
    logger.info("Training Starts...\n")
    val_loss_best = 1e6
    best_mAP = -1.0
    val_start_epoch = cfg.workflow.get("val_start_epoch", 0)

    for epoch in range(resume_epoch + 1, max_epoch):
        # ═══ Re-sample empty clips for this epoch ═══
        if hasattr(train_loader.dataset, "set_epoch"):
            train_loader.dataset.set_epoch(epoch)
        if hasattr(val_loader.dataset, "set_epoch"):
            val_loader.dataset.set_epoch(epoch)

        train_loader.sampler.set_epoch(epoch)

        # train for one epoch
        train_one_epoch(
            train_loader,
            model,
            optimizer,
            scheduler,
            epoch,
            logger,
            model_ema=model_ema,
            clip_grad_l2norm=cfg.solver.clip_grad_norm,
            logging_interval=cfg.workflow.logging_interval,
            scaler=scaler,
        )

        # save checkpoint
        if (epoch == max_epoch - 1) or ((epoch + 1) % cfg.workflow.checkpoint_interval == 0):
            if args.rank == 0:
                save_checkpoint(model, model_ema, optimizer, scheduler, epoch, work_dir=cfg.work_dir)

        # val loss monitoring
        if (cfg.workflow.val_loss_interval > 0) and ((epoch + 1) % cfg.workflow.val_loss_interval == 0):
            val_loss = val_one_epoch(
                val_loader,
                model,
                logger,
                args.rank,
                epoch,
                model_ema=model_ema,
                use_amp=use_amp,
            )
            if val_loss < val_loss_best:
                val_loss_best = val_loss
                logger.info(f"Lowest val_loss so far: {val_loss:.4f} at epoch {epoch} (monitoring only)")

        # mAP evaluation — on ALL validation videos (including negatives)
        if (cfg.workflow.val_eval_interval > 0) and ((epoch + 1) % cfg.workflow.val_eval_interval == 0):
            metrics_dict = eval_one_epoch(
                eval_val_loader,
                model,
                cfg,
                logger,
                args.rank,
                model_ema=model_ema,
                use_amp=use_amp,
                world_size=args.world_size,
                not_eval=args.not_eval,
            )

            # save best checkpoint based on validation mAP
            if metrics_dict is not None:
                current_mAP = metrics_dict.get("average_mAP", 0.0)
                if current_mAP > best_mAP:
                    logger.info(
                        f"New best val average_mAP: {current_mAP * 100:.2f}% "
                        f"(prev: {best_mAP * 100:.2f}%) at epoch {epoch}"
                    )
                    best_mAP = current_mAP
                    save_best_checkpoint(model, model_ema, epoch, work_dir=cfg.work_dir)

    logger.info("Training Over...\n")


if __name__ == "__main__":
    main()