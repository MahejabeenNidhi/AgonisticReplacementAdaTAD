import os
import numpy as np
import random
import shutil
import torch
import torch.distributed as dist


def set_seed(seed, disable_deterministic=False):
    """Set random seed for pytorch, numpy, and python random for full
    reproducibility.

    NOTE: CUBLAS_WORKSPACE_CONFIG and PYTHONHASHSEED must be set as
    environment variables BEFORE any CUDA context is created (i.e., at the
    top of the main script or via shell export).
    """
    # FIX: pin PYTHONHASHSEED to guarantee dict/set iteration order
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU determinism

    if disable_deterministic:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    else:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Only set env var if CUDA hasn't been initialized yet
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    # FIX: Use warn_only=False to CRASH on non-deterministic ops during debugging.
    # Once you've patched all non-deterministic ops, you can set this back to True.
    # For production with Mamba (which cannot be made deterministic), use True.
    torch.use_deterministic_algorithms(True, warn_only=True)


def worker_init_fn(worker_id):
    """Seed numpy and random in each DataLoader worker for reproducibility."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def update_workdir(cfg, exp_id, gpu_num):
    cfg.work_dir = os.path.join(cfg.work_dir, f"gpu{gpu_num}_id{exp_id}/")
    return cfg


def create_folder(folder_path):
    dir_name = os.path.expanduser(folder_path)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name, mode=0o777, exist_ok=True)


def save_config(cfg, folder_path):
    shutil.copy2(cfg, folder_path)


def reduce_loss(loss_dict):
    # reduce loss when distributed training, only for logging
    for loss_name, loss_value in loss_dict.items():
        loss_value = loss_value.data.clone()
        dist.all_reduce(loss_value.div_(dist.get_world_size()))
        loss_dict[loss_name] = loss_value
    return loss_dict


class AverageMeter(object):
    """Computes and stores the average and current value."""

    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = 0.0

    def initialize(self, val, n):
        self.val = val
        self.avg = val
        self.sum = val * n
        self.count = n
        self.initialized = True

    def update(self, val, n=1):
        if not self.initialized:
            self.initialize(val, n)
        else:
            self.add(val, n)

    def add(self, val, n):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count