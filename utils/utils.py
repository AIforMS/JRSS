import os
import random
import sys
import logging
import math
from typing import List

import numpy as np
from medpy import metric

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, _LRScheduler

import warnings

warnings.filterwarnings("ignore")


class ImgTransform:
    """
    Image intensity normalization.

    Params:
        :scale_type: normalization way, default mean-std scaled
        :img: ndarray or tensor
    Return:
        scaled img
    """
    def __init__(self, scale_type="mean-std"):
        assert scale_type in ["mean-std", "max-min", "old-way", None], \
            f"scale type include ['mean-std', 'max-min', 'old-way', 'None'], but got {scale_type}"
        self.scale_type = scale_type

    def __call__(self, img):
        if self.scale_type == "mean-std":
            return (img - img.mean()) / img.std()
        if self.scale_type == "max-min":
            return (img - img.min()) / (img.max() - img.min())
        if self.scale_type == "old-way":
            return img / 1024.0 + 1.0
        if self.scale_type is None:
            return img


def get_cosine_schedule_with_warmup(optimizer,
                                    warmup_steps,
                                    total_steps,
                                    num_cycles=7. / 16.,
                                    last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        no_progress = float(current_step - warmup_steps) / \
                      float(max(1, total_steps - warmup_steps))
        return max(0., math.cos(math.pi * num_cycles * no_progress))

    return LambdaLR(optimizer=optimizer,
                    lr_lambda=_lr_lambda,
                    last_epoch=last_epoch, )


class LinearWarmupCosineAnnealingLR(_LRScheduler):

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int,
        max_epochs: int,
        warmup_start_lr: float = 0.0,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        """
        Args:
            optimizer (Optimizer): Wrapped optimizer.
            warmup_epochs (int): Maximum number of iterations for linear warmup
            max_epochs (int): Maximum number of iterations
            warmup_start_lr (float): Learning rate to start the linear warmup. Default: 0.
            eta_min (float): Minimum learning rate. Default: 0.
            last_epoch (int): The index of last epoch. Default: -1.
        """
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min

        super(LinearWarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """
        Compute learning rate using chainable form of the scheduler
        """
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.",
                UserWarning,
            )

        if self.last_epoch == 0:
            return [self.warmup_start_lr] * len(self.base_lrs)
        elif self.last_epoch < self.warmup_epochs:
            return [
                group["lr"] + (base_lr - self.warmup_start_lr) / (self.warmup_epochs - 1)
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]
        elif self.last_epoch == self.warmup_epochs:
            return self.base_lrs
        elif (self.last_epoch - 1 - self.max_epochs) % (2 * (self.max_epochs - self.warmup_epochs)) == 0:
            return [
                group["lr"] + (base_lr - self.eta_min) *
                (1 - math.cos(math.pi / (self.max_epochs - self.warmup_epochs))) / 2
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]

        return [
            (1 + math.cos(math.pi * (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs))) /
            (
                1 +
                math.cos(math.pi * (self.last_epoch - self.warmup_epochs - 1) / (self.max_epochs - self.warmup_epochs))
            ) * (group["lr"] - self.eta_min) + self.eta_min for group in self.optimizer.param_groups
        ]

    def _get_closed_form_lr(self) -> List[float]:
        """
        Called when epoch is passed as a param to the `step` function of the scheduler.
        """
        if self.last_epoch < self.warmup_epochs:
            return [
                self.warmup_start_lr + self.last_epoch * (base_lr - self.warmup_start_lr) / (self.warmup_epochs - 1)
                for base_lr in self.base_lrs
            ]

        return [
            self.eta_min + 0.5 * (base_lr - self.eta_min) *
            (1 + math.cos(math.pi * (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)))
            for base_lr in self.base_lrs
        ]


def init_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


def countParam(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


def get_logger(output, name='train', log_level=1):
    log_levels = {
        0: logging.WARNING,
        1: logging.INFO,
        2: logging.DEBUG
    }
    logging.basicConfig(format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                        datefmt="%m/%d/%Y %H:%M:%S",
                        level=log_levels[log_level],
                        filename=f'{output}/{name}.log',
                        filemode='a')

    msg_log_level = 'log_level option {} is invalid. Valid options are {}.'.format(log_level,
                                                                                   log_levels.keys())
    assert log_level in log_levels, msg_log_level
    logger = logging.getLogger(__name__)
    chlr = logging.StreamHandler()  # 输出到控制台的handler
    logger.addHandler(chlr)
    return logger


def setup_seed(seed=3407):
    os.environ['PYTHONHASHSEED'] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    np.random.seed(seed)
    random.seed(seed)

    # 以下操作会降低运行效率
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.enabled = False


if __name__ == "__main__":
    import nibabel as nib

    gtp = r"D:\code_sources\from_github\Medical Images Seg & Reg\MICCAI2020\vm_troch\dataset\LPBA40\label\S1.delineation.structure.label.nii.gz"
    predp = r"D:\code_sources\from_github\Medical Images Seg & Reg\MICCAI2020\vm_troch\dataset\LPBA40\label\S2.delineation.structure.label.nii.gz"
    dice = dice_coeff(torch.from_numpy(nib.load(predp).get_fdata()),
                      torch.from_numpy(nib.load(gtp).get_fdata()))
    print(f'Dice validation: {dice}', f'Avg. {dice.mean() :.3f}',
          f'Std. {dice.std() :.3f}')
