import os

import apex
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch_optimizer as optim
from torch.optim.lr_scheduler import _LRScheduler
from adabelief_pytorch import AdaBelief
from dllogger import JSONStreamBackend, Logger, StdOutBackend, Verbosity
from monai.inferers import sliding_window_inference
from utils.utils import flip, is_main_process

from models.dice import Dice
from models.loss import Loss
from models.unet import UNet
from models.unet3d import Unet3D


class NNUnet(pl.LightningModule):
    def __init__(self, args):
        super(NNUnet, self).__init__()
        self.args = args
        self.save_hyperparameters()
        self.build_nnunet()
        self.loss = Loss(ce_weights=args.ce_weights)
        self.dice = Dice(self.n_class)
        self.best_sum = 0
        self.eval_dice = 0
        self.best_sum_epoch = 0
        self.best_dice = self.n_class * [0]
        self.best_epoch = self.n_class * [0]
        self.best_sum_dice = self.n_class * [0]
        self.learning_rate = args.learning_rate
        self.dllogger = Logger(
            backends=[
                JSONStreamBackend(Verbosity.VERBOSE, os.path.join(args.results, "logs.json")),
                StdOutBackend(Verbosity.VERBOSE, step_format=lambda step: f"Epoch: {step} "),
            ]
        )

        self.tta_flips = (
            [[2], [3], [2, 3]] if self.args.dim == 2 else [[2], [3], [4], [2, 3], [2, 4], [3, 4], [2, 3, 4]]
        )

    def forward(self, img):
        return self.model(img)

    def training_step(self, batch, _):
        img, label, rloss = self.train_pre_step(batch)
        pred = self.forward(img)
        loss = self.compute_loss(pred, label)
        return loss

    def validation_step(self, batch, _):
        img, label = self.validation_pre_step(batch)
        pred = self.inference(img)
        loss = self.loss(pred, label)
        dice = self.dice(pred, label[:, 0])
        return {"val_dice": dice, "val_loss": loss}

    def test_step(self, batch, _):
        img = self.test_pre_step(batch)
        pred = self.inference(img)
        if self.args.save_preds:
            fname = batch["name"]
            self.save_mask(pred, fname)
        if not self.args.exec_mode == "benchmark":
            label = batch["label"][:, 0]
            return {"test_dice": self.dice(pred, label)}

    def build_unet(self, in_channels, out_channels, kernels, strides):
        return UNet(
            in_channels=in_channels,
            out_channels=out_channels,
            kernels=kernels,
            strides=strides,
            normalization_layer=self.args.norm,
            negative_slope=self.args.negative_slope,
            deep_supervision=self.args.deep_supervision,
            dimension=self.args.dim,
        )

    def build_nnunet(self):
        size = 320 if "320" in self.args.model else 512
        self.model = Unet3D(in_channels=1, n_class=3, normalization=self.args.norm, size=size)
        self.n_class = 2

    def compute_loss(self, preds, label):
        return self.loss(preds, label)

    def inference(self, img):
        return self.tta_inference(img) if self.args.tta else self.do_inference(img)

    def do_inference(self, data):
        return sliding_window_inference(
            inputs=data,
            roi_size=self.args.input_shape,
            sw_batch_size=self.args.val_batch_size,
            predictor=self.forward,
            overlap=self.args.overlap,
            mode=self.args.val_mode,
        )

    def tta_inference(self, img):
        pred = self.do_inference(img)
        for flip_idx in self.tta_flips:
            pred += flip(self.do_inference(flip(img, flip_idx)), flip_idx)
        pred /= len(self.tta_flips) + 1
        return pred

    @staticmethod
    def metric_mean(name, outputs):
        return torch.stack([out[name] for out in outputs]).mean(dim=0)

    def validation_epoch_end(self, outputs):
        dice, loss = 100 * self.metric_mean("val_dice", outputs), self.metric_mean("val_loss", outputs)
        dice_sum = torch.sum(dice)
        if dice_sum >= self.best_sum:
            self.best_sum = dice_sum
            self.best_sum_dice = dice[:]
            self.best_sum_epoch = self.current_epoch
        for i, dice_i in enumerate(dice):
            if dice_i > self.best_dice[i]:
                self.best_dice[i], self.best_epoch[i] = dice_i, self.current_epoch

        if is_main_process():
            metrics = {f"L{i+1}": round(m.item(), 2) for i, m in enumerate(dice)}
            metrics.update({"val_loss": round(loss.item(), 4)})
            metrics.update({f"TOP_L{i+1}": round(m.item(), 2) for i, m in enumerate(self.best_sum_dice)})
            self.dllogger.log(step=self.current_epoch, data=metrics)
            self.dllogger.flush()

        self.log("val_loss", loss)
        self.log("dice_sum", dice_sum)

    def test_epoch_end(self, outputs):
        if self.args.exec_mode != "benchmark":
            self.eval_dice = 100 * self.metric_mean("test_dice", outputs)

    def configure_optimizers(self):
        optimizer = {
            "sgd": torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=self.args.momentum, nesterov=True),
            "adam": torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.args.weight_decay),
            "adamw": torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.args.weight_decay),
            "radam": optim.RAdam(self.parameters(), lr=self.learning_rate, weight_decay=self.args.weight_decay),
            "adabelief": AdaBelief(self.parameters(), lr=self.learning_rate),
            "novograd": optim.NovoGrad(self.parameters(), lr=self.learning_rate, weight_decay=self.args.weight_decay),
            "fused_adam": apex.optimizers.FusedAdam(
                self.parameters(), lr=self.learning_rate, weight_decay=self.args.weight_decay
            ),
        }[self.args.optimizer.lower()]

        scheduler = {
            "none": None,
            "multistep": torch.optim.lr_scheduler.MultiStepLR(optimizer, self.args.steps, gamma=self.args.factor),
            "cosine": torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.args.max_epochs),
            "plateau": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, factor=self.args.factor, patience=self.args.lr_patience
            ),
            "poly": PolynomialLRDecay(optimizer, self.args.max_epochs, end_learning_rate=0.0001, power=0.9)
        }[self.args.scheduler.lower()]

        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

    def train_pre_step(self, batch):
        return batch["image"], batch["label"], None

    def validation_pre_step(self, batch):
        return batch["image"], batch["label"]

    def test_pre_step(self, batch):
        return batch["image"]

    def save_mask(self, pred, name):
        name = os.path.basename(name[0]).replace("_y", "_pred")
        _, pred = torch.max(pred, 1)
        pred = pred.cpu().detach().numpy()
        np.save(os.path.join(self.save_dir, name), pred, allow_pickle=False)



class PolynomialLRDecay(_LRScheduler):
    """Polynomial learning rate decay until step reach to max_decay_step

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        max_decay_steps: after this step, we stop decreasing learning rate
        end_learning_rate: scheduler stoping learning rate decay, value of learning rate must be this value
        power: The power of the polynomial.
    """

    def __init__(self, optimizer, max_decay_steps, end_learning_rate=0.0001, power=1.0):
        if max_decay_steps <= 1.:
            raise ValueError('max_decay_steps should be greater than 1.')
        self.max_decay_steps = max_decay_steps
        self.end_learning_rate = end_learning_rate
        self.power = power
        self.last_step = 0
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_step > self.max_decay_steps:
            return [self.end_learning_rate for _ in self.base_lrs]

        return [(base_lr - self.end_learning_rate) *
                ((1 - self.last_step / self.max_decay_steps) ** (self.power)) +
                self.end_learning_rate for base_lr in self.base_lrs]

    def step(self, step=None):
        if step is None:
            step = self.last_step + 1
        self.last_step = step if step != 0 else 1
        if self.last_step <= self.max_decay_steps:
            decay_lrs = [(base_lr - self.end_learning_rate) *
                         ((1 - self.last_step / self.max_decay_steps) ** (self.power)) +
                         self.end_learning_rate for base_lr in self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, decay_lrs):
                param_group['lr'] = lr