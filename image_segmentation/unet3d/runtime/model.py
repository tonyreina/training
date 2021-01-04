import torch
import monai
from monai.metrics import compute_meandice

from pytorch_lightning import LightningModule
from pytorch_lightning.metrics.classification import DiceCoefficient

from data_loading.synthetic_loader import SyntheticDataset
from data_loading.dali_loader import get_dali_loader
from data_loading.monai_loader import get_monai_loader
from model.unet3d import Unet3D


class Model(LightningModule):
    def __init__(self, params):
        super().__init__()
        model = Unet3D(4, 4, "instancenorm3d")

        self._params = params
        self._model = model
        self._loss = monai.losses.DiceLoss(to_onehot_y=True, softmax=True)
        self._dice_fn = DiceCoefficient(reduction="none", reduce_op="mean")

    def forward(self, x):
        return self._model(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self._model.parameters(), 1e-4)

    def train_dataloader(self):
        if self._params.loader == "synthetic":
            dataset = SyntheticDataset(batch_size=self._params.batch_size * self._params.gpus,
                                       shape=self._params.input_shape)
            return torch.utils.data.DataLoader(dataset, batch_size=self._params.batch_size,
                                               shuffle=False, sampler=None, batch_sampler=None,
                                               num_workers=self._params.threads, pin_memory=True, drop_last=True)
        elif self._params.loader == "dali":
            return get_dali_loader(self._params)
        elif self._params.loader == "monai":
            return get_monai_loader(self._params, is_training=True)

    def test_dataloader(self):
        if self._params.loader == 'synthetic':
            dataset = SyntheticDataset(batch_size=self._params.batch_size * self._params.gpus,
                                       shape=self._params.input_shape)
            return torch.utils.data.DataLoader(dataset, batch_size=self._params.batch_size,
                                               shuffle=False, sampler=None, batch_sampler=None,
                                               num_workers=self._params.threads, pin_memory=True, drop_last=True)
        elif self._params.loader == "dali":
            return get_dali_loader(self._params)

    def val_dataloader(self):
        return get_monai_loader(self._params, is_training=False)

    def validation_step(self, batch, batch_idx):
        x, y = batch['image'], batch['label']
        y_hat = self(x)
        loss = self._loss(y_hat, y)
        # mean_dice = compute_meandice(y_pred=y_hat, y=y, include_background=False,
        #                              mutually_exclusive=True, to_onehot_y=True)
        mean_dice = self._dice_fn(pred=y_hat, target=y)
        tensorboard_logs = {'eval_loss': loss, 'eval_dice': mean_dice}
        return tensorboard_logs

    def validation_epoch_end(self, outputs):
        val_acc_mean = 0
        i = 0
        for output in outputs:
            if not torch.isnan(output['eval_dice']).any():
                val_acc_mean += torch.mean(output['eval_dice'])
                i += 1
            else:
                print("NaN found in", output['eval_dice'])

        val_acc_mean /= i
        print("Validation dice:", val_acc_mean.item())
        tqdm_dict = {'eval_dice': val_acc_mean.item()}

        # show val_acc in progress bar but only log val_loss
        results = {
            'progress_bar': tqdm_dict,
            'log': {'eval_dice': val_acc_mean.item()}
        }
        return results

    def training_step(self, batch, batch_idx):
        if self._params.loader == "dali":
            batch = batch[0]
        x, y = batch['image'], batch['label']
        y_hat = self(x)
        loss = self._loss(y_hat, y)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def test_step(self, batch, batch_nb):
        if self._params.loader == "dali":
            batch = batch[0]
        x, _ = batch['image'], batch['label']
        pred = self(x).cpu()

