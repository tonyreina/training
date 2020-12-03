import torch
from pytorch_lightning.metrics.functional import stat_scores
from pytorch_lightning.metrics.functional.reduction import reduce
from pytorch_lightning.metrics.metric import Metric


def compute_dice(pred, target):
    num_classes = pred.shape[1]
    _bg = 1
    scores = torch.zeros(num_classes - _bg, device=pred.device, dtype=torch.float32)
    pred_unit8 = torch.zeros_like(target, device=pred.device, dtype=torch.uint8)
    for j in range(pred.shape[2]):
        pred_unit8[:, j] = torch.squeeze(torch.argmax(pred[:, :, j], dim=1), 0)
    for i in range(_bg, num_classes):
        _tp, _fp, _tn, _fn, _ = stat_scores(pred=pred_unit8, target=target, class_index=i)
        denom = (2 * _tp + _fp + _fn).to(torch.float)
        score_cls = (2 * _tp).to(torch.float) / denom if torch.is_nonzero(denom) else 0.0
        scores[i - _bg] += score_cls
    return reduce(scores, reduction="none")


class Dice(Metric):
    def __init__(self, nclass):
        super().__init__(dist_sync_on_step=True)
        self.add_state("dice", default=torch.zeros((nclass,)), dist_reduce_fx="mean")

    def update(self, pred, target):
        self.dice += compute_dice(pred, target)

    def compute(self):
        return self.dice
