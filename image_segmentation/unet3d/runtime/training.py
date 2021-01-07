import numpy as np
from tqdm import tqdm

import torch
import torch.distributed as dist
from torch.optim import Adam, SGD
from torch.cuda.amp import autocast, GradScaler


from runtime.setup import get_rank
from runtime.inference import evaluate


def get_optimizer(params, name: str, learning_rate: float, momentum: float):
    if name == "adam":
        optim = Adam(params, lr=learning_rate)
    elif name == "sgd":
        optim = SGD(params, lr=learning_rate, momentum=momentum, nesterov=True)
    else:
        raise ValueError("Optimizer {} unknown.".format(name))
    return optim


def train(flags, model, train_loader, val_loader, loss_fn, score_fn, callbacks, is_distributed):
    
    """
    Detect if GPU/CUDA is available. If not, then fallback to CPU.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    rank = get_rank()
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    optimizer = get_optimizer(model.parameters(), flags.optimizer, flags.learning_rate, flags.momentum)
    scaler = GradScaler()

    model.to(device)
    if flags.normalization == "syncbatchnorm":
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    if is_distributed:
        model = torch.nn.parallel.DistributedDataParallel(model,
                                                          device_ids=[flags.local_rank],
                                                          output_device=flags.local_rank)

    for epoch in range(1, flags.epochs + 1):
        cumulative_loss = []
        for i, batch in enumerate(tqdm(train_loader, disable=(rank != 0) or not flags.verbose)):
            image, label = batch
            image, label = image.to(device), label.to(device)
            for callback in callbacks:
                callback.on_batch_start()

            optimizer.zero_grad()
            with autocast(enabled=flags.amp):
                output = model(image)
                loss_value = loss_fn(output, label)

            if flags.amp:
                scaler.scale(loss_value).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss_value.backward()
                optimizer.step()

            cumulative_loss.append(np.sum(loss_value.detach().cpu().numpy()))

        if ((epoch % flags.evaluate_every) == 0) and not flags.benchmark:
            if is_distributed:
                dist.barrier()
            del output
            eval_metrics = evaluate(flags, model, val_loader, loss_fn, score_fn, epoch)
            eval_metrics["train_loss"] = round(sum(cumulative_loss) / len(cumulative_loss), 4)

            for callback in callbacks:
                callback.on_epoch_end(epoch, eval_metrics, model, optimizer)
            model.train()

    for callback in callbacks:
        callback.on_fit_end()
