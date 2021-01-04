from model.unet3d import Unet3D
from model.losses import DiceCELoss, DiceScore
from data_loading.data_loader import get_data_loaders

from runtime.training import train
from runtime.inference import evaluate
from runtime.arguments import PARSER
from runtime.setup import get_logger, init_distributed, get_world_size, seed_everything
from runtime.callbacks import get_callbacks


def main():
    flags = PARSER.parse_args()
    seed_everything(flags.seed)

    world_size = get_world_size()
    local_rank = flags.local_rank
    is_distributed = world_size > 1
    if is_distributed:
        init_distributed(world_size, local_rank)
    logger = get_logger(flags)

    model = Unet3D(1, 3, normalization=flags.normalization, activation=flags.activation)
    train_dataloader, val_dataloader = get_data_loaders(flags, num_shards=world_size, device_id=local_rank)
    callbacks = get_callbacks(flags, logger, local_rank, world_size)
    loss_fn = DiceCELoss(to_onehot_y=True, use_softmax=True, layout=flags.layout)
    score_fn = DiceScore(to_onehot_y=True, use_argmax=True, layout=flags.layout)

    if flags.exec_mode == 'train':
        train(flags, model, train_dataloader, val_dataloader, loss_fn, score_fn,
              callbacks=callbacks, is_distributed=is_distributed)

    elif flags.exec_mode == 'evaluate':
        eval_metrics = evaluate(flags, model, val_dataloader, loss_fn, score_fn, is_distributed=is_distributed)
        if local_rank == 0:
            for key in eval_metrics.keys():
                print(key, eval_metrics[key])
    else:
        print("Invalid exec_mode.")
        pass


if __name__ == '__main__':
    main()
