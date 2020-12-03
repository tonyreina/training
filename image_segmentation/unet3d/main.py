import os

from dllogger import JSONStreamBackend, Logger, StdOutBackend, Verbosity
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from data_loading.data_module import DataModule
from models.nn_unet import NNUnet
from utils.gpu_affinity import set_affinity
from utils.logger import LoggingCallback
from utils.utils import get_main_args, is_main_process, make_empty_dir, set_cuda_devices, verify_ckpt_path


def log(logname, dice, epoch=None, dice_tta=None):
    dllogger = Logger(
        backends=[
            JSONStreamBackend(Verbosity.VERBOSE, os.path.join(args.results, logname)),
            StdOutBackend(Verbosity.VERBOSE, step_format=lambda step: ""),
        ]
    )
    metrics = {}
    if epoch is not None:
        metrics.update({"Epoch": epoch})
    metrics.update({"Mean dice": round(dice.mean().item(), 2)})
    if dice_tta is not None:
        metrics.update({"Mean TTA dice": round(dice_tta.mean().item(), 2)})
    metrics.update({f"L{j+1}": round(m.item(), 2) for j, m in enumerate(dice)})
    if dice_tta is not None:
        metrics.update({f"TTA_L{j+1}": round(m.item(), 2) for j, m in enumerate(dice_tta)})
    dllogger.log(step=(), data=metrics)
    dllogger.flush()


if __name__ == "__main__":
    args = get_main_args()
    if args.affinity != "disabled":
        affinity = set_affinity(os.getenv("LOCAL_RANK", 0), args.affinity)

    set_cuda_devices(args)
    if args.cascade:
        assert args.dim == 3, "Cascade model works with 3D data only."
    if is_main_process():
        print(f"{args.exec_mode.upper()} TASK {args.task} FOLD {args.fold} SEED {args.seed}")
    seed_everything(args.seed)
    data_module = DataModule(args)
    ckpt_path = verify_ckpt_path(args)

    callbacks = None
    if args.exec_mode == "train":
        model = NNUnet(args)
        model_ckpt = ModelCheckpoint(filepath="/tmp", monitor="dice_sum", mode="max", save_last=True)
        callbacks = [EarlyStopping(monitor="dice_sum", patience=args.patience, verbose=True, mode="max")]
    elif args.exec_mode in ["evaluate", "predict"]:
        data_module.setup()
        model = NNUnet.load_from_checkpoint(ckpt_path)
    else:
        data_module.setup()
        model = NNUnet(args)
        mode = args.benchmark_mode
        batch_size = args.batch_size if mode == "train" else args.val_batch_size
        log_dir = os.path.join(args.results, args.logname if args.logname is not None else "perf.json")
        callbacks = [
            LoggingCallback(
                log_dir=log_dir,
                global_batch_size=batch_size * args.gpus * args.num_nodes,
                mode=mode,
                warmup=args.warmup,
                dim=args.dim,
            )
        ]

    trainer = Trainer(
        logger=False,
        gpus=args.gpus,
        num_nodes=args.num_nodes,
        precision=16 if args.amp else 32,
        benchmark=True,
        deterministic=args.deterministic,
        min_epochs=args.min_epochs,
        max_epochs=args.max_epochs,
        check_val_every_n_epoch=args.eval_every,
        callbacks=callbacks,
        sync_batchnorm=args.norm == "batch",
        num_sanity_val_steps=0,
        default_root_dir=args.results,
        resume_from_checkpoint=ckpt_path,
        distributed_backend="ddp" if args.gpus > 1 else None,
        checkpoint_callback=model_ckpt if args.exec_mode == "train" else None,
        limit_train_batches=1.0 if args.train_batches == 0 else args.train_batches,
        limit_test_batches=1.0 if args.test_batches == 0 else args.test_batches,
    )

    if args.exec_mode == "train":
        trainer.fit(model, data_module)
        model.args.exec_mode = "evaluate"
        model.args.tta = True
        trainer.interrupted = False
        trainer.test(test_dataloaders=data_module.test_dataloader())
        if is_main_process():
            log_name = args.logname if args.logname is not None else "train_log.json"
            log(log_name, model.best_sum_dice, model.best_sum_epoch, model.eval_dice)
    elif args.exec_mode == "evaluate":
        model.args = args
        trainer.test(model, test_dataloaders=data_module.test_dataloader())
        if is_main_process():
            log(args.logname if args.logname is not None else "eval_log.json", model.eval_dice)
    elif args.exec_mode == "predict":
        model.args = args
        if args.save_preds:
            save_dir = os.path.join(args.results, f"preds_{args.task}_{args.dim}")
            model.save_dir = save_dir
            make_empty_dir(save_dir)
        trainer.test(model, test_dataloaders=data_module.test_dataloader())
    else:
        if args.benchmark_mode == "predict":
            trainer.test(model, test_dataloaders=data_module.test_dataloader())
        else:
            trainer.fit(model, train_dataloader=data_module.train_dataloader())
