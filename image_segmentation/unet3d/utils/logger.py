import operator
import time

import dllogger as logger
import numpy as np
from dllogger import JSONStreamBackend, StdOutBackend, Verbosity
from pytorch_lightning import Callback

from utils.utils import is_main_process


class LoggingCallback(Callback):
    def __init__(self, log_dir, global_batch_size, mode, warmup, dim):
        logger.init(backends=[JSONStreamBackend(Verbosity.VERBOSE, log_dir), StdOutBackend(Verbosity.VERBOSE)])
        self.warmup_steps = warmup
        self.global_batch_size = global_batch_size
        self.step = 0
        self.dim = dim
        self.mode = mode
        self.timestamps = []
        self.batch_sizes = []

    def do_step(self):
        self.step += 1
        if self.step >= self.warmup_steps:
            self.timestamps.append(time.time())

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        self.do_step()

    def on_test_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        if self.dim == 2 and self.step >= self.warmup_steps:
            batch_size = batch["image"].shape[2] - batch["image"].shape[2] % self.global_batch_size
            self.batch_sizes.append(batch_size)
        self.do_step()

    def process_performance_stats(self, deltas):
        def _round3(val):
            return round(val, 3)

        if self.batch_sizes:
            numerator = np.sum(self.batch_sizes)
        else:
            numerator = (self.step - self.warmup_steps) * self.global_batch_size

        throughput_imgps = _round3((numerator / np.sum(deltas)))
        timestamps_ms = 1000 * deltas
        stats = {
            f"throughput_{self.mode}": throughput_imgps,
            f"latency_{self.mode}_mean": _round3(timestamps_ms.mean()),
        }
        for level in [90, 95, 99]:
            stats.update({f"latency_{self.mode}_{level}": _round3(np.percentile(timestamps_ms, level))})

        return stats

    def teardown(self, trainer, pl_module, stage):
        if is_main_process():
            deltas = np.array(list(map(operator.sub, self.timestamps[1:], self.timestamps[:-1])))
            stats = self.process_performance_stats(deltas)
            logger.log(step=(), data=stats)
            logger.flush()
