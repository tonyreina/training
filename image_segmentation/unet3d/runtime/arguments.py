import argparse

PARSER = argparse.ArgumentParser(description="UNet-3D")

PARSER.add_argument('--data_dir', dest='data_dir', required=True)
PARSER.add_argument('--log_dir', dest='log_dir', type=str)
PARSER.add_argument('--save_ckpt_path', dest='save_ckpt_path', type=str, default="")
PARSER.add_argument('--load_ckpt_path', dest='load_ckpt_path', type=str, default="")
PARSER.add_argument('--loader', dest='loader', default="dali", type=str)
PARSER.add_argument("--local_rank", default=0, type=int)

PARSER.add_argument('--epochs', dest='epochs', type=int, default=1)
PARSER.add_argument('--warmup_steps', dest='warmup_steps', type=int, default=4)
PARSER.add_argument('--log_every', dest='log_every', type=int, default=25)
PARSER.add_argument('--batch_size', dest='batch_size', type=int, default=1)
PARSER.add_argument('--val_batch_size', dest='val_batch_size', type=int, default=2)
PARSER.add_argument('--layout', dest='layout', type=str, choices=['NCDHW', 'NDHWC'], default='NCDHW')
PARSER.add_argument('--input_shape', nargs='+', type=int, default=[128, 128, 128])
PARSER.add_argument('--val_input_shape', nargs='+', type=int, default=[128, 128, 128])
PARSER.add_argument('--seed', dest='seed', default=1, type=int)
PARSER.add_argument('--num_workers', dest='num_workers', type=int, default=8)
PARSER.add_argument('--fold', dest='fold', type=int, choices=[0, 1, 2, 3, 4], default=3)
PARSER.add_argument('--num_folds', dest='num_folds', type=int, default=5)
PARSER.add_argument('--exec_mode', dest='exec_mode', choices=['train', 'evaluate'], default='train')

PARSER.add_argument('--benchmark', dest='benchmark', action='store_true', default=False)
PARSER.add_argument('--amp', dest='amp', action='store_true', default=False)
PARSER.add_argument('--optimizer', dest='optimizer', default="sgd", type=str)
PARSER.add_argument('--learning_rate', dest='learning_rate', type=float, default=1.0)
PARSER.add_argument('--momentum', dest='momentum', type=float, default=0.9)
PARSER.add_argument('--evaluate_every', '--eval_every', dest='evaluate_every', type=int, default=25)
PARSER.add_argument('--verbose', '-v', dest='verbose', action='store_true', default=False)
PARSER.add_argument('--normalization', dest='normalization', type=str,
                    choices=['instancenorm', 'batchnorm', "syncbatchnorm"], default='instancenorm')

PARSER.add_argument('--oversampling', dest='oversampling', type=float, default=0.7)
PARSER.add_argument('--overlap', dest='overlap', type=float, default=0.5)
PARSER.add_argument('--activation', dest='activation', type=str, choices=['relu', 'leaky_relu'], default='relu')
