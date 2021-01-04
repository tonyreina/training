# U-Net3D Pytorch implementation for MLPerf v1.0

## Build the docker environment
In order to pre-process the data and run training you will need to build the container. This can be achieved by running:
```bash
docker build -t unet3d:mlperf .
```
now create directories which will contain the dataset and artifacts
```bash
mkdir data_raw data results
```
and run the container mounting the directories
```bash
docker run -it --ipc=host --rm -v ${PWD}/data_raw:/data_raw -v ${PWD}/data:/data -v ${PWD}/results:/results unet3d:mlperf
```
## Data preparation
This section presents steps required to download and pre-process the KiTS19 dataset. 

### Data download
In order to download the data please go https://github.com/neheller/kits19/tree/master and follow the instructions:
1. git clone https://github.com/neheller/kits19
2. cd kits19
3. pip3 install -r requirements.txt
4. python3 -m starter_code.get_imaging
    
This will start downloading data into the repository under `./data` directory. Now you can copy the downloaded data to the `/data_raw` directory in the container (which is mounted to the host)

### Data pre-processing
To pre-process the data you will have to run the `preprocess_dataset.py` script. The script accepts two parameters:
- `--data_dir` which is a path to the directory with the raw dataset, i.e. `/data_raw`
- `--results_dir` which is a path to the pre-processed data directory, i.e. `/data`

Run the script:
```bash
python preprocess_dataset.py --data_dir /data_raw --results_dir /data
```

The process of converting `nifti` files into `numpy` files will start and the results will be visible in the `/data` directory. There are some global variables that you might want to modify later, but it might influence the final score, so you might not be able to reproduce the results.

```python
EXCLUDED_CASES = [23, 68, 125, 133, 15, 37] # Exclude cases with uncertain quality of segmentation (Fabian mentioned that this might have been fixed by now)
MAX_ID = 210                                # Max ID for preprocessing. There are 210 volumes with labels, the remaining are test set without labels, there are not useful for us.
MEAN_VAL = 101.0                            # Mean intensity value for z-score, taken from https://arxiv.org/pdf/1908.02182.pdf
STDDEV_VAL = 76.9                           # Standard deviation of the intensity value for z-score, taken from https://arxiv.org/pdf/1908.02182.pdf
MIN_CLIP_VAL = -79.0                        # Min intensity clip value for z-score, taken from https://arxiv.org/pdf/1908.02182.pdf
MAX_CLIP_VAL = 304.0                        # Max intensity clip value for z-score, taken from https://arxiv.org/pdf/1908.02182.pdf
TARGET_SPACING = [1.6, 1.0, 1.0]            # Target spacing directly influences the final shape of the volume. Setting it to [1.6, 1.0, 1.0] will provide volumes approx. 300 x 300 x 300
```
 
## Run training and evaluation

### Training

After the pre-processing is completed you can run the training by invoking:
```bash
python main.py --data_dir /data --epochs 6000 --batch_size 2 -v --eval_every 25
```

or to train on 8 GPUs:
```bash
python -m torch.distributed.launch --nproc_per_node=8 main.py --data_dir /data --epochs 6000 --batch_size 2 -v --eval_every 25
```

where relevant parameters are:
- `--data_dir` path to the directory containing the dataset
- `--epochs` number of epochs to train
- `--batch_size` batch size
- `-v` enable displaying tqdm progress bar (remove to disable)
  
additional avalable parameters:
- `--fold` allows to select a fold. Default: 3
- `--amp` enables mixed-precision training
- `--norm` allows to select normalization. Only `instancenorm`, `batchnorm`, and `syncbatchnorm` are supported. Default: `instancenorm`.
- `--optimizer` allows to select an optimizer. Only `adam` and `sgd` are supported. Default: `sgd`.
- `--learning_rate` allows to select the learning rate. Default: 1.0.
- `--momentum` allows to select the momentum. Default: 0.9.
- `--eval_every` allows to select the interval of evaluation. Please mind that the evaluation takes a long time. Default: 25.


### Evaluation

Evaluation is performed in the intervals controlled by the `--eval_every` parameter. The typical output reported looks like:
```
DLL 2021-01-04 16:37:28.435525 - 1.0 epoch : 1.0  L1 dice : 0.0  L2 dice : 0.0  mean_dice : 0.0  eval_loss : 0.5126  train_loss : 0.6036  TOP_epoch : 1.0  TOP_L1 dice : 0.0  TOP_L2 dice : 0.0  TOP_mean_dice : 0.0  TOP_eval_loss : 0.5126  TOP_train_loss : 0.6036 
```
where,
- `DLL 2021-01-04 16:37:28.435525` is a timestamp.
- `1.0 epoch` shows the current epoch.
- `L1 dice : 0.0` shows the current dice score for the first non-background class (kidney).
- `L2 dice : 0.0` shows the current dice score for the second non-background class (kidney tumor).
- `mean_dice : 0.0` shows the current mean dice score
- `eval_loss : 0.5126` shows the current evaluation loss
- `train_loss : 0.6036` shows the current trailing loss

Additionally:
- `TOP_*` show the same metric measured at the best epoch (`TOP_epoch`) based on `mean_dice` metric.






