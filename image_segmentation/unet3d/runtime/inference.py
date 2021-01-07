import math
import collections
from typing import Union, Sequence, Callable, Optional, Any, Tuple, List, cast

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

from runtime.setup import reduce_tensor, get_world_size, get_rank


def evaluate(flags, model, loader, loss_fn, score_fn, epoch=0, is_distributed=False):
    """
    Detect if CUDA/GPU is available. If not, then fallback to CPU.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    rank = get_rank()
    world_size = get_world_size()
    model.to(device)
    if flags.load_ckpt_path:
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
        checkpoint = torch.load(flags.load_ckpt_path, map_location=map_location)
        epoch = checkpoint['epoch']
        if is_distributed:
            model = torch.nn.parallel.DistributedDataParallel(model,
                                                              device_ids=[flags.local_rank],
                                                              output_device=flags.local_rank)

        model.load_state_dict(checkpoint['best_model_state_dict'])

    model.eval()

    eval_loss = []
    scores = []
    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, disable=(rank != 0) or not flags.verbose)):
            image, label = batch
            image, label = image.to(device), label.to(device)
            if image.numel() == 0:
                continue
            with autocast(enabled=flags.amp):
                output = sliding_window_inference(
                    inputs=image,
                    roi_size=flags.input_shape,
                    sw_batch_size=flags.val_batch_size,
                    predictor=model,
                    overlap=flags.overlap,
                    mode="gaussian"
                )
                eval_loss_value = loss_fn(output, label)
                scores.append(score_fn(output, label))
            eval_loss.append(eval_loss_value)
            del output

    scores = reduce_tensor(torch.mean(torch.stack(scores, dim=0), dim=0), world_size)
    eval_loss = reduce_tensor(torch.mean(torch.stack(eval_loss, dim=0), dim=0), world_size)

    scores, eval_loss = scores.cpu().numpy(), float(eval_loss.cpu().numpy())
    eval_metrics = {"epoch": epoch,
                    "L1 dice": round(scores[0], 4),
                    "L2 dice": round(scores[1], 4),
                    "mean_dice": round((scores[0] + scores[1]) / 2, 4),
                    "eval_loss": round(eval_loss, 4)}

    return eval_metrics


def issequenceiterable(obj: Any) -> bool:
    return isinstance(obj, collections.abc.Iterable) and not isinstance(obj, str)


def ensure_tuple(vals: Any) -> Tuple[Any, ...]:
    if not issequenceiterable(vals):
        vals = (vals,)
    return tuple(vals)


def ensure_tuple_size(tup: Any, dim: int, pad_val: Any = 0) -> Tuple[Any, ...]:
    tup = ensure_tuple(tup) + (pad_val,) * dim
    return tuple(tup[:dim])


def ensure_tuple_rep(tup: Any, dim: int) -> Tuple[Any, ...]:
    if not issequenceiterable(tup):
        return (tup,) * dim
    elif len(tup) == dim:
        return tuple(tup)

    raise ValueError(f"Sequence must have length {dim}, got {len(tup)}.")


def get_valid_patch_size(image_size: Sequence[int], patch_size: Union[Sequence[int], int]) -> Tuple[int, ...]:
    """
    Given an image of dimensions `image_size`, return a patch size tuple taking the dimension from `patch_size` if this is
    not 0/None. Otherwise, or if `patch_size` is shorter than `image_size`, the dimension from `image_size` is taken. This ensures
    the returned patch size is within the bounds of `image_size`. If `patch_size` is a single number this is interpreted as a
    patch of the same dimensionality of `image_size` with that size in each dimension.
    """
    ndim = len(image_size)
    patch_size_ = ensure_tuple_size(patch_size, ndim)

    # ensure patch size dimensions are not larger than image dimension, if a dimension is None or 0 use whole dimension
    return tuple(min(ms, ps or ms) for ms, ps in zip(image_size, patch_size_))


def _get_scan_interval(image_size: Sequence[int], roi_size: Sequence[int],
                       num_spatial_dims: int, overlap: float) -> Tuple[int, ...]:
    if len(image_size) != num_spatial_dims:
        raise ValueError("image coord different from spatial dims.")
    if len(roi_size) != num_spatial_dims:
        raise ValueError("roi coord different from spatial dims.")

    scan_interval = []
    for i in range(num_spatial_dims):
        if roi_size[i] == image_size[i]:
            scan_interval.append(int(roi_size[i]))
        else:
            interval = int(roi_size[i] * (1 - overlap))
            scan_interval.append(interval if interval > 0 else 1)
    return tuple(scan_interval)


def first(iterable, default=None):
    """
    Returns the first item in the given iterable or `default` if empty, meaningful mostly with 'for' expressions.
    """
    for i in iterable:
        return i
    return default


def dense_patch_slices(image_size: Sequence[int], patch_size: Sequence[int],
                       scan_interval: Sequence[int]) -> List[Tuple[slice, ...]]:
    """
    Enumerate all slices defining 2D/3D patches of size `patch_size` from an `image_size` input image.

    Args:
        image_size: dimensions of image to iterate over
        patch_size: size of patches to generate slices
        scan_interval: dense patch sampling interval

    Raises:
        ValueError: When ``image_size`` length is not one of [2, 3].

    Returns:
        a list of slice objects defining each patch

    """
    num_spatial_dims = len(image_size)
    if num_spatial_dims not in (2, 3):
        raise ValueError(f"Unsupported image_size length: {len(image_size)}, available options are [2, 3]")
    patch_size = get_valid_patch_size(image_size, patch_size)
    scan_interval = ensure_tuple_size(scan_interval, num_spatial_dims)

    scan_num = list()
    for i in range(num_spatial_dims):
        if scan_interval[i] == 0:
            scan_num.append(1)
        else:
            num = int(math.ceil(float(image_size[i]) / scan_interval[i]))
            scan_dim = first(d for d in range(num) if d * scan_interval[i] + patch_size[i] >= image_size[i])
            scan_num.append(scan_dim + 1)

    slices: List[Tuple[slice, ...]] = []
    for i in range(scan_num[0]):
        start_i = i * scan_interval[0]
        start_i -= max(start_i + patch_size[0] - image_size[0], 0)
        slice_i = slice(start_i, start_i + patch_size[0])

        for j in range(scan_num[1]):
            start_j = j * scan_interval[1]
            start_j -= max(start_j + patch_size[1] - image_size[1], 0)
            slice_j = slice(start_j, start_j + patch_size[1])

            for k in range(0, scan_num[2]):
                start_k = k * scan_interval[2]
                start_k -= max(start_k + patch_size[2] - image_size[2], 0)
                slice_k = slice(start_k, start_k + patch_size[2])
                slices.append((slice_i, slice_j, slice_k))
    return slices


def gaussian_1d(sigma: float, truncated: float = 4.0) -> np.ndarray:

    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}.")

    tail = int(sigma * truncated + 0.5)
    sigma2 = sigma * sigma
    x = np.arange(-tail, tail + 1)
    out = np.exp(-0.5 / sigma2 * x ** 2)
    out /= out.sum()
    return out


def same_padding(kernel_size: Union[Sequence[int], int],
                 dilation: Union[Sequence[int], int] = 1) -> Union[Tuple[int, ...], int]:
    """
    Return the padding value needed to ensure a convolution using the given kernel size produces an output of the same
    shape as the input for a stride of 1, otherwise ensure a shape of the input divided by the stride rounded down.

    Raises:
        NotImplementedError: When ``np.any((kernel_size - 1) * dilation % 2 == 1)``.

    """

    kernel_size_np = np.atleast_1d(kernel_size)
    dilation_np = np.atleast_1d(dilation)

    if np.any((kernel_size_np - 1) * dilation % 2 == 1):
        raise NotImplementedError(
            f"Same padding not available for kernel_size={kernel_size_np} and dilation={dilation_np}."
        )

    padding_np = (kernel_size_np - 1) / 2 * dilation_np
    padding = tuple(int(p) for p in padding_np)

    return padding if len(padding) > 1 else padding[0]


class GaussianFilter(nn.Module):
    def __init__(self, spatial_dims: int, sigma: Union[Sequence[float], float], truncated: float = 4.0) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions of the input image.
                must have shape (Batch, channels, H[, W, ...]).
            sigma: std.
            truncated: spreads how many stds.
        """
        super().__init__()
        self.spatial_dims = int(spatial_dims)
        _sigma = ensure_tuple_rep(sigma, self.spatial_dims)
        self.kernel = [
            torch.nn.Parameter(torch.as_tensor(gaussian_1d(s, truncated), dtype=torch.float), False) for s in _sigma
        ]
        self.padding = [cast(int, (same_padding(k.size()[0]))) for k in self.kernel]
        self.conv_n = [F.conv1d, F.conv2d, F.conv3d][spatial_dims - 1]
        for idx, param in enumerate(self.kernel):
            self.register_parameter(f"kernel_{idx}", param)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: in shape [Batch, chns, H, W, D].

        Raises:
            TypeError: When ``x`` is not a ``torch.Tensor``.

        """
        if not torch.is_tensor(x):
            raise TypeError(f"x must be a torch.Tensor but is {type(x).__name__}.")
        chns = x.shape[1]
        sp_dim = self.spatial_dims
        x = x.clone()  # no inplace change of x

        def _conv(input_: torch.Tensor, d: int) -> torch.Tensor:
            if d < 0:
                return input_
            s = [1] * (sp_dim + 2)
            s[d + 2] = -1
            kernel = self.kernel[d].reshape(s)
            kernel = kernel.repeat([chns, 1] + [1] * sp_dim)
            padding = [0] * sp_dim
            padding[d] = self.padding[d]
            return self.conv_n(input=_conv(input_, d - 1), weight=kernel, padding=padding, groups=chns)

        return _conv(x, sp_dim - 1)


def compute_importance_map(
    patch_size: Tuple[int, ...],
    mode: str = "constant",
    sigma_scale: float = 0.125,
    device: Optional[torch.device] = None) -> torch.Tensor:
    """Get importance map for different weight modes.

    Args:
        patch_size: Size of the required importance map. This should be either H, W [,D].
        mode: {``"constant"``, ``"gaussian"``}
            How to blend output of overlapping windows. Defaults to ``"constant"``.

            - ``"constant``": gives equal weight to all predictions.
            - ``"gaussian``": gives less weight to predictions on edges of windows.

        sigma_scale: Sigma_scale to calculate sigma for each dimension
            (sigma = sigma_scale * dim_size). Used for gaussian mode only.
        device: Device to put importance map on.

    Raises:
        ValueError: When ``mode`` is not one of ["constant", "gaussian"].

    Returns:
        Tensor of size patch_size.

    """
    if mode == "constant":
        importance_map = torch.ones(patch_size, device=device).float()
    elif mode == "gaussian":
        center_coords = [i // 2 for i in patch_size]
        sigmas = [i * sigma_scale for i in patch_size]

        importance_map = torch.zeros(patch_size, device=device)
        importance_map[tuple(center_coords)] = 1
        pt_gaussian = GaussianFilter(len(patch_size), sigmas).to(device=device, dtype=torch.float)
        importance_map = pt_gaussian(importance_map.unsqueeze(0).unsqueeze(0))
        importance_map = importance_map.squeeze(0).squeeze(0)
        importance_map = importance_map / torch.max(importance_map)
        importance_map = importance_map.float()

        # importance_map cannot be 0, otherwise we may end up with nans!
        importance_map[importance_map == 0] = torch.min(importance_map[importance_map != 0])
    else:
        raise ValueError(f'Unsupported mode: {mode}, available options are ["constant", "gaussian"].')

    return importance_map


def sliding_window_inference(inputs: torch.Tensor,
                             roi_size: Union[Sequence[int], int],
                             sw_batch_size: int,
                             predictor: Callable[[torch.Tensor], torch.Tensor],
                             overlap: float = 0.25,
                             mode: str = "gaussian",
                             padding_mode: str = "constant",
                             cval: float = 0.0,
                             device: Optional[torch.device] = None) -> torch.Tensor:

    num_spatial_dims = len(inputs.shape) - 2
    assert 0 <= overlap < 1, "overlap must be >= 0 and < 1."

    # determine image spatial size and batch size
    # Note: all input images must have the same image size and batch size
    image_size_ = list(inputs.shape[2:])
    batch_size = inputs.shape[0]

    if device is None:
        device = inputs.device

    # in case that image size is smaller than roi size
    image_size = tuple(max(image_size_[i], roi_size[i]) for i in range(num_spatial_dims))
    pad_size = []
    for k in range(len(inputs.shape) - 1, 1, -1):
        diff = max(roi_size[k - 2] - inputs.shape[k], 0)
        half = diff // 2
        pad_size.extend([half, diff - half])
    inputs = F.pad(inputs, pad=pad_size, mode=padding_mode, value=cval)

    scan_interval = _get_scan_interval(image_size, roi_size, num_spatial_dims, overlap)

    # Store all slices in list
    slices = dense_patch_slices(image_size, roi_size, scan_interval)

    slice_batches = []
    for slice_index in range(0, len(slices), sw_batch_size):
        slice_index_range = range(slice_index, min(slice_index + sw_batch_size, len(slices)))
        input_slices = []
        for curr_index in slice_index_range:
            curr_slice = slices[curr_index]
            if len(curr_slice) == 3:
                input_slices.append(inputs[0, :, curr_slice[0], curr_slice[1], curr_slice[2]])
            else:
                input_slices.append(inputs[0, :, curr_slice[0], curr_slice[1]])
        slice_batches.append(torch.stack(input_slices))

    # Perform predictions
    output_rois = list()
    for data in slice_batches:
        seg_prob = predictor(data)  # batched patch segmentation
        output_rois.append(seg_prob.to(device))

    # stitching output image
    output_classes = output_rois[0].shape[1]
    output_shape = [batch_size, output_classes] + list(image_size)

    # Create importance map
    importance_map = compute_importance_map(get_valid_patch_size(image_size, roi_size), mode=mode, device=device)

    # allocate memory to store the full output and the count for overlapping parts
    output_image = torch.zeros(output_shape, dtype=torch.float32, device=device)
    count_map = torch.zeros(output_shape, dtype=torch.float32, device=device)

    for window_id, slice_index in enumerate(range(0, len(slices), sw_batch_size)):
        slice_index_range = range(slice_index, min(slice_index + sw_batch_size, len(slices)))

        # store the result in the proper location of the full output. Apply weights from importance map.
        for curr_index in slice_index_range:
            curr_slice = slices[curr_index]
            if len(curr_slice) == 3:
                output_image[0, :, curr_slice[0], curr_slice[1], curr_slice[2]] += (
                    importance_map * output_rois[window_id][curr_index - slice_index, :]
                )
                count_map[0, :, curr_slice[0], curr_slice[1], curr_slice[2]] += importance_map
            else:
                output_image[0, :, curr_slice[0], curr_slice[1]] += (
                    importance_map * output_rois[window_id][curr_index - slice_index, :]
                )
                count_map[0, :, curr_slice[0], curr_slice[1]] += importance_map

    # account for any overlapping sections
    output_image = output_image / count_map

    return output_image[
           ...,
           pad_size[4]: image_size_[0] + pad_size[4],
           pad_size[2]: image_size_[1] + pad_size[2],
           pad_size[0]: image_size_[2] + pad_size[0]
           ]
