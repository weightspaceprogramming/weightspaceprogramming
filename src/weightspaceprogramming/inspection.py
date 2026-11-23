"""Inspect finite real NPZ weights and visualize input-channel magnitudes."""
from pathlib import Path

import numpy as np


def channel_norms(kernel, grid_shape):
    """L2 norms over hidden units, with HWC inputs; result has shape C,H,W."""
    if len(grid_shape) != 3 or any(type(n) is not int or n <= 0 for n in grid_shape):
        raise ValueError("grid_shape must contain three positive integers: height, width, channels")
    values = np.asarray(kernel)
    if values.ndim != 2 or values.shape[1] == 0:
        raise ValueError("Kernel must have shape (input features, hidden units)")
    if values.dtype.kind not in "fiu" or not np.isfinite(values).all():
        raise ValueError("Kernel must contain finite real numbers")
    if values.shape[0] != np.prod(grid_shape):
        raise ValueError("Kernel input dimension does not equal height * width * channels")
    try:
        with np.errstate(over="raise", invalid="raise"):
            values = values.astype(np.float64)
            shaped = values.reshape(*grid_shape, values.shape[1])
            return np.sqrt(np.sum(shaped * shaped, axis=-1)).transpose(2, 0, 1)
    except FloatingPointError as error:
        raise ValueError("Squared kernel magnitudes exceed float64 range") from error


def load_arrays(path):
    """Read an NPZ checkpoint without pickle deserialization."""
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def plot_channel_norms(norms, labels=None, output=None):
    """Draw channel maps on one shared color scale; requires the plot extra.

    Return the matplotlib Figure. If output is provided, save it to that path.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise ImportError('Install plotting with pip install "weightspaceprogramming[plot]"') from error
    maps = np.asarray(norms)
    if maps.ndim != 3 or 0 in maps.shape or maps.dtype.kind not in "fiu" or not np.isfinite(maps).all() or (maps < 0).any():
        raise ValueError("norms must be a nonempty, finite, nonnegative (channels,height,width) array")
    labels = list(labels) if labels is not None else [f"Channel {i}" for i in range(len(maps))]
    if len(labels) != len(maps):
        raise ValueError("supply one label per channel")
    figure, axes = plt.subplots(1, len(maps), squeeze=False, figsize=(3 * len(maps), 3))
    for axis, values, label in zip(axes[0], maps, labels):
        image = axis.imshow(values, vmin=0, vmax=float(maps.max()) or 1.0, cmap="viridis", interpolation="nearest")
        axis.set_title(label)
    figure.colorbar(image, ax=list(axes[0]), label="L2 norm across hidden units", shrink=0.7)
    if output is not None:
        figure.savefig(Path(output), bbox_inches="tight")
    return figure
