"""Small networks with row-vector inputs: y = x @ W + b."""
from dataclasses import dataclass

import numpy as np


def _array(value, name):
    result = np.asarray(value)
    if result.dtype.kind not in "biuf" or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain finite real numbers")
    return result.astype(np.float64)


def _activate(x, activation):
    if activation == "linear":
        return x
    if activation == "step":
        return (x >= 0).astype(np.int64)
    if activation == "relu":
        return np.maximum(x, 0)
    if activation == "tanh":
        return np.tanh(x)
    raise ValueError("activation must be linear, step, relu or tanh")


def neuron(x, weights, bias=0.0, activation="linear"):
    """Evaluate a neuron; the final input axis holds its features.

    A step activation returns 1 at zero and above, and 0 below zero.
    """
    x, weights, bias = _array(x, "x"), _array(weights, "weights"), _array(bias, "bias")
    if weights.ndim != 1 or weights.size == 0:
        raise ValueError("weights must be a nonempty vector")
    if x.ndim == 0 or x.shape[-1] != weights.size:
        raise ValueError("the final input dimension must match the weights")
    if bias.ndim != 0:
        raise ValueError("bias must be a scalar")
    return _activate(x @ weights + bias, activation)


def binary_inputs(n_features=2):
    """All binary inputs, ordered from all zeros to all ones (1–16 features)."""
    if type(n_features) is not int or not 1 <= n_features <= 16:
        raise ValueError("n_features must be an integer between 1 and 16")
    rows = np.arange(2 ** n_features, dtype=np.int64)[:, None]
    return (rows >> np.arange(n_features - 1, -1, -1)) & 1


def _layers(weights, biases):
    if len(weights) == 0 or len(weights) != len(biases):
        raise ValueError("supply one bias vector for each nonempty weight layer")
    ws, bs = [], []
    for index, (weight, bias) in enumerate(zip(weights, biases)):
        w, b = _array(weight, "weights"), _array(bias, "biases")
        if w.ndim != 2 or 0 in w.shape:
            raise ValueError("each weight matrix must have shape (inputs, outputs)")
        if b.shape != (w.shape[1],):
            raise ValueError("each bias must have one element per output")
        if index and ws[-1].shape[1] != w.shape[0]:
            raise ValueError("adjacent weight matrices must have matching widths")
        ws.append(w)
        bs.append(b)
    return ws, bs


def mlp(x, weights, biases, activation="tanh", output_activation="linear"):
    """Run dense layers, applying activation between layers and output_activation last."""
    ws, bs = _layers(weights, biases)
    output = _array(x, "x")
    if output.ndim == 0 or output.shape[-1] != ws[0].shape[0]:
        raise ValueError("the final input dimension must match the first layer")
    _activate(np.array(0.0), activation)
    for index, (weight, bias) in enumerate(zip(ws, bs)):
        output = _activate(output @ weight + bias,
                           output_activation if index == len(ws) - 1 else activation)
    return output


def compile_affine(scale, shift):
    """Construct W and b for the exact one-input program f(x) = scale*x + shift."""
    scale, shift = _array(scale, "scale"), _array(shift, "shift")
    if scale.ndim or shift.ndim:
        raise ValueError("scale and shift must be scalars")
    return np.array([[float(scale)]]), np.array([float(shift)])


@dataclass
class LinearWeightGenerator:
    """A fitted affine map from numerical descriptors to target weight arrays."""
    coefficients: np.ndarray
    weight_shape: tuple

    def generate(self, descriptor):
        """Generate weights for one descriptor or a batch of descriptors."""
        x = _array(descriptor, "descriptor")
        if x.ndim not in (1, 2) or x.shape[-1] != self.coefficients.shape[0] - 1:
            raise ValueError("descriptor width must match the generator's training inputs")
        design = np.concatenate([x, np.ones((*x.shape[:-1], 1))], axis=-1)
        return (design @ self.coefficients).reshape((*x.shape[:-1], *self.weight_shape))


def fit_linear_generator(descriptors, target_weights):
    """Fit descriptor→weights by least squares, with a learned constant offset."""
    x, y = _array(descriptors, "descriptors"), _array(target_weights, "target_weights")
    if x.ndim != 2 or 0 in x.shape or y.ndim < 2 or x.shape[0] != y.shape[0] or 0 in y.shape:
        raise ValueError("supply nonempty descriptor and target batches with the same row count")
    design = np.column_stack([x, np.ones(len(x))])
    if np.linalg.matrix_rank(design) < design.shape[1]:
        raise ValueError("descriptors must identify each input dimension and the constant offset")
    coefficients = np.linalg.lstsq(design, y.reshape(len(y), -1), rcond=None)[0]
    return LinearWeightGenerator(coefficients, y.shape[1:])


def apply_lora(weights, a, b, scale=1.0):
    """Return W + scale*A@B; W is (in,out), A (in,rank), B (rank,out).

    The input arrays are unchanged. This convention matches row-vector x @ W.
    """
    w, a, b, scale = (_array(weights, "weights"), _array(a, "a"),
                       _array(b, "b"), _array(scale, "scale"))
    if w.ndim != 2 or a.ndim != 2 or b.ndim != 2 or 0 in w.shape or 0 in a.shape or 0 in b.shape:
        raise ValueError("weights, a and b must be nonempty matrices")
    if a.shape[0] != w.shape[0] or b.shape[1] != w.shape[1] or a.shape[1] != b.shape[0]:
        raise ValueError("LoRA factors must have shapes (inputs, rank) and (rank, outputs)")
    if scale.ndim:
        raise ValueError("scale must be a scalar")
    return w + float(scale) * (a @ b)


def permute_hidden(weights, biases, permutation, layer=0):
    """Reorder one hidden layer and its connections; return copied parameters."""
    ws, bs = _layers(weights, biases)
    if type(layer) is not int or not 0 <= layer < len(ws) - 1:
        raise ValueError("layer must name a hidden layer with a following weight matrix")
    p = np.asarray(permutation)
    width = ws[layer].shape[1]
    if p.ndim != 1 or p.dtype.kind not in "iu" or not np.array_equal(np.sort(p), np.arange(width)):
        raise ValueError("permutation must contain every hidden-unit index exactly once")
    ws, bs = [w.copy() for w in ws], [b.copy() for b in bs]
    ws[layer] = ws[layer][:, p]
    bs[layer] = bs[layer][p]
    ws[layer + 1] = ws[layer + 1][p, :]
    return ws, bs
