"""Learn weight-space programming with small NumPy experiments."""
from .core import (LinearWeightGenerator, apply_lora, binary_inputs, compile_affine,
                   fit_linear_generator, mlp, neuron, permute_hidden)
from .inspection import channel_norms, load_arrays, plot_channel_norms
from .lessons import lesson_names, lesson_source

__version__ = "0.1.0.dev1"
__all__ = ["LinearWeightGenerator", "apply_lora", "binary_inputs", "compile_affine",
           "fit_linear_generator", "mlp", "neuron", "permute_hidden", "channel_norms",
           "load_arrays", "plot_channel_norms", "lesson_names", "lesson_source"]
