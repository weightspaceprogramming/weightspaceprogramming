# weightspaceprogramming

Learn weight-space programming by changing small networks and seeing what they compute. Run seven NumPy lessons, inspect NPZ weights, generate channel heatmaps, and check that a consistent hidden-unit permutation preserves a network's output.

[Interactive guide](https://weightspaceprogramming.tech/learn/) · [PyPI](https://pypi.org/project/weightspaceprogramming/) · [Examples](https://github.com/weightspaceprogramming/examples)

## Install

```bash
python -m pip install weightspaceprogramming
weightspaceprogramming lesson list
weightspaceprogramming lesson lora
```

Python 3.10+ and NumPy. The lessons run locally on CPU without accounts, API keys or model downloads.

## Write a function into weights

```python
from weightspaceprogramming import binary_inputs, neuron

x = binary_inputs(2)
print(neuron(x, [1, 1], bias=-1.5, activation="step").tolist())
# [0, 0, 0, 1] — AND

print(neuron(x, [1, 1], bias=-0.5, activation="step").tolist())
# [0, 1, 1, 1] — OR
```

The same inputs produce a different function after changing one bias. A step neuron returns 1 when its preactivation is zero or positive.

## Lessons

- `neuron`: change the slope and offset of a linear neuron.
- `boolean`: encode AND and OR in threshold-neuron weights.
- `xor`: construct a two-layer XOR network.
- `compile`: construct an affine program's weights, then reuse them on new inputs.
- `generator`: fit a numerical descriptor-to-weights map by least squares and generate a new neuron program.
- `lora`: apply a rank-one update and compare the original and edited outputs.
- `norms`: inspect a synthetic input-layer kernel and permute hidden units while preserving its function.

```bash
weightspaceprogramming lesson xor
weightspaceprogramming lesson xor --show-code
```

The Python API exposes the exact lesson source for notebooks and browser runners:

```python
from weightspaceprogramming import lesson_source
print(lesson_source("xor"))
```

## Apply a LoRA update

The package uses row-vector inputs: `output = input @ W`. For a weight matrix with shape `(inputs, outputs)`, the low-rank factors have shapes `(inputs, rank)` and `(rank, outputs)`.

```python
import numpy as np
from weightspaceprogramming import apply_lora

w = np.eye(2)
edited = apply_lora(w, [[1.0], [0.0]], [[0.0, 2.0]])
x = np.array([[3.0, 4.0]])
print((x @ w).tolist())       # [[3.0, 4.0]]
print((x @ edited).tolist())  # [[3.0, 10.0]]
```

`apply_lora` returns `W + scale * A @ B` and leaves the supplied arrays unchanged.

## Read input-channel weights

For an HWC input grid, `channel_norms` computes the L2 norm across the first layer's hidden units at every cell and channel. It returns an array with shape `(channels, height, width)`.

```python
from weightspaceprogramming import channel_norms

print(channel_norms([[3, 4], [0, 2]], (1, 2, 1)).tolist())
# [[[5.0, 2.0]]]
```

Inspect your own NPZ checkpoint:

```bash
weightspaceprogramming inspect network.npz \
  --key actor/dense1/kernel --shape 10 10 5 --output norms.json
```

NPZ files load with pickle disabled. The kernel must be a finite real matrix with one row per flattened HWC input and one column per hidden unit.

Add plotting when needed:

```bash
python -m pip install 'weightspaceprogramming[plot]'
weightspaceprogramming inspect network.npz \
  --key actor/dense1/kernel --shape 10 10 5 --plot heatmaps.svg
```

`permute_hidden(weights, biases, permutation, layer=0)` copies the parameters and reorders a hidden layer's incoming columns, biases and outgoing rows together. `mlp` evaluates the original and reordered networks with the same activation functions.

## From small networks to research implementations

[Program-as-Weights](https://weightspaceprogramming.tech/research/program-as-weights/) introduces a programming paradigm for fuzzy functions, a compiler–interpreter interface and FuzzyBench. PAW specifically trains a compiler to generate LoRA adapters that edit a frozen interpreter's weights from a natural-language function specification. [Run the PAW urgency and routing examples](https://github.com/weightspaceprogramming/examples#2-compile-and-reuse-an-email-urgency-function-with-paw) with the official SDK.

[Tracr](https://github.com/weightspaceprogramming/examples#1-compile-a-token-counter-with-tracr) compiles exact RASP programs into transformer weights. The [Sparks weight-reading experiment](https://weightspaceprogramming.tech/build/code-and-implementations/#weight-reading) applies the channel-norm computation to four published game policies and reports their permutation checks. PAW and Tracr keep their own installation instructions and dependencies.

Explore the [field guide](https://weightspaceprogramming.tech/) and [code and implementations](https://weightspaceprogramming.tech/build/code-and-implementations/).

## Development

Get the [source code](https://github.com/weightspaceprogramming/weightspaceprogramming) or [report an issue](https://github.com/weightspaceprogramming/weightspaceprogramming/issues).

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
```

Implementation provenance ships with the distribution in `SOURCES.md`. Package code is licensed under Apache-2.0.
