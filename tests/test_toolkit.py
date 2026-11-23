import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from weightspaceprogramming import (apply_lora, binary_inputs, channel_norms, compile_affine,
                                   fit_linear_generator, lesson_names, lesson_source,
                                   load_arrays, mlp, neuron, permute_hidden)


class ToolkitTests(unittest.TestCase):
    def test_neuron_and_boolean_tables(self):
        self.assertEqual(neuron([[0], [1], [2]], [3], -1).tolist(), [-1, 2, 5])
        x = binary_inputs(2)
        self.assertEqual(x.tolist(), [[0, 0], [0, 1], [1, 0], [1, 1]])
        self.assertEqual(neuron(x, [1, 1], -1.5, "step").tolist(), [0, 0, 0, 1])
        self.assertEqual(neuron(x, [1, 1], -0.5, "step").tolist(), [0, 1, 1, 1])
        self.assertEqual(neuron([0], [1], activation="step"), 1)

    def test_xor_written_in_weights(self):
        weights = [[[1, 1], [1, 1]], [[1], [-2]]]
        biases = [[-0.5, -1.5], [-0.5]]
        actual = mlp(binary_inputs(), weights, biases, "step", "step")[:, 0]
        np.testing.assert_array_equal(actual, [0, 1, 1, 0])

    def test_exact_compilation_reused_on_new_inputs(self):
        for scale, shift in [(3, 2), (-2.5, 0.3), (0, -4)]:
            w, b = compile_affine(scale, shift)
            inputs = np.array([[-7.0], [0], [2], [13]])
            np.testing.assert_allclose(mlp(inputs, [w], [b]), scale * inputs + shift)

    def test_learned_generator_predicts_unseen_weights(self):
        x = np.array([[-2, -1], [-1, 1], [1, -1], [2, 1]])
        y = np.column_stack([2 * x[:, 0], 0.5 * x[:, 1] - 1])
        generator = fit_linear_generator(x, y)
        np.testing.assert_allclose(generator.generate([3, 2]), [6, 0], atol=1e-12)
        np.testing.assert_allclose(generator.generate([[3, 2], [-4, 3]]), [[6, 0], [-8, 0.5]], atol=1e-12)
        tensor_generator = fit_linear_generator(x, y.reshape(4, 1, 2))
        self.assertEqual(tensor_generator.generate([3, 2]).shape, (1, 2))
        with self.assertRaises(ValueError):
            fit_linear_generator([[1], [1]], [[2], [3]])

    def test_lora_matches_factorized_application_and_preserves_base(self):
        rng = np.random.default_rng(7)
        w, a, b = rng.normal(size=(4, 3)), rng.normal(size=(4, 2)), rng.normal(size=(2, 3))
        original = w.copy()
        x = rng.normal(size=(8, 4))
        edited = apply_lora(w, a, b, scale=0.25)
        np.testing.assert_allclose(x @ edited, x @ w + 0.25 * (x @ a) @ b, atol=1e-12)
        np.testing.assert_array_equal(w, original)
        self.assertFalse(np.shares_memory(w, edited))

    def test_channel_norms_against_independent_cell_loop(self):
        rng = np.random.default_rng(5)
        kernel = rng.normal(size=(3 * 4 * 2, 7))
        expected = np.empty((2, 3, 4))
        for channel in range(2):
            for row in range(3):
                for column in range(4):
                    vector = kernel[(row * 4 + column) * 2 + channel]
                    expected[channel, row, column] = sum(float(v) ** 2 for v in vector) ** 0.5
        np.testing.assert_allclose(channel_norms(kernel, (3, 4, 2)), expected, atol=1e-12)
        np.testing.assert_allclose(channel_norms([[3, 4], [0, 2]], (1, 2, 1)), [[[5, 2]]])

    def test_integer_norms_and_overflow_rejection(self):
        np.testing.assert_allclose(channel_norms(np.array([[30000, 40000]], dtype=np.int64), (1, 1, 1)), [[[50000]]])
        with self.assertRaises(ValueError):
            channel_norms([[1e308]], (1, 1, 1))

    def test_hidden_permutation_preserves_composed_network(self):
        rng = np.random.default_rng(19)
        weights = [rng.normal(size=(6, 5)), rng.normal(size=(5, 4)), rng.normal(size=(4, 2))]
        biases = [rng.normal(size=(5,)), rng.normal(size=(4,)), rng.normal(size=(2,))]
        originals = [w.copy() for w in weights]
        inputs = rng.normal(size=(32, 6))
        for activation in ["tanh", "relu", "step"]:
            for layer, permutation in [(0, [3, 0, 4, 1, 2]), (1, [2, 0, 3, 1])]:
                changed, bs = permute_hidden(weights, biases, permutation, layer)
                np.testing.assert_allclose(mlp(inputs, weights, biases, activation), mlp(inputs, changed, bs, activation), atol=1e-12)
                for a, b in zip(originals, weights):
                    np.testing.assert_array_equal(a, b)
        changed, _ = permute_hidden(weights, biases, [3, 0, 4, 1, 2])
        np.testing.assert_allclose(channel_norms(weights[0], (2, 3, 1)), channel_norms(changed[0], (2, 3, 1)), atol=1e-12)

    def test_invalid_shapes_and_nonfinite_inputs(self):
        calls = [
            lambda: neuron([1, 2], [1]),
            lambda: neuron([1], [float("nan")]),
            lambda: neuron([1], [1], activation="sigmoid"),
            lambda: neuron([1j], [1]),
            lambda: binary_inputs(0),
            lambda: binary_inputs(True),
            lambda: mlp([[1]], [[[1, 2]], [[3]]], [[0, 0], [0]]),
            lambda: mlp([[1]], [[[1]]], [[0, 1]]),
            lambda: apply_lora([[1, 2]], [[1]], [[1]]),
            lambda: channel_norms([[1, 2]], (1, 2, 1)),
            lambda: channel_norms([[float("inf")]], (1, 1, 1)),
            lambda: channel_norms([[1]], (1, 1, True)),
            lambda: permute_hidden([[[1, 2]], [[1], [2]]], [[0, 0], [0]], [0, 0]),
            lambda: permute_hidden([[[1]]], [[0]], [0]),
        ]
        for call in calls:
            with self.subTest(call=call):
                with self.assertRaises(ValueError):
                    call()

    def test_npz_reader_rejects_pickled_objects(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "weights.npz"
            np.savez(path, w=np.eye(2))
            np.testing.assert_array_equal(load_arrays(path)["w"], np.eye(2))
            np.savez(path, w=np.array([{"value": 1}], dtype=object))
            with self.assertRaises(ValueError):
                load_arrays(path)

    def test_all_packaged_lessons_execute(self):
        expected = {
            "neuron": "[-1.0, 1.0, 3.0, 5.0]",
            "boolean": "AND: [0, 0, 0, 1]",
            "xor": "XOR: [0, 1, 1, 0]",
            "compile": "outputs: [2.0, 5.0, 14.0]",
            "generator": "outputs: [0.0, 6.0, 12.0]",
            "lora": "edited output: [[3.0, 10.0]]",
            "norms": "same outputs: True",
        }
        self.assertEqual(set(lesson_names()), set(expected))
        for name in lesson_names():
            with self.subTest(lesson=name), contextlib.redirect_stdout(io.StringIO()) as out:
                exec(compile(lesson_source(name), name, "exec"), {"__name__": "__main__"})
                self.assertIn(expected[name], out.getvalue())

    def test_cli_inspects_local_file(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "weights.npz"
            np.savez(path, kernel=np.array([[3, 4], [0, 2]]))
            result = subprocess.run([sys.executable, "-m", "weightspaceprogramming", "inspect", str(path),
                                     "--key", "kernel", "--shape", "1", "2", "1"], capture_output=True, text=True, check=True)
            self.assertEqual(json.loads(result.stdout)["norm_maps_chw"], [[[5.0, 2.0]]])


if __name__ == "__main__":
    unittest.main()
