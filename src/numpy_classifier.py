"""Lightweight NumPy inference for the preserved Keras 2.3.1 classifier.

The historical network uses only valid convolutions, ReLU, max pooling, dense
layers, and softmax. Reimplementing those inference operations lets the saved
weights run on current Python versions without installing legacy TensorFlow.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


class NumpyClassifier:
    """Run the original three-class CNN directly from its HDF5 weights."""

    def __init__(self, weights_path: Path):
        try:
            import h5py
        except ImportError as exc:
            raise ImportError(
                "NumPy inference requires h5py; install requirements.txt"
            ) from exc

        with h5py.File(str(weights_path), "r") as handle:
            self.conv1_kernel = handle["conv1/conv1_2/kernel:0"][()].astype(np.float32)
            self.conv1_bias = handle["conv1/conv1_2/bias:0"][()].astype(np.float32)
            self.conv2_kernel = handle["conv2/conv2_2/kernel:0"][()].astype(np.float32)
            self.conv2_bias = handle["conv2/conv2_2/bias:0"][()].astype(np.float32)
            self.fc1_kernel = handle["fc_1/fc_1_2/kernel:0"][()].astype(np.float32)
            self.fc1_bias = handle["fc_1/fc_1_2/bias:0"][()].astype(np.float32)
            self.fc2_kernel = handle["fc_2/fc_2_2/kernel:0"][()].astype(np.float32)
            self.fc2_bias = handle["fc_2/fc_2_2/bias:0"][()].astype(np.float32)
            self.output_kernel = handle["softmax/softmax_2/kernel:0"][()].astype(
                np.float32
            )
            self.output_bias = handle["softmax/softmax_2/bias:0"][()].astype(np.float32)

    @staticmethod
    def _conv2d(inputs: np.ndarray, kernel: np.ndarray, bias: np.ndarray) -> np.ndarray:
        kernel_height, kernel_width = kernel.shape[:2]
        windows = sliding_window_view(
            inputs,
            (kernel_height, kernel_width),
            axis=(0, 1),
        )
        # sliding_window_view returns H, W, C, KH, KW for a channels-last input.
        windows = windows.transpose(0, 1, 3, 4, 2)
        outputs = np.tensordot(
            windows,
            kernel,
            axes=((2, 3, 4), (0, 1, 2)),
        )
        return outputs + bias

    @staticmethod
    def _max_pool2d(inputs: np.ndarray) -> np.ndarray:
        windows = sliding_window_view(inputs, (2, 2), axis=(0, 1))
        return windows[::2, ::2].max(axis=(-2, -1))

    @staticmethod
    def _relu(inputs: np.ndarray) -> np.ndarray:
        return np.maximum(inputs, 0)

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        shifted = logits - np.max(logits)
        exponentials = np.exp(shifted)
        return exponentials / exponentials.sum()

    def _forward(self, image: np.ndarray) -> np.ndarray:
        values = np.asarray(image, dtype=np.float32)
        if values.shape != (45, 45, 3):
            raise ValueError(f"Expected one 45 x 45 x 3 crop, got {values.shape}")

        values = self._relu(self._conv2d(values, self.conv1_kernel, self.conv1_bias))
        values = self._max_pool2d(values)
        values = self._relu(self._conv2d(values, self.conv2_kernel, self.conv2_bias))
        values = self._max_pool2d(values)
        values = values.reshape(-1)
        values = self._relu(values @ self.fc1_kernel + self.fc1_bias)
        values = self._relu(values @ self.fc2_kernel + self.fc2_bias)
        logits = values @ self.output_kernel + self.output_bias
        return self._softmax(logits)

    def predict(self, batch: np.ndarray, verbose: int = 0) -> np.ndarray:
        """Match the small subset of the Keras predict API used by the pipeline."""

        del verbose
        images = np.asarray(batch)
        if images.ndim != 4:
            raise ValueError(f"Expected a batch of images, got shape {images.shape}")
        return np.stack([self._forward(image) for image in images])
