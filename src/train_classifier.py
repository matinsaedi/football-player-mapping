"""Train the original three-class football-player CNN on local folders."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np
import tensorflow as tf


IMAGE_SIZE = (45, 45)
TRAIN_CLASSES: Dict[str, int] = {"Red": 0, "Blue": 1, "Referee": 2}
TEST_CLASSES: Dict[str, int] = {"Red_test": 0, "Blue_test": 1, "Referee_test": 2}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        required=True,
        type=Path,
        help="Directory containing training_set/ and test_set/",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model-json", type=Path, default=Path("models/model.json"))
    parser.add_argument("--model-weights", type=Path, default=Path("models/model.h5"))
    parser.add_argument(
        "--history",
        type=Path,
        default=Path("artifacts/training-history.json"),
    )
    return parser.parse_args()


def load_split(root: Path, class_map: Dict[str, int]) -> Tuple[np.ndarray, np.ndarray]:
    images = []
    labels = []

    for directory_name, class_id in class_map.items():
        directory = root / directory_name
        paths = sorted(directory.glob("*.jpg"))
        if not paths:
            raise FileNotFoundError(f"No JPEG images found in {directory}")

        for path in paths:
            image = cv2.imread(str(path))
            if image is None:
                raise ValueError(f"Could not read image: {path}")
            if (image.shape[1], image.shape[0]) != IMAGE_SIZE:
                image = cv2.resize(image, IMAGE_SIZE)
            images.append(image)
            labels.append(class_id)

    return np.asarray(images), np.asarray(labels, dtype=np.uint8)


def shuffled(images: np.ndarray, labels: np.ndarray, seed: int):
    order = np.random.default_rng(seed).permutation(len(images))
    return images[order], labels[order]


def build_model() -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(45, 45, 3), name="input")
    x = tf.keras.layers.Conv2D(6, 5, padding="valid", name="conv1")(inputs)
    x = tf.keras.layers.Activation("relu", name="conv1_relu")(x)
    x = tf.keras.layers.MaxPooling2D(2, strides=2, name="pool1")(x)
    x = tf.keras.layers.Conv2D(14, 3, padding="valid", name="conv2")(x)
    x = tf.keras.layers.Activation("relu", name="conv2_relu")(x)
    x = tf.keras.layers.MaxPooling2D(2, strides=2, name="pool2")(x)
    x = tf.keras.layers.Flatten(name="flatten")(x)
    x = tf.keras.layers.Dense(150, activation="relu", name="fc1")(x)
    x = tf.keras.layers.Dense(80, activation="relu", name="fc2")(x)
    outputs = tf.keras.layers.Dense(3, activation="softmax", name="classification")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="football_classifier")


def main() -> int:
    args = parse_args()
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    train_images, train_labels = load_split(
        args.data_root / "training_set", TRAIN_CLASSES
    )
    test_images, test_labels = load_split(args.data_root / "test_set", TEST_CLASSES)
    train_images, train_labels = shuffled(train_images, train_labels, args.seed)
    test_images, test_labels = shuffled(test_images, test_labels, args.seed)

    # The historical model was trained directly on OpenCV BGR byte values.
    # We preserve that convention so newly trained weights match the pipeline.
    train_targets = tf.keras.utils.to_categorical(train_labels, num_classes=3)
    test_targets = tf.keras.utils.to_categorical(test_labels, num_classes=3)

    model = build_model()
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()
    history = model.fit(
        train_images,
        train_targets,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_data=(test_images, test_targets),
    )
    test_loss, test_accuracy = model.evaluate(test_images, test_targets, verbose=0)

    args.model_json.parent.mkdir(parents=True, exist_ok=True)
    args.model_weights.parent.mkdir(parents=True, exist_ok=True)
    args.history.parent.mkdir(parents=True, exist_ok=True)
    args.model_json.write_text(model.to_json(), encoding="utf-8")
    model.save_weights(str(args.model_weights))

    history_payload = {
        key: [float(value) for value in values]
        for key, values in history.history.items()
    }
    history_payload["test_loss"] = float(test_loss)
    history_payload["test_accuracy"] = float(test_accuracy)
    args.history.write_text(
        json.dumps(history_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Saved architecture to {args.model_json}")
    print(f"Saved weights to {args.model_weights}")
    print(f"Saved training history to {args.history}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
