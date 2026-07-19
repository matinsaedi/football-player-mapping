"""Detect football players, classify them, and map them to a top-down field.

This module is a cleaned version of a 2021 undergraduate computer-vision
project. Detection intentionally follows the original classical approach:
KNN background subtraction, morphology, and connected components. A small
CNN is used only for team/referee classification.
"""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import cv2
import numpy as np


BASE_FRAME_SIZE = (1280, 960)
BASE_MAP_SIZE = (1050, 680)

# Manually selected correspondences from the original project.
SOURCE_POINTS = np.array(
    [
        (144, 166),
        (1136, 116),
        (873, 780),
        (639, 110),
        (673, 200),
        (490, 210),
        (857, 192),
        (660, 162),
        (692, 251),
    ],
    dtype=np.float32,
)

MAP_POINTS = np.array(
    [
        (164, 147),
        (886, 147),
        (525, 676),
        (525, 4),
        (525, 340),
        (430, 340),
        (618, 340),
        (525, 250),
        (525, 430),
    ],
    dtype=np.float32,
)

OPEN_KERNEL = np.array(
    [
        [0, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 0],
        [0, 1, 1, 1, 1, 1, 1, 0],
        [0, 1, 1, 1, 1, 1, 1, 0],
        [0, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 1, 1, 1, 1, 0, 0],
        [0, 0, 1, 1, 1, 1, 0, 0],
        [0, 0, 1, 1, 1, 1, 0, 0],
    ],
    dtype=np.uint8,
)

CLOSE_KERNEL = np.array(
    [
        [0, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 0],
        [0, 1, 1, 1, 1, 1, 1, 0],
        [0, 1, 1, 1, 1, 1, 1, 0],
        [0, 1, 1, 1, 1, 1, 1, 0],
        [0, 1, 1, 1, 1, 1, 1, 0],
        [0, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 1, 1, 1, 1, 0, 0],
        [0, 0, 1, 1, 1, 1, 0, 0],
        [0, 0, 1, 1, 1, 1, 0, 0],
    ],
    dtype=np.uint8,
)

# OpenCV colors are BGR.
CLASS_COLORS: Dict[int, Tuple[int, int, int]] = {
    0: (0, 0, 255),
    1: (255, 0, 0),
    2: (0, 255, 255),
}
CLASS_NAMES = {0: "red", 1: "blue", 2: "referee"}


@dataclass(frozen=True)
class Detection:
    """One connected foreground component."""

    bbox: Tuple[int, int, int, int]
    foot: Tuple[float, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Input match video")
    parser.add_argument(
        "--field-map",
        type=Path,
        default=Path("assets/field-map.png"),
        help="Top-down field image",
    )
    parser.add_argument(
        "--model-json",
        type=Path,
        default=Path("models/model.json"),
        help="Saved Keras architecture",
    )
    parser.add_argument(
        "--model-weights",
        type=Path,
        default=Path("models/model.h5"),
        help="Saved Keras weights",
    )
    parser.add_argument("--output", type=Path, help="Optional output MP4")
    parser.add_argument(
        "--output-view",
        choices=("combined", "map"),
        default="combined",
        help="Write the annotated frame and map together, or the map alone",
    )
    parser.add_argument("--display", action="store_true", help="Show live OpenCV windows")
    parser.add_argument(
        "--detection-only",
        action="store_true",
        help="Skip TensorFlow and draw every accepted component in red",
    )
    parser.add_argument(
        "--disable-upper-mask",
        action="store_true",
        help="Do not mask the moving upper edge of the original scene",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        help="Stop early after this many frames (useful for smoke tests)",
    )
    args = parser.parse_args()
    if not args.display and args.output is None:
        parser.error("select --display, --output, or both")
    if args.max_frames is not None and args.max_frames <= 0:
        parser.error("--max-frames must be positive")
    return args


def load_classifier(model_json: Path, model_weights: Path):
    """Load the original Keras architecture and weights lazily."""

    from tensorflow.keras.models import model_from_json

    architecture = model_json.read_text(encoding="utf-8")
    model = model_from_json(architecture)
    model.load_weights(str(model_weights))
    return model


def _scale_points(
    points: np.ndarray,
    from_size: Tuple[int, int],
    to_size: Tuple[int, int],
) -> np.ndarray:
    scaled = points.copy()
    scaled[:, 0] *= to_size[0] / from_size[0]
    scaled[:, 1] *= to_size[1] / from_size[1]
    return scaled


def build_homography(
    frame_size: Tuple[int, int], map_size: Tuple[int, int]
) -> np.ndarray:
    """Estimate the original nine-point homography at the current resolutions."""

    source = _scale_points(SOURCE_POINTS, BASE_FRAME_SIZE, frame_size)
    target = _scale_points(MAP_POINTS, BASE_MAP_SIZE, map_size)
    homography, _ = cv2.findHomography(source, target, cv2.RANSAC, 5.0)
    if homography is None:
        raise RuntimeError("OpenCV could not estimate the field homography")
    return homography


def make_foreground_mask(
    frame: np.ndarray,
    subtractor,
    mask_upper_edge: bool,
) -> np.ndarray:
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)

    if mask_upper_edge:
        height, width = blurred.shape[:2]
        start = (0, round(120 * height / BASE_FRAME_SIZE[1]))
        end = (
            width,
            round(57 * height / BASE_FRAME_SIZE[1]),
        )
        thickness = max(1, round(40 * height / BASE_FRAME_SIZE[1]))
        cv2.line(blurred, start, end, color=(0, 0, 0), thickness=thickness)

    foreground = subtractor.apply(blurred)
    _, foreground = cv2.threshold(foreground, 128, 255, cv2.THRESH_BINARY)
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, OPEN_KERNEL)
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, CLOSE_KERNEL)
    return foreground


def _minimum_component_area(top: int, frame_size: Tuple[int, int]) -> float:
    """Apply the original perspective-aware area thresholds."""

    width, height = frame_size
    scale = (width * height) / (BASE_FRAME_SIZE[0] * BASE_FRAME_SIZE[1])
    if top <= 0.25 * height:
        return 20 * scale
    if top <= 0.50 * height:
        return 320 * scale
    return 1600 * scale


def detect_components(mask: np.ndarray) -> Iterable[Detection]:
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    frame_size = (mask.shape[1], mask.shape[0])

    for index in range(1, count):
        x = int(stats[index, cv2.CC_STAT_LEFT])
        y = int(stats[index, cv2.CC_STAT_TOP])
        width = int(stats[index, cv2.CC_STAT_WIDTH])
        height = int(stats[index, cv2.CC_STAT_HEIGHT])
        area = int(stats[index, cv2.CC_STAT_AREA])

        if area <= _minimum_component_area(y, frame_size):
            continue
        if width <= 0 or height <= 0:
            continue

        # A person's contact point with the field is better represented by the
        # bottom-center of the component than by its visual centroid.
        foot = (x + width / 2.0, y + height)
        yield Detection(bbox=(x, y, width, height), foot=foot)


def classify_crop(model, frame: np.ndarray, detection: Detection) -> int:
    x, y, width, height = detection.bbox
    crop = frame[y : y + height, x : x + width]
    if crop.size == 0:
        return 0
    crop = cv2.resize(crop, (45, 45))
    batch = np.expand_dims(crop, axis=0)
    try:
        prediction = model.predict(batch, verbose=0)
    except TypeError:  # Compatibility with older Keras releases.
        prediction = model.predict(batch)
    return int(np.argmax(prediction[0]))


def project_to_map(point: Tuple[float, float], homography: np.ndarray) -> Tuple[int, int]:
    source = np.array(point, dtype=np.float32).reshape(1, 1, 2)
    target = cv2.perspectiveTransform(source, homography).reshape(2)
    return int(round(float(target[0]))), int(round(float(target[1])))


def combined_view(frame: np.ndarray, field: np.ndarray) -> np.ndarray:
    """Place the annotated camera frame and field map side by side."""

    target_height = field.shape[0]
    target_width = int(round(frame.shape[1] * target_height / frame.shape[0]))
    if target_width % 2:
        target_width += 1
    camera = cv2.resize(frame, (target_width, target_height))
    return np.hstack((camera, field))


def _opencv_safe_output_path(path: Path) -> Path:
    """Return an ASCII temporary path when OpenCV cannot handle the destination.

    Some Windows OpenCV video backends fail on otherwise valid Unicode paths.
    The completed temporary video is moved to the requested path after release.
    """

    absolute = path.resolve()
    try:
        str(absolute).encode("ascii")
        return absolute
    except UnicodeEncodeError:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix="football-player-mapping-",
            suffix=path.suffix or ".mp4",
        )
        os.close(descriptor)
        return Path(temporary_name)


def open_writer(path: Path, fps: float, frame: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    writer_path = _opencv_safe_output_path(path)
    codec = "MJPG" if path.suffix.lower() == ".avi" else "mp4v"
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(
        str(writer_path),
        fourcc,
        fps,
        (frame.shape[1], frame.shape[0]),
    )
    if not writer.isOpened():
        writer_path.unlink(missing_ok=True)
        raise RuntimeError(f"Could not open output video: {path}")
    return writer, writer_path


def run(args: argparse.Namespace) -> int:
    capture = cv2.VideoCapture(str(args.input))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open input video: {args.input}")

    field_source = cv2.imread(str(args.field_map))
    if field_source is None:
        raise FileNotFoundError(f"Could not read field map: {args.field_map}")
    field_source = cv2.resize(field_source, BASE_MAP_SIZE)

    model = None
    if not args.detection_only:
        model = load_classifier(args.model_json, args.model_weights)

    subtractor = cv2.createBackgroundSubtractorKNN()
    writer = None
    writer_path: Optional[Path] = None
    processed = 0

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            if processed == 0:
                frame_size = (frame.shape[1], frame.shape[0])
                map_size = (field_source.shape[1], field_source.shape[0])
                homography = build_homography(frame_size, map_size)

            mask = make_foreground_mask(
                frame,
                subtractor,
                mask_upper_edge=not args.disable_upper_mask,
            )
            annotated = frame.copy()
            field = field_source.copy()
            class_counts = {name: 0 for name in CLASS_NAMES.values()}

            for detection in detect_components(mask):
                class_id = 0 if model is None else classify_crop(model, frame, detection)
                color = CLASS_COLORS.get(class_id, CLASS_COLORS[0])
                class_name = CLASS_NAMES.get(class_id, "unknown")
                class_counts[class_name] = class_counts.get(class_name, 0) + 1

                x, y, width, height = detection.bbox
                cv2.rectangle(annotated, (x, y), (x + width, y + height), color, 2)
                map_x, map_y = project_to_map(detection.foot, homography)
                if 0 <= map_x < field.shape[1] and 0 <= map_y < field.shape[0]:
                    cv2.circle(field, (map_x, map_y), 7, color, thickness=-1)

            status = "  ".join(f"{key}: {value}" for key, value in class_counts.items())
            cv2.putText(
                field,
                status,
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (20, 20, 20),
                2,
                cv2.LINE_AA,
            )

            output_frame = field if args.output_view == "map" else combined_view(annotated, field)
            if args.output is not None:
                if writer is None:
                    fps = capture.get(cv2.CAP_PROP_FPS)
                    if not fps or fps <= 0:
                        fps = 30.0
                    writer, writer_path = open_writer(args.output, fps, output_frame)
                writer.write(output_frame)

            processed += 1
            if args.display:
                cv2.imshow("Annotated frame", annotated)
                cv2.imshow("Foreground mask", mask)
                cv2.imshow("Top-down map", field)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break

            if args.max_frames is not None and processed >= args.max_frames:
                break
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if args.output is not None and writer_path is not None:
            requested_path = args.output.resolve()
            if writer_path.resolve() != requested_path:
                requested_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(writer_path), str(requested_path))
        if args.display:
            cv2.destroyAllWindows()

    print(f"Processed {processed} frames")
    if args.output is not None:
        print(f"Wrote {args.output}")
    return 0


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
