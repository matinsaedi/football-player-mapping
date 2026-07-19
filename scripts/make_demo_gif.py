"""Convert a processed pipeline video into an optimized README GIF."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--start", type=float, default=4.0, help="Start time in seconds")
    parser.add_argument("--duration", type=float, default=8.0, help="GIF duration")
    parser.add_argument("--fps", type=float, default=8.0, help="Output frame rate")
    parser.add_argument("--width", type=int, default=980, help="Output width")
    parser.add_argument("--colors", type=int, default=128, help="Global palette size")
    return parser.parse_args()


def load_frames(args: argparse.Namespace):
    capture = cv2.VideoCapture(str(args.input))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open {args.input}")

    source_fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    start_frame = round(args.start * source_fps)
    end_frame = round((args.start + args.duration) * source_fps)
    frame_step = max(1, round(source_fps / args.fps))
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frames = []
    index = start_frame
    try:
        while index < end_frame:
            ok, frame = capture.read()
            if not ok:
                break
            if (index - start_frame) % frame_step == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                height = round(frame.shape[0] * args.width / frame.shape[1])
                image = Image.fromarray(frame).resize(
                    (args.width, height),
                    Image.Resampling.LANCZOS,
                )
                frames.append(image)
            index += 1
    finally:
        capture.release()

    if not frames:
        raise RuntimeError("No frames were decoded")
    return frames, frame_step / source_fps


def save_gif(frames, frame_duration: float, output: Path, colors: int) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    colors = max(16, min(256, colors))
    adaptive = frames[len(frames) // 2].quantize(
        colors=colors,
        method=Image.Quantize.MEDIANCUT,
    )
    # Reserve exact visualization colors so class markers remain red, blue,
    # and yellow after GIF palette reduction.
    special_colors = [
        (255, 0, 0),
        (0, 0, 255),
        (255, 255, 0),
        (0, 0, 0),
        (255, 255, 255),
    ]
    palette_values = [component for color in special_colors for component in color]
    adaptive_values = adaptive.getpalette() or []
    remaining_colors = 256 - len(special_colors)
    palette_values.extend(adaptive_values[: remaining_colors * 3])
    palette_values.extend([0] * (768 - len(palette_values)))
    palette = Image.new("P", (1, 1))
    palette.putpalette(palette_values)
    quantized = [
        frame.quantize(palette=palette, dither=Image.Dither.FLOYDSTEINBERG)
        for frame in frames
    ]
    quantized[0].save(
        output,
        save_all=True,
        append_images=quantized[1:],
        duration=round(frame_duration * 1000),
        loop=0,
        optimize=True,
        disposal=2,
    )


def main() -> int:
    args = parse_args()
    frames, frame_duration = load_frames(args)
    save_gif(frames, frame_duration, args.output, args.colors)
    print(f"Wrote {len(frames)} frames to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
