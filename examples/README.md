# Example input

The original pipeline was calibrated against a 1280 x 960 fixed-camera football clip at approximately 30 frames per second. Match footage is not included because its redistribution rights are unknown.

To reproduce the preserved setup locally, copy the original 24-second `output.mp4` into this directory as `input.mp4`, then run:

```bash
python src/pipeline.py \
  --input examples/input.mp4 \
  --output artifacts/mapped-output.mp4
```

Despite its historical filename, `output.mp4` was the pipeline's input clip, not a generated result.
