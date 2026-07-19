# Tracking experiment

`tracking_experiment.ipynb` is the preserved optional-bonus experiment from 2021. It performs foreground detection on every tenth frame and draws a connection to the nearest detection from the previous processed frame in top-down map coordinates.

Known issues:

- It skips intermediate frames instead of propagating bounding boxes through them.
- Connected-component statistics and centroid indices are offset by one.
- Nearest-neighbor distances were computed with integer arrays and produced overflow warnings in the saved run.
- It does not maintain persistent identities or resolve crossings and occlusions.

The notebook remains unchanged so the historical work is not confused with the cleaned pipeline.
