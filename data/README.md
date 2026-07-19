# Dataset

The image dataset is intentionally not committed. It consists of player/referee crops extracted from the fixed-camera match footage and may not be redistributable.

To retrain the classifier, place or link the data locally using this structure:

```text
Dataset/
├── training_set/
│   ├── Red/
│   ├── Blue/
│   └── Referee/
└── test_set/
    ├── Red_test/
    ├── Blue_test/
    └── Referee_test/
```

Every labeled image in the preserved dataset is a 45 x 45 RGB JPEG. OpenCV reads these as BGR arrays, which is the convention used by both the historical training code and the cleaned pipeline.

Preserved counts:

| Split | Red | Blue | Referee | Total |
|---|---:|---:|---:|---:|
| Training | 2,503 | 1,173 | 714 | 4,390 |
| Test | 432 | 298 | 91 | 821 |

The original directory also contains 5,112 images in `other`, `other2`, and `other3`. These appear to be unlabeled or rejected connected-component crops and are not consumed by the training program.

Run training from the repository root with:

```bash
python src/train_classifier.py --data-root path/to/Dataset
```
