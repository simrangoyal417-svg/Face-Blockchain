# HH Goa 2026 Task 3: Face Identification

This repository contains Simran's local face-identification module for the HH Goa 2026 blockchain verification pipeline.

## Pipeline

```text
Input image -> face detection -> SFace embedding -> genuine web/social candidates
						 -> best face match -> Pranesh's hash and blockchain verification
```

Vedant's search module should provide candidate records containing at least `image_path`. The face module does not depend on how those images were discovered. Pranesh's blockchain module can consume the selected candidate's URL and metadata after matching.

## Team responsibilities

- Simran: face detection, embeddings, matching, and integration interface
- Vedant: genuine web/social-media search and candidate post discovery
- Pranesh: hashing, blockchain upload, and verification

## Technology

- Python 3.13+
- OpenCV 4.12 with YuNet for face detection
- OpenCV SFace for local face embeddings and comparison
- NumPy for numerical operations

YuNet and SFace are CPU-friendly, avoid a large TensorFlow installation, and keep input images local. An embedding is a numerical representation of facial features; it is compared with another embedding rather than comparing image pixels directly.

## Installation

```bash
python3 -m venv venv
venv/bin/python -m pip install -r requirements.txt
mkdir -p models
curl -L -o models/face_detection_yunet_2023mar.onnx \
	https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
curl -L -o models/face_recognition_sface_2021dec.onnx \
	https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx
```

The model files are intentionally ignored by Git because they are downloaded dependencies, not secrets. Never commit `.env`, API keys, passwords, or private keys.

## Run

Detection only:

```bash
venv/bin/python -m face.detector input/sample.jpg
```

Compare an input against candidate images:

```bash
venv/bin/python main.py input/sample.jpg candidates/person-a.jpg candidates/person-b.jpg
```

The result is JSON containing `best_match`, `score`, `is_match`, `threshold`, and per-candidate results. Candidate arguments may also be dictionaries in Python code:

```python
from face.matcher import find_best_match

candidates = [
		{"url": "https://example.test/post/1", "image_path": "downloads/post-1.jpg", "caption": "..."}
]
result = find_best_match("input/sample.jpg", candidates)
```

## Matching method

SFace embeddings are compared with cosine similarity from 0 to 1. The default threshold is `0.363`, the OpenCV SFace cosine benchmark threshold for its published LFW evaluation. It is configurable because camera quality, pose, lighting, cropping, and the candidate source can change performance:

```bash
venv/bin/python main.py input/sample.jpg candidates/person-a.jpg --threshold 0.45
```

A candidate image containing multiple faces is accepted; each detected face is compared and the strongest similarity represents that candidate. An input image currently uses its first detected face. Missing, invalid, unsupported, no-face, and model errors return clear exceptions or JSON error entries rather than silently claiming a match.

## Tests

```bash
venv/bin/python -m pytest -q
```

The tests cover same-person-style embeddings, different embeddings, threshold configuration, invalid thresholds, and missing images. Real-image accuracy tests require consented sample photos and the two model files.

## Known limitations

- Similarity is not proof of identity; use a human review and the wider task's verification steps.
- Poor lighting, extreme pose, masks, occlusion, and very small faces can reduce accuracy.
- The detector does not identify a person's name and does not validate whether a social post is genuine.
- Model binaries are downloaded locally and are not stored in the repository.
