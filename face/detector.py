"""Local face detection helpers backed by OpenCV YuNet."""

from pathlib import Path
from typing import Any

import cv2
import numpy as np


class FaceDetectionError(ValueError):
    """Raised when an image cannot be loaded or does not contain a face."""


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DETECTOR_MODEL = PROJECT_ROOT / "models" / "face_detection_yunet_2023mar.onnx"


def load_image(image_path: str | Path):
    """Load and normalize an image through an in-memory PNG buffer."""
    path = Path(image_path)
    if not path.is_file():
        raise FaceDetectionError(f"Image not found: {path}")

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        try:
            encoded = path.read_bytes()
            image = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
        except (OSError, cv2.error):
            image = None
    if image is None:
        raise FaceDetectionError(f"Invalid or unsupported image: {path}")

    # A PNG round trip makes WebP and other decoder-specific formats consistent.
    success, png_buffer = cv2.imencode(".png", image)
    if success:
        normalized = cv2.imdecode(png_buffer, cv2.IMREAD_COLOR)
        if normalized is not None:
            return normalized
    return image


def _detector_attempts(image):
    """Yield the original, enlarged, and padded variants used for retry."""
    height, width = image.shape[:2]
    yield image, 0.9
    if min(height, width) < 900:
        yield cv2.resize(image, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC), 0.75
    padding_y = max(12, int(height * 0.15))
    padding_x = max(12, int(width * 0.15))
    yield cv2.copyMakeBorder(
        image,
        padding_y,
        padding_y,
        padding_x,
        padding_x,
        cv2.BORDER_REPLICATE,
    ), 0.75


def detect_raw_faces(
    image_path: str | Path,
    model_path: str | Path | None = None,
):
    """Return the processed image and raw detections from the first successful attempt."""
    path = Path(image_path)
    detector_model = Path(model_path) if model_path else DEFAULT_DETECTOR_MODEL
    if not detector_model.is_file() and not Path(model_path or "").is_file():
        raise FaceDetectionError(
            f"Face detector model not found: {detector_model}. "
            "Download face_detection_yunet_2023mar.onnx into models/."
        )

    image = load_image(path)
    last_error = None
    for attempt, score_threshold in _detector_attempts(image):
        try:
            detector = cv2.FaceDetectorYN.create(
                str(detector_model), "", (attempt.shape[1], attempt.shape[0]), score_threshold, 0.3, 5000
            )
            _, detections = detector.detect(attempt)
            if detections is not None:
                return attempt, detections
        except cv2.error as error:
            last_error = error
    if last_error is not None:
        raise FaceDetectionError(f"Could not detect a face in '{path}': {last_error}") from last_error
    raise FaceDetectionError(f"No face detected in '{path}'")


def detect_faces(
    image_path: str | Path,
    model_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Detect faces in an image and return bounding boxes and confidence.

    The YuNet model is loaded locally and the image is never uploaded.
    """
    path = Path(image_path)
    _, detections = detect_raw_faces(path, model_path)

    return [
        {
            "facial_area": {
                "x": int(face[0]),
                "y": int(face[1]),
                "w": int(face[2]),
                "h": int(face[3]),
            },
            "confidence": float(face[14]),
        }
        for face in detections
    ]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Detect faces in a local image.")
    parser.add_argument("image", help="Path to the image to scan")
    arguments = parser.parse_args()

    print("Loading image...")
    print("Detecting face...")
    try:
        faces = detect_faces(arguments.image)
    except FaceDetectionError as error:
        print(f"Face detected: NO\nError: {error}")
    else:
        print(f"Face detected: YES\nNumber of faces: {len(faces)}")
        for index, face in enumerate(faces, start=1):
            print(f"Face {index}: {face['facial_area']}")