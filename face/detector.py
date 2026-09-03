"""Local face detection helpers backed by OpenCV YuNet."""

from pathlib import Path
from typing import Any

import cv2


class FaceDetectionError(ValueError):
    """Raised when an image cannot be loaded or does not contain a face."""


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DETECTOR_MODEL = PROJECT_ROOT / "models" / "face_detection_yunet_2023mar.onnx"


def detect_faces(
    image_path: str | Path,
    model_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Detect faces in an image and return bounding boxes and confidence.

    The YuNet model is loaded locally and the image is never uploaded.
    """
    path = Path(image_path)
    if not path.is_file():
        raise FaceDetectionError(f"Image not found: {path}")

    detector_model = Path(model_path) if model_path else DEFAULT_DETECTOR_MODEL
    if not detector_model.is_file() and not Path(model_path or "").is_file():
        raise FaceDetectionError(
            f"Face detector model not found: {detector_model}. "
            "Download face_detection_yunet_2023mar.onnx into models/."
        )

    image = cv2.imread(str(path))
    if image is None:
        raise FaceDetectionError(f"Invalid or unsupported image: {path}")

    try:
        detector = cv2.FaceDetectorYN.create(
            str(detector_model), "", (image.shape[1], image.shape[0]), 0.9, 0.3, 5000
        )
        _, detections = detector.detect(image)
    except cv2.error as error:
        raise FaceDetectionError(f"Could not detect a face in '{path}': {error}") from error

    if detections is None:
        raise FaceDetectionError(f"No face detected in '{path}'")

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