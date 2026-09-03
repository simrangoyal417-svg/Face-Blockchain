"""Face embedding comparison and candidate ranking."""

from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .encoder import FaceEncodingError, encode_faces


DEFAULT_COSINE_THRESHOLD = 0.363


def compare_faces(
    face1: np.ndarray,
    face2: np.ndarray,
    threshold: float = DEFAULT_COSINE_THRESHOLD,
) -> dict[str, Any]:
    """Compare two SFace embeddings using cosine similarity."""
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")
    first_norm = np.linalg.norm(face1)
    second_norm = np.linalg.norm(face2)
    if first_norm == 0 or second_norm == 0:
        raise ValueError("face embeddings must not be zero vectors")
    similarity = float(np.dot(face1.ravel(), face2.ravel()) / (first_norm * second_norm))
    return {
        "similarity": similarity,
        "score_percent": round(max(0.0, similarity) * 100, 2),
        "threshold": threshold,
        "is_match": similarity >= threshold,
    }


def find_best_match(
    input_image: str | Path,
    candidate_images: Iterable[str | Path | dict[str, Any]],
    threshold: float = DEFAULT_COSINE_THRESHOLD,
) -> dict[str, Any]:
    """Rank candidate images and return the strongest match with metadata."""
    input_embedding = encode_faces(input_image)[0]
    results = []
    for candidate in candidate_images:
        path = Path(candidate["image_path"] if isinstance(candidate, dict) else candidate)
        try:
            candidate_embeddings = encode_faces(path)
            comparisons = [compare_faces(input_embedding, embedding, threshold) for embedding in candidate_embeddings]
            best = max(comparisons, key=lambda result: result["similarity"])
            results.append({"candidate": candidate, **best})
        except FaceEncodingError as error:
            results.append({"candidate": candidate, "error": str(error)})
    valid_results = [result for result in results if "similarity" in result]
    if not valid_results:
        raise FaceEncodingError("No candidate image could be encoded.")
    best = max(valid_results, key=lambda result: result["similarity"])
    return {"best_match": best["candidate"], "score": best["score_percent"], "is_match": best["is_match"], "threshold": threshold, "candidates": results}